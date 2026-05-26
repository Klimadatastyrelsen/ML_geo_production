# Main script: process_images.py
from osgeo import gdal  # noqa: F401 — before fastai to avoid libtiff/jpeg conflicts
from fastai.vision.all import *
import torch
import numpy as np
import rasterio
import os
import math
import argparse
import json
import geopandas as gpd
from pathlib import Path
import multiprocessing
import threading
import queue
import time
import sys # Import sys for argument parsing adjustment
import geopandas as gpd
import rasterio
from rasterio.errors import RasterioIOError
from rasterio.windows import Window
from rasterio.transform import from_origin
from rasterio.warp import reproject, Resampling

from multi_channel_dataset_creation import geopackage_to_label_im

# Import modules
from ML_geo_production.patch_dataset import get_dataloader
from ML_geo_production.image_utils import load_central_window, load_dummy_mask
from ML_geo_production.model_utils import create_dummy_dls, load_unet_from_state, preload_model_states
from ML_geo_production.geo_utils import filter_images_by_shapefile, compute_target_dimensions, filter_images_by_bounds
from ML_geo_production.processing_utils import (
    reproject_patch,
    merge_worker,
    save_output_data,
    save_predictions_data,
)
from ML_geo_production import create_diff_polygons, model_utils
from ML_geo_production import count_overlaps

class ImageProcessingError(Exception):
    """Custom exception for errors in processing images."""
    def __init__(self, message, filepath):
        super().__init__(message)
        self.filepath = filepath

def verify_images(image_paths):
    """
    Verifies that all images in the provided list can be read without errors.
    This function iterates over all block windows of the first band.
    Raises an ImageProcessingError if any image is corrupted.
    """
    for path in image_paths:
        #print("verifying image :"+str(path))
        try:
            with rasterio.open(path) as src:
                # Iterate over each block window in band 1 to force block reads.
                for _, window in src.block_windows(1):
                    _ = src.read(1, window=window)
        except RasterioIOError as e:
            raise ImageProcessingError(f"Failed to read image block in verification: {path}", path) from e


def process_images_from_dict(parsed_json):
    return process_images(
        image_paths=parsed_json.get("image_paths"),
        data_folders=parsed_json.get("data_types"),
        channels=parsed_json.get("channels"),
        bounds=parsed_json.get("bounds"),
        resolution=parsed_json.get("resolution"),
        path_to_labels=parsed_json.get("path_to_labels"),
        remove_matching_label=parsed_json.get("remove_matching_label", False),
        patch_size=parsed_json.get("patch_size", 1000),
        overlap=parsed_json.get("overlap", 40),
        batch_size=parsed_json.get("batch_size", 4),
        num_workers=parsed_json.get("num_workers", 4),
        saved_models=parsed_json.get("saved_models"),
        model_names=parsed_json.get("model_names"),
        n_classes=parsed_json.get("n_classes", 3),
        pixel_buffer=parsed_json.get("pixel_buffer", 0),
        means=parsed_json.get("means"),
        stds=parsed_json.get("stds"),
        model_states=parsed_json.get("model_states"),
        only_use_these_models_index=parsed_json.get("only_use_these_models_index"),
    )

def process_images(image_paths, data_folders, channels, bounds, resolution=None, 
                  path_to_labels=None, remove_matching_label=False,
                  patch_size=1000, overlap=40, batch_size=8, num_workers=4,
                  saved_models=[], model_names="", means=[[0.485, 0.456, 0.406]], stds=[[0.229, 0.224, 0.225]],
                  n_classes=3, pixel_buffer=0, debug=False, model_states=None, verify_images=False, 
                  only_use_these_models_index=None):
    """
    Core function to process a list of images using an ensemble of models and create a combined output.
    The models are loaded and processed one by one to save GPU memory.

    Parameters:
    -----------
    image_paths : list of str
        List of paths to input images
    bounds : tuple
        (minx, miny, maxx, maxy) bounds defining the target area
    model_states : preloaded model states
    resolution : float
        Resolution (map units per pixel) for the target grid
    # ... (other parameters)
        
    Returns:
    --------
    tuple
        (final_array: np.ndarray, final_transform: rasterio.transform.Affine, dst_crs: rasterio.crs.CRS)
        The final unbuffered array of averaged probabilities and its GeoTIFF metadata.
    """
    print("#"*20)
    print("core process for inference on images started")
    print("#"*20)
    start_time = time.time()

    if model_states ==None:
        print("preloading all model weights into main memory for fast transfer to GPU later")
        model_states= preload_model_states(saved_models)
    if verify_images:
        verify_start_time = time.time()
        verify_images(image_paths)
        print("verifying all images are ok took: " + str((time.time()-verify_start_time)/60) + " minutes")

    # Setup device for processing
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")


    # 1. Define the bounds for the unbuffered area
    # bounds is (minx, miny, maxx, maxy) from the shapefile or user input
    minx, miny, maxx, maxy = bounds

    # 2. Define the bounds for the buffered area
    buffered_minx = minx - pixel_buffer
    buffered_miny = miny - pixel_buffer
    buffered_maxx = maxx + pixel_buffer
    buffered_maxy = maxy + pixel_buffer
    buffered_bounds = (buffered_minx, buffered_miny, buffered_maxx, buffered_maxy)

    # 3. Create transform for unbuffered bounds
    # The transform starts at the upper-left corner (minx, maxy)
    width_unbuffered = math.ceil((maxx - minx) / resolution)
    height_unbuffered = math.ceil((maxy - miny) / resolution)
    transform_unbuffered = from_origin(minx, maxy, resolution, resolution)

    # 4. Create transform for buffered_bounds
    # The transform starts at the upper-left corner of the buffered area (buffered_minx, buffered_maxy)
    target_width = math.ceil((buffered_maxx - buffered_minx) / resolution)
    target_height = math.ceil((buffered_maxy - buffered_miny) / resolution)
    transform_buffered = from_origin(buffered_minx, buffered_maxy, resolution, resolution)

    # CRS from the first image 
    if not image_paths:
        raise Exception("No images found to process.")
    with rasterio.open(image_paths[0]) as src:
        dst_crs = src.crs

    # Count how many times each pixel is covered by a geotiff
    # This geotiff_count_array is for the unbuffered area (bounds)
    print("creating an array counting number of times each output pixel is covered by a geotiff")
    # If shape_file is None, we need a way to get the geometry from bounds for count_overlaps.
    # For now, assuming shape_file is provided or a default mechanism is in place in count_overlaps
    # or that the user provides a shapefile when this function is used for full functionality.
    #OLD! if shape_file:

    # OLD     _, shape_geom = count_overlaps.get_shape_bounds(shape_file)
    #else:
    #     # Placeholder logic if shape_file is absent and bounds is the only input
    #     # This needs proper implementation within geo_utils or count_overlaps
    #     print("Warning: shape_file is None. count_overlaps might not work as intended.")
    #     shape_geom = None # Replace with logic to create geometry from bounds if needed
    geotiff_count_array = count_overlaps.rasterize_overlap_count(image_paths,  bounds, resolution, transform_unbuffered)
    
    print("creation of count_array done")

    # Process labels if provided
    # Assuming path_to_images is the directory that *contained* these images originally
    path_to_images_dir = Path(image_paths[0]).parent if image_paths else None # Placeholder, not strictly used below
    label_paths = prepare_label_paths(path_to_images_dir, path_to_labels, image_paths, remove_matching_label)

    # Initialize merge process and shared memory (n_classes = softmax channels).
    shared_memory, merge_queue, merge_process = initialize_merge_process(
        n_classes, target_height, target_width
    )

    ready_to_do_inference = time.time()
    print("now doing inference on all data in area, model by model")

    # Track the actual number of models used
    models_used_count = 0
    
    # Iterate through each model, load it, process all images, and then unload it.
    for idx, model_state in enumerate(model_states):
        # Skip models not in only_use_these_models_index if the list is provided
        if only_use_these_models_index is not None and idx not in only_use_these_models_index:
            print(f"Skipping model {idx+1}/{len(saved_models)} (index {idx}) - not in only_use_these_models_index")
            continue
        
        models_used_count += 1
        print(f"Loading model {idx+1}/{len(saved_models)} from memory to GPU memory (path : {saved_models[idx]})")
        load_start = time.time()
        # Each model can have its own channels configuration and normalization parameters
        n_in = len(means[idx])
        learner = load_unet_from_state(
            model_state,
            model_names[idx],
            path_to_images_dir,
            n_classes=n_classes,
            device=device,
            n_in=n_in,
            sample_image_path=image_paths[0] if image_paths else None,
            norm_means=means[idx],
            norm_stds=stds[idx],
        )
        learner.model.to(device)
        learner.model.eval()
        print("loading model took : "+str(time.time()-load_start))

        create_data_loader_start  = time.time()

        # Prepare dataloader for the current model
        dataloader = prepare_dataloader(
            image_paths,
            data_folders[idx],
            channels[idx],
            label_paths,
            batch_size,
            num_workers,
            patch_size,
            overlap,
            bounds,
            pixel_buffer
        )
        print("crating dataloader took: "+str(time.time()-create_data_loader_start))

        # Process all images with the current model and accumulate results into shared_memory
        # transform_buffered is used for reprojection into the target array
        print("running inference..")
        run_inference_start= time.time()
        run_inference_and_accumulate(
            dataloader,
            learner,
            device,
            means[idx],
            stds[idx],
            transform_buffered,
            dst_crs,
            merge_queue,
            target_width,
            target_height,
        )
        print("running inference with model: "+str(saved_models[idx])+ " , took: "+str(time.time()-run_inference_start))

        del_learner_start_time= time.time()
        # Explicitly delete the learner and clear GPU memory after processing with the model
        del learner
        if torch.cuda.is_available():
            print("emptying cache")
            torch.cuda.empty_cache()
        print("deleting learner and cleaning cache took: "+str(time.time()-del_learner_start_time))
        print(f"Finished processing with model {idx+1}. Model unloaded and GPU memory cleared.")

    done_with_inference = time.time()
    print("running inference on all models took: "+str(time.time()-ready_to_do_inference))
    print(f"Total models used: {models_used_count}")

    # Finalize processing after all models have contributed their predictions
    # Capture the returned array and metadata from finalize_output
    finalizing_output_start = time.time()
    final_array, final_transform, dst_crs_out = finalize_output(
        shared_memory,
        merge_queue,
        merge_process,
        n_classes,
        target_height,
        target_width,
        pixel_buffer,
        bounds,
        resolution,
        dst_crs,
        geotiff_count_array,
        transform_unbuffered,
        models_used_count,
    )
    merge_process.join()
    del merge_process
    print("finalizing output took : "+str(time.time()-finalizing_output_start))
    
    # Print summary
    end_time = time.time()
    print_summary(start_time, end_time, len(image_paths), ready_to_do_inference, done_with_inference)
    
    # Return the final array and its metadata for saving outside the function
    return final_array, final_transform, dst_crs_out


def process_images_in_folder(path_to_images, data_folders, channels,shape_file, resolution=None,
                  path_to_labels=None, remove_matching_label=False,
                  patch_size=1000, overlap=40, batch_size=8, num_workers=4,
                  saved_models=[], model_names="", means=[[0.485, 0.456, 0.406]], stds=[[0.229, 0.224, 0.225]],
                  n_classes=3, pixel_buffer=0,debug=False,bounds_dict=None):
    """
    Function to handle image filtering/selection before calling the core processing logic.
    
    NOTE: This function is now mostly for backward compatibility or when a directory/shapefile
    is the main input source for image selection and boundary definition.
    """
    print("#"*20)
    print("main process for inference on images started (folder-based filtering)")
    print("#"*20)

    # Filter images by shapefile overlap
    print("Finding files that overlap with the shapefile")
    filter_start_time = time.time()
    image_paths, shapefile_bounds = filter_images_by_shapefile(path_to_images, shape_file,bounds_dict=bounds_dict)
    print("finding images took : " + str((time.time()-filter_start_time)/60) + " minutes")
    print(f"Filtered images: {len(image_paths)} overlap with the shapefile extent")
    
    if not image_paths:
        raise Exception("No images found that overlap with the shapefile extent.")

    # Call the core processing function with the filtered list of images
    return process_images(
        image_paths, 
        data_folders, 
        channels, 
        shapefile_bounds, # Pass bounds directly
        resolution,
        path_to_labels, 
        remove_matching_label,
        patch_size, 
        overlap, 
        batch_size, 
        num_workers,
        saved_models, 
        model_names, 
        means, 
        stds,
        n_classes, 
        pixel_buffer,
        debug,
    )


def prepare_label_paths(path_to_images, path_to_labels, image_paths, remove_matching_label):
    """Prepare label paths if provided"""
    if not path_to_labels:
        return None

    # Get filenames of the images being processed
    image_files = [Path(path).name for path in image_paths]
    
    # Check for labels matching the image names
    label_paths = []
    for img_file in image_files:
        label_path = Path(path_to_labels) / img_file # Assuming label names match image names
        if label_path.exists():
            label_paths.append(str(label_path))
        elif remove_matching_label:
             # If a label is required but missing, remove the image path from the list 
             # (This logic might need refinement if image_paths is supposed to be modified in place)
             pass 
    
    # Original logic adapted for the new structure (relies on path_to_images being the dir)
    # The original implementation modified image_paths in place, which is tricky in the new structure.
    # For now, keeping a more robust label list generation:
    if remove_matching_label:
        # A full implementation of `remove_matching_label` would need to re-filter image_paths
        # This is a placeholder to prevent errors if the image_paths list cannot be mutated easily here.
        pass # The logic of removing images/labels is complex to implement outside of the filter/dataloader.

    return label_paths


def initialize_merge_process(n_channels, target_height, target_width):
    """Initialize shared memory and merge process.

    n_channels must be the softmax / class count (same as n_classes).
    """
    # Create shared memory array to store the sum of probabilities from all models
    target_size = n_channels * target_height * target_width
    shared_target = multiprocessing.Array('f', target_size)  # 'f' for float

    # Create lock for updating shared_target
    update_lock = multiprocessing.Lock()

    # Create merge queue and process
    merge_queue = multiprocessing.JoinableQueue()
    target_shape = (n_channels, target_height, target_width)
    merge_process = multiprocessing.Process(
        target=merge_worker,
        args=(merge_queue, shared_target, target_shape, update_lock)
    )
    merge_process.start()

    return shared_target, merge_queue, merge_process


def prepare_dataloader(image_paths, data_folders, channels, label_paths, batch_size, num_workers,
                     patch_size, overlap, shape_bounds, pixel_buffer):
    """Prepare the dataloader with the specified parameters"""
    dataloader = get_dataloader(
        image_paths, data_folders, channels, label_paths,
        batch_size=batch_size,
        num_workers=num_workers,
        patch_size=patch_size,
        overlap=overlap,
        shape_bounds=shape_bounds,
        pixel_buffer=pixel_buffer
    )
    return dataloader


def run_inference_and_accumulate(dataloader, learner, device, model_mean, model_std, dst_transform, dst_crs,
                             merge_queue, target_width, target_height, verbose=False):
    """
    Process batches from a single dataloader through its corresponding model.
    The reprojected probabilities for each image are submitted to the merge_queue
    to be summed into the global shared memory.
    """
    # Concurrency setup for asynchronous reprojection tasks.
    MAX_CONCURRENT_TASKS = 6
    semaphore = multiprocessing.Semaphore(MAX_CONCURRENT_TASKS)
    pool = multiprocessing.Pool(processes=5)  # Adjust as needed

    # Create one iterator for the single dataloader.
    dataloader_iterator = iter(dataloader)

    # Variables to keep track of the current image and its accumulator.
    current_img_idx = None
    temp_image_array = None # Accumulates probabilities for patches within the current image
    count_array_same_image = None # Counts how many patches cover a pixel within the current image
    src_transform, src_crs = None, None

    def submit_current_image():
        """
        Submit the current image's accumulated probabilities for reprojection.
        These probabilities are for a single model and have intra-image patch overlap handled.
        """
        nonlocal temp_image_array, count_array_same_image, src_transform, src_crs, current_img_idx
        if temp_image_array is not None:
            # Normalize by the number of times each pixel within the image was covered by a patch
            # Ensure no division by zero for count_array_same_image
            count_array_same_image_safe = np.where(count_array_same_image == 0, 1, count_array_same_image)
            final_array = temp_image_array / count_array_same_image_safe

            # Submit the reprojection task asynchronously.
            semaphore.acquire()
            pool.apply_async(
                reproject_patch,
                args=(final_array, src_transform, src_crs, dst_transform, dst_crs,
                      (final_array.shape[0], target_height, target_width)),
                callback=lambda res: (merge_queue.put(res), semaphore.release())
            )
        temp_image_array = None
        count_array_same_image = None

    # Process batches until the dataloader is exhausted.
    while True:
        try:
            batch = next(dataloader_iterator)
        except StopIteration:
            # If the dataloader is exhausted, we're done with this model's processing.
            break

        images = batch["image"].to(device, non_blocking=True) / 255.0
        # Normalization using the specific mean and std for this model
        mean_tensor = torch.tensor(model_mean).view(1, -1, 1, 1).to(device)
        std_tensor = torch.tensor(model_std).view(1, -1, 1, 1).to(device)
        normalized_images = (images - mean_tensor) / std_tensor

        with torch.no_grad():
            preds = learner.model(normalized_images)  # Logits output
            preds = torch.softmax(preds, dim=1) # Apply softmax here to get probabilities

        # Process each patch in the batch.
        batch_size = images.shape[0]
        for i in range(batch_size):
            img_idx = batch["img_idx"][i].item() if isinstance(batch["img_idx"][i], torch.Tensor) else batch["img_idx"][i]
            x = batch["x"][i].item() if isinstance(batch["x"][i], torch.Tensor) else batch["x"][i]
            y = batch["y"][i].item() if isinstance(batch["y"][i], torch.Tensor) else batch["y"][i]
            patch_probs = preds[i].cpu().numpy()  # shape: (n_classes, patch_h, patch_w)

            # If we're on a new image, submit the previous one for reprojection.
            if current_img_idx is None or img_idx != current_img_idx:
                if temp_image_array is not None:
                    submit_current_image()
                current_img_idx = img_idx
                meta = dataloader.dataset.image_metadata[current_img_idx]
                width, height = meta["width"], meta["height"]
                src_transform = meta["transform"]
                src_crs = meta["crs"]
                temp_image_array = np.zeros((patch_probs.shape[0], height, width), dtype=np.float32)
                # Count how many image patches overlap with a position within this specific image.
                count_array_same_image = np.zeros((height, width), dtype=np.float32)

            # Accumulate the patch probabilities into the proper location.
            patch_h, patch_w = patch_probs.shape[1], patch_probs.shape[2]
            temp_image_array[:, y:y+patch_h, x:x+patch_w] += patch_probs
            # Keep track of how many patches have been contributing
            count_array_same_image[y:y+patch_h, x:x+patch_w] += 1

    # Submit the final image for this model, if any patches were processed.
    if temp_image_array is not None:
        submit_current_image()

    # Cleanup: close and wait for all asynchronous tasks started by this function.
    pool.close()
    pool.join()
    del pool

def finalize_output(shared_memory, merge_queue, merge_process, n_channels, target_height, target_width,
                  pixel_buffer, unbuffered_bounds, resolution, dst_crs, geotiff_count_array, final_transform, num_models):
    """
    Finalize processing and return the final unbuffered array, transform, and CRS.
    Disk saving is explicitly removed here.

    n_channels: class probability channels (n_classes), matching merge_worker buffer.
    """
    # Wait for all merge tasks to complete
    merge_queue.join()

    # Signal merge process to terminate
    merge_queue.put(None)
    merge_process.join()

    # Reshape shared memory back into the original 3D array
    # shared_memory now contains the sum of probabilities from all models (buffered size)
    data_array = np.ctypeslib.as_array(shared_memory.get_obj())
    final_array_buffered = data_array.reshape(n_channels, target_height, target_width)

    # Average the probabilities across the ensemble (divide by total number of models)
    final_array_buffered = final_array_buffered / num_models

    # Ensure no division by zero for pixels not covered by any geotiff
    geotiff_count_array_safe = np.where(geotiff_count_array == 0, 1, geotiff_count_array)

    # Get the target dimensions for the unbuffered area from geotiff_count_array_safe
    unbuffered_height, unbuffered_width = geotiff_count_array_safe.shape

    # Calculate the starting and ending indices for cropping the buffered final_array
    minx, miny, maxx, maxy = unbuffered_bounds
    
    # Calculate offset in map units
    offset_x_map = minx - (minx - pixel_buffer) # Should be pixel_buffer
    offset_y_map = (maxy + pixel_buffer) - maxy # Should be pixel_buffer

    # Convert to pixels (rows/cols)
    start_x = int(offset_x_map / resolution)
    start_y = int(offset_y_map / resolution)
    
    end_x = start_x + unbuffered_width
    end_y = start_y + unbuffered_height
    
    # Safety check for dimensions
    if end_y > target_height or end_x > target_width:
        print("Warning: Calculated crop indices exceed buffered array dimensions. Recalculating crop indices based on center.")
        start_y = (target_height - unbuffered_height) // 2
        end_y = start_y + unbuffered_height
        start_x = (target_width - unbuffered_width) // 2
        end_x = start_x + unbuffered_width
    
    # Crop final_array to the unbuffered dimensions
    final_array = final_array_buffered[:, start_y:end_y, start_x:end_x]

    # Now, final_array and geotiff_count_array_safe have compatible dimensions.
    # Perform the division by geotiff_count_array to handle overlapping input geotiffs
    final_array = final_array / geotiff_count_array_safe

    return final_array, final_transform, dst_crs # Return the final array and its metadata


def print_summary(start_time, end_time, num_images, ready_to_do_inference, done_with_inference):
    """Print processing summary"""
    output_status = f"Final array returned in memory. "
    
    print("#" * 30)
    print(output_status)
    print(f"Processed {num_images} large images")
    print(f"Average time per large image: {(end_time - start_time) / num_images:.2f} seconds")
    print(f"Inference time: {(done_with_inference - ready_to_do_inference) / 60:.2f} minutes")
    print("#" * 10)
    print(f"Total processing time: {(end_time - start_time) / 60:.2f} minutes")
    print("#" * 30)

def parse_list_of_lists(value):
    """
    Custom type function to parse a string into a list of lists of integers.

    Args:
        value (str): Input string representing a list of lists

    Returns:
        list: Parsed list of lists of integers
    """
    try:
        parsed = json.loads(value)
        if not isinstance(parsed, list) or \
           not all(isinstance(sublist, list) and
                   all(isinstance(item, int) for item in sublist)
                   for sublist in parsed):
            raise argparse.ArgumentTypeError("Input must be a list of lists of integers")
        return parsed
    except json.JSONDecodeError:
        raise argparse.ArgumentTypeError("Invalid JSON format")

def parse_arguments():
    """Parse command line arguments"""
    example_usage = """
    python process_images.py --json /path/to/config.json --bounding_box /path/to/extent.shp --image_paths /path/to/tif1.tif /path/to/tif2.tif
    python process_images.py --json /path/to/config.json --bounding_box '[100.0, 500.0, 200.0, 600.0]' --image_paths /path/to/tif1.tif
    
    JSON config file structure:
    {
        "probs_path": "/mnt/T/mnt/random_files/probs_20250404.tif",
        "resolution": 1.0,
        "remove_matching_label": false,
        "remove_tmp_files": false,
        "patch_size": 1000,
        "overlap": 40,
        "batch_size": 4,
        "num_workers": 4,
        "saved_model": ["/path/to/model1.pth", "/path/to/model2.pth"],
        "n_classes": 3,
        "means": [[0.485, 0.456, 0.406], [0.485, 0.456, 0.406]],
        "stds": [[0.229, 0.224, 0.225], [0.229, 0.224, 0.225]],
        "channels": [[0, 1, 2], [0, 1, 2]],
        "data_folders": ["rooftop_rgb", "rooftop_rgb"],
        "model_name": ["tf_efficientnetv2_l.in21k", "tf_efficientnetv2_l.in21k"],
        "pixel_buffer": 0.0,
        "geopackage": "/path/to/geopackage.gpkg",
        "polygon_output_folder": "/mnt/T/mnt/random_files/"
    }
    """

    parser = argparse.ArgumentParser(
        description="Run ensemble model inference using a JSON configuration file and direct image paths.",
        epilog=example_usage,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    # Optional arguments that might override the JSON config 
    parser.add_argument("--json", type=str, required=True, help="Path to the JSON configuration file.")
    parser.add_argument("--bounding_box", type=str,    help="Path to a shapefile (.shp) to use its extent, OR a JSON string for a bounding box '[minx, miny, maxx, maxy]'.")
    parser.add_argument("--image_paths", nargs='+', type=str,  
                        help="A series of paths to input GeoTIFFs (e.g., /path/to/tif1.tif /path/to/tif2.tif).")


    return parser.parse_args()


def init(parsed_json):
    """Initialization: preloads model weights and geopackage"""


    print("Preloading all model weights into main memory...")
    parsed_json["model_states"] = model_utils.preload_model_states(parsed_json["saved_models"])
    print("loading the geopackage")
    parsed_json["geopackage"] = gpd.read_file(parsed_json["geopackage"])

    print("Initialization complete.")
    return parsed_json


# Helper function to parse the bounding box argument
def parse_bounding_box(bounding_box_arg):
    """
    Parses the bounding_box argument.
    If it's a .shp file, returns its extent.
    If it's a JSON string, returns the parsed tuple (minx, miny, maxx, maxy).
    """
    path_obj = Path(bounding_box_arg)
    
    if path_obj.exists() and path_obj.suffix.lower() == '.shp':
        print(f"Parsing bounds from shapefile: {bounding_box_arg}")
        gdf = gpd.read_file(bounding_box_arg)
        minx, miny, maxx, maxy = gdf.total_bounds
        return (minx, miny, maxx, maxy)
    else:
        # parse as a JSON string for bounds
        bounds_list = json.loads(bounding_box_arg)
        if isinstance(bounds_list, list) and len(bounds_list) == 4 and all(isinstance(x, (int, float)) for x in bounds_list):
            minx, miny, maxx, maxy = bounds_list
            # Validate bounds order
            if minx >= maxx or miny >= maxy:
                 raise ValueError("Bounding box coordinates are invalid (minx >= maxx or miny >= maxy).")
            print(f"Using bounding box from JSON string: {tuple(bounds_list)}")
            return tuple(bounds_list)
        else:
            sys.exit("bounding_box_arg:"+str(bounding_box_arg) + "is not of corect format")

def get_overlapping_geotif_paths(parsed_json):
    # Filter images by shapefile overlap
    print("Finding files that overlap with the shapefile")
    filter_start_time = time.time()
    image_paths, shapefile_bounds = filter_images_by_bounds(parsed_json["path_to_images"],parsed_json["bounds"],bounds_dict=parsed_json.get("bounds_dict"))
    print("finding images took : " + str((time.time()-filter_start_time)/60) + " minutes")
    print(f"Filtered images: {len(image_paths)} overlap with the shapefile extent")

    if not image_paths:
        raise Exception("No images found that overlap with the shapefile extent.")
    return image_paths




if __name__ == "__main__":
    # 1. Parse command line arguments (json, bounding_box, image_paths)
    args = parse_arguments()

    # 2. Load the JSON configuration file
    with open(args.json, 'r') as f:
        parsed_json = json.load(f)

    # extract boindingbox from shapefile
    if parsed_json.get("shapefile"):
        # bounds is (minx, miny, maxx, maxy) tuple
        # shapefile_path is the path string if a .shp was used, bounds can be derved from a shapefile
        parsed_json["bounds"]= parse_bounding_box(parsed_json["shapefile"])

    # 3. Parse the flexible bounding_box argument
    if args.bounding_box:
        # bounds is (minx, miny, maxx, maxy) tuple
        # shapefile_path is the path string if a .shp was used, otherwise None
        parsed_json["bounds"]= parse_bounding_box(args.bounding_box)
    print('parsed_json["bounds"]: '+str(parsed_json["bounds"]))



    if args.image_paths:
        # 
        parsed_json["image_paths"]= args.image_paths
    elif parsed_json.get("image_paths"):
        print("use the iamge paths defined in the json")
    else:
        parsed_json["image_paths"]= get_overlapping_geotif_paths(parsed_json)
    # 4. preload modle weights and load geopackages
    parsed_json = init(parsed_json)

    process_images_start_time = time.time()
    
    # 5. Run the ensemble inference using the new function signature.
    # Arguments from JSON are extracted and passed explicitly.
    # Note: path_to_images is no longer used in process_images_in_folder, so
    # we directly call the core process_images function.
    
    # Check for required fields in the JSON that are not explicitly arguments now
    required_json_keys = ["data_types", "channels", "resolution", "probs_path", "saved_models",
                          "means", "stds", "polygon_output_folder", "geopackage","bounds"]
    for key in required_json_keys:
        if key not in parsed_json:
            print(f"Error: Required key '{key}' missing from JSON configuration file.")
            sys.exit(1)


    final_probability_array, final_transform, dst_crs = process_images_from_dict(parsed_json)
    
    # 6. Save the array to disk
    save_output_data(final_probability_array, final_transform, dst_crs, parsed_json["probs_path"])
    print(f'Saved output GeoTIFF to {parsed_json["probs_path"]}')

    if parsed_json.get("save_preds"):
        pred_path = parsed_json.get("pred_path")
        if not pred_path:
            print("Error: save_preds is true but pred_path is missing from JSON.")
            sys.exit(1)
        preds_2d = np.argmax(final_probability_array, axis=0).astype(np.uint8)
        save_predictions_data(preds_2d, final_transform, dst_crs, pred_path)
        print(f"Saved prediction GeoTIFF (argmax classes) to {pred_path}")

    save_original_polygons = parsed_json.get("save_original_polygons") or False
    save_buffer_in_polygons = parsed_json.get("save_buffer_in_polygons") or False
    save_buffer_in_buffer_out_polygons = (
        parsed_json.get("save_buffer_in_buffer_out_polygons") or False
    )
    polygon_debug_output_folder = None
    if (
        save_original_polygons
        or save_buffer_in_polygons
        or save_buffer_in_buffer_out_polygons
    ):
        polygon_debug_output_folder = parsed_json["polygon_output_folder"]

    # 7. create polygons
    diff_polygons_df = create_diff_polygons.create_diff_polygons(
        probs=final_probability_array,
        geopackage_data=parsed_json["geopackage"],
        transform=final_transform,
        crs=dst_crs,
        bounds=parsed_json["bounds"],
        change_detection_threshold=parsed_json["change_detection_threshold"],
        unknown_boarder_size=parsed_json.get("unknown_boarder_size", 1.5),
        path_to_mapping=parsed_json.get(
            "path_to_mapping",
            "/mnt/T/mnt/trainingdata/bygningsudpegning/iter_4/roof_no_roof_mapping.txt",
        ),
        create_new_mapping=parsed_json.get("create_new_mapping", False),
        extra_atributes=parsed_json.get("extra_atributes"),
        save_original_polygons=save_original_polygons,
        save_buffer_in_polygons=save_buffer_in_polygons,
        save_buffer_in_buffer_out_polygons=save_buffer_in_buffer_out_polygons,
        polygon_debug_output_folder=polygon_debug_output_folder,
    )

    print("created polygons: " + str(diff_polygons_df))

    Path(parsed_json["polygon_output_folder"]).mkdir(parents=True, exist_ok=True)
    diff_shp_path = (
        Path(parsed_json["polygon_output_folder"])
        / Path(parsed_json["probs_path"]).name
    ).with_suffix(".shp")
    diff_polygons_df.to_file(diff_shp_path)
    print(f"Saved diff polygons to {diff_shp_path}")

    print(
        "process_images took in all : "
        + str((time.time() - process_images_start_time) / 60)
        + " minutes"
    )
