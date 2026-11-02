from torch.utils.data import Dataset, DataLoader
from torch.utils.data._utils.collate import default_collate
import torch
import numpy as np
import rasterio
import os
import math
from pathlib import Path
from rasterio.windows import Window
from rasterio.transform import from_origin
import geopandas as gpd


class LargeImageDataset(Dataset):
    def __init__(self, image_paths, data_folders, channels, label_paths=None, patch_size=1000, overlap=40, shape_bounds=None, pixel_buffer=0):
        """
        Parameters:
          image_paths: list of image file paths (used to derive metadata and filenames).
          data_folders: list of folder names containing geotiff images. The full path is computed as Path(image_path).parent / folder_name.
          channels: list of lists of integers. Each sublist indicates which channels to load from the corresponding folder.
          label_paths: list of label file paths.
          patch_size: size of each patch.
          overlap: overlapping pixels between patches.
          shape_bounds: allowed geographic bounds as (minx, miny, maxx, maxy).
          pixel_buffer: buffer (in map- units) to expand the allowed area.
        """
        self.image_paths = image_paths
        self.data_folders = data_folders
        self.channels = channels
        self.label_paths = label_paths
        self.patch_size = patch_size
        self.overlap = overlap
        self.stride = patch_size - overlap
        self.shape_bounds = shape_bounds  # in geographic coordinates
        self.pixel_buffer = pixel_buffer  # in pixel units
        self.image_metadata = self._compute_image_metadata()
        self.patches = self._compute_patches()
    
    def _compute_image_metadata(self):
        metadata = []
        with rasterio.Env():
            for path in self.image_paths:
                with rasterio.open(path) as src:
                    bounds = src.bounds  # (left, bottom, right, top)
                    metadata.append({
                        "width": src.width,
                        "height": src.height,
                        "transform": src.transform,
                        "crs": str(src.crs),
                        "filename": os.path.basename(path),  # used to locate matching files in data_folders
                        "bounds": (bounds.left, bounds.bottom, bounds.right, bounds.top)
                    })
        return metadata


    def _compute_patches(self):
        patches = []
        for img_idx, meta in enumerate(self.image_metadata):
            width, height = meta["width"], meta["height"]
            transform = meta["transform"]
            
            # Get image origin and resolution
            image_left = transform.c
            image_top = transform.f
            pixel_width = transform.a
            pixel_height = abs(transform.e)
            
            # Determine allowed geographic bounds for this image
            if self.shape_bounds is not None:
                a_minx, a_miny, a_maxx, a_maxy = self.shape_bounds
                # Convert pixel_buffer to geographic units
                #this was a mistake buffer_geo = self.pixel_buffer * pixel_width
                buffer_geo = self.pixel_buffer
                allowed_minx = a_minx - buffer_geo
                allowed_maxx = a_maxx + buffer_geo
                allowed_miny = a_miny - buffer_geo
                allowed_maxy = a_maxy + buffer_geo
                # Intersect with the image bounds
                img_minx, img_miny, img_maxx, img_maxy = meta["bounds"]
                final_minx = max(allowed_minx, img_minx)
                final_maxx = min(allowed_maxx, img_maxx)
                final_miny = max(allowed_miny, img_miny)
                final_maxy = min(allowed_maxy, img_maxy)
            else:
                final_minx, final_miny, final_maxx, final_maxy = meta["bounds"]
            
            # Convert geographic boundaries to pixel indices
            col_start = int(math.floor((final_minx - image_left) / pixel_width))
            col_end = int(math.ceil((final_maxx - image_left) / pixel_width))
            
            row_start = int(math.floor((image_top - final_maxy) / pixel_height))
            row_end = int(math.ceil((image_top - final_miny) / pixel_height))
            
            # Clamp indices to valid image ranges (important to prevent mirroring)
            col_start = max(0, col_start)
            row_start = max(0, row_start)
            col_end = min(width, col_end)
            row_end = min(height, row_end)
            
            # Find the last valid starting position for a patch that stays within the image
            max_col_start = width - self.patch_size
            max_row_start = height - self.patch_size
            
            # Generate patches within the valid image area
            for y in range(row_start, min(row_end, max_row_start + 1), self.stride):
                for x in range(col_start, min(col_end, max_col_start + 1), self.stride):
                    # Add this patch (we've already ensured it's within image bounds)
                    patches.append((img_idx, x, y))
            
            # Handle edge cases for right and bottom boundaries
            # If our last patch doesn't reach the boundary and we still have valid pixels
            if col_end > min(col_end, max_col_start + 1) and max_col_start >= 0:
                # Add patches at the right edge
                x = max_col_start  # Start the patch at the last valid position
                for y in range(row_start, min(row_end, max_row_start + 1), self.stride):
                    patches.append((img_idx, x, y))
            
            if row_end > min(row_end, max_row_start + 1) and max_row_start >= 0:
                # Add patches at the bottom edge
                y = max_row_start  # Start the patch at the last valid position
                for x in range(col_start, min(col_end, max_col_start + 1), self.stride):
                    patches.append((img_idx, x, y))
            
            # Add the bottom-right corner patch if needed
            if col_end > min(col_end, max_col_start + 1) and row_end > min(row_end, max_row_start + 1) and max_col_start >= 0 and max_row_start >= 0:
                patches.append((img_idx, max_col_start, max_row_start))
            
        return patches
    def __len__(self):
        return len(self.patches)

    def __getitem__(self, index):
        img_idx, x, y = self.patches[index]
        meta = self.image_metadata[img_idx]
        window = Window(x, y, self.patch_size, self.patch_size)
        
        # Use the filename from metadata to load the corresponding patch from each folder.
        filename = meta["filename"]
        patch_list = []
        # Get the grand parent directory of the image used to compute metadata.
        base_folder = Path(self.image_paths[img_idx]).parent.parent
        with rasterio.Env():
            # Loop over each folder name and its corresponding channels.
            for folder_idx, folder_name in enumerate(self.data_folders):
                file_path = base_folder / folder_name / filename
                with rasterio.open(file_path) as src:
                    # Read only the channels specified for this folder.
                    # Note: rasterio's read function expects 1-indexed channel numbers.
                    ch_indexes = [ch + 1 for ch in self.channels[folder_idx]]
                    # Important: Set boundless=False to prevent mirroring at the edges
                    data_patch = src.read(indexes=ch_indexes, window=window, boundless=False, fill_value=0).astype(np.float32)
                    patch_list.append(data_patch)
        
        # Concatenate along the channel axis.
        merged_patch = np.concatenate(patch_list, axis=0)
        
        # Compute patch transform correctly preserving rotation
        transform_patch = rasterio.Affine(
            meta["transform"].a,         # a: width of a pixel
            meta["transform"].b,         # b: row rotation (typically zero)
            meta["transform"].c + x * meta["transform"].a,  # c: x-coordinate of upper-left corner
            meta["transform"].d,         # d: column rotation (typically zero)
            meta["transform"].e,         # e: height of a pixel (negative)
            meta["transform"].f + y * meta["transform"].e   # f: y-coordinate of upper-left corner
        )
        
        data = {
            "img_idx": img_idx,
            "image": torch.tensor(merged_patch, dtype=torch.float32),
            "transform": transform_patch,
            "crs": meta["crs"],
            "filename": filename,
            "x": x,
            "y": y
        }
        
        if self.label_paths:
            with rasterio.open(self.label_paths[img_idx]) as src:
                # Also use boundless=False for labels
                label_patch = src.read(window=window, boundless=False, fill_value=0).astype(np.float32)
            data["label"] = torch.tensor(label_patch, dtype=torch.float32)
        
        return data


def custom_collate_fn(batch):
    batch_dict = {}
    for key in batch[0]:
        # If key is 'crs' (or any non-batchable type), collect as list.
        if key == "crs":
            batch_dict[key] = [item[key] for item in batch]
        elif isinstance(batch[0][key], (torch.Tensor, np.ndarray)):
            batch_dict[key] = default_collate([item[key] for item in batch])
        else:
            batch_dict[key] = [item[key] for item in batch]
    return batch_dict


def get_dataloader(image_paths, data_folders, channels, label_paths=None, batch_size=8, num_workers=4, patch_size=1000, overlap=40, 
                   shape_bounds=None, pixel_buffer=0, prefetch_factor=2):
    dataset = LargeImageDataset(image_paths, data_folders, channels, label_paths, patch_size=patch_size, overlap=overlap,
                               shape_bounds=shape_bounds, pixel_buffer=pixel_buffer)
    # torch rejects prefetch_factor when num_workers is 0, so num_workers=0 (everything in the
    # main process, useful when the workers hang on slow network storage) needs it left out
    extra_kwargs = {"prefetch_factor": prefetch_factor} if num_workers > 0 else {}
    return DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=False,
        num_workers=num_workers, 
        pin_memory=True,
        collate_fn=custom_collate_fn,
        persistent_workers=False,  # Keeps workers alive across epochs/inference loops. Keeping it at true casue 'to many open files' error
        **extra_kwargs
    )
if __name__ == "__main__":
    import argparse
    import json
    from tqdm import tqdm
    print("#"*10)
    example_usage= "python patch_dataset.py --data_folders rooftop_rgb --channels [[0,1,2]] --patch_size 1000 --overlap 40 --image_paths /mnt/T/mnt/trainingdata/bygningsudpegning/1km2data_for_benchmarking_demo_komune_areas_20250205/data/rooftop_rgb/O2024_84_41_06_0278.tif --shapefile /mnt/T/mnt/trainingdata/bygningsudpegning/demo_komunes_areas_1_shapes/1km_6202_701.shp --pixel_buffer 40 --output_folder /mnt/T/mnt/random_files/dataloaderdebbuging"
    print("example_usage")
    print(example_usage)
    print("#"*10)

    parser = argparse.ArgumentParser()
    parser.add_argument('--image_paths', nargs='+', required=True, help='Paths to reference images used for patch alignment')
    parser.add_argument('--data_folders', nargs='+', required=True, help='Names of folders containing corresponding image bands')
    parser.add_argument('--channels', type=str, required=True, help='JSON string or path to JSON file defining channels per folder')
    parser.add_argument('--label_paths', nargs='+', default=None, help='Optional label image paths')
    parser.add_argument('--patch_size', type=int, default=1000, help='Patch size in pixels')
    parser.add_argument('--overlap', type=int, default=40, help='Overlap between patches in pixels')
    parser.add_argument('--shapefile', type=str,  default=None,help='shapefile describing the area to extract patches from')
    parser.add_argument('--pixel_buffer', type=int, default=0, help='Buffer in pixels to apply around shape_bounds')
    parser.add_argument('--output_folder', type=str, required=True, help='Folder to save output GeoTIFF patches')
    args = parser.parse_args()

    # Load the shapefile and compute bounds
    gdf = gpd.read_file(args.shapefile)
    bounds = gdf.total_bounds  # (minx, miny, maxx, maxy)

    # Load channels from JSON string or file
    if os.path.isfile(args.channels):
        with open(args.channels, 'r') as f:
            channels = json.load(f)
    else:
        channels = json.loads(args.channels)

    os.makedirs(args.output_folder, exist_ok=True)

    dataset = LargeImageDataset(
        image_paths=args.image_paths,
        data_folders=args.data_folders,
        channels=channels,
        label_paths=args.label_paths,
        patch_size=args.patch_size,
        overlap=args.overlap,
        shape_bounds=bounds,
        pixel_buffer=args.pixel_buffer
    )

    for idx in tqdm(range(len(dataset)), desc="Extracting GeoTIFF patches"):
        sample = dataset[idx]
        image_tensor = sample["image"]
        transform = sample["transform"]
        crs = sample["crs"]
        filename = sample["filename"]
        x = sample["x"]
        y = sample["y"]

        out_filename = f"{Path(filename).stem}_x{x}_y{y}.tif"
        out_path = os.path.join(args.output_folder, out_filename)

        with rasterio.open(
            out_path,
            'w',
            driver='GTiff',
            height=image_tensor.shape[1],
            width=image_tensor.shape[2],
            count=image_tensor.shape[0],
            dtype='float32',
            crs=crs,
            transform=transform
        ) as dst:
            dst.write(image_tensor.numpy())
