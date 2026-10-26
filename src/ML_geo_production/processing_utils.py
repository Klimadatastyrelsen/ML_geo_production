# processing_utils.py
import math
import numpy as np
import rasterio
from rasterio.transform import array_bounds
from rasterio.warp import reproject, transform_bounds, Resampling
from rasterio.windows import Window
from rasterio.windows import from_bounds as window_from_bounds
from rasterio.windows import transform as window_transform
from pathlib import Path
import time

def merge_worker(merge_queue, shared_target,target_shape, lock, verbose=False):
    """
    A dedicated process that reads reprojected patch arrays from merge_queue and safely
    adds them to the shared target array.
    
    Parameters:
    -----------
    merge_queue : multiprocessing.JoinableQueue
        Queue to receive arrays from
    shared_target : multiprocessing.Array
        Shared memory array to accumulate results
    target_shape : tuple
        Shape of the target array
    lock : multiprocessing.Lock
        Lock for safe access to shared memory
    """
    # Create a numpy view of the shared target array
    target_np = np.frombuffer(shared_target.get_obj(), dtype=np.float32).reshape(target_shape)
    while True:
        if verbose:print("Merge worker is getting data from queue")
        item = merge_queue.get()
        if verbose:print("Merge worker GOT data")
        if item is None:
            if verbose:print("Merge data is None, exiting")
            break  # Exit signal

        # reproject_patch only returns the part of the target grid that the source image
        # covers, so the array is added at its offset instead of over the whole target
        row_off, col_off, array = item
        height, width = array.shape[1], array.shape[2]

        # Acquire lock to update the shared target array safely
        if verbose:print("Merge worker is acquiring lock")
        with lock:
            target_np[:, row_off:row_off + height, col_off:col_off + width] += array
        if verbose:print("Merge worker is ready for new iteration")
        merge_queue.task_done()
        if verbose:print("Merge marked task done")


def _crs_missing(crs):
    return crs is None or crs == "None" or crs == "none"


def reproject_patch(patch_array, src_transform, src_crs, dst_transform, dst_crs, dst_shape,verbose=False):
    """
    Reprojects a complete image patch (all channels) into the destination coordinate system.
    
    Parameters:
    -----------
    patch_array : numpy.ndarray
        Source array to reproject
    src_transform : affine.Affine
        Source transform
    src_crs : CRS
        Source coordinate reference system
    dst_transform : affine.Affine
        Destination transform
    dst_crs : CRS
        Destination coordinate reference system
    dst_shape : tuple
        Shape of the full destination grid, (channels, height, width)

    Returns:
    --------
    tuple
        (row_off, col_off, array) where array only covers the part of the destination
        grid that the source actually reaches. Returning the full destination grid
        instead means pickling channels * height * width * 4 bytes back to the caller
        for every source image, which is gigabytes per image for a 0.1 m tile.
    """
    if verbose:print("Reproject_patch started")
    if (
        patch_array.shape == dst_shape
        and src_transform == dst_transform
        and (_crs_missing(src_crs) or _crs_missing(dst_crs))
    ):
        return 0, 0, patch_array.astype(np.float32).copy()

    n_channels, dst_height, dst_width = dst_shape
    row_off, col_off, height, width = _destination_window(
        patch_array.shape, src_transform, src_crs, dst_transform, dst_crs, dst_height, dst_width
    )

    # Create a temporary array to hold the reprojected data
    temp = np.zeros((n_channels, height, width), dtype=np.float32)
    if height == 0 or width == 0:
        # the source falls outside the destination grid; reproject cannot take an empty
        # destination, and adding an empty array at the offset is a no-op anyway
        if verbose:print("Reproject_patch: source does not reach the destination grid")
        return row_off, col_off, temp
    window_dst_transform = window_transform(Window(col_off, row_off, width, height), dst_transform)

    for i in range(patch_array.shape[0]):
        reproject(
            source=patch_array[i],
            destination=temp[i],
            src_transform=src_transform,
            src_crs=src_crs,
            dst_transform=window_dst_transform,
            dst_crs=dst_crs,
            resampling=Resampling.nearest
        )
    if verbose:print("Reproject_patch ENDED")
    return row_off, col_off, temp


def _destination_window(src_shape, src_transform, src_crs, dst_transform, dst_crs, dst_height, dst_width):
    """The (row_off, col_off, height, width) of the destination grid the source covers."""
    bounds = array_bounds(src_shape[1], src_shape[2], src_transform)
    if not _crs_missing(src_crs) and not _crs_missing(dst_crs) and str(src_crs) != str(dst_crs):
        bounds = transform_bounds(src_crs, dst_crs, *bounds)

    window = window_from_bounds(*bounds, transform=dst_transform)
    # one pixel of slack on each side so nearest resampling cannot fall just outside
    col_off = min(max(math.floor(window.col_off) - 1, 0), dst_width)
    row_off = min(max(math.floor(window.row_off) - 1, 0), dst_height)
    col_end = min(max(math.ceil(window.col_off + window.width) + 1, col_off), dst_width)
    row_end = min(max(math.ceil(window.row_off + window.height) + 1, row_off), dst_height)
    return row_off, col_off, row_end - row_off, col_end - col_off

def save_probabilities_data(data_array: np.ndarray, transform, crs, filename: str, output_dtype=np.float32):
    """
    Save a 3D prediction probability array (channels, y, x) as a GeoTIFF.

    The function explicitly casts the input data to the specified numpy float type
    to minimize file size.

    Parameters
    ----------
    data_array : np.ndarray
        3D array with shape (channels, height, width) and float values.
    transform : rasterio.transform.Affine
        Geo-transform describing pixel location.
    crs : rasterio.crs.CRS or str
        Coordinate reference system.
    filename : str
        Output file path (.tif).
    output_dtype : numpy.dtype or str, optional
        The data type to use for the output GeoTIFF bands (e.g., np.float16, np.float32).
        Defaults to np.float16. Note: float16 compatibility depends on the underlying
        GDAL/rasterio version.
    """
    # Check that the array is 3-dimensional
    if data_array.ndim != 3:
        raise ValueError(f"Expected a 3D array (channels, y, x), but got {data_array.ndim} dimensions.")

    # Determine dimensions
    num_channels, height, width = data_array.shape

    # 1. Cast data explicitly to the specified dtype
    # We use the argument output_dtype directly
    output_data = data_array.astype(output_dtype)

    print(f"Saving GeoTIFF with {num_channels} channels, size {height}x{width}, and forced dtype {output_dtype}")

    # 2. Open and write the file
    with rasterio.open(
        filename,
        "w",
        driver="GTiff",
        height=height,
        width=width,
        count=num_channels,
        dtype=output_data.dtype, # Use the actual dtype after casting
        transform=transform,
        crs=crs,
    ) as dst:
        # The data is in the (count, height, width) format required by rasterio.
        dst.write(output_data)

    print(f"Successfully saved data to {filename} using {output_data.dtype}.")

def save_predictions_data(data_array, transform, crs, filename):
    """
    Save a 2D prediction array as a uint8 GeoTIFF.

    Parameters
    ----------
    data_array : np.ndarray
        2D array with shape (height, width) and integer values.
    transform : rasterio.transform.Affine
        Geo-transform describing pixel location.
    crs : rasterio.crs.CRS or str
        Coordinate reference system.
    filename : str
        Output file path (.tif).
    """

    # Ensure correct dtype
    data_uint8 = data_array.astype(np.uint8)

    height, width = data_uint8.shape

    with rasterio.open(
        filename,
        "w",
        driver="GTiff",
        height=height,
        width=width,
        count=1,
        dtype="uint8",
        transform=transform,
        crs=crs,
    ) as dst:
        dst.write(data_uint8, 1)


def save_output_data(data_array, transform, crs, filename):
    """
    Save output data as a GeoTIFF with geospatial metadata.
    
    Parameters:
    -----------
    data_array : numpy.ndarray with probs for an area
        Array to save (shape: channels, height, width)
    transform : affine.Affine
        Geospatial transform
    crs : CRS
        Coordinate reference system
    filename : str
        Output filename
    """
    start_time = time.time()
    Path(filename).parent.mkdir(parents=True, exist_ok=True)


    print("data cominhg in to save_output_data")
    #print(data_array)
    print("lowest")
    print(data_array.flatten().min())
    print("max")
    print(data_array.flatten().max())
    print("shape")
    print(data_array.shape)
    #print("- min / (max-min)")



    # Normalize and convert data

    summed = data_array.sum(axis=0, keepdims=True)
    summed[summed == 0] = 1  # Avoid division by zero
    normalized_data = data_array / summed
    normalized_data = np.nan_to_num(normalized_data, nan=0)  # Handle NaNs
    normalized_data = normalized_data.astype(np.float32)  # Use float32 for speed

    # Write to GeoTIFF
    with rasterio.open(
        filename,
        "w",
        driver="GTiff",
        height=normalized_data.shape[1],
        width=normalized_data.shape[2],
        count=normalized_data.shape[0],
        dtype="float32",
        crs=crs,
        transform=transform,
        tiled=True,
        blockxsize=256,
        blockysize=256,
        compress=None  # No compression = fastest read/write
    ) as dst:
        dst.write(normalized_data)
    print("data saved to :"+str(filename)+ "took : "+str(time.time()-start_time) + "seconds" )
