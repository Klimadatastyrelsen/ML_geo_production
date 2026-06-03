# image_utils.py
from pathlib import Path

import numpy as np
import rasterio
import torch
from rasterio.windows import Window
from fastai.vision.all import PILImage, PILMask, TensorImage


def _to_uint8_hwc(arr):
    """Convert (H, W, C) array to uint8 for fastai PILImage."""
    if arr.dtype != np.uint8:
        arr = arr.astype(np.float32)
        arr = (arr - arr.min()) / (arr.max() - arr.min() + 1e-6) * 255
        arr = arr.astype(np.uint8)
    return arr

def load_central_window(fn, window_size=1000, n_channels=None):
    """
    Uses rasterio to load only the central window from the image.
    
    Parameters:
    -----------
    fn : str
        Path to the image file
    window_size : int
        Size of the window to extract
    n_channels : int, optional
        If set, read at most this many raster bands (1..n_channels), so dummy
        dataloaders match a model with n_in channels. If None, read all bands.
        
    Returns:
    --------
    PILImage
        The extracted window as a FastAI PILImage
    """
    with rasterio.open(fn) as src:
        width, height = src.width, src.height
        # Compute the top-left corner of the central window.
        col_off = max((width - window_size) // 2, 0)
        row_off = max((height - window_size) // 2, 0)
        window = Window(col_off, row_off, window_size, window_size)
        # Read only the window from the file (optionally first n_channels bands only).
        if n_channels is None:
            arr = src.read(window=window)
        else:
            n = max(1, min(int(n_channels), src.count))
            arr = src.read(indexes=list(range(1, n + 1)), window=window)
        arr = np.moveaxis(arr, 0, -1)
        arr = _to_uint8_hwc(arr)
    return PILImage.create(arr)


def load_multi_channel_central_window(fn, data_folders, channels, window_size=1000):
    """
    Load a central window by concatenating bands from multiple data folders,
    matching the inference path in patch_dataset.LargeImageDataset.

    Parameters:
    -----------
    fn : str or Path
        Reference GeoTIFF path (used for window placement and filename lookup).
    data_folders : sequence of str
        Subfolder names under the dataset data root (e.g. rgb, cir, DSM).
    channels : sequence of sequences
        Band indexes (0-based) to read from each folder.
    window_size : int
        Size of the central window to extract.

    Returns:
    --------
    TensorImage
        The merged multi-channel window (C, H, W) as a FastAI TensorImage.
    """
    fn = Path(fn)
    filename = fn.name
    base_folder = fn.parent.parent

    with rasterio.open(fn) as ref_src:
        width, height = ref_src.width, ref_src.height
        col_off = max((width - window_size) // 2, 0)
        row_off = max((height - window_size) // 2, 0)
        window = Window(col_off, row_off, window_size, window_size)

    patch_list = []
    with rasterio.Env():
        for folder_idx, folder_name in enumerate(data_folders):
            file_path = base_folder / folder_name / filename
            with rasterio.open(file_path) as src:
                ch_indexes = [ch + 1 for ch in channels[folder_idx]]
                data_patch = src.read(
                    indexes=ch_indexes,
                    window=window,
                    boundless=False,
                    fill_value=0,
                )
                patch_list.append(data_patch)

    arr = np.concatenate(patch_list, axis=0).astype(np.float32)
    return TensorImage(torch.from_numpy(arr))


def load_dummy_mask(fn, window_size=1000):
    """
    Creates a dummy mask of zeros with dimensions window_size x window_size.
    
    Parameters:
    -----------
    fn : str
        Path to the image file (unused, kept for compatibility)
    window_size : int
        Size of the mask to create
        
    Returns:
    --------
    PILMask
        A dummy mask as a FastAI PILMask
    """
    mask_arr = np.zeros((window_size, window_size), dtype=np.uint8)
    return PILMask.create(mask_arr)
