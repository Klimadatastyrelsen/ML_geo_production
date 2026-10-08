"""Deprecated: use download_upload_models_hf.py for downloading models from Hugging Face."""
import os
import requests
from typing import List, Dict, Union

# Define the file mapping. The key is the final desired filename.
# The value is either a single URL (for single files) or a list of URLs
# (for multi-part files, in the correct order for concatenation).
FILE_MAP: Dict[str, Union[str, List[str]]] = {
    # Single files
    "andringsudpegning_1km2benchmark_iter_52_24.pth":
        "https://github.com/Klimadatastyrelsen/ML_geo_production/releases/download/v1.0.0/andringsudpegning_1km2benchmark_iter_52_24.pth",
    "andringsudpegning_1km2benchmark_iter_63_24.pth":
        "https://github.com/Klimadatastyrelsen/ML_geo_production/releases/download/v1.0.0/andringsudpegning_1km2benchmark_iter_63_24.pth",

    # Concatenated file: The parts are listed in order
    "andringsudpegning_1km2benchmark_iter_62_24.pth": [
        "https://github.com/Klimadatastyrelsen/ML_geo_production/releases/download/v1.0.0/andringsudpegning_1km2benchmark_iter_62_24.pth.partaa",
        "https://github.com/Klimadatastyrelsen/ML_geo_production/releases/download/v1.0.0/andringsudpegning_1km2benchmark_iter_62_24.pth.partab",
    ]
}

def download_file(url: str, output_path: str):
    """Downloads a single file from a URL and saves it to output_path."""
    print(f"  Downloading: {os.path.basename(output_path)}")
    
    # Add a User-Agent header to prevent issues with GitHub's CDN/redirects
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}

    try:
        # Use stream=True for efficient handling of large files
        with requests.get(url, stream=True, headers=headers) as r:
            r.raise_for_status() # Raise an exception for bad status codes (4xx or 5xx)

            # Write data in chunks to prevent high memory usage
            with open(output_path, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192): # 8 KiB chunks
                    if chunk:
                        f.write(chunk)

            print(f"  Successfully downloaded {os.path.basename(output_path)}.")
    except requests.exceptions.RequestException as e:
        print(f"  ERROR: Could not download {url}. Reason: {e}")
        # Clean up partial file on error
        if os.path.exists(output_path):
            os.remove(output_path)
        raise # Re-raise the exception so the caller can handle continuing or stopping

def concatenate_files(part_paths: List[str], final_path: str):
    """Concatenates a list of file parts into a single final file."""
    print(f"  Concatenating {len(part_paths)} parts into {os.path.basename(final_path)}...")
    try:
        # Open the final file in binary write mode
        with open(final_path, 'wb') as outfile:
            for part_path in part_paths:
                print(f"    Adding part: {os.path.basename(part_path)}")
                # Open each part in binary read mode
                with open(part_path, 'rb') as infile:
                    # Read and write in chunks (8MB)
                    while True:
                        chunk = infile.read(8192 * 1024)
                        if not chunk:
                            break
                        outfile.write(chunk)
        print(f"  Successfully created {os.path.basename(final_path)}.")

        # Clean up temporary part files after successful concatenation
        for part_path in part_paths:
            os.remove(part_path)
            print(f"  Cleaned up part: {os.path.basename(part_path)}")

    except Exception as e:
        print(f"  ERROR: Failed to concatenate files into {os.path.basename(final_path)}. Reason: {e}")
        # Clean up the final file if concatenation failed
        if os.path.exists(final_path):
            os.remove(final_path)
        raise # Re-raise the exception

def download_models(output_dir: str = "./models/"):
    """
    Downloads model files from the specified GitHub release, handling
    single files and concatenated multi-part files. This function is resilient
    and will skip failed downloads to continue processing other files.

    Args:
        output_dir: The directory where the downloaded models will be saved.
    """
    print(f"Starting model download process to directory: {output_dir}")

    # 1. Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)
    print("Output directory ensured.")

    # 2. Iterate through the files to download
    for final_filename, urls in FILE_MAP.items():
        print(f"\nProcessing file: {final_filename}")

        final_path = os.path.join(output_dir, final_filename)

        # Check if the final file already exists and skip if so
        if os.path.exists(final_path):
            print(f"  Skipping: File already exists at {final_path}")
            continue

        if isinstance(urls, str):
            # Case 1: Single file download
            try:
                download_file(urls, final_path)
            except Exception:
                # Logged in download_file, now skip to the next file
                continue 

        elif isinstance(urls, list) and urls:
            # Case 2: Multi-part file download and concatenation
            part_paths = []
            try:
                # a. Download all parts
                all_parts_successful = True
                for url in urls:
                    part_filename = url.split("/")[-1] # Extract filename from URL
                    part_path = os.path.join(output_dir, part_filename)
                    part_paths.append(part_path)

                    # Only download if the part file doesn't exist (supports resuming)
                    if os.path.exists(part_path):
                         print(f"  Part file already exists: {os.path.basename(part_path)}. Skipping download.")
                         continue

                    try:
                        download_file(url, part_path)
                    except Exception:
                        all_parts_successful = False
                        break # Stop downloading parts if one fails

                # b. Concatenate the parts only if all parts were successfully downloaded
                if all_parts_successful:
                    concatenate_files(part_paths, final_path)

            except Exception:
                # Handle cleanup of parts and continue to the next file
                print(f"Download or concatenation failed for {final_filename}. Cleaning up partial files.")
                for p_path in part_paths:
                    if os.path.exists(p_path):
                        os.remove(p_path)
                        print(f"  Cleaned up failed part: {os.path.basename(p_path)}")
                continue # Continue to the next file in FILE_MAP

    print("\nModel download and assembly process completed successfully!")


if __name__ == '__main__':
    # You can call the function here to test it:
    # Requires 'requests' package: pip install requests
    try:
        download_models()
    except Exception:
        print("\nProcess ended due to an unhandled error.")
