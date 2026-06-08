# ML_geo_production

**ML_geo_production** is a geospatial ML pipeline that runs one or more
PyTorch semantic-segmentation models over GeoTIFF imagery, ensembles
their outputs, and produces combined, normalized probability arrays for
areas of interest. It processes only the area of the raster that intersect a
defined bounding box, and can optionally compare
predictions with polygon data (e.g GeoPackage) for change detection.

For questions about the repo, email rajoh@kds.dk 

------------------------------------------------------------------------

![change_detection2](https://github.com/user-attachments/assets/d9607467-81fb-4e05-b0ac-246103c8c07a)

------------------------------------------------------------------------
## Features

-   Model ensembling for multiple PyTorch semantic-segmentation models\
-   Processes only GeoTIFF regions that intersect the area of interest\
-   Outputs a single combined & normalized probability array for the
    AOI\
-   Optional comparison against reference polygons for change
    detection\
-   Example configs work with [https://github.com/SDFIdk/multi_channel_dataset_creation]dataset and pretrained models

------------------------------------------------------------------------

## Installation

### Conda version

Use **conda** or **mamba** (Miniforge includes conda; mamba is optional). Clone all four shared-env repos as siblings (`ML_Production`, `ML_geo_production`, `ML_sdfi_fastai2`, `multi_channel_dataset_creation`), then run the steps below **from this repository root**. The same files and commands exist in each repo and produce the same `ML_sdfi` environment.

```sh
conda env create --file environment.yml   # once
conda activate ML_sdfi

bash install_pytorch.sh
pip install --pre --no-build-isolation -r requirements_pip.txt
bash install_local_repos.sh
pip install -r requirements_extra.txt
```

`install_pytorch.sh` auto-selects the PyTorch CUDA build (nightly cu128 for Blackwell / sm_12.0, stable cu124 for other NVIDIA GPUs). CUDA is required. Override with e.g. `PYTORCH_CUDA=cu121 bash install_pytorch.sh` (see [pytorch.org/get-started/locally](https://pytorch.org/get-started/locally)).

**Verify CUDA support:**

```sh
python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('Device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')"
```

**Windows:** After the steps above, run once: `pip install --force-reinstall pillow rasterio` so PIL and rasterio use pip's Windows wheels.

### Docker version

Pull the prebuilt image and run with this repo as working directory:

```bash
docker pull rasmuspjohansson/kds_cuda_pytorch:latest

docker run --gpus all --shm-size=100g -it \
  -v /path/to/your/projects:/home/projects \
  -w /home/projects/ML_geo_production \
  rasmuspjohansson/kds_cuda_pytorch:20260302 \
  bash
```

If you need all four shared-env repos installed in the container, run once from ML_Production (e.g. first run with `-w /home/projects/ML_Production` and `sh install_local_repos.sh && pip install -r requirements_extra.txt`), then use `-w /home/projects/ML_geo_production` for this repo.

Example run after setup:

```bash
python src/ML_geo_production/process_images.py --json config_files/save_probs_preds_and_change_detection.json
```

### Clone the example dataset repo side-by-side

Clone the dataset repository so it sits next to this repository in the
same parent directory:

``` bash
git clone https://github.com/SDFIdk/multi_channel_dataset_creation
```

The example config files in `config_files/` work **out of the box** with
the dataset included in that repository.

------------------------------------------------------------------------

### Download example models

Place a Hugging Face token file in the repo root (e.g. `my_hugging_face_token.txt`) or use the same token file as ML_Production. When repos are cloned as siblings, models are stored once under **`../ML_Production/models/`** (this repo's `./models/` may be a symlink). Then run:

```bash
python src/ML_geo_production/download_upload_models_hf.py --download --token_file ./my_hugging_face_token.txt --file_path ../ML_Production/models/
```

This downloads all `.pth` model files from the default Hugging Face repo. Use `--repo_id` if your models are in a different repo.

**Note:** The example models were trained using the training code from: https://github.com/SDFIdk/ML_model_training

------------------------------------------------------------------------

## Basic Usage

Main processing script:

``` bash
python src/ML_geo_production/process_images.py --json config_files/save_probs_preds_and_change_detection.json
```
Example above use an ensamble of three models, save both probs, preds and change detection.


Example workflows:

``` bash
python src/ML_geo_production/do_change_detection.py --json config_files/change_detection.json
python src/ML_geo_production/create_prediction_raster.py --json config_files/raster_production.json
python src/ML_geo_production/process_many_areas.py --json config_files/process_many_areas.json  --shapefile ../multi_channel_dataset_creation/example_dataset/shape_files/many_areas.shp
```
The process_many_areas.py example above shows an example of how to process many areas after each other.

------------------------------------------------------------------------

## Verify that everything works

Set `export GTIFF_SRS_SOURCE=EPSG` before running (optional; suppresses PROJ/GDAL warnings).

Manual check — run the Quick Start steps (clone dataset repo, download models if needed), then:

```bash
python src/ML_geo_production/process_images.py --json config_files/save_probs_preds_and_change_detection.json
```

There should be no errors in the output.

Automated verification (CUDA check — fails if CUDA unavailable; downloads models to shared `../ML_Production/models/` when `my_hugging_face_token.txt` is present; runs the config above; writes `verification.log`):

```bash
python verify_functionality.py
python check_logs.py
```

------------------------------------------------------------------------
## Model evaluation and summarization

### evaluate_models.py

Runs one or more configs over shapefile-defined areas: builds label rasters from the config’s geopackage, runs inference, and writes per-area classification stats (IoU, pixel accuracy, F1, etc.) plus prediction and difference rasters.

**Arguments:**

-   `--config`: One or more JSON config paths; glob patterns supported (e.g. `path/to/change_detection_5_models_2026_SOTA_*`). Default: `config_files/change_detection.json`.
-   `--shape`: One or more `.shp` or `.gpkg` paths (required).
-   `--image_folder`: Path to folder containing input images (required).
-   `--output_folder`: Path where label, stats, prediction and diff files are written (required).

**Example:**

``` bash
python src/ML_geo_production/evaluate_models.py \
  --config config_files/change_detection.json \
  --shape /path/to/areas.shp \
  --image_folder /path/to/rooftop_rgb \
  --output_folder /path/to/evaluations
```

**Outputs:** For each (config, shape, feature): label `.tif`, stats `.md`, prediction `.tif` (`_pred_im.tif`), and difference `.tif` (`_label_pred_diff_im.tif`; 0=agree, 1=FP, 2=FN, 3=wrong class). The config must include a `geopackage` key for label creation.

### evaluate_ensamble_in_list_of_images.py

Evaluates a **complete ensemble** defined in a JSON config on a labeled benchmark image list. Computes global pixel accuracy (same metric as `ML_sdfi_fastai2/eval.py`): fraction of non-ignored label pixels predicted correctly, pooled across all images. Unlike `evaluate_models.py`, no shapefiles or geopackage are required — labels come from `path_to_labels` and the image list from `path_to_all_benchmarkset_txt`.

**Arguments:**

-   `--config` / `-c`: One or more JSON config paths; each ensemble is evaluated in order (stdout only).

**JSON must include:** ensemble keys (`saved_models`, `model_names`, `means`, `stds`, `channels`, `data_types`, `n_classes`, `resolution`, `patch_size`, `overlap`, `batch_size`) plus `path_to_images`, `path_to_labels`, `path_to_all_benchmarkset_txt`. Optional: `ignore_index` (default `0`), `im_type` (default `.tif`), `pixel_buffer`, `only_use_these_models_index`.

**Example:**

``` bash
python src/ML_geo_production/evaluate_ensamble_in_list_of_images.py \
  --config config_files/evaluate_ensemble_example.json

python src/ML_geo_production/evaluate_ensamble_in_list_of_images.py \
  --config ensemble_a.json ensemble_b.json
```

**Output:** Prints global pixel accuracy per config to stdout (no files written).

### summarize_evaluations.py

Reads evaluation `.md` files from a folder, extracts a chosen metric and inference time, and writes a summary markdown table (score, inference minutes, filename) plus model-index mapping.

**Arguments:**

-   `--folder`: Folder containing evaluation `.md` files (default: `/mnt/T/mnt/ML_output/building_change_detection_2026/evaluations`).
-   `--area`: Substring to filter files, e.g. `parcellhuse` (default: `parcellhuse`).
-   `--output_directory`: Where to write the summary `.md` (default: same as folder).
-   `--statistic`: Metric to extract and sort by (default: `Pixel accuracy`).
-   `--original_config`: JSON with `model_names` for the index mapping (default: `config_files/change_detection_5_models_2026_SOTA.json`).

**Example:**

``` bash
python src/ML_geo_production/summarize_evaluations.py \
  --folder /path/to/evaluations --area parcellhuse \
  --statistic "Pixel accuracy"
```

**Output:** One markdown file per area/statistic (e.g. `parcellhuse-Pixel_accuracy.md`) with a table of score, inference (min), and filename, plus a model index mapping section.

------------------------------------------------------------------------

## Config Files

See examples in `config_files/` --- these are ready to run using the
dataset from\
https://github.com/SDFIdk/multi_channel_dataset_creation

------------------------------------------------------------------------

## Inputs & Requirements

-   GeoTIFFs must share a compatible CRS\
-   Models must output probability tensors\
-   Optional polygon layers must match the CRS (or will be reprojected
    automatically)

------------------------------------------------------------------------

## Outputs

-   Combined, normalized probability arrays\
-   Prediction raster outputs\
-   Change-detection results when polygon comparison is enabled

------------------------------------------------------------------------

## License (MIT)

    MIT License

    Copyright (c) 2025

    Permission is hereby granted, free of charge, to any person obtaining a copy
    of this software and associated documentation files (the "Software"), to deal
    in the Software without restriction, including without limitation the rights
    to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
    copies of the Software, and to permit persons to whom the Software is
    furnished to do so, subject to the following conditions:

    The above copyright notice and this permission notice shall be included in all
    copies or substantial portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
    OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
    SOFTWARE.
