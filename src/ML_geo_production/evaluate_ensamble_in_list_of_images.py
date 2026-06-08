#!/usr/bin/env python
"""
Evaluate an ensemble of segmentation models on a labeled benchmark set.

Computes global pixel classification accuracy (matching ML_sdfi_fastai2 eval.py):
fraction of non-ignored label pixels predicted correctly, pooled across all images.

Which images are evaluated
--------------------------
All images listed in the JSON key ``path_to_all_benchmarkset_txt``. Each non-empty
line is resolved under ``path_to_images``. Lines that do not contain ``im_type``
(typically ``.tif``) are skipped. Labels are loaded from ``path_to_labels`` using
the same filename as each image.

The JSON must define the complete ensemble (saved_models, model_names, means,
stds, channels, data_types, etc.) plus evaluation paths. No shapefiles or
geopackage are required.
"""

import argparse
import json
import sys
from copy import deepcopy
from pathlib import Path

import numpy as np
import rasterio
import torch
from rasterio.warp import Resampling, reproject

from ML_geo_production import model_utils
from ML_geo_production import process_images

ENSEMBLE_KEYS = (
    "saved_models",
    "model_names",
    "means",
    "stds",
    "channels",
    "data_types",
    "n_classes",
    "resolution",
    "patch_size",
    "overlap",
    "batch_size",
)

EVAL_KEYS = (
    "path_to_images",
    "path_to_labels",
    "path_to_all_benchmarkset_txt",
)


def load_eval_config(config_path):
    """Load JSON config, validate required keys, normalize workers -> num_workers."""
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")

    with open(config_path, "r") as f:
        cfg = json.load(f)

    missing = [k for k in ENSEMBLE_KEYS + EVAL_KEYS if k not in cfg]
    if missing:
        raise ValueError(f"Config {config_path} missing required keys: {missing}")

    n_models = len(cfg["saved_models"])
    for key in ("model_names", "means", "stds", "channels", "data_types"):
        if len(cfg[key]) != n_models:
            raise ValueError(
                f"Config {config_path}: length of '{key}' ({len(cfg[key])}) "
                f"must match saved_models ({n_models})"
            )

    if "num_workers" not in cfg and "workers" in cfg:
        cfg["num_workers"] = cfg["workers"]
    cfg.setdefault("num_workers", 4)
    cfg.setdefault("pixel_buffer", 0)
    cfg.setdefault("ignore_index", 0)
    cfg.setdefault("im_type", ".tif")
    cfg.setdefault("remove_matching_label", False)
    cfg.setdefault("only_use_these_models_index", None)

    cfg["path_to_images"] = str(Path(cfg["path_to_images"]).resolve())
    cfg["path_to_labels"] = str(Path(cfg["path_to_labels"]).resolve())
    cfg["path_to_all_benchmarkset_txt"] = str(
        Path(cfg["path_to_all_benchmarkset_txt"]).resolve()
    )

    return cfg


def resolve_benchmark_images(cfg):
    """
    Read path_to_all_benchmarkset_txt, filter by im_type, resolve under path_to_images.
    Skips missing files with a warning.
    """
    txt_path = Path(cfg["path_to_all_benchmarkset_txt"])
    im_type = cfg["im_type"]
    path_to_images = Path(cfg["path_to_images"])

    lines = txt_path.read_text().split("\n")
    image_paths = []
    for line in lines:
        line = line.strip()
        if not line or im_type not in line:
            continue
        resolved = (path_to_images / Path(line)).resolve()
        if not resolved.exists():
            print(f"Warning: image not found, skipping: {resolved}")
            continue
        image_paths.append(str(resolved))

    return image_paths


def reproject_label_to_pred_grid(label_path, pred_shape, pred_transform, pred_crs):
    """Reproject label band 1 onto the prediction grid (nearest-neighbor)."""
    with rasterio.open(label_path) as src:
        src_data = src.read(1)
        src_transform = src.transform
        src_crs = src.crs

    dst = np.zeros(pred_shape, dtype=np.int64)
    reproject(
        source=src_data,
        destination=dst,
        src_transform=src_transform,
        src_crs=src_crs,
        dst_transform=pred_transform,
        dst_crs=pred_crs,
        resampling=Resampling.nearest,
    )
    return dst


def accumulate_pixel_accuracy(pred_im, label_im, ignore_index):
    """Return (correct, total) for pixels where label != ignore_index."""
    ignore_index = int(ignore_index)
    mask = label_im != ignore_index
    total = int(mask.sum())
    if total == 0:
        return 0, 0
    correct = int((pred_im[mask] == label_im[mask]).sum())
    return correct, total


def evaluate_image(cfg, image_path, model_states):
    """
    Run ensemble inference on one image and compare to its label raster.
    Returns (correct, total) pixel counts.
    """
    image_path = Path(image_path)
    label_path = Path(cfg["path_to_labels"]) / image_path.name
    if not label_path.exists():
        raise FileNotFoundError(f"Label not found for {image_path.name}: {label_path}")

    with rasterio.open(image_path) as src:
        b = src.bounds
        bounds = (b.left, b.bottom, b.right, b.top)

    parsed = deepcopy(cfg)
    parsed["image_paths"] = [str(image_path.resolve())]
    parsed["bounds"] = bounds
    parsed["model_states"] = model_states

    final_probability_array, final_transform, dst_crs = (
        process_images.process_images_from_dict(parsed)
    )
    pred_im = final_probability_array.argmax(axis=0).astype(np.int64)
    label_im = reproject_label_to_pred_grid(
        label_path,
        pred_im.shape,
        final_transform,
        dst_crs,
    )
    return accumulate_pixel_accuracy(pred_im, label_im, cfg["ignore_index"])


def print_device_banner():
    print("##########################################")
    if torch.cuda.is_available():
        device_name = torch.cuda.get_device_name(torch.cuda.current_device())
        print(f"PyTorch is using GPU: {device_name}")
    else:
        print("PyTorch is using CPU")
    print("##########################################")


def run_eval(config_path):
    """Evaluate one JSON ensemble config; print global pixel accuracy."""
    cfg = load_eval_config(config_path)
    image_paths = resolve_benchmark_images(cfg)

    if not image_paths:
        print(
            f"No images to evaluate for config {config_path} "
            f"(check path_to_all_benchmarkset_txt and path_to_images).",
            file=sys.stderr,
        )
        return 0.0

    print_device_banner()

    print(
        f"Evaluating {len(image_paths)} images listed in path_to_all_benchmarkset_txt "
        f"({cfg['path_to_all_benchmarkset_txt']}), resolved under path_to_images "
        f"({cfg['path_to_images']}), im_type={cfg['im_type']}"
    )

    print(f"Preloading model weights for {Path(config_path).name}...")
    model_states = model_utils.preload_model_states(cfg["saved_models"])

    correct = 0
    total = 0
    n_evaluated = 0

    for image_path in image_paths:
        print(f"\n--- {Path(image_path).name} ---")
        try:
            img_correct, img_total = evaluate_image(cfg, image_path, model_states)
        except Exception as exc:
            print(f"Error evaluating {image_path}: {exc}", file=sys.stderr)
            continue
        correct += img_correct
        total += img_total
        n_evaluated += 1
        if img_total > 0:
            print(
                f"  image pixel accuracy: {img_correct / img_total:.6f} "
                f"({img_correct}/{img_total} pixels)"
            )
        else:
            print("  image pixel accuracy: no valid pixels (all ignored)")

    accuracy = (correct / total) if total > 0 else 0.0

    print(f"\nImages evaluated: {n_evaluated}")
    print(f"ignore_index: {cfg['ignore_index']}")
    print(f"Pixel accuracy: {accuracy:.6f} ({accuracy * 100:.2f}%)")
    return accuracy


if __name__ == "__main__":
    usage_example = (
        "Example usage:\n"
        "python src/ML_geo_production/evaluate_ensamble_in_list_of_images.py \\\n"
        "  --config config_files/evaluate_ensemble_example.json\n"
        "python src/ML_geo_production/evaluate_ensamble_in_list_of_images.py \\\n"
        "  --config ensemble_a.json ensemble_b.json\n"
    )
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate ensemble pixel accuracy on a labeled benchmark image list "
            "(JSON config per ensemble)."
        ),
        epilog=usage_example,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "-c",
        "--config",
        nargs="+",
        required=True,
        help="One or more paths to JSON ensemble evaluation config files",
    )
    args = parser.parse_args()

    for config_path in args.config:
        print(f"\n{'=' * 60}")
        print(f"Config: {config_path}")
        print("=" * 60)
        run_eval(config_path)
