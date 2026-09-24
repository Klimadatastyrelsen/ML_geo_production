# Note on input size: crop, patch and overlap settings

Applies to `ML_sdfi_fastai2` (training, validation) and `ML_geo_production`
(evaluation, production inference). The same copy of this note lives in both repos.

## The rule

**Every size the network sees must be a multiple of 32, and inference size should
equal training size.**

All segmentation architectures used here (fastai UNet on timm EfficientNet,
Swin + UPerNet, ConvNeXt + UPerNet, ResNet UNet) downsample by a factor of 32.
Sizes that are not multiples of 32 force the network to pad, drop or resample
rows somewhere in the middle, and those operations differ from what the model
saw during training.

| Size | 32-compatible | Fits in a 1000x1000 training tile | Use |
|------|---------------|-----------------------------------|-----|
| 512  | yes | yes | training when memory is tight (BatchNorm models) |
| 992  | yes | yes | preferred: training, validation, evaluation, production |
| 1000 | **no** | yes | do not use; this was the source of the 1000-vs-512 accuracy gaps |
| 1024 | yes | **no** (tiler skips tiles smaller than the patch) | production on large orthophotos only |

## Why 1000 hurt, per model

Sizes at each stage for a 1000 px input: 500, 250, 125, 63, 32.

- **fastai UNet + `tf_efficientnetv2_*`** (largest effect, ~2 pp accuracy):
  the decoder upsamples by exactly 2x and must nearest-neighbour resample
  64 to 63 and 126 to 125 to match the skip connections. Deep features drift up to
  one stride-32 pixel (~30 px) out of register toward the bottom-right of each
  patch. The TF "SAME" padding also switches from asymmetric to symmetric at
  the odd stages. None of this happens at 512 or 992.
- **ConvNeXt + UPerNet** (moderate): the 2x2/stride-2 downsampling drops the last
  row at odd sizes; deep maps cover 992 px but are stretched over 1000.
- **Swin + UPerNet** (small): Swin pads internally and UPerNet fuses bilinearly,
  so registration survives. The remaining gap was the validation population
  (see below), not the size.

At 992 the stages are 496, 248, 124, 62, 31: exact everywhere.

## Training (`ML_sdfi_fastai2`, `.ini` configs)

```ini
transforms = [..., "crop", ...]
crop_size  = [992, 992]      # Swin/ConvNeXt UPerNet, and any LayerNorm model
# crop_size = [512, 512]     # fastai UNet + EfficientNet (BatchNorm), see below
```

The `crop` transform installs a **random** crop for the training split and a
**centre** crop for the validation split (`transforms/sdfi_transforms.py`).
Consequences:

- With `crop_size = 512` the logged `pixel_accuracy` scores only the centre
  26% of each 1000x1000 validation tile, none of it near a tile border. It is
  optimistic compared with a full-tile evaluation and is **not comparable**
  with numbers from `evaluate_ensamble_in_list_of_images.py`.
- With `crop_size = 992` validation covers 98% of the tile and the two numbers
  measure almost the same thing.

Per architecture:

- **Swin / ConvNeXt + UPerNet: train at 992.** UPerNet's pyramid pooling module
  average-pools the whole feature map, so the model learns scene statistics at
  the training size. Train at the size you infer at.
  **The backbones are LayerNorm, but the UPerNet decoder head contains 13
  `BatchNorm2d` layers** (pyramid pooling and FPN fusion). Therefore:
  - `batch_size = 1` does not run in train mode at all: the PPM pools to 1x1
    and BatchNorm raises `Expected more than 1 value per channel`.
  - `batch_size = 2` runs but the head's BN statistics come from 2 samples.
  - Use a physical `batch_size` of at least 4, preferably 8, and set `n_acc`
    to keep the effective batch (Swin was 4x20 = 80 at 512; use 8x10 or 4x20
    at 992). Gradient accumulation does not help BN.
  - If memory forces batch 2, freeze the head's BN running statistics
    (`set_bn_eval` on the decode head / fastai `BnFreeze`).
  Measured peak memory at 992, fp16, one training step (3-channel input;
  extra input channels change only the first conv): Swin-small 8.7 GB at
  batch 2, ConvNeXt-base 9.1 GB at batch 2; roughly 4.4 GB per additional
  sample, so batch 4 ~18 GB and batch 8 ~35 GB. A 48 GB card that is not
  shared handles batch 8.
- **fastai UNet + EfficientNet: train at 512, infer at 992.** It has no global
  component, so once the input is a multiple of 32 the interior of a 992 patch
  is processed exactly as a 512 patch. Training at 992 would force
  `batch_size` ~2, which starves BatchNorm (gradient accumulation does not help
  BN). If you do train it at 992, freeze BN statistics (`BnFreeze`).
  Its logged `pixel_accuracy` stays a centre-512 number.
- **Epoch time** scales with pixels per crop: 992 is ~3.75x slower per epoch
  than 512. Random cropping shrinks to an 8 px range at 992 and stops being an
  augmentation; expect somewhat earlier overfitting and rely on the per-epoch
  checkpoints.

## Evaluation on the 1000x1000 tile set (`ML_geo_production`, `eval_*.json`)

```json
"patch_size": 992,
"overlap": 64
```

- The tiler (`patch_dataset.py`) only emits patches with
  `patch_size <= image size`. **1024 yields zero patches on 1000 px tiles and
  the tile is silently skipped.** Use 992 (or 512) on the tile set.
- 992 on a 1000 tile gives four patches (offsets 0 and 8), 4x the compute of a
  single patch; harmless for a benchmark.
- Use the same `patch_size` for all models in one comparison, and the same size
  the models were trained at (992 for UPerNet models; 992 is also correct for a
  512-trained EfficientNet UNet).
- Inference memory is small. Measured peak at 992 (11 classes):

  | model | fp32, batch 1 | fp16, batch 1 | fp16, batch 2 |
  |---|---|---|---|
  | swin-small-upernet | 1.5 GB | 1.8 GB | 3.3 GB |
  | convnext-base-upernet | 1.6 GB | 2.0 GB | 3.5 GB |
  | convnext-tiny-upernet | 1.4 GB | 1.6 GB | 3.1 GB |

  About 1.5 GB per extra patch, so `batch_size: 8` is ~12 GB. A 20 GB
  inference GPU is fine at 992 for all of these.

## Production on large orthophotos (`ML_geo_production`, production `.json`)

```json
"patch_size": 992,
"overlap": 128
```

- 992 rather than 1024 so inference is at the training scale of the UPerNet
  models. 1024 is only acceptable for models with no global component
  (EfficientNet/ResNet UNet).
- Overlapping patches are currently fused by **plain averaging**
  (`run_inference_and_accumulate` in `process_images.py`); border predictions
  are not discarded, only averaged at half weight in the overlap strip.
  Noted improvement: weight each patch by a ramp that is 0 at the border and
  1 at a margin `m` inward, and set `overlap >= 2m` (m of 64-128 px for
  512/992-trained models).

## Checklist when changing a size

1. Is it a multiple of 32?
2. Does it fit inside the smallest image the tiler will see?
3. Do training crop and inference patch match for UPerNet (global pooling) models?
4. Is the physical `batch_size` at least 4, preferably 8? This applies to
   EfficientNet/ResNet UNets **and** to UPerNet models (BatchNorm in the
   decoder head). `batch_size = 1` fails for UPerNet in train mode.
5. Are all models in a comparison evaluated at the same `patch_size`?
6. Remember that a 512 centre-crop validation number is not a full-tile number.
