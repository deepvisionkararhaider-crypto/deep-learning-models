# Model Status

**Generated from `models/manifest.json` — do not edit by hand.**

- Generated: `2026-10-18T20:04:29Z`
- Hugging Face repo: [`KararDeepVision/deep-learning-models`](https://huggingface.co/KararDeepVision/deep-learning-models)
- Pinned revision: `aeab7749122941325aa52f916e942b01b5011dcd`
- Total models: **32**

## Status summary

| Status | Count | Meaning |
|---|---|---|
| `TRAINED_VERIFIED` | 1 | Weights published and re-verified by fresh download + inference |
| `TRAINED_UNVERIFIED` | 16 | Weights exist locally; not yet published/verified |
| `ARCH_ONLY` | 13 | Architecture defined and importable; no weights |
| `BLOCKED` | 1 | Cannot proceed; blocker recorded in the manifest `note` |
| `DEFERRED` | 1 | Deliberately postponed |

## Per-model status

| Model | Area | Task | Modality | Status | HF path |
|---|---|---|---|---|---|
| `brain_cnn` | medical | classification | image | `TRAINED_VERIFIED` | `medical/brain_cnn/model.pt` |
| `brain_densenet` | medical | classification | image | `ARCH_ONLY` | `medical/brain_densenet/model.pt` |
| `brain_vit` | medical | classification | image | `ARCH_ONLY` | `medical/brain_vit/model.pt` |
| `brain_unet` | medical | segmentation | image | `ARCH_ONLY` | `medical/brain_unet/model.pt` |
| `brain_yolo` | medical | detection | image | `ARCH_ONLY` | `medical/brain_yolo/model.pt` |
| `brain_maskrcnn` | medical | instance_segmentation | image | `DEFERRED` | `medical/brain_maskrcnn/model.pt` |
| `plant_cnn` | agriculture | classification | image | `ARCH_ONLY` | `agriculture/plant_cnn/model.pt` |
| `plant_densenet` | agriculture | classification | image | `ARCH_ONLY` | `agriculture/plant_densenet/model.pt` |
| `plant_vit` | agriculture | classification | image | `ARCH_ONLY` | `agriculture/plant_vit/model.pt` |
| `plant_unet` | agriculture | segmentation | image | `ARCH_ONLY` | `agriculture/plant_unet/model.pt` |
| `plant_yolo` | agriculture | detection | image | `ARCH_ONLY` | `agriculture/plant_yolo/model.pt` |
| `plant_maskrcnn` | agriculture | instance_segmentation | image | `BLOCKED` | `agriculture/plant_maskrcnn/model.pt` |
| `lab01` | lab | tabular_classification | tabular | `TRAINED_UNVERIFIED` | `laboratories/lab01/model.pt` |
| `lab02` | lab | classification | image | `TRAINED_UNVERIFIED` | `laboratories/lab02/model.pt` |
| `lab03` | lab | sequence_classification | sequence | `TRAINED_UNVERIFIED` | `laboratories/lab03/model.pt` |
| `lab04` | lab | sequence_classification | sequence | `TRAINED_UNVERIFIED` | `laboratories/lab04/model.pt` |
| `lab05` | lab | sequence_classification | sequence | `TRAINED_UNVERIFIED` | `laboratories/lab05/model.pt` |
| `lab06` | lab | reconstruction | image | `TRAINED_UNVERIFIED` | `laboratories/lab06/model.pt` |
| `lab07` | lab | reconstruction | image | `TRAINED_UNVERIFIED` | `laboratories/lab07/model.pt` |
| `lab08` | lab | binary_classification | image | `TRAINED_UNVERIFIED` | `laboratories/lab08/model.pt` |
| `lab09` | lab | text_classification | text | `TRAINED_UNVERIFIED` | `laboratories/lab09/model.pt` |
| `lab10` | lab | text_classification | text | `TRAINED_UNVERIFIED` | `laboratories/lab10/model.pt` |
| `lab11` | lab | language_modeling | text | `TRAINED_UNVERIFIED` | `laboratories/lab11/model.pt` |
| `lab12` | lab | classification | image | `ARCH_ONLY` | `laboratories/lab12/model.pt` |
| `lab13` | lab | classification | image | `ARCH_ONLY` | `laboratories/lab13/model.pt` |
| `lab14` | lab | classification | image | `ARCH_ONLY` | `laboratories/lab14/model.pt` |
| `lab15` | lab | segmentation | image | `TRAINED_UNVERIFIED` | `laboratories/lab15/model.pt` |
| `lab16` | lab | detection | image | `ARCH_ONLY` | `laboratories/lab16/model.pt` |
| `lab17` | lab | metric_learning | image_pair | `TRAINED_UNVERIFIED` | `laboratories/lab17/model.pt` |
| `lab18` | lab | sequence_to_sequence | text | `TRAINED_UNVERIFIED` | `laboratories/lab18/model.pt` |
| `lab19` | lab | tabular_denoising | tabular | `TRAINED_UNVERIFIED` | `laboratories/lab19/model.pt` |
| `lab20` | lab | graph_classification | graph | `TRAINED_UNVERIFIED` | `laboratories/lab20/model.pt` |

## Verified models

### `brain_cnn` — CNN (brain MRI)

- Architecture: `cnn` (PyTorch (from scratch))
- Task: classification · Modality: image
- Classes: glioma, meningioma, no-tumor, pituitary
- accuracy: **0.8585** (test n=537)
- precision_macro: **0.858** (test n=537)
- recall_macro: **0.8557** (test n=537)
- f1_macro: **0.8555** (test n=537)
- Weights: 980608 bytes · sha256 `97e2b9eb767e2af0687346b63a8d302d64bb074184ab42771166bd46fcfb8bd5`
- HF: `KararDeepVision/deep-learning-models/medical/brain_cnn/model.pt` @ `aeab7749122941325aa52f916e942b01b5011dcd`
- Limitations: Trained from scratch; MRI is grayscale replicated to 3 channels.

## Blocked and deferred

- **`brain_maskrcnn`** (DEFERRED): —
- **`plant_maskrcnn`** (BLOCKED): No agriculture instance-segmentation dataset is available on the HuggingFace datasets-server parquet mirror. daiv05/corn-leaf-instance-segmentation exists but has no parquet conversion (0 rows, config-size failed), so it cannot be fetched offline. This model has no trained weights and is not selectable.

## Remaining work

31 of 32 models still require training and/or publishing via GitHub Actions:

```text
  brain_densenet  brain_vit  brain_unet  brain_yolo  brain_maskrcnn  plant_cnn
  plant_densenet  plant_vit  plant_unet  plant_yolo  plant_maskrcnn  lab01
  lab02  lab03  lab04  lab05  lab06  lab07
  lab08  lab09  lab10  lab11  lab12  lab13
  lab14  lab15  lab16  lab17  lab18  lab19
  lab20
```
