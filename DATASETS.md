# DATASETS.md

Real public datasets used by the platform. **Row counts and schemas below were verified by
inspecting the actual parquet files** (via the HuggingFace `datasets-server` parquet/size endpoints
and direct schema reads). Datasets are downloaded **only** by the offline trainer — the Streamlit app
never downloads or trains.

> Licences are the responsibility of the original dataset authors. Check each source before
> redistribution or commercial use. Restricted/registered datasets (e.g. BraTS) are **not**
> auto-downloaded; where such data is needed the required manual step is documented.

## Computer-vision datasets (medical & agriculture)

| key | Source (HuggingFace) | Task | Rows (verified) | Splits | Classes | Annotation |
|---|---|---|---|---|---|---|
| `brain_mri_cls` | `PranomVignesh/MRI-Images-of-Brain-Tumor` | 4-class MRI classification | 5,369 | train / validation / test | glioma, meningioma, notumor, pituitary | image + int label |
| `plant_cls` | `Dipra444/plant-leaf-disease-jharkhand` | 25-class leaf classification | 6,045 | single `train` → stratified 70/15/15 (seed 42) | 25 (PlantVillage label set) | image (**path-only**, fetched from the HF resolve endpoint) + int label |
| `brain_det` | `mmenendezg/brain-tumor-object-detection` | tumour detection | 1,101 | train / validation / test | 2 un-named numeric labels (`label_0`, `label_1`) | COCO boxes (`objects.bbox` x,y,w,h + `label`) |
| `cotton_det` | `Francesco/cotton-plant-disease` | cotton disease detection | 1,798 | train / validation / test | `cotton-plant-disease`, `dc` | COCO boxes (`objects.bbox` + `category`). **Boxes are ~2.5×2.5 px in 640×640 images** |
| `plant_seg` | `Mauregato/leaf_disease_segmentation` | leaf-disease segmentation | 2,940 | train (2 shards) / val / test | background, disease | **pixel masks** (PNG `mask` column), class-index encoded (0/38) |
| `brain_instseg` | `dwb2023/brain-tumor-image-dataset-semantic-segmentation` | tumour instance segmentation | 2,146 | train / valid / test | 2 categories (`category_id` 1 and 2) | **COCO polygons** (`segmentation` + `bbox` + `category_id`) |

### Notes on individual datasets

* **`plant_cls`** stores images as a *path* (`hf://datasets/…`) rather than inline bytes. The loader
  percent-encodes the path, fetches it from the HuggingFace resolve endpoint, and **caches it on disk**
  under `.cache/images/` (with retries) so a multi-epoch run does not re-download the same JPEG.
* **`cotton_det`** annotates disease regions as tiny spots. The registry documents this: a single-scale
  14×14-grid detector cannot represent a sub-cell object, so `plant_yolo` mAP@50 is expected to be very
  low. This is a **data-scale limitation**, reported honestly, not a training bug.
* **`brain_instseg`** `category_id` values are 1 and 2 (two tumour categories); the loader maps them to
  0-based class indices. Rows whose polygon fails to rasterise or whose box is degenerate are dropped.
* **`plant_maskrcnn` is BLOCKED**: the only candidate agriculture instance-segmentation dataset
  (`daiv05/corn-leaf-instance-segmentation`) has **no parquet conversion** on the datasets-server
  (0 files), so it cannot be fetched offline. Semantic masks from `plant_seg` are **not** converted into
  fake object instances.

### Medical dataset candidates considered

* **BraTS** (brain tumour segmentation) — the canonical MRI segmentation benchmark. Requires
  **registration**; it is *not* auto-downloadable. To use it, download manually, place NIfTI files
  under a local path, and point the loader at them. Documented, not faked.

### Leakage prevention

* Classification uses the **source dataset's own train/validation/test split** where provided
  (`brain_mri_cls`, `brain_det`, `cotton_det`, `plant_seg`, `brain_instseg`).
* Where a dataset has a single split (`plant_cls`), a **class-stratified split with a fixed seed (42)**
  is applied and recorded in the artifact metadata (`split_seed`).
* Cached parquet files are named `<split>-<index>.parquet` and `files_for(key, split)` canonicalises
  `validation` / `valid` / `val` → `validation`, so a loader that asks for one split **never sees
  another split's rows**. This is covered by `test_files_for_never_returns_other_splits` and
  `test_missing_split_raises_not_silently_empty`.
* Medical models use the source splits; **patient-disjointness is inherited from the source dataset**
  and is not independently re-verified here — a documented limitation.

## Laboratory datasets (the 20 models)

| key | Source | Task | Notes |
|---|---|---|---|
| `breast_cancer` | UCI Breast Cancer Wisconsin (Diagnostic), via scikit-learn | binary classification / tabular diffusion | 569 samples, 30 features, bundled with scikit-learn (no network) |
| `mnist` | MNIST (LeCun et al.), via torchvision | 10-class digit classification / reconstruction / GAN | 60k train / 10k test |
| `mnist_seg` | derived from MNIST | foreground segmentation | mask = intensity > 0.15 |
| `mnist_det` | derived from MNIST | digit localisation | box = ink extent, normalised xywh |
| `har` | UCI Smartphone Human Activity Recognition | 6-class sequence classification | raw inertial signals; UCI archive |
| `newsgroups` | `SetFit/20_newsgroups` (HF mirror of 20 Newsgroups) | 4-class text classification / char LM | filtered to 4 categories via `label_text` |
| `cora` | Planetoid `ind.cora.*` (Yang et al.) | 7-class node classification | 2,708 nodes, 1,433 features |

## How splits and preprocessing are recorded

Each artifact's `artifacts/metadata/<id>.json` stores:

```json
{
  "dataset": "...", "dataset_key": "...", "classes": ["..."],
  "input": {"size": 224, "channels": 3, "mean": [0.485,0.456,0.406], "std": [0.229,0.224,0.225]},
  "metrics": {"accuracy": 0.0},
  "split_manifest": {"train": 0, "validation": 0, "test": 0},
  "training": {"epochs": 4, "lr": 0.001, "split_seed": 42, "backbone": "..."},
  "artifact_version": "2.0.0", "git_commit": "..."
}
```

Inference reads `input.size` / `input.mean` / `input.std` from this file, so preprocessing is
guaranteed identical between training and prediction.

Each run also writes a structured per-model record (`artifacts/status/<id>.json`) and a run summary
(`artifacts/status/summary.json`) with the status (`TRAINED_VERIFIED` / `DEFERRED` / `BLOCKED` / …),
checkpoint path + size, measured metrics and duration.
