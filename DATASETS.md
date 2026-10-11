# DATASETS.md

Real public datasets used by the platform. **All statistics below were verified by fetching the
actual files** (row counts from the HuggingFace `datasets-server` size endpoint; schemas inspected
directly from the parquet). Datasets are downloaded **only** by the offline trainer — the Streamlit
app never downloads or trains.

> Licences are the responsibility of the original dataset authors. Check each source before
> redistribution or commercial use. Restricted/registered datasets (e.g. BraTS) are **not**
> auto-downloaded; where such data is needed the required manual step is documented.

## Computer-vision datasets (medical & agriculture)

| key | Source (HuggingFace) | Task | Rows (verified) | Classes | Annotation | Licence note |
|---|---|---|---|---|---|---|
| `brain_mri_cls` | `PranomVignesh/MRI-Images-of-Brain-Tumor` | 4-class MRI classification | 5,369 | glioma, meningioma, notumor, pituitary | image + int label; ships train/validation/test splits | Kaggle-derived; verify original terms |
| `plant_cls` | `Dipra444/plant-leaf-disease-jharkhand` | 25-class leaf classification | 6,045 | 25 (PlantVillage label set) | image (path-only) + int label | PlantVillage-style; CC BY-SA 3.0 lineage |
| `brain_det` | `mmenendezg/brain-tumor-object-detection` | tumour detection | 1,101 | tumor | COCO boxes (`objects.bbox` x,y,w,h + label) | Derived from Roboflow; verify terms |
| `cotton_det` | `Francesco/cotton-plant-disease` | cotton disease detection | 1,798 | disease | COCO boxes (`objects.bbox` + category) | Roboflow-derived; verify terms |
| `plant_seg` | `Mauregato/leaf_disease_segmentation` | leaf-disease segmentation | 2,940 | background, disease | **pixel masks** (PNG `mask` column) | Verify original terms |
| `brain_instseg` | `dwb2023/brain-tumor-image-dataset-semantic-segmentation` | tumour instance segmentation | 2,146 | tumor | **COCO polygons** (`segmentation` + `bbox` + `category_id`) | Roboflow-derived; verify terms |

### Medical dataset candidates considered

* **BraTS** (brain tumour segmentation) — the canonical MRI segmentation benchmark. Requires
  **registration** at the BraTS challenge site; it is *not* auto-downloadable. To use it, download
  manually, place NIfTI files under a local path, and point the loader at them. Documented, not faked.
* **PranomVignesh/MRI-Images-of-Brain-Tumor** — used for classification (ships patient-style splits).
* **mmenendezg/brain-tumor-object-detection**, **dwb2023/…** — used for detection / instance
  segmentation because they provide genuine boxes/polygons.

### Leakage prevention

* Classification uses the **source dataset's own train/validation/test split** where provided
  (`brain_mri_cls`). Where a dataset has a single split (`plant_cls`), a **class-stratified split with
  a fixed seed (42)** is applied and recorded in the artifact metadata (`split_seed`).
* Medical models are documented as using the source splits; **patient-disjointness is inherited from
  the source dataset** and is not independently re-verified here — a documented limitation.

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

## How the splits and preprocessing are recorded

Each artifact's `artifacts/metadata/<id>.json` stores:

```json
{
  "dataset": "...", "dataset_key": "...", "classes": ["..."],
  "input": {"size": 224, "channels": 3, "mean": [0.485,0.456,0.406], "std": [0.229,0.224,0.225]},
  "metrics": {"accuracy": 0.0, "...": "..."},
  "training": {"epochs": 4, "lr": 0.001, "split_seed": 42},
  "artifact_version": "2.0.0", "git_commit": "..."
}
```

Inference reads `input.size` / `input.mean` / `input.std` from this file, so preprocessing is
guaranteed identical between training and prediction.
