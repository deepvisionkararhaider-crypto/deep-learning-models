# Deep Learning · Medical AI · Smart Agriculture — Streamlit Platform

A single **Streamlit** application that combines three things:

1. **A 20-architecture deep-learning laboratory** (MLP, CNN, RNN/LSTM/GRU, Autoencoder, VAE, GAN,
   Transformer, BERT-style, GPT-style, ResNet, DenseNet, ViT, U-Net, YOLO-style, Siamese, Seq2Seq,
   Diffusion, GNN).
2. **Medical AI** — brain-MRI tumour **classification**, **segmentation**, **detection** and
   **instance segmentation**.
3. **Smart Agriculture AI** — plant-leaf **disease classification** (25 classes), leaf-disease
   **segmentation**, and cotton disease **detection**.

The six required computer-vision model types — **CNN, DenseNet121, YOLO-style, Mask R-CNN, U-Net,
Vision Transformer** — are instantiated for both the medical and agricultural domains.

> **Inference only.** The public app never trains and never downloads datasets. Models are trained
> **offline** by scripts in `scripts/`, evaluated, and shipped as committed artifacts under
> `artifacts/`. When weights are missing, the UI says **"training required"** instead of faking a
> result.

---

## Quick start (local)

```bash
pip install -r requirements.txt          # runtime (inference only, CPU torch)
streamlit run app.py                     # http://localhost:8501
```

## Deploy to Streamlit Community Cloud (free) — do this yourself

1. Push this repository to GitHub (branch `main`).
2. Go to <https://share.streamlit.io> → **New app** → **Deploy a public app from GitHub**.
3. Choose this repository, branch **main**, and set **Main file path** to `app.py`.
4. **Deploy.** Streamlit installs `requirements.txt` (CPU-only PyTorch) and starts the app.
5. No secrets, no database, no Docker, no Render required.

> Models whose weights are not committed will display **"training required"** — that is expected and
> honest. See *Training* below to produce them.

---

## Architecture

```
real public dataset
      │  (scripts/_fetch_data.py  →  .cache/parquet)
      ▼
OFFLINE TRAINING            scripts/train_models.py   (medical + agriculture CV)
      │                     scripts/train_lab.py      (the 20 lab models)
      ▼
EVALUATION  →  SAVE WEIGHTS  →  SAVE METADATA
      │            artifacts/<area>/<id>.pt
      │            artifacts/metadata/<id>.json     (classes, size, mean/std, metrics, provenance)
      ▼
COMMITTED ARTIFACTS
      ▼
STREAMLIT  app.py  →  src.inference / src.lab_inference   (loads weights, eval(), no_grad)
      ▼
VISITOR UPLOADS IMAGE  →  INFERENCE  →  PREDICTION + VISUALISATION
```

Key guarantees enforced in code:

* **Preprocessing parity** — `src/imaging.to_tensor()` is the *only* image→tensor path, and the
  normalisation constants are read back from each artifact's metadata. Training and inference cannot
  drift.
* **No training at inference** — `src/inference.py` and `src/lab_inference.py` contain no
  `torch.optim`, no `.backward()` (asserted by a CI test).
* **Explicit IO contracts** — every architecture documents and validates its tensor shapes. The old
  `mat1 and mat2 shapes cannot be multiplied (256x20 and 30x64)` bug is covered by
  `test_ann_input_contract_is_30`.
* **Honest status** — a model is "trained" only when its weights load and an inference smoke test
  passes (`scripts/verify_models.py`).

---

## Repository structure

```
app.py                     Streamlit entry point (main file path)
requirements.txt           runtime deps (CPU torch, streamlit, …)
requirements-train.txt     extra deps for offline training (pyarrow, …)
src/
  registry.py              single source of truth: 32 models, tasks, datasets, IO contracts
  artifacts.py             save/load weights + metadata; graceful missing handling
  imaging.py               shared preprocessing (train == inference)
  inference.py             CV predictors (classification / segmentation / detection / instances)
  lab_inference.py         the 20 lab predictors, each with its own input contract
  models/                  classifiers.py · segmentation.py · detection.py · lab20.py
  datasets/                loaders.py (CV) · lab_datasets.py (20-model datasets)
  ui/                      home · medical · agriculture · lab · info pages
scripts/
  _fetch_data.py           resumable dataset downloader
  train_models.py          train medical/agriculture CV models
  train_lab.py             train the 20 lab models
  verify_models.py         report REAL status of every model
tests/test_platform.py     registry, shapes, IO contracts, no-training-at-inference
artifacts/                 medical/ · agriculture/ · lab/ · metadata/
DATASETS.md                dataset sources, licences, annotation formats
```

---

## Training (offline, your machine or CI)

```bash
pip install -r requirements-train.txt

python scripts/_fetch_data.py                 # cache the CV datasets
python scripts/train_models.py --list
python scripts/train_models.py --model brain_densenet --epochs 4
python scripts/train_models.py --model all --epochs 4
python scripts/train_lab.py --model all        # the 20 lab models

python scripts/verify_models.py                # verify load + inference for every model
pytest -q tests/test_platform.py
```

Every run writes `artifacts/<area>/<id>.pt` and `artifacts/metadata/<id>.json` containing the model
config, class labels, input size, mean/std, dataset id, training config, measured metrics, artifact
version and git commit.

### Heavy models
Mask R-CNN (and ViT-B/16) are large. They are marked **training required** unless trained with
`--allow-heavy`; the platform documents their cost instead of pretending they are ready.

---

## Model → dataset → task (computer-vision models)

| Model | Domain | Task | Dataset | Metric |
|---|---|---|---|---|
| CNN | medical | 4-class MRI classification | Brain MRI (PranomVignesh) | accuracy, F1 |
| DenseNet121 | medical | 4-class MRI classification | Brain MRI | accuracy, F1 |
| ViT (compact) | medical | 4-class MRI classification | Brain MRI | accuracy, F1 |
| U-Net | medical | tumour segmentation | BraTS-derived masks | Dice, IoU |
| YOLO-style | medical | tumour detection | Brain tumour boxes (COCO) | mAP@50 |
| Mask R-CNN | medical | tumour instances | Brain tumour polygons (COCO) | mask AP@50 |
| CNN | agriculture | 25-class leaf classification | Plant leaf disease (Jharkhand) | accuracy, F1 |
| DenseNet121 | agriculture | 25-class leaf classification | Plant leaf disease | accuracy, F1 |
| ViT (compact) | agriculture | 25-class leaf classification | Plant leaf disease | accuracy, F1 |
| U-Net | agriculture | leaf-disease segmentation | Leaf disease segmentation | Dice, IoU |
| YOLO-style | agriculture | cotton disease detection | Cotton plant disease (COCO) | mAP@50 |
| Mask R-CNN | agriculture | cotton disease instances | cotton instances | mask AP@50 |

Full sources, licences and annotation formats: **[DATASETS.md](DATASETS.md)**.

---

## Evaluation

`📊 Model Evaluation` shows **only measured** metrics from each artifact's held-out test set.
`scripts/verify_models.py` prints `READY / ARCH-ONLY / INFER-FAILED` for all 32 models. Run it any
time — it is the source of truth for "how many models actually work".

---

## Responsible AI

**Medical:** research/educational prototype, **not a medical device**. Outputs are not diagnoses and
must not drive treatment. Consult a clinician.
**Agriculture:** decision-support only; predictions can be wrong. Verify important findings with an
agronomist. No personal or medical data is collected or stored.

## Limitations

* CPU-only free tier: some large detectors are marked *training required*.
* Metrics reflect source-dataset splits and biases; no external-cohort validation.
* PlantVillage-style lab images are easier than real field photos.

## Troubleshooting

| Symptom | Fix |
|---|---|
| "training required" badge | Train offline: `python scripts/train_models.py --model <id>` |
| `ArtifactIncompatible` | Architecture changed — retrain that model |
| Slow first load | Weights load once per session (`st.cache_resource`) |
| Streamlit Cloud build fails | Ensure `requirements.txt` is at the repo root and uses CPU torch |

## Licence

Code: MIT. Datasets retain their own licences — see [DATASETS.md](DATASETS.md).
