# Training

Training runs on **GitHub Actions runners**, never inside the app and never in the
Streamlit front end. Checkpoints are produced on the runner and published to the
Hugging Face Hub.

## Requirements

- Training dependencies: `pip install -r requirements-train.txt` (inference deps plus
  `pyarrow` and `matplotlib`).
- Datasets are fetched from Hugging Face datasets-server **parquet mirrors** by
  `scripts/_fetch_data.py`. Datasets are never committed. See `DATASETS.md`.
- Publishing needs `HF_TOKEN` (write scope) and `HF_MODEL_REPO`.

## Running

### Local (small models only)

```bash
python3 scripts/train_models.py --model brain_cnn --epochs 3 --batch-size 8
python3 scripts/train_models.py --all-cv --allow-heavy          # all 12 CV models
python3 scripts/train_models.py --model plant_yolo --allow-heavy # heavy models need this flag
```

### GitHub Actions (recommended)

Dispatch **Train models** from the Actions tab. Inputs:

| Input | Values | Meaning |
|---|---|---|
| `train_cv` | true/false | Train the 12 medical + agriculture CV models |
| `train_lab` | true/false | Train the 20 laboratory models |
| `model_group` | `all` / `medical` / `agriculture` / `single` | Which CV subset |
| `model_id` | e.g. `brain_vit` | Used when `model_group=single` |
| `cv_epochs` | 2 / 3 / 4 | Epochs per CV model |
| `batch_size` | integer | Batch size |
| `include_heavy` | true/false | Include models flagged heavy (YOLO, Mask R-CNN, ViT) |
| `resume` | true/false | Resume from the last valid checkpoint |
| `publish` | true/false | Run the 12-step publish protocol after training |
| `verify_only` | true/false | Skip training; only run inference verification |

## Resume

`src/training_state.py` writes a resumable state file after **every epoch**:

- model, optimizer and scheduler state dicts
- epoch, global step, best metric, history
- Python / NumPy / PyTorch RNG state

Writes are **atomic** (`tmp` file then `os.replace`), so an interrupted run can never
leave a half-written checkpoint. `load_state()` raises `InvalidCheckpoint` for a
missing, corrupt, architecture-mismatched or already-finished state, so `--resume` can
never silently restart a completed model.

## Heavy models

Models flagged heavy (`brain_yolo`, `brain_maskrcnn`, `plant_yolo`, `plant_maskrcnn`,
the ViTs, and the larger lab models) are skipped by default and recorded as `DEFERRED`
rather than being reported as failures. Pass `include_heavy: true` (or `--allow-heavy`)
to attempt them.

## Honest outcomes

- A skipped heavy model is recorded `DEFERRED`, never `TRAINED_*`.
- A model that fails verification is recorded `FAILED` and the workflow exits non-zero
  when it was a **required** model (`scripts/verify_models.py --require cv`).
- `--require none` is used for optional/partial runs; the run still fails on any
  `infer_failed`, so a broken checkpoint cannot pass as green.

## Reproducibility

Training uses a fixed `split_seed` (default 42) and records it in the manifest. Splits
are computed from the raw file list before any capping, and train/val/test are disjoint
by construction (regression-tested). Metrics reported in the manifest come from the
**held-out test split**; validation metrics are stored separately under `metrics.val`.
