# Inference

## Guarantees

- No training, no dataset download, and no checkpoint download happens at app start.
- Only the model the visitor selects is downloaded, pinned to `HF_REVISION`.
- Loaded objects are cached with `st.cache_resource(max_entries=3)`; the manifest is
  cached with `st.cache_data(ttl=300)`.
- Every forward pass runs under `model.eval()` and `torch.inference_mode()`.
- Checkpoints are loaded with `safetensors` when the file is `.safetensors`, otherwise
  `torch.load(weights_only=True)`, with a `state_dict` unwrap fallback.
- The app never loads an arbitrary user-supplied repo id or path; the location comes
  from `models/manifest.json` only.
- A model without published weights raises `NotPublished` and is shown as not ready —
  it is never given placeholder or random predictions.

## Pipeline

```
manifest entry → input contract → form → preprocessing → model → decoder → renderer
```

| Stage | Module |
|---|---|
| Manifest read | `src/services/manifest_service.py` |
| Form generation | `src/forms/prediction_forms.py` (`form_kind`, `schema`) |
| Checkpoint resolution + load | `src/hf_loader.py` (`resolve_checkpoint`, `load_model`) |
| Caching | `src/services/model_loader.py` |
| Task dispatch | `src/services/prediction_service.py` → `src/inference.py` / `src/lab_inference.py` |
| Rendering | `src/ui/results.py` |

## Task-specific behaviour

| Task | Input | Output |
|---|---|---|
| Image classification | Image upload (≤15 MB), preview | Top-k classes, probability bar chart, limitations |
| Image segmentation | Image upload | Original / mask / overlay side by side, class names, mask PNG download |
| Object detection | Image upload, confidence threshold slider | Boxes, labels, confidences, table of objects, empty-state when nothing is detected |
| Text / language modelling | Labelled textarea (≤20 000 chars), example input, empty-input validation | Label probabilities or generated tokens |
| Tabular | Form generated from the real feature schema | Prediction with the model's exact preprocessing |
| Sequence / time-series | CSV upload, feature-count and length validation | Class probabilities |
| Image pair (metric learning) | Two image uploads | Distance / similarity |
| Graph | Node/edge input | Class probabilities |

Probability values are presented as **model scores**. They are not described as
calibrated probabilities anywhere in the UI, because calibration has not been evaluated.
A generic CNN is never labelled "YOLO", and a detector is never labelled "Mask R-CNN".

## Command line

```bash
python3 scripts/fetch_models.py --list                      # what is published
python3 scripts/fetch_models.py --model brain_cnn --out ./models_local
python3 scripts/verify_models.py --require none             # inference verification
```

## Failure modes

| Situation | Behaviour |
|---|---|
| `HF_MODEL_REPO` unset | Manifest-driven browse still works; prediction reports the config error |
| Model has no published weights | Shown as not ready, with the reason from the manifest |
| Download fails | Retried 3× with backoff; auth and 404 errors fail fast |
| Checksum mismatch | Rejected (`ChecksumMismatch`) and the file is not used |
| Corrupt / incompatible checkpoint | Actionable error surfaced to the visitor; no silent fallback |
| Oversized or invalid upload | Rejected before any model work |
