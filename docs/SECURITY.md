# Security and Privacy

## Credentials

- `HF_TOKEN` is stored **only** as a GitHub Actions secret (and locally at
  `/home/user/.secrets/hf_token`, mode `600`, outside the repository).
- The token is never written to source, the README, docs, logs, or chat. `.env` and
  `.streamlit/secrets.toml` are gitignored; `.env.example` contains empty placeholders.
- The app needs **no write token** — all published artifacts are public and read
  anonymously. No write credential is ever shipped to the front end.
- If a token is ever shared in plaintext, rotate it: Hugging Face → Settings →
  Access Tokens → *Invalidate*, then re-create and update the `HF_TOKEN` secret.

## Untrusted input

- Uploads are validated for type and size (images ≤ 15 MB, text ≤ 20 000 chars,
  sequences ≤ 20 000 values) before any model work.
- Uploaded files are **never executed** and never written to a served directory.
- Filenames, MIME types, model ids and artifact paths from the browser are not trusted.
  Artifact locations come exclusively from `models/manifest.json`, so an arbitrary
  repository id or path cannot be injected into a download.
- Preprocessing applies the model's own recorded mean/std and input size; it never
  infers them from the uploaded bytes.

## Data handling

- Visitor images and text are held in memory for the duration of the request only.
  They are not persisted, logged, or sent anywhere except the local model forward pass.
- No analytics, telemetry, or third-party calls are made with visitor content.
- Downloaded weights are cached locally under `~/.cache/dl-models-hf` (or
  `HF_ARTIFACT_CACHE`) so repeat requests do not re-download.

## Medical responsibility

Medical models are research and educational demonstrations. The app shows a prominent
disclaimer on every medical page and result:

- Not a medical device and not for diagnostic use.
- Outputs are model estimates that can be wrong, including confidently wrong.
- Must not replace assessment by a qualified professional.
- Trained on small public research datasets with the limitations listed per model.

## Agriculture responsibility

Agriculture models state their intended dataset and domain (plant-disease leaf imagery,
cotton/weed detection, leaf segmentation). Predictions are estimates and may not
generalise to other crops, lighting, growth stages, or regions.

## Repository hygiene

- No checkpoint larger than 100 MB may be tracked; CI asserts this and warns on
  oversized files.
- Binary weight patterns (`*.pt`, `*.pth`, `*.safetensors`, `*.bin`, `*.ckpt`, `*.onnx`,
  `*.h5`, `*.pkl`, `*.joblib`) are gitignored, with JSON metadata re-included via a
  negation rule.
- Inference modules are asserted by CI to contain no optimiser or `backward(` call, so
  the app can never train.
- History has not been rewritten and no destructive cleanup has been performed.
