# Deployment

Three independent surfaces: **GitHub** (source), **Hugging Face Hub** (weights), and
**Streamlit Community Cloud** (app). This project is **not** deployed to Render or
Cloudflare.

## Current status

| Surface | Status |
|---|---|
| GitHub source | Active — `deepvisionkararhaider-crypto/deep-learning-models` |
| Hugging Face weights | Active — `KararDeepVision/deep-learning-models` (public) |
| Streamlit app | **Not deployed yet** |

## 1. Adding the workflows

The GitHub App used to push this repository does not have the `workflows` permission,
so files under `.github/workflows/` cannot be written by automation. The improved
workflows are committed at:

- `ci/train-models.yml`
- `ci/validation-ci.yml`

**Manual step (one time):** in the GitHub web UI open each file in `ci/`, copy its
contents, then create `.github/workflows/train-models.yml` and
`.github/workflows/validation-ci.yml` with those contents and commit via the UI.

## 2. Repository secrets and variables

Settings → Secrets and variables → Actions:

| Kind | Name | Value |
|---|---|---|
| Secret | `HF_TOKEN` | A Hugging Face **write** token (fine-grained, scoped to the model repo) |
| Variable | `HF_MODEL_REPO` | `KararDeepVision/deep-learning-models` |

Never paste the token into source, issues, the README, or chat. The Streamlit app never
needs a write token — public Hub reads are anonymous.

## 3. Running the pipeline

Actions → **Train models** → *Run workflow*, choose the inputs (see
[`TRAINING.md`](TRAINING.md)). The workflow:

1. `validate` — install, compile, pytest, smoke test, resolve the model list.
2. `train-cv` / `train-lab` — matrix jobs, `fail-fast: false`, `max-parallel: 2`.
3. `finalize` — merge status artifacts, run `verify_models.py --require none`,
   rebuild the manifest with `--remote`, commit only
   `models/manifest.json`, `artifacts/status/**`, `artifacts/metadata/**` with `[skip ci]`.

## 4. Streamlit Community Cloud

1. Push this repository to GitHub (already done for the existing commit).
2. Go to <https://share.streamlit.io> → **New app** → *Deploy a public app from GitHub*.
3. Repository: `deepvisionkararhaider-crypto/deep-learning-models`; Branch: `main`;
   Main file path: **`streamlit_app.py`** (or `app.py` — both work).
4. Under **Advanced settings → Secrets**, add:

   ```toml
   HF_MODEL_REPO = "KararDeepVision/deep-learning-models"
   HF_REVISION   = "aeab7749122941325aa52f916e942b01b5011dcd"
   ```

   `HF_TOKEN` is **not** required — all published artifacts are public.
5. Deploy. First load installs `requirements.txt` (inference-only, CPU wheels).

Runtime notes: the app performs **no training** and downloads **only** the model you
select, pinned to `HF_REVISION`, cached under `~/.cache/dl-models-hf`. `st.cache_resource`
keeps at most 3 models in memory.

## 5. After deploying

Update the README status table from `Not deployed yet` to the real
`https://<app>.streamlit.app` URL — only once the app is live and verified.
