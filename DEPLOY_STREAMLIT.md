# Streamlit Community Cloud — deploy each model as its own app

All 20 demos run on **free** Streamlit Community Cloud. Each model folder has its
own entry point, so you create one app per model against the same repository
(this is Streamlit's supported multi-app mode).

- **Repository:** `deepvisionkararhaider-crypto/deep-learning-models`
- **Branch:** `main`
- **Python:** 3.11 (set in the Advanced settings of the deploy form)

## Entry points (one app each)

| # | App name suggestion | Main file path |
|--:|---|---|
| — | `ai-model-lab` (dashboard) | `dashboard.py` |
| 01 | `ann-breast-cancer` | `01_ann/app.py` |
| 02 | `cnn-mnist` | `02_cnn/app.py` |
| 03 | `rnn-har` | `03_rnn/app.py` |
| 04 | `lstm-har` | `04_lstm/app.py` |
| 05 | `gru-har` | `05_gru/app.py` |
| 06 | `autoencoder-mnist` | `06_autoencoder/app.py` |
| 07 | `vae-mnist` | `07_vae/app.py` |
| 08 | `gan-mnist` | `08_gan/app.py` |
| 09 | `transformer-topic` | `09_transformer/app.py` |
| 10 | `bert-topic` | `10_bert/app.py` |
| 11 | `gpt-textgen` | `11_gpt/app.py` |
| 12 | `resnet-mnist` | `12_resnet/app.py` |
| 13 | `densenet-mnist` | `13_densenet/app.py` |
| 14 | `vit-mnist` | `14_vit/app.py` |
| 15 | `unet-segmentation` | `15_unet/app.py` |
| 16 | `yolo-detection` | `16_yolo/app.py` |
| 17 | `siamese-digits` | `17_siamese/app.py` |
| 18 | `seq2seq-text` | `18_seq2seq/app.py` |
| 19 | `diffusion-tabular` | `19_diffusion/app.py` |
| 20 | `gnn-cora` | `20_gnn/app.py` |

The repository-root `requirements.txt` is intentionally small
(`streamlit`, `torch`, `numpy`, `pillow`). Streamlit Community Cloud can only use
one requirements file for a branch — that root file is normalised so **every**
app can use it. Each folder also keeps its own `requirements.txt` for local use.

## Steps (once, ~2 minutes per app)

1. Sign in at <https://share.streamlit.io> with the GitHub account that owns the repo.
2. **Create app → Deploy a public app from GitHub**.
3. Fill in:
   - Repository: `deepvisionkararhaider-crypto/deep-learning-models`
   - Branch: `main`
   - Main file path: e.g. `01_ann/app.py`
   - Advanced settings → Python version: **3.11**
4. Deploy. Each app gets a URL of the form
   `https://<app-subdomain>.streamlit.app`.
5. Repeat for the remaining rows in the table above.

> **No secrets or API keys are required.** The models are committed as small
> artifacts and inference is CPU-only.

## Wiring the dashboard to your live links

Once your apps exist, set an environment variable on the **dashboard** app
(`dashboard.py` → Settings → Secrets, or the deploy form's “Advanced settings”):

```toml
STREAMLIT_APP_BASE = "https://<your-dashboard-subdomain>.streamlit.app"
```

The dashboard derives each model link as
`<STREAMLIT_APP_BASE>/<folder>` (e.g. `.../01_ann`). This is only possible if
each app's subdomain matches its folder name, so name each app after its folder
(the “App URL” can be edited in the app's settings). Until this variable is set,
the dashboard shows the exact `streamlit run …` command for each app instead of
a link that would not resolve.

## Verify locally before deploying

```bash
pip install -r requirements.txt
streamlit run dashboard.py          # portfolio
streamlit run 01_ann/app.py         # an individual model
```

## Model artifacts

Each `model/` folder carries the trained weights (`model.pt`), the exact
preprocessing (`preprocess.pkl`), a model card (`meta.json`) and example inputs
(`samples.json`). They are small enough to commit directly — no Git LFS needed.
To rebuild any model from scratch:

```bash
python tools/build_artifacts.py --only <id>
python tools/finalize_artifacts.py
```
