# Deep Learning Models — 20 Trained Models with Real Inference Apps

A professional portfolio of **20 deep-learning architectures**. Each one is
trained once, committed as a small artifact, and served through a real inference
pipeline with its own **Streamlit** frontend and **FastAPI** backend.

Students enter **new, unseen data**, the app runs the **real trained model**, and
shows the **real prediction** — nothing is retrained on page load and no
prediction is hard-coded.

- 🧠 20 models · 4 training tasks · CPU-only · free to host
- 🔮 Every model: real artifact → real preprocessing → real inference
- 🎛️ Streamlit UI per model + a portfolio dashboard
- ⚙️ FastAPI `POST /predict` service per model
- 🧪 All 20 apps and backends covered by automated smoke tests

---

## Live demos

Each model is a **separate** Streamlit Community Cloud app built from this one
repository. The links below are created by you in Streamlit's dashboard (see
[`DEPLOY_STREAMLIT.md`](DEPLOY_STREAMLIT.md)); they are intentionally left blank
until they resolve, so this table never contains an unverified URL.

| # | Model | Framework | Task | App entry point | Live demo |
|--:|---|---|---|---|---|
| 1 | ANN / MLP | PyTorch | Breast-cancer binary classification | `01_ann/app.py` |https://deepvisionkararhaider-crypto-deep-learning-mod-01-annapp-1bfjpi.streamlit.app/ |
| 2 | CNN | PyTorch | MNIST digit classification | `02_cnn/app.py` | https://deepvisionkararhaider-crypto-deep-learning-mod-02-cnnapp-zphbhe.streamlit.app/ |
| 3 | RNN | PyTorch | HAR activity recognition | `03_rnn/app.py` |https://deepvisionkararhaider-crypto-deep-learning-mod-03-rnnapp-qgt25o.streamlit.app/|
| 4 | LSTM | PyTorch | HAR activity recognition | `04_lstm/app.py` | https://deepvisionkararhaider-crypto-deep-learning-mo-04-lstmapp-yfripw.streamlit.app/ |
| 5 | GRU | PyTorch | HAR activity recognition | `05_gru/app.py` |https://deepvisionkararhaider-crypto-deep-learning-mod-05-gruapp-mlaovw.streamlit.app/ |
| 6 | Autoencoder | PyTorch | Image reconstruction | `06_autoencoder/app.py` |https://deepvisionkararhaider-crypto-deep-lear-06-autoencoderapp-adqpjc.streamlit.app/ |
| 7 | VAE | PyTorch | Variational reconstruction | `07_vae/app.py` | https://deepvisionkararhaider-crypto-deep-learning-mod-07-vaeapp-4tp6ow.streamlit.app/ |
| 8 | GAN | PyTorch | Real-vs-generated discrimination | `08_gan/app.py` |https://deepvisionkararhaider-crypto-deep-learning-mod-08-ganapp-o7e27r.streamlit.app/ |
| 9 | Transformer | PyTorch | 4-class topic classification | `09_transformer/app.py` | https://deepvisionkararhaider-crypto-deep-lear-09-transformerapp-bjjabz.streamlit.app/ |
| 10 | BERT-style | PyTorch | 4-class topic classification | `10_bert/app.py` | https://deepvisionkararhaider-crypto-deep-learning-mo-10-bertapp-ont3nu.streamlit.app/ |
| 11 | GPT-style | PyTorch | Autoregressive text generation | `11_gpt/app.py` | https://deepvisionkararhaider-crypto-deep-learning-mod-11-gptapp-dzavsz.streamlit.app/ |
| 12 | ResNet | PyTorch | MNIST digit classification | `12_resnet/app.py` | https://deepvisionkararhaider-crypto-deep-learning--12-resnetapp-ldlsgq.streamlit.app/ |
| 13 | DenseNet | PyTorch | MNIST digit classification | `13_densenet/app.py` |https://deepvisionkararhaider-crypto-deep-learnin-13-densenetapp-ikma6j.streamlit.app/ |
| 14 | ViT | PyTorch | MNIST digit classification | `14_vit/app.py` |https://deepvisionkararhaider-crypto-deep-learning-mod-14-vitapp-pbq21u.streamlit.app/|
| 15 | U-Net | PyTorch | Binary image segmentation | `15_unet/app.py` | https://deepvisionkararhaider-crypto-deep-learning-mo-15-unetapp-jav18t.streamlit.app/ |
| 16 | YOLO-style | PyTorch | Object detection | `16_yolo/app.py` | https://deepvisionkararhaider-crypto-deep-learning-mo-16-yoloapp-6v1d0u.streamlit.app/ |
| 17 | Siamese | PyTorch | Same/different digit verification | `17_siamese/app.py` | https://deepvisionkararhaider-crypto-deep-learning-17-siameseapp-xlopa0.streamlit.app/ |
| 18 | Seq2Seq | PyTorch | Text token reconstruction | `18_seq2seq/app.py` | https://deepvisionkararhaider-crypto-deep-learning-18-seq2seqapp-y51bd9.streamlit.app/ |
| 19 | Diffusion MLP | PyTorch | Tabular denoising | `19_diffusion/app.py` |https://deepvisionkararhaider-crypto-deep-learni-19-diffusionapp-c3r5cg.streamlit.app/ |
| 20 | GNN | PyTorch | Cora node classification | `20_gnn/app.py` |https://deepvisionkararhaider-crypto-deep-learning-mod-20-gnnapp-y5vt7s.streamlit.app/ |
| — | **AI Model Lab** (dashboard) | Streamlit | Portfolio of all 20 | `dashboard.py` |https://deepvisionkararhaider-crypto-deep-learning-mod-dashboard-d9ztuh.streamlit.app/ |

> “_pending_” means the app is fully prepared and tested locally; the URL is
> created when you deploy it on Streamlit Community Cloud. No placeholder or
> invented URL is ever shown here.

---

## How the inference path works

```text
Streamlit UI (per model)
      │  user enters NEW data
      ▼
app_shared.predict(model_id, payload)
      │  load <folder>/model/model.pt        (real trained weights)
      │  load <folder>/model/preprocess.pkl  (exact training preprocessing)
      │  model forward pass
      ▼
real prediction / probabilities
      │  JSON
      ▼
Streamlit renders the result
```

The **same** `predict()` is used by the FastAPI backend, so the browser and the
API always return identical results.

---

## Repository layout

```text
deep-learning-models/
├── app_shared.py            # shared inference runtime (loads artifacts, preprocesses)
├── models_arch.py           # self-contained architecture definitions
├── dashboard.py             # portfolio dashboard (Streamlit)
├── requirements.txt         # small runtime deps used by every Streamlit app
├── DEPLOY_STREAMLIT.md      # step-by-step free deployment guide
├── tools/
│   ├── build_artifacts.py   # trains each model once, exports artifacts
│   ├── finalize_artifacts.py# copies artifacts into each folder + samples
│   ├── app_templates.py     # backend / frontend templates
│   └── generate_apps.py     # writes every per-model app
├── 01_ann/
│   ├── model.py             # ORIGINAL training script (unchanged)
│   ├── requirements-train.txt  # ORIGINAL training dependencies (preserved)
│   ├── model/               # REAL trained artifact
│   │   ├── model.pt         #   weights
│   │   ├── preprocess.pkl   #   preprocessing (scalers, class names, tokenizer cfg)
│   │   ├── meta.json        #   model card
│   │   └── samples.json     #   real example inputs
│   ├── backend/             # FastAPI service
│   │   ├── main.py          #   POST /predict
│   │   ├── predictor.py     #   inference layer
│   │   ├── preprocessing.py #   transform documentation
│   │   └── requirements.txt
│   ├── frontend/app.py      # Streamlit UI
│   ├── app.py               # Streamlit Community Cloud entry point
│   ├── requirements.txt
│   └── README_APP.md        # per-model app docs
├── 02_cnn/ … 20_gnn/        # same structure for every model
├── plots/, predictions.csv  # ORIGINAL outputs (preserved)
└── DATASETS.md              # dataset attribution
```

---

## Input format per model

| Input kind | Models | What you provide |
|---|---|---|
| Tabular | 01, 19 | 30 comma-separated numeric Breast-Cancer features |
| Image (28×28) | 02, 12, 13, 14 | upload PNG/JPG, or pick a bundled sample digit |
| Image → 8×8 | 06, 07, 08, 17 | same as above (auto-pooled to 8×8) |
| Image (64×64) | 15, 16 | upload PNG/JPG, or a bundled sample |
| Sensor sequence | 03, 04, 05 | upload a CSV with ≥ 558 numeric values (93×6), or a sample |
| Text | 09, 10, 18 | type/paste a headline or short article |
| Text prompt | 11 | type a prompt to continue |
| Graph nodes | 20 | 1433 comma-separated node features, or a sample row |
| Image pair | 17 | two images (samples or uploads) |

**Text models** use the same deterministic tokeniser as training
(lowercase → whitespace → first 32 tokens → `md5(word) % 1022 + 1` hash IDs),
so any free text works without a vocabulary file.

---

## Datasets

Real public benchmarks, attributed in [`DATASETS.md`](DATASETS.md). Some folders
named MNIST/CIFAR in their original scripts; the deployment artifacts use a
single consistent, documented source per task:

| Task | Source used |
|---|---|
| Breast-cancer tabular | UCI Breast Cancer Wisconsin (via scikit-learn) |
| Handwritten digits | MNIST 28×28 (LeCun / cvdf mirror) |
| Sensor sequences | UCI Human Activity Recognition Using Smartphones |
| 4-class news topics | 20 Newsgroups (fallback: AG News) |
| Segmentation / detection | the synthetic generators in `15_unet/`, `16_yolo/` |
| Graph nodes | Cora (Planetoid) |

---

## Run locally

```bash
pip install -r requirements.txt

streamlit run dashboard.py        # portfolio dashboard
streamlit run 01_ann/app.py       # any single model
```

FastAPI backend for a model:

```bash
pip install -r 01_ann/backend/requirements.txt
cd 01_ann && uvicorn backend.main:app --reload --port 8000
curl -X POST http://localhost:8000/predict \
     -H 'Content-Type: application/json' \
     -d '{"features": [ ... 30 values ... ]}'
```

---

## Retrain / rebuild artifacts

```bash
python tools/build_artifacts.py            # train all 20 once (some minutes)
python tools/build_artifacts.py --only 16  # a single model
python tools/finalize_artifacts.py         # copy artifacts into the folders
python tools/generate_apps.py              # (re)generate backend + frontend
```

Training is deterministic (fixed seed). Existing valid artifacts are **not**
retrained — `build_artifacts.py` skips any model that already has `model.pt`.

---

## Deployment

Free, per-model, from this repository: see
**[`DEPLOY_STREAMLIT.md`](DEPLOY_STREAMLIT.md)**.

- Repository: `deepvisionkararhaider-crypto/deep-learning-models`
- Branch: `main`
- Python: `3.11`
- Main file: `<folder>/app.py` (dashboard: `dashboard.py`)
- No secrets or API keys required.

---

## Testing

- All 20 Streamlit apps render headlessly (`streamlit.testing.v1.AppTest`).
- Every backend answers `/health`, `/info`, `/predict` and rejects malformed or
  missing input with `400`/`422`.
- Real predictions verified end-to-end on unseen sample data for all 20 models.

---

## Preserving the original work

The original `model.py`, `plots/`, `predictions.csv` and per-folder training
requirements are kept. The application layer is **added around** them; the
original training dependencies are preserved alongside the new ones in
`requirements-train.txt`.

## License

Educational/research portfolio. Review each model folder and the upstream dataset
sources for licensing and attribution.
