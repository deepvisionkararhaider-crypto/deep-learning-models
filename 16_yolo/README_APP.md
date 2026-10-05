# 16 — YOLO-style

**Task:** Object detection + classification
**Dataset:** Synthetic multi-shape
**Input:** 64x64 grayscale image
**Framework:** PyTorch (trained here; the original `model.py` in this folder is
the reference training script it mirrors)

## Layout

```
16_yolo/
├── model/                 # REAL trained artifact (committed)
│   ├── model.pt           #   torch state_dict
│   ├── preprocess.pkl     #   exact preprocessing / metadata
│   ├── meta.json          #   human-readable model card
│   └── samples.json       #   a few real example inputs
├── backend/
│   ├── preprocessing.py   #   transform documentation
│   ├── predictor.py       #   inference layer (loads the real model)
│   ├── main.py            #   FastAPI service (POST /predict)
│   └── requirements.txt
├── frontend/
│   └── app.py             #   Streamlit UI
├── app.py                 #   Streamlit Community Cloud entry point
├── requirements.txt
└── README.md
```

## Run the Streamlit app locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Run the FastAPI backend locally

```bash
pip install -r backend/requirements.txt
uvicorn backend.main:app --reload --port 8000
curl -X POST http://localhost:8000/predict -H 'Content-Type: application/json' -d '{"features": [...]}'
```

## Train / rebuild the artifact

```bash
# from the repository root
python tools/build_artifacts.py --only 16
python tools/finalize_artifacts.py
```

The training pipeline is reproducible and deterministic (fixed seed).

## Model card

- **Architecture:** Single-stage detection head (8x8 grid)
- **Framework:** PyTorch (mirrors 16_yolo)
- **Reported metric:** detection recall (test, conf>=0.3) = 1.0 (train/eval split, not a leaderboard claim)
