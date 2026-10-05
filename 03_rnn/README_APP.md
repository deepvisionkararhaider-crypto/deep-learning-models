# 03 — RNN

**Task:** 6-class activity recognition
**Dataset:** UCI Smartphone HAR
**Input:** 93x6 sensor window
**Framework:** PyTorch (trained here; the original `model.py` in this folder is
the reference training script it mirrors)

## Layout

```
03_rnn/
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
python tools/build_artifacts.py --only 3
python tools/finalize_artifacts.py
```

The training pipeline is reproducible and deterministic (fixed seed).

## Model card

- **Architecture:** Recurrent Neural Network
- **Framework:** PyTorch (mirrors 03_rnn)
- **Reported metric:** train accuracy = 0.5001 (train/eval split, not a leaderboard claim)
