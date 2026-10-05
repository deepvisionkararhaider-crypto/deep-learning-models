"""FastAPI backend for this model.

Run locally:
    uvicorn backend.main:app --reload --port 8000
Then:
    curl -X POST http://localhost:8000/predict -H 'Content-Type: application/json' \
         -d '{"features": [...]}'
"""
import io
import base64
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_PROJECT = os.path.dirname(_HERE)
_ROOT = os.path.dirname(_PROJECT)
for p in (_ROOT, _PROJECT):
    if p not in sys.path:
        sys.path.insert(0, p)

from fastapi import FastAPI, HTTPException, UploadFile, File          # noqa: E402
from fastapi.middleware.cors import CORSMiddleware                    # noqa: E402
from pydantic import BaseModel                                        # noqa: E402

import predictor                                                      # noqa: E402

MODEL_ID = 10

app = FastAPI(title="Deep Learning Model API", version="1.0.0",
              description="Real inference API for model 10.")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"],
                   allow_headers=["*"])


class TabularRequest(BaseModel):
    features: list[float]


class TextRequest(BaseModel):
    text: str


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "model_id": MODEL_ID}


@app.get("/info")
def info() -> dict:
    try:
        return predictor.info()
    except Exception as exc:                                          # noqa: BLE001
        raise HTTPException(status_code=503, detail=str(exc))


@app.post("/predict")
def predict_tabular(req: TabularRequest) -> dict:
    """Tabular / graph features entry point."""
    try:
        return predictor.predict({"features": req.features})
    except Exception as exc:                                          # noqa: BLE001
        raise HTTPException(status_code=400, detail=str(exc))


@app.post("/predict/text")
def predict_text(req: TextRequest) -> dict:
    try:
        return predictor.predict({"text": req.text})
    except Exception as exc:                                          # noqa: BLE001
        raise HTTPException(status_code=400, detail=str(exc))


@app.post("/predict/image")
async def predict_image(file: UploadFile = File(...)) -> dict:
    try:
        return predictor.predict({"image": io.BytesIO(await file.read())})
    except Exception as exc:                                          # noqa: BLE001
        raise HTTPException(status_code=400, detail=str(exc))


@app.post("/predict/image-base64")
def predict_image_b64(payload: dict) -> dict:
    try:
        raw = base64.b64decode(payload["image_base64"])
        return predictor.predict({"image": io.BytesIO(raw)})
    except Exception as exc:                                          # noqa: BLE001
        raise HTTPException(status_code=400, detail=str(exc))
