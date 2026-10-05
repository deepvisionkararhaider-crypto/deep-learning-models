"""Code templates for the per-model backend + Streamlit frontend.

Each template contains a single ``__MODEL_ID__`` token; every other detail
(name, framework, task, dataset, class list, input kind) is resolved at runtime
from ``app_shared`` / the model's ``meta.json``, so templates cannot drift.
"""
from __future__ import annotations

# --------------------------------------------------------------------------- #
# backend/predictor.py — the inference layer (real preprocessing + real model)
# --------------------------------------------------------------------------- #
PREDICTOR = r'''"""Inference layer for this model.

Loads the REAL trained artifact committed next to it (model/model.pt +
preprocess.pkl + meta.json) and applies exactly the preprocessing used during
training. `predict()` is the single entry point used by both the FastAPI
service (backend/main.py) and the Streamlit frontend (frontend/app.py).
"""
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_HERE))          # repository root
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import app_shared as _A                                       # noqa: E402

MODEL_ID = __MODEL_ID__


def info() -> dict:
    """Model + training metadata (no inference)."""
    bundle = _A.load(MODEL_ID)
    return {**_A.MODELS[MODEL_ID], "meta": bundle["meta"], "preprocess_kind": bundle["pre"]["kind"]}


def predict(payload: dict) -> dict:
    """Run REAL inference on user-supplied data.

    ``payload`` keys depend on the model's input kind:
      tabular  -> {"features": [float, ...]}
      image    -> {"image": <bytes | file-like>}
      sequence -> {"csv": <bytes>}  or  {"tensor": torch.Tensor}
      text     -> {"text": str}
      pair     -> {"image_a": ..., "image_b": ...}
      graph    -> {"features": [float, ...]}   (1433-dim)
    """
    return _A.predict(MODEL_ID, payload)
'''

# --------------------------------------------------------------------------- #
# backend/preprocessing.py
# --------------------------------------------------------------------------- #
PREPROCESSING = r'''"""Preprocessing helpers for this model (thin, documented re-exports).

The exact transforms live in ``app_shared`` so training and inference can never
diverge; this module documents which transform this specific model uses.
"""
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_HERE))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from app_shared import (                       # noqa: F401,E402
    MODEL_ID_PLACEHOLDER,  # replaced at generation time
)

MODEL_ID = __MODEL_ID__


def describe() -> dict:
    """Return the preprocessing configuration of this model."""
    import app_shared as _A
    return _A.load(MODEL_ID)["pre"]
'''

# --------------------------------------------------------------------------- #
# backend/main.py — FastAPI service
# --------------------------------------------------------------------------- #
FASTAPI_MAIN = r'''"""FastAPI backend for this model.

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

MODEL_ID = __MODEL_ID__

app = FastAPI(title="Deep Learning Model API", version="1.0.0",
              description="Real inference API for model __MODEL_ID__.")
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
'''

# --------------------------------------------------------------------------- #
# folder-root app.py — the Streamlit Community Cloud entry point
# --------------------------------------------------------------------------- #
ROOT_APP = r'''"""Streamlit Community Cloud entry point for this model.

Set the app's "Main file path" to this file (e.g. ``01_ann/app.py``).
The user interface lives in ``frontend/app.py``.
"""
import os
import runpy
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

_TARGET = os.path.join(_HERE, "frontend", "app.py")
runpy.run_path(_TARGET, run_name="__main__")
'''

# --------------------------------------------------------------------------- #
# requirements
# --------------------------------------------------------------------------- #
REQ_FRONTEND = """streamlit>=1.39,<2
torch>=2.2,<3
numpy>=1.26,<3
pillow>=10,<12
"""

REQ_BACKEND = """fastapi>=0.110,<1
uvicorn[standard]>=0.29,<1
pydantic>=2.6,<3
torch>=2.2,<3
numpy>=1.26,<3
pillow>=10,<12
"""

REQ_ROOT = """streamlit>=1.39,<2
torch>=2.2,<3
numpy>=1.26,<3
pillow>=10,<12
"""

# --------------------------------------------------------------------------- #
# per-model README
# --------------------------------------------------------------------------- #
README = """# __MODEL_INDEX__ — __MODEL_NAME__

**Task:** __TASK__
**Dataset:** __DATASET__
**Input:** __INPUT__
**Framework:** PyTorch (trained here; the original `model.py` in this folder is
the reference training script it mirrors)

## Layout

```
__FOLDER__/
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
python tools/build_artifacts.py --only __MODEL_ID__
python tools/finalize_artifacts.py
```

The training pipeline is reproducible and deterministic (fixed seed).
"""


def render(template: str, model_id: int) -> str:
    """Substitute the single token that templates declare."""
    return template.replace("__MODEL_ID__", str(model_id))
