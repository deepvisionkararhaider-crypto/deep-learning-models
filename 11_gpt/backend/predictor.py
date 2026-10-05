"""Inference layer for this model.

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

MODEL_ID = 11


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
