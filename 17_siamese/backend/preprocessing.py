"""Preprocessing helpers for this model (thin, documented re-exports).

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
    )

MODEL_ID = 17


def describe() -> dict:
    """Return the preprocessing configuration of this model."""
    import app_shared as _A
    return _A.load(MODEL_ID)["pre"]
