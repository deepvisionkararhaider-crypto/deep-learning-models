"""Streamlit Community Cloud entry point for this model.

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
