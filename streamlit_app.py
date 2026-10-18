"""Alternate entry point for Streamlit Community Cloud.

Some deployments are configured with ``streamlit_app.py`` as the main file path.
This shim keeps both entry points working without duplicating the app.
"""
from __future__ import annotations

import runpy
from pathlib import Path

runpy.run_path(str(Path(__file__).resolve().parent / "app.py"), run_name="__main__")
