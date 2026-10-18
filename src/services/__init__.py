"""Service layer for the Streamlit app.

Thin, testable wrappers that keep Streamlit specifics out of the core modules:

* :mod:`src.services.manifest_service` — read/build the model manifest
* :mod:`src.services.model_loader`    — ``st.cache_resource`` model loading
* :mod:`src.services.prediction_service` — cached, uniform prediction calls
* :mod:`src.services.artifact_service`   — cache/integrity helpers for artifacts

The app is inference-only: nothing here trains, and nothing downloads at import
time — a model is fetched only when a visitor selects it.
"""
from . import manifest_service, model_loader, prediction_service  # noqa: F401

__all__ = ["manifest_service", "model_loader", "prediction_service"]
