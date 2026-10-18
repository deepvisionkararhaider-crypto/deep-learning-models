"""``st.cache_resource`` model loading — lazy, bounded, Hub-first.

Memory discipline (Streamlit Community Cloud runs in ~1 GB):

* A model is loaded **only** when a visitor selects it; the app never preloads.
* ``st.cache_resource`` keeps one instance per ``(model_id, area)`` for the
  process lifetime, so repeat predictions do not reload or re-download.
* Nothing here trains, and no dataset is ever touched.

Errors are surfaced as :class:`ModelUnavailable` with an actionable message
(missing token, unpublished artifact, incompatible checkpoint, download failure)
so the UI can show something honest instead of a stack trace.
"""
from __future__ import annotations

from typing import Any

import streamlit as st

from .. import artifacts as A
from .. import hf_loader
from ..registry import ModelSpec

#: Upper bound on simultaneously cached model objects. Checkpoints are ≤ ~120 MB,
#: but several concurrent models could still exhaust a small instance, so we clear
#: the resource cache before it can grow without bound.
MAX_CACHED_MODELS = 3


class ModelUnavailable(RuntimeError):
    """The selected model cannot be served (missing/unpublished/incompatible)."""


@st.cache_resource(show_spinner=False, max_entries=MAX_CACHED_MODELS)
def _load(model_id: str, area: str, allow_download: bool = True):
    """Cache one loaded (eval-mode) model. Keyed by id+area, not by object identity."""
    from ..registry import BY_ID
    spec = BY_ID[model_id]
    if spec.area == "lab":
        from .. import lab_inference as LI
        loader = LI.load
    else:
        from .. import hf_loader as HL
        loader = lambda s: HL.load_model(s, allow_download=allow_download)  # noqa: E731
    try:
        return loader(spec)
    except A.ArtifactMissing as exc:
        raise ModelUnavailable(
            f"'{model_id}' has no trained checkpoint. {exc}"
        ) from exc
    except A.ArtifactIncompatible as exc:
        raise ModelUnavailable(f"'{model_id}' checkpoint is incompatible: {exc}") from exc
    except Exception as exc:                                        # noqa: BLE001
        raise ModelUnavailable(
            f"Could not load '{model_id}': {type(exc).__name__}: {exc}"
        ) from exc


def load(spec: ModelSpec, allow_download: bool = True):
    """Return ``(model, config)`` for a spec, cached for the process lifetime."""
    return _load(spec.id, spec.area, allow_download)


def is_ready(spec: ModelSpec) -> bool:
    return hf_loader.is_ready(spec)


def readiness(spec: ModelSpec) -> dict[str, Any]:
    """Why a model is (not) ready — used for honest badges and empty states."""
    return {
        "ready": hf_loader.is_ready(spec),
        "reason": hf_loader.readiness_reason(spec),
        "local": A.has_weights(spec.id, spec.area),
    }


def clear_cache() -> None:
    _load.clear()


__all__ = ["load", "is_ready", "readiness", "ModelUnavailable", "clear_cache",
           "MAX_CACHED_MODELS"]
