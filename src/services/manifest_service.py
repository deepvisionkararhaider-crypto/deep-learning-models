"""Manifest access for the UI.

The manifest is the app's single source of truth for *what exists* and *what is
ready*. The UI never hardcodes model counts or readiness — it reads them here.
"""
from __future__ import annotations

from typing import Any

import streamlit as st

from .. import manifest as M
from ..registry import ALL_MODELS, BY_ID, ModelSpec

MODEL_COUNT = len(ALL_MODELS)          # 32 — computed, never a magic literal in the UI


@st.cache_data(show_spinner=False, ttl=300)
def load_manifest() -> dict[str, Any]:
    """Manifest from disk, rebuilt on the fly when the file is absent.

    Cached for 5 minutes so a redeploy/publish is picked up without a restart,
    while normal reruns stay cheap.
    """
    return M.load_or_build()


def entries() -> dict[str, dict[str, Any]]:
    return M.entries_by_id(load_manifest())


def entry(spec: ModelSpec) -> dict[str, Any]:
    return entries().get(spec.id) or M.fallback_entry(spec)


def counts() -> dict[str, Any]:
    m = load_manifest()
    return {
        "models": m.get("totals", {}).get("models", MODEL_COUNT),
        "laboratory": m.get("totals", {}).get("laboratory", 20),
        "medical": m.get("totals", {}).get("medical", 6),
        "agriculture": m.get("totals", {}).get("agriculture", 6),
        "status_counts": m.get("status_counts", {}),
        "ready": len(m.get("ready_models") or []),
    }


def ready_count() -> int:
    return counts()["ready"]


def hf_repo() -> str | None:
    return load_manifest().get("hf_repo")


def models_for(area: str | None = None, task: str | None = None,
               only_ready: bool = False) -> list[ModelSpec]:
    """Registry models filtered by area/task/readiness (readiness from the manifest)."""
    e = entries()
    out = []
    for spec in ALL_MODELS:
        if area and spec.area != area:
            continue
        if task and spec.task != task:
            continue
        if only_ready and (e.get(spec.id, {}).get("status") != "TRAINED_VERIFIED"):
            continue
        out.append(spec)
    return out


def status_of(spec: ModelSpec) -> str:
    return entry(spec).get("status", "ARCH_ONLY")


def is_ready(spec: ModelSpec) -> bool:
    return status_of(spec) == "TRAINED_VERIFIED"


__all__ = ["MODEL_COUNT", "load_manifest", "entries", "entry", "counts", "ready_count",
           "hf_repo", "models_for", "status_of", "is_ready", "BY_ID"]
