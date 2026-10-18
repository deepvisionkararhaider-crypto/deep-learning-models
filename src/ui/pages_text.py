"""Text & Tabular Prediction — text, feature-vector, sequence and graph models."""
from __future__ import annotations

import streamlit as st

from ..forms.prediction_forms import form_kind
from ..registry import ALL_MODELS
from ..services import manifest_service as MS
from .common import hero, inject_css
from .predict import studio

NON_IMAGE_KINDS = ("text", "text_prompt", "tabular", "tabular_denoise", "graph", "sequence")


def render() -> None:
    inject_css()
    hero("Text & Tabular Prediction",
         "Models that genuinely accept text, numeric feature vectors, sensor sequences or graph "
         "features. Forms are generated from each model's declared contract — an image model never "
         "appears here.", "⌨️")

    entries = MS.entries()
    candidates = []
    for s in ALL_MODELS:
        try:
            if form_kind(s, entries.get(s.id, {})) in NON_IMAGE_KINDS:
                candidates.append(s)
        except Exception:                                           # noqa: BLE001
            continue

    if not candidates:
        st.warning("No text/tabular models are registered.")
        return

    ready = [s for s in candidates if entries.get(s.id, {}).get("status") == "TRAINED_VERIFIED"]
    st.caption(f"**{len(ready)} / {len(candidates)}** text, tabular, sequence and graph models "
               "are ready for inference.")

    st.divider()
    studio(candidates, key_prefix="txt", area_label="text/tabular")

    st.divider()
    st.subheader("Supported non-image contracts")
    rows = []
    for s in candidates:
        e = entries.get(s.id, {})
        rows.append({"model": s.name, "id": s.id, "input kind": form_kind(s, e),
                     "task": s.task, "status": e.get("status", "ARCH_ONLY"),
                     "contract": s.outputs})
    st.dataframe(rows, use_container_width=True, hide_index=True)
    st.caption("Feature ordering and dtype are fixed by the model's training contract; the form "
               "validates the count before running inference.")
