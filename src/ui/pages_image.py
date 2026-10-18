"""Image Prediction — any image model (classification, segmentation, detection, instances)."""
from __future__ import annotations

import streamlit as st

from ..registry import ALL_MODELS
from ..services import manifest_service as MS
from .common import hero, inject_css, medical_disclaimer, agri_disclaimer
from .predict import studio

IMAGE_TASKS = ("classification", "segmentation", "detection", "instance_segmentation")


def render() -> None:
    inject_css()
    hero("Image Prediction",
         "Upload an image and run a verified model — classification, segmentation, object "
         "detection or instance segmentation. Only models with a verified checkpoint are shown.",
         "🖼️")

    entries = MS.entries()
    candidates = [s for s in ALL_MODELS
                  if s.task in IMAGE_TASKS and not s.is_blocked]
    ready = [s for s in candidates if entries.get(s.id, {}).get("status") == "TRAINED_VERIFIED"]

    if not ready:
        st.info("No image model is ready yet. The catalogue below shows exactly why each one is "
                "not available — nothing is faked.")
    else:
        st.success(f"{len(ready)} image model(s) ready for inference.")

    st.caption("Medical outputs are informational and **not** a diagnosis. Agricultural outputs "
               "are decision support only.")
    medical_disclaimer()
    agri_disclaimer()
    st.divider()

    studio(candidates, key_prefix="img", area_label="image")

    st.divider()
    st.subheader("All image models")
    rows = []
    for s in candidates:
        e = entries.get(s.id, {})
        rows.append({"model": s.name, "id": s.id, "domain": e.get("domain", s.area),
                     "task": s.task, "status": e.get("status", "ARCH_ONLY"),
                     "reason": (e.get("note") or e.get("limitations") or "")[:100]})
    st.dataframe(rows, use_container_width=True, hide_index=True)
