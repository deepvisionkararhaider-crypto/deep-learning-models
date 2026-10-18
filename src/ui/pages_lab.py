"""Deep Learning Labs — the original 20-model catalogue with per-model prediction studio."""
from __future__ import annotations

import streamlit as st

from ..registry import LAB_BY_INDEX
from ..services import manifest_service as MS
from .catalog import readiness_table, status_legend
from .common import hero, inject_css, status_badge
from .predict import studio

LAB_SPECS = [LAB_BY_INDEX[i] for i in range(1, 21)]


def render() -> None:
    inject_css()
    hero("Deep Learning Laboratory — 20 Models",
         "The original catalogue. Every architecture keeps its own real dataset, input/output "
         "contract and training objective; prediction always uses committed weights.", "🧪")

    entries = MS.entries()
    ready = [s for s in LAB_SPECS if entries.get(s.id, {}).get("status") == "TRAINED_VERIFIED"]
    st.caption(f"**{len(ready)} / 20** laboratory models are ready for inference.")

    tab_cat, tab_predict, tab_eval = st.tabs(["📦 Catalogue", "🔮 Prediction Studio", "📊 Evaluation"])

    with tab_cat:
        for spec in LAB_SPECS:
            e = entries.get(spec.id, {})
            with st.expander(f"{spec.name}  —  {spec.task}", expanded=False):
                st.markdown(status_badge(e.get("status", "ARCH_ONLY")), unsafe_allow_html=True)
                st.write(f"**Dataset:** {e.get('dataset', spec.dataset_name)} (`{spec.dataset_key}`)")
                st.write(f"**Input:** {spec.input_desc}")
                st.write(f"**Output:** {spec.outputs}")
                st.write(f"**Framework:** {spec.framework}")
                if e.get("classes"):
                    st.write(f"**Classes:** {', '.join(e['classes'])}")
                if e.get("note"):
                    st.caption(e["note"])
                metrics = {k: v for k, v in (e.get("metrics") or {}).items()
                           if isinstance(v, (int, float))}
                if metrics:
                    st.write(metrics)

    with tab_predict:
        studio(LAB_SPECS, key_prefix="lab", area_label="laboratory")

    with tab_eval:
        readiness_table(LAB_SPECS, entries)
        status_legend()
