"""Explore Models — filterable catalogue of every registered model."""
from __future__ import annotations

import streamlit as st

from ..registry import ALL_MODELS
from ..services import manifest_service as MS
from .catalog import filter_bar, model_card, readiness_table, status_legend
from .common import GITHUB, hero, inject_css, metric_card


def render() -> None:
    inject_css()
    hero("Explore Models",
         "Every model in the registry with its real contract, dataset, status and measured metric. "
         "Filter by domain, task or status — counts update with the manifest.", "🔎")

    entries = MS.entries()
    c = MS.counts()
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Models", c["models"])
    m2.metric("Ready", c["ready"])
    m3.metric("Laboratory", c["laboratory"])
    m4.metric("Medical + Agriculture", c["medical"] + c["agriculture"])

    st.divider()
    specs = filter_bar(ALL_MODELS, entries, key="explore")
    st.caption(f"Showing **{len(specs)}** of {len(ALL_MODELS)} models.")

    tab_cards, tab_table = st.tabs(["🗂️ Cards", "📋 Table"])
    with tab_cards:
        if not specs:
            st.warning("No models match those filters.")
        for spec in specs:
            model_card(spec, entries.get(spec.id, {}))
    with tab_table:
        readiness_table(specs, entries)
        st.download_button("Download manifest.json",
                           data=_manifest_bytes(),
                           file_name="manifest.json", mime="application/json")

    status_legend()
    st.divider()
    st.caption(f"Registry and dataset details: {GITHUB}/blob/main/DATASETS.md")


def _manifest_bytes() -> bytes:
    import json
    return json.dumps(MS.load_manifest(), indent=2).encode("utf-8")
