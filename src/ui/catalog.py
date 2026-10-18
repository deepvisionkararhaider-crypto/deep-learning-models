"""Reusable catalogue components: model cards, readiness tables, filters.

All counts and statuses come from the manifest — the UI never hardcodes a number
that could drift when a model is published or retrained.
"""
from __future__ import annotations

import streamlit as st

from ..registry import ModelSpec
from .common import domain_chip, status_badge, task_chip

#: metric keys we surface first, in priority order
_METRIC_PRIORITY = ("accuracy", "f1_macro", "dice", "iou", "mAP@50", "mask_AP@50",
                    "pair_accuracy", "recon_mse", "perplexity")


def primary_metric(entry: dict) -> str:
    metrics = entry.get("metrics") or {}
    for k in _METRIC_PRIORITY:
        if isinstance(metrics.get(k), (int, float)):
            return f"{k}={metrics[k]}"
    for k, v in metrics.items():
        if isinstance(v, (int, float)):
            return f"{k}={v}"
    return "—"


def model_card(spec: ModelSpec, entry: dict, *, expanded: bool = False) -> None:
    """One model as a compact card: identity, contract, status, primary metric."""
    status = entry.get("status", "ARCH_ONLY")
    classes = entry.get("classes") or []
    with st.container(border=True):
        st.markdown(
            f"**{spec.name}** &nbsp; {status_badge(status)}<br>"
            f'{domain_chip(spec.area)} {task_chip(spec.task)} '
            f'<span class="chip">{len(classes)} classes</span>',
            unsafe_allow_html=True,
        )
        st.caption(f"`{spec.id}` · {spec.dataset_name}")
        c1, c2, c3 = st.columns(3)
        c1.metric("Primary metric", primary_metric(entry))
        c2.metric("Input", f"{(entry.get('input') or {}).get('size') or '—'}"
                           f"×{(entry.get('input') or {}).get('channels') or '—'}")
        size = entry.get("weights_bytes")
        c3.metric("Checkpoint", f"{size/1e6:.1f} MB" if size else "—")
        if entry.get("note"):
            st.caption(f"_{entry['note']}_")


def readiness_table(specs: list[ModelSpec], entries: dict[str, dict]) -> None:
    rows = []
    for s in specs:
        e = entries.get(s.id, {})
        rows.append({
            "id": s.id,
            "model": s.name,
            "domain": e.get("domain", s.area),
            "task": s.task,
            "dataset": e.get("dataset", s.dataset_name),
            "status": e.get("status", "ARCH_ONLY"),
            "primary metric": primary_metric(e),
            "test n": e.get("test_set_size") or "—",
        })
    st.dataframe(rows, use_container_width=True, hide_index=True)


def filter_bar(specs: list[ModelSpec], entries: dict[str, dict], *, key: str) -> list[ModelSpec]:
    """Domain / task / status filters driven by the actual registry + manifest."""
    domains = sorted({s.area for s in specs})
    tasks = sorted({s.task for s in specs})
    statuses = sorted({entries.get(s.id, {}).get("status", "ARCH_ONLY") for s in specs})

    c1, c2, c3, c4 = st.columns([1, 1.4, 1.4, 1.2])
    dom = c1.multiselect("Domain", domains, default=domains, key=f"{key}_dom")
    task = c2.multiselect("Task", tasks, default=[], key=f"{key}_task",
                          help="Leave empty to include every task.")
    status = c3.multiselect("Status", statuses, default=[], key=f"{key}_status",
                            help="Leave empty to include every status.")
    only_ready = c4.toggle("Ready only", value=False, key=f"{key}_ready",
                           help="Only TRAINED_VERIFIED models can run inference.")

    out = []
    for s in specs:
        e = entries.get(s.id, {})
        st_status = e.get("status", "ARCH_ONLY")
        if s.area not in dom:
            continue
        if task and s.task not in task:
            continue
        if status and st_status not in status:
            continue
        if only_ready and st_status != "TRAINED_VERIFIED":
            continue
        out.append(s)
    return out


def status_legend() -> None:
    with st.expander("What the statuses mean", expanded=False):
        st.markdown(
            "- **TRAINED_VERIFIED** — a checkpoint exists, loads, and passed an inference smoke "
            "test (and a fresh-download check when published to the Hub). Only these can run.\n"
            "- **TRAINED_UNVERIFIED** — a checkpoint exists locally but the fresh-download "
            "verification has not been recorded yet.\n"
            "- **ARCH_ONLY** — the architecture is implemented but no trained weights exist.\n"
            "- **DEFERRED** — intentionally postponed (e.g. too heavy for the current runner).\n"
            "- **BLOCKED** — cannot be trained; the exact blocker is recorded.\n"
            "- **FAILED** — the last attempt failed; the error is recorded."
        )
