"""Shared prediction studio.

One implementation used by the Image, Text/Tabular, Medical, Agriculture and Lab
pages: select a model → render the task-appropriate form → run inference → render
the result. The *form* is chosen from the model's metadata, so no page has to know
which control a model needs.
"""
from __future__ import annotations

import streamlit as st

from ..registry import ModelSpec
from ..services import manifest_service as MS
from ..services import model_loader, prediction_service
from ..forms import prediction_forms as PF
from . import results
from .common import domain_chip, status_badge, task_chip
from .catalog import primary_metric


def _header(spec: ModelSpec, entry: dict) -> None:
    st.markdown(
        f"#### {spec.name} &nbsp; {status_badge(entry.get('status', 'ARCH_ONLY'))}<br>"
        f'{domain_chip(spec.area)} {task_chip(spec.task)} '
        f'<span class="chip">{len(entry.get("classes") or [])} classes</span>',
        unsafe_allow_html=True,
    )
    st.caption(f"`{spec.id}` · {entry.get('dataset', spec.dataset_name)} · "
               f"input: {spec.input_desc or 'see contract'} · output: {spec.outputs or '—'}")


def _empty_state(specs: list[ModelSpec], entries: dict, area_label: str) -> None:
    if not specs:
        st.warning(f"No {area_label} models are registered.")
        return
    st.warning(f"No {area_label} model currently has a verified checkpoint, so inference is not "
               "available. This is reported honestly rather than served with random weights.")
    rows = []
    for s in specs:
        e = entries.get(s.id, {})
        rows.append({"model": s.name, "id": s.id, "task": s.task,
                     "status": e.get("status", "ARCH_ONLY"),
                     "reason": (e.get("note") or e.get("limitations") or "")[:110]})
    st.dataframe(rows, use_container_width=True, hide_index=True)
    st.caption("Train and publish with `python scripts/train_models.py --model <id>` then "
               "`python scripts/publish_models.py --model <id>` (or dispatch the GitHub Action).")


def studio(specs: list[ModelSpec], *, key_prefix: str, area_label: str,
           show_selector: bool = True, preselect: str | None = None) -> None:
    """Render a complete prediction studio for a set of candidate models."""
    entries = MS.entries()
    ready = [s for s in specs if entries.get(s.id, {}).get("status") == "TRAINED_VERIFIED"]

    st.caption(f"**{len(ready)} / {len(specs)}** {area_label} models are ready for inference.")
    if not ready:
        _empty_state(specs, entries, area_label)
        return

    if show_selector:
        labels = {f"{s.name}  ·  {s.task}": s for s in ready}
        default = 0
        if preselect:
            keys = list(labels)
            default = next((i for i, k in enumerate(keys) if labels[k].id == preselect), 0)
        choice = st.selectbox("Select a model", list(labels), index=default,
                              key=f"{key_prefix}_model")
        spec = labels[choice]
    else:
        spec = ready[0]

    entry = entries.get(spec.id, {})
    _header(spec, entry)

    c1, c2, c3 = st.columns(3)
    c1.metric("Primary metric", primary_metric(entry))
    c2.metric("Test set", str(entry.get("test_set_size") or "—"))
    c3.metric("Checkpoint", f"{(entry.get('weights_bytes') or 0)/1e6:.1f} MB"
                            if entry.get("weights_bytes") else "—")

    if entry.get("limitations"):
        st.caption(f"_{entry['limitations']}_")

    st.divider()

    try:
        kind = PF.form_kind(spec, entry)
    except Exception:                                               # noqa: BLE001
        st.error(f"This model's input contract is unknown, so no form can be generated for "
                 f"`{spec.id}`. It is reported as not-ready rather than given invented fields.")
        return

    payload, submitted = PF.build(spec, entry, key=f"{key_prefix}_{spec.id}")
    if not submitted or payload is None:
        if payload is None and not submitted:
            st.caption("Nothing runs until you press Predict — no training and no dataset download "
                       "happens in this app.")
        return

    _run(spec, kind, payload, key_prefix)


def _run(spec: ModelSpec, kind: str, payload: dict, key_prefix: str) -> None:
    try:
        with st.spinner("Loading model and running inference…"):
            model_loader.load(spec)
            if kind == "image" or kind == "image_pair":
                envelope = prediction_service.run(spec, image=_image(payload),
                                                  image_b=_image_b(payload))
            elif kind in ("tabular", "graph", "sequence"):
                envelope = prediction_service.run(spec, values=payload.get("values"),
                                                  noise=payload.get("noise"))
            elif kind == "tabular_denoise":
                envelope = prediction_service.run(spec, values=payload.get("values"),
                                                  noise=payload.get("noise"))
            elif kind in ("text", "text_prompt"):
                envelope = prediction_service.run(spec, text=payload.get("text"))
            else:
                envelope = prediction_service.run(spec, image=_image(payload))
    except model_loader.ModelUnavailable as exc:
        st.error(str(exc))
        st.caption("The model could not be loaded. Check that the checkpoint is published and "
                   "that the manifest's revision/sha match what is on the Hub.")
        return
    except Exception as exc:                                        # noqa: BLE001
        st.error(f"Could not start inference: {type(exc).__name__}: {exc}")
        return

    meta = MS.entry(spec)
    results.render(spec, envelope, meta)


def _image(payload: dict):
    from ..imaging import load_image
    raw = payload.get("image")
    return load_image(raw) if raw else None


def _image_b(payload: dict):
    from ..imaging import load_image
    raw = payload.get("image_b")
    return load_image(raw) if raw else None
