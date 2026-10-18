"""Model Details, Evaluation, Datasets, About & Documentation pages."""
from __future__ import annotations

import streamlit as st

from .. import hf_hub
from ..registry import ALL_MODELS
from ..services import manifest_service as MS
from .catalog import primary_metric, status_legend
from .common import (GITHUB, domain_chip, hf_url, hero, inject_css, metric_card,
                     show_metrics, show_provenance, status_badge, task_chip)


# --------------------------------------------------------------------------- #
# Model Details
# --------------------------------------------------------------------------- #
def render_model_details() -> None:
    inject_css()
    hero("Model Details",
         "The full declared contract for any model: architecture, dataset, preprocessing, splits, "
         "metrics, checkpoint location and limitations.", "📄")

    entries = MS.entries()
    labels = {f"{s.name}  ·  {s.id}": s for s in ALL_MODELS}
    choice = st.selectbox("Choose a model", list(labels), key="details_model")
    spec = labels[choice]
    e = entries.get(spec.id, {})

    st.markdown(f"## {spec.name}")
    st.markdown(f"{status_badge(e.get('status', 'ARCH_ONLY'))} {domain_chip(spec.area)} "
                f"{task_chip(spec.task)}", unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Primary metric", primary_metric(e))
    c2.metric("Test set", str(e.get("test_set_size") or "—"))
    c3.metric("Checkpoint", f"{(e.get('weights_bytes') or 0)/1e6:.2f} MB"
                            if e.get("weights_bytes") else "—")
    c4.metric("Classes", str(len(e.get("classes") or [])))

    st.divider()
    a, b = st.columns(2)
    with a:
        st.subheader("Contract")
        st.write({
            "id": spec.id,
            "area": spec.area,
            "task": spec.task,
            "framework": spec.framework,
            "architecture": spec.builder,
            "build_kwargs": spec.build_kwargs,
            "input": spec.input_desc,
            "output": spec.outputs,
            "classes": e.get("classes") or spec.classes,
        })
    with b:
        st.subheader("Preprocessing")
        inp = e.get("input") or {}
        st.write({
            "modality": inp.get("modality"),
            "size": inp.get("size"),
            "channels": inp.get("channels"),
            "mean": inp.get("mean"),
            "std": inp.get("std"),
            "normalisation source": "artifact metadata (identical to training)",
        })
        st.subheader("Checkpoint")
        ck = e.get("checkpoint") or {}
        st.write({
            "kind": ck.get("kind"),
            "local_path": ck.get("local_path"),
            "hf_repo": ck.get("hf_repo"),
            "hf_path": ck.get("hf_path"),
            "hf_revision": ck.get("hf_revision"),
            "weights_sha256": (e.get("weights_sha256") or "")[:20] or None,
        })

    st.divider()
    st.subheader("Dataset & splits")
    st.write({
        "dataset": e.get("dataset", spec.dataset_name),
        "dataset_key": e.get("dataset_key", spec.dataset_key),
        "split_manifest": e.get("split_manifest") or {},
        "training": e.get("training") or {},
    })

    st.subheader("Measured metrics (held-out test split)")
    show_metrics(e.get("metrics") or {})

    st.subheader("Status & limitations")
    st.write({"status": e.get("status"), "note": e.get("note"),
              "limitations": e.get("limitations")})
    status_legend()

    url = hf_url()
    if url:
        ck = e.get("checkpoint") or {}
        path = ck.get("hf_path")
        st.link_button("🤗 Open the model on Hugging Face",
                       f"{url}/blob/{ck.get('hf_revision') or 'main'}/{path}" if path else url)


# --------------------------------------------------------------------------- #
# Evaluation
# --------------------------------------------------------------------------- #
def render_evaluation() -> None:
    inject_css()
    hero("Model Evaluation",
         "Measured metrics only. Every number comes from a trained artifact's held-out test "
         "evaluation — nothing here is estimated or copied from a paper.", "📊")

    entries = MS.entries()
    trained = {k: v for k, v in entries.items() if v.get("status") in
               ("TRAINED_VERIFIED", "TRAINED_UNVERIFIED")}
    if not trained:
        st.warning("No trained artifacts are recorded in the manifest yet.")
        st.caption("Train with `python scripts/train_models.py --model all` (CV) or "
                   "`python scripts/train_lab.py --model all` (laboratory), then rebuild the "
                   "manifest with `python scripts/build_manifest.py`.")
        return

    st.caption(f"{len(trained)} trained artifact(s) in the manifest.")
    rows = []
    for mid, e in trained.items():
        rows.append({"id": mid, "model": e.get("display_name"), "domain": e.get("domain"),
                     "task": e.get("task"), "status": e.get("status"),
                     "primary metric": primary_metric(e),
                     "test n": e.get("test_set_size") or "—",
                     "checkpoint MB": round((e.get("weights_bytes") or 0) / 1e6, 2)})
    st.dataframe(rows, use_container_width=True, hide_index=True)

    for mid, e in sorted(trained.items()):
        with st.expander(f"{e.get('display_name')}  —  {e.get('task')}  ({e.get('dataset')})"):
            show_metrics(e.get("metrics") or {})
            st.caption(f"status={e.get('status')} · "
                       f"epochs={(e.get('training') or {}).get('epochs')} · "
                       f"split={e.get('split_manifest')}")
            show_provenance(e)


# --------------------------------------------------------------------------- #
# Datasets
# --------------------------------------------------------------------------- #
def render_datasets() -> None:
    inject_css()
    hero("Dataset Explorer",
         "The real public datasets behind every model, their task and annotation type. Licences "
         "are the responsibility of the original dataset authors — see DATASETS.md.", "🗂️")
    seen: dict[tuple, list] = {}
    for s in ALL_MODELS:
        seen.setdefault((s.dataset_key, s.dataset_name), []).append(s)
    rows = []
    for (key, name), specs in sorted(seen.items()):
        rows.append({"dataset": name, "key": key, "task": specs[0].task,
                     "used_by": ", ".join(sp.id for sp in specs)})
    st.dataframe(rows, use_container_width=True, hide_index=True)
    st.info("Datasets are downloaded **only** by the offline trainer / GitHub Action. The "
            "Streamlit app never downloads a dataset and never trains.")


# --------------------------------------------------------------------------- #
# About & documentation
# --------------------------------------------------------------------------- #
def render_about() -> None:
    inject_css()
    hero("About & Documentation",
         "What this platform is, how it is built, and what it must not be used for.", "ℹ️")

    st.markdown(f"""
### Purpose
A single Streamlit platform combining a 20-architecture deep-learning laboratory with medical
(brain-MRI) and agricultural (plant-leaf) computer-vision models. It is an **educational and
research prototype**, not a clinical or commercial product.

### How it works
```
real public dataset → offline training (scripts/ or GitHub Actions)
    → evaluation on a held-out split → checkpoint + config + metrics + model card
    → published to the Hugging Face Hub (SHA-256 + pinned revision)
    → this app downloads only the selected model, verifies the checksum, runs inference
```
The public app **never trains** and **never downloads datasets**. A model with no verified
checkpoint is reported as *training required*; random weights are never presented as trained.

### Status vocabulary
`TRAINED_VERIFIED` · `TRAINED_UNVERIFIED` · `ARCH_ONLY` · `BLOCKED` · `DEFERRED` · `FAILED`
— the full definitions are in the Explore Models page and in `docs/MODEL_STATUS.md`.
""")

    c1, c2, c3, c4 = st.columns(4)
    counts = MS.counts()
    metric_card("Registered models", str(counts["models"]))
    metric_card("Ready", str(counts["ready"]))
    metric_card("Laboratory", str(counts["laboratory"]))
    metric_card("CV models", str(counts["medical"] + counts["agriculture"]))

    st.divider()
    st.markdown("### Responsible AI")
    st.markdown("""
- **Medical:** not a medical device. Outputs are not diagnoses; do not act on them clinically.
- **Agriculture:** decision support only; verify with an agronomist.
- **Scores are not calibrated probabilities** — calibration was not evaluated on external data.
- **No visitor data is stored.** Uploads are processed in-session and never persisted.
- **Predictions can be wrong.** Always verify against the original image and professional judgement.
""")

    st.markdown("### Links")
    links = [("⭐ GitHub repository", GITHUB),
             ("📚 DATASETS.md", f"{GITHUB}/blob/main/DATASETS.md"),
             ("📐 docs/ARCHITECTURE.md", f"{GITHUB}/blob/main/docs/ARCHITECTURE.md"),
             ("🚀 docs/DEPLOYMENT.md", f"{GITHUB}/blob/main/docs/DEPLOYMENT.md")]
    url = hf_url()
    if url:
        links.append(("🤗 Hugging Face model hub", url))
    cols = st.columns(2)
    for i, (label, target) in enumerate(links):
        with cols[i % 2]:
            st.link_button(label, target, use_container_width=True)

    st.divider()
    st.markdown("### Configuration")
    repo = hf_hub.model_repo() if hf_hub.is_configured() else None
    st.write({
        "HF_MODEL_REPO": repo or "not set",
        "artifact cache": str(hf_hub.DEFAULT_CACHE),
        "HF_TOKEN required by this app": "no (public artifacts are read anonymously)",
    })
