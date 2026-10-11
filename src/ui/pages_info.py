"""Evaluation, Dataset Explorer and About pages."""
from __future__ import annotations

import streamlit as st

from .. import artifacts as A
from ..registry import ALL_MODELS, CV_MODELS
from .common import GITHUB, hero, inject_css, metric_card, show_metrics


def render_evaluation():
    inject_css()
    hero("Model Evaluation", "Measured metrics only — every number below comes from a trained "
         "artifact's held-out test evaluation.", "📊")
    metas = A.registry_summary()
    if not metas:
        st.warning("No trained artifacts yet. Train models with "
                   "`python scripts/train_models.py --model all` and `python scripts/train_lab.py --model all`.")
        return
    st.caption(f"{len(metas)} trained artifacts on disk.")
    for m in metas:
        mid = m.get("model_id")
        spec = next((s for s in ALL_MODELS if s.id == mid), None)
        title = spec.name if spec else mid
        with st.expander(f"{title}  —  {m.get('task')}  ({m.get('dataset')})"):
            show_metrics(m.get("metrics", {}))
            st.caption(f"trained_at={m.get('saved_at')}  ·  weights={m.get('weights_file')}  ·  "
                       f"commit={(m.get('git_commit') or '')[:10]}")


def render_datasets():
    inject_css()
    hero("Dataset Explorer", "The real public datasets behind every model, with their task and "
         "annotation type. See DATASETS.md in the repo for licences.", "🗂️")
    seen = {}
    for s in ALL_MODELS:
        seen.setdefault((s.dataset_key, s.dataset_name), []).append(s)
    rows = []
    for (key, name), specs in seen.items():
        rows.append({"dataset": name, "key": key, "used_by": ", ".join(sp.id for sp in specs),
                     "task": specs[0].task})
    st.dataframe(rows, use_container_width=True, hide_index=True)
    st.info("Datasets are downloaded on demand by the offline trainer only. The Streamlit app never "
            "downloads datasets and never trains.")


def render_about():
    inject_css()
    hero("About & Limitations", "What this platform is, what it is not, and how it is built.", "ℹ️")
    st.markdown(f"""
### Purpose
A single Streamlit platform combining a 20-architecture deep-learning laboratory with medical
(brain-MRI) and agricultural (plant-leaf) computer-vision models. It is an **educational and
research prototype**.

### How it works
```
real public dataset → offline training (scripts/) → evaluation → saved weights + metadata
        → committed artifacts/ → Streamlit loads weights → visitor uploads image → inference
```
The public app **never trains** and **never downloads datasets**. Missing weights are reported
honestly; random weights are never presented as trained.

### Responsible AI
- **Medical:** not a medical device; outputs are not diagnoses; do not act on them clinically.
- **Agriculture:** decision support only; verify with an agronomist.
- Confidence scores are model scores, not guarantees of correctness.
- No personal or medical data is collected or stored; uploads are processed in-session.

### Known limitations
- CPU-only; heavy detectors (Mask R-CNN) may be marked *training required* where CPU training is
  impractical in the free tier.
- Metrics reflect the source datasets' splits and biases.
- Not validated on external cohorts or field data.

### Links
- Repository: {GITHUB}
- Datasets: {GITHUB}/blob/main/DATASETS.md
""")
    c1, c2, c3 = st.columns(3)
    with c1:
        metric_card("Registered models", str(len(ALL_MODELS)))
    with c2:
        metric_card("CV models", str(len(CV_MODELS)))
    with c3:
        metric_card("Lab models", "20")
