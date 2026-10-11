"""Original 20-model laboratory: catalogue + prediction studio with per-model inputs."""
from __future__ import annotations

import numpy as np
import streamlit as st

from .. import artifacts as A
from .. import lab_inference as LI
from ..imaging import load_image
from ..registry import LAB_BY_INDEX
from .common import badge, get_model, hero, inject_css, metric_card, show_metrics, show_provenance

INPUT_LABEL = {
    "tabular": "Numeric feature vector", "tabular_denoise": "Numeric feature vector + noise level",
    "graph": "Node feature vector", "image": "Image upload", "image_pair": "Two image uploads",
    "sequence": "Sensor sequence (CSV or text)", "text": "Free text",
    "text_prompt": "Text prompt",
}


def render():
    inject_css()
    hero("20-Model Deep-Learning Laboratory",
         "Every architecture keeps its own real dataset, input/output contract and training objective. "
         "Prediction is always from committed weights — never a training run.", "🧪")

    ready = [s for s in LAB_BY_INDEX.values() if A.has_weights(s.id, s.area)]
    st.caption(f"**{len(ready)} / 20** models have verified trained weights.")

    tab_cat, tab_predict, tab_eval = st.tabs(["📦 Catalogue", "🔮 Prediction Studio", "📊 Evaluation"])

    with tab_cat:
        for i in range(1, 21):
            s = LAB_BY_INDEX[i]
            meta = A.read_metadata(s.id) or {}
            ok = A.has_weights(s.id, s.area)
            with st.expander(f"{s.name}  —  {s.task}", expanded=False):
                st.markdown(f"{badge(ok)}", unsafe_allow_html=True)
                st.write(f"**Dataset:** {s.dataset_name}")
                st.write(f"**Input:** {s.input_desc}")
                st.write(f"**Output:** {s.outputs}")
                st.write(f"**Framework:** {s.framework}")
                if s.notes:
                    st.caption(s.notes)
                if ok and (meta.get("metrics")):
                    st.write({k: v for k, v in meta["metrics"].items() if isinstance(v, (int, float))})

    with tab_predict:
        _studio(ready)

    with tab_eval:
        rows = []
        for i in range(1, 21):
            s = LAB_BY_INDEX[i]
            m = (A.read_metadata(s.id) or {}).get("metrics") or {}
            rows.append({"model": s.name, "task": s.task,
                         "metric": next(iter([f"{k}={v}" for k, v in m.items()
                                              if isinstance(v, (int, float))]), "—"),
                         "status": "✔ trained" if A.has_weights(s.id, s.area) else "✖ training required"})
        st.dataframe(rows, use_container_width=True, hide_index=True)


def _studio(ready):
    if not ready:
        st.warning("No laboratory models have trained weights yet. "
                   "Run `python scripts/train_lab.py --model all`.")
        return
    name = st.selectbox("Choose a model", [s.name for s in ready])
    spec = next(s for s in ready if s.name == name)
    kind = LI.INPUT_KIND[spec.id]
    st.caption(f"**Dataset:** {spec.dataset_name}  ·  **Input:** {spec.input_desc}  ·  "
               f"**Contract:** {spec.outputs}")
    st.markdown(f"Input type: **{INPUT_LABEL[kind]}**")

    try:
        _run(spec, kind)
    except Exception as exc:  # noqa: BLE001
        st.error(f"{type(exc).__name__}: {exc}")


def _run(spec, kind):
    model, meta = get_model(spec)

    if kind == "tabular":
        n = (meta.get("input") or {}).get("features", 30)
        raw = st.text_area(f"{n} comma-separated values", ",".join(["0.0"] * n), height=90)
        if st.button("🔮 Predict", type="primary"):
            feats = [float(x) for x in raw.replace("\n", ",").split(",") if x.strip()]
            st.json(LI.predict_tabular(spec, feats))

    elif kind == "tabular_denoise":
        n = (meta.get("input") or {}).get("features", 30)
        raw = st.text_area(f"{n} comma-separated values", ",".join(["0.0"] * n), height=90)
        noise = st.slider("Noise level", 0.0, 1.0, 0.5, 0.05)
        if st.button("🔮 Denoise", type="primary"):
            feats = [float(x) for x in raw.replace("\n", ",").split(",") if x.strip()]
            res = LI.denoise_tabular(spec, feats, noise)
            st.metric("MSE to input", res["mse_to_input"])

    elif kind == "graph":
        n = (meta.get("input") or {}).get("features", 1433)
        st.caption(f"Paste {n} node-feature values (e.g. a Cora row).")
        raw = st.text_area("features", ",".join(["0.0"] * 8), height=90,
                           help=f"Provide exactly {n} values.")
        if st.button("🔮 Classify node", type="primary"):
            feats = [float(x) for x in raw.replace("\n", ",").split(",") if x.strip()]
            st.json(LI.gnn_predict(spec, feats))

    elif kind == "image":
        up = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg", "bmp", "webp"])
        if up:
            img = load_image(up.getvalue())
            st.image(img, width=200)
            if st.button("🔮 Predict", type="primary"):
                _dispatch_image(spec, img)

    elif kind == "image_pair":
        c1, c2 = st.columns(2)
        a = c1.file_uploader("Image A", type=["png", "jpg", "jpeg"], key="pair_a")
        b = c2.file_uploader("Image B", type=["png", "jpg", "jpeg"], key="pair_b")
        if a and b and st.button("🔮 Compare", type="primary"):
            st.json(LI.siamese_pair(spec, load_image(a.getvalue()), load_image(b.getvalue())))

    elif kind == "sequence":
        st.caption("Paste 128x9 = 1152 numeric values (whitespace or comma separated).")
        raw = st.text_area("sequence values", "", height=120)
        if st.button("🔮 Predict", type="primary") and raw.strip():
            vals = [float(x) for x in raw.replace(",", " ").split()]
            st.json(LI.predict_sequence(spec, vals))

    elif kind == "text":
        txt = st.text_area("Text", "The space shuttle launched successfully into orbit.", height=120)
        if st.button("🔮 Classify", type="primary") and txt.strip():
            st.json(LI.text_classify(spec, txt))

    elif kind == "text_prompt":
        txt = st.text_area("Prompt", "the mission control team reported", height=120)
        if st.button("🔮 Next tokens", type="primary") and txt.strip():
            st.json(LI.generate_text(spec, txt))

    st.divider()
    show_metrics((meta or {}).get("metrics", {}))
    show_provenance(meta)


def _dispatch_image(spec, img):
    if spec.id in ("lab02", "lab12", "lab13", "lab14"):
        st.json(LI.predict_image_class(spec, img))
    elif spec.id in ("lab06", "lab07"):
        res = LI.reconstruct(spec, img)
        c1, c2 = st.columns(2)
        c1.image((res["input"] * 255).astype("uint8"), caption="Input", clamp=True)
        c2.image((res["reconstruction"] * 255).astype("uint8"), caption="Reconstruction", clamp=True)
        st.metric("Reconstruction MSE", res["mse"])
    elif spec.id == "lab08":
        st.json(LI.gan_score(spec, img))
    elif spec.id == "lab15":
        res = LI.segmentation_28(spec, img)
        c1, c2 = st.columns(2)
        c1.image((res["input"] * 255).astype("uint8"), caption="Input", clamp=True)
        c2.image((res["binary"] * 255).astype("uint8"), caption="Predicted mask", clamp=True)
    elif spec.id == "lab16":
        from ..imaging import draw_boxes
        res = LI.detection_28(spec, img)
        disp = load_image(img).resize((28, 28))
        st.image(draw_boxes(disp, res["boxes"], spec.classes).resize((220, 220)),
                 caption=f"{res['count']} detection(s)")
        st.json(res["boxes"])
