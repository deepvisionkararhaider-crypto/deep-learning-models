"""Interactive Streamlit demo for GRU.

Dataset: UCI Smartphone HAR
Task:    6-class activity recognition
Input:   93x6 sensor window
"""
import os
import sys

import numpy as np
import streamlit as st

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_HERE))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import app_shared as A                                       # noqa: E402

MODEL_ID = 5
INFO = A.MODELS[MODEL_ID]
st.set_page_config(page_title=f"{INFO['name']} · Model 05",
                   page_icon="🤖", layout="wide")
_SAMPLE_LOAD_ERROR = None


@st.cache_resource(show_spinner="Loading the trained model…")
def _warm():
    A.load(MODEL_ID)
    return True


def _load_samples():
    import json
    p = os.path.join(_HERE, "..", "model", "samples.json")
    try:
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as exc:  # noqa: BLE001
        global _SAMPLE_LOAD_ERROR
        _SAMPLE_LOAD_ERROR = str(exc)
        return {}


def _sample_image(i=0):
    import numpy as np
    for rel, key in (("sample_images.npy", "images"),):
        p = os.path.join(_HERE, "..", "model", rel)
        if os.path.exists(p):
            arr = np.load(p)
            return arr[i]
    return None


def header():
    top = st.container()
    with top:
        c1, c2 = st.columns([3, 1])
        with c1:
            st.title(f"🤖 {INFO['name']}")
            st.caption(f"Model 05 · {INFO['task']}")
        with c2:
            try:
                m = A.load(MODEL_ID)["meta"]
                st.metric(m.get("metric_name", "metric"), m.get("metric_value", "n/a"))
            except Exception:  # noqa: BLE001
                pass
    with st.expander("Model details"):
        st.markdown(f"- **Model:** {INFO['name']}\n- **Framework:** {INFO['framework']}"
                    f"\n- **Task:** {INFO['task']}\n- **Dataset:** {INFO['dataset']}"
                    f"\n- **Input:** {INFO['input']}")
        try:
            st.json(A.load(MODEL_ID)["meta"])
        except Exception as exc:  # noqa: BLE001
            st.warning(f"Model card unavailable: {exc}")


def show_result(r):
    kind = r.get("kind")
    if kind == "classification":
        st.success(f"### Prediction\n## {r['label']}")
        conf = r.get("confidence")
        if conf is not None and conf > 0:
            st.metric("Confidence", f"{conf * 100:.1f}%")
        probs = r.get("probabilities") or {}
        if probs:
            st.write("**Class probabilities**")
            st.bar_chart({"probability": probs})
    elif kind == "reconstruction":
        st.success(f"### Reconstructed output\nMSE to input: {r['mse']}")
        st.image(np.asarray(r["image"]), clamp=True, width=180, caption="reconstruction")
        st.caption(r.get("note", ""))
    elif kind == "segmentation":
        st.success(f"###Foreground mask — {r['foreground_ratio'] * 100:.1f}% of pixels are foreground")
        st.image((np.asarray(r["mask"]) * 255).astype("uint8"), width=240, caption="mask")
    elif kind == "detection":
        st.success(f"### {r['count']} object(s) detected")
        names = r.get("class_names") or []
        for b in r["boxes"]:
            label = names[b["class"]] if 0 <= b["class"] < len(names) else str(b["class"])
            st.write(f"- **{label}** · confidence {b['score']} · box ({b['x1']}, {b['y1']}, "
                     f"{b['x2']}, {b['y2']})")
        if not r["boxes"]:
            st.info("No object above the confidence threshold. Try lowering it.")
    elif kind == "generation":
        st.success("### Predicted next tokens")
        st.markdown(f"**Next token:** `{r['next_token_word']}`")
        st.write("**Top candidates**")
        st.dataframe([{"token": t["word"], "probability": t["prob"]} for t in r["top_tokens"]],
                     use_container_width=True)
    elif kind == "pair":
        st.success(f"### {r['label']}   (squared-L2 distance {r['distance']}, threshold {r['threshold']})")
        st.metric("Confidence", f"{r['confidence'] * 100:.1f}%")
    elif kind == "denoise":
        st.success("### Denoised feature row")
        st.write("**Cleaned features (standardised space)**")
        st.dataframe({"value": r["clean"]}, use_container_width=True)
        st.metric("MSE to input row", r["mse_to_input"])
    else:
        st.json(r)


def run_demo():
    header()
    st.info("Every model in this lab was trained once and committed. Enter new, unseen "
            "data below — nothing is retrained when you click Predict.")
    samples = _load_samples()
    try:
        A.predict
    except Exception:  # noqa: BLE001
        pass

    st.subheader("Sensor-window input")
    st.caption("Provide a CSV with at least 558 numeric values (93 timesteps x 6 channels).")
    presets = (samples or {}).get("sequences", [])
    use_sample = False
    if presets:
        if st.checkbox(A.SAMPLE_LABEL.get(MODEL_ID, "Use a bundled real sample")):
            use_sample = True
            st.caption(f"Sample label: {(samples.get('labels') or ['?'])[0]}")
    up = st.file_uploader("Upload HAR CSV", type=["csv", "txt"])
    if st.button(A.DEMO_LABEL.get(MODEL_ID, "Predict"), type="primary"):
        try:
            if use_sample and presets:
                tens = __import__("torch").tensor([p for p in presets[0]], dtype=__import__("torch").float32).view(1, 93, 6)
                show_result(A.predict(MODEL_ID, {"tensor": tens}))
            elif up is not None:
                show_result(A.predict(MODEL_ID, {"csv": up.getvalue()}))
            else:
                st.error("Upload a CSV or tick the bundled sample.")
        except Exception as exc:  # noqa: BLE001
            st.error(str(exc))


if __name__ == "__main__":
    try:
        _warm()
    except Exception as exc:  # noqa: BLE001
        st.error(f"The trained artifact could not be loaded: {exc}")
        st.stop()
    run_demo()
    st.divider()
    st.caption("Trained once · committed weights · real inference · "
               "part of the 20-model Deep Learning Lab")
