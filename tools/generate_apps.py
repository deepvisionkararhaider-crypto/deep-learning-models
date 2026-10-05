"""Generate the backend + Streamlit frontend for every model folder.

    python tools/generate_apps.py

Idempotent: re-running refreshes generated files from the templates while
leaving the original hand-written `model.py`, `plots/`, `predictions.csv` and
per-folder data untouched. Existing non-generated files are never deleted.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools import app_templates as T  # noqa: E402
from app_shared import MODELS        # noqa: E402


def write(path: Path, text: str, created: list, updated: list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    existed = path.exists()
    if existed and path.read_text(encoding="utf-8") == text:
        return
    path.write_text(text, encoding="utf-8")
    (updated if existed else created).append(str(path.relative_to(ROOT)))


def generate(model_id: int) -> tuple[list, list]:
    info = MODELS[model_id]
    folder = ROOT / info["folder"]
    created, updated = [], []
    meta = _load_meta(folder)

    write(folder / "backend" / "predictor.py", T.render(T.PREDICTOR, model_id), created, updated)
    raw_pre = T.PREPROCESSING.replace("MODEL_ID_PLACEHOLDER,  # replaced at generation time\n", "")
    write(folder / "backend" / "preprocessing.py", T.render(raw_pre, model_id), created, updated)
    write(folder / "backend" / "main.py", T.render(T.FASTAPI_MAIN, model_id), created, updated)
    write(folder / "backend" / "requirements.txt", T.REQ_BACKEND, created, updated)
    write(folder / "frontend" / "app.py", _frontend(model_id, info, meta), created, updated)
    write(folder / "app.py", T.render(T.ROOT_APP, model_id), created, updated)
    write(folder / "requirements.txt", T.REQ_ROOT, created, updated)
    write(folder / "README_APP.md", _readme(model_id, info, meta), created, updated)
    return created, updated


def _load_meta(folder: Path) -> dict:
    import json
    p = folder / "model" / "meta.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


def _readme(model_id, info, meta) -> str:
    body = (T.README
            .replace("__MODEL_INDEX__", f"{model_id:02d}")
            .replace("__MODEL_NAME__", info["name"])
            .replace("__FOLDER__", info["folder"])
            .replace("__TASK__", info["task"])
            .replace("__DATASET__", info["dataset"])
            .replace("__INPUT__", info["input"])
            .replace("__MODEL_ID__", str(model_id)))
    if meta:
        body += (f"\n## Model card\n\n- **Architecture:** {meta.get('architecture','')}\n"
                 f"- **Framework:** {meta.get('framework','')}\n"
                 f"- **Reported metric:** {meta.get('metric_name','')} = "
                 f"{meta.get('metric_value','')} (train/eval split, not a leaderboard claim)\n")
    return body


# --------------------------------------------------------------------------- #
# the per-model Streamlit UI
# --------------------------------------------------------------------------- #
_HEAD = '''"""Interactive Streamlit demo for {name}.

Dataset: {dataset}
Task:    {task}
Input:   {input}
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

MODEL_ID = {mid}
INFO = A.MODELS[MODEL_ID]
st.set_page_config(page_title=f"{{INFO['name']}} · Model {mid:02d}",
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
        return {{}}


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
            st.title(f"🤖 {{INFO['name']}}")
            st.caption(f"Model {mid:02d} · {{INFO['task']}}")
        with c2:
            try:
                m = A.load(MODEL_ID)["meta"]
                st.metric(m.get("metric_name", "metric"), m.get("metric_value", "n/a"))
            except Exception:  # noqa: BLE001
                pass
    with st.expander("Model details"):
        st.markdown(f"- **Model:** {{INFO['name']}}\\n- **Framework:** {{INFO['framework']}}"
                    f"\\n- **Task:** {{INFO['task']}}\\n- **Dataset:** {{INFO['dataset']}}"
                    f"\\n- **Input:** {{INFO['input']}}")
        try:
            st.json(A.load(MODEL_ID)["meta"])
        except Exception as exc:  # noqa: BLE001
            st.warning(f"Model card unavailable: {{exc}}")


def show_result(r):
    kind = r.get("kind")
    if kind == "classification":
        st.success(f"### Prediction\\n## {{r['label']}}")
        conf = r.get("confidence")
        if conf is not None and conf > 0:
            st.metric("Confidence", f"{{conf * 100:.1f}}%")
        probs = r.get("probabilities") or {{}}
        if probs:
            st.write("**Class probabilities**")
            st.bar_chart({{"probability": probs}})
    elif kind == "reconstruction":
        st.success(f"### Reconstructed output\\nMSE to input: {{r['mse']}}")
        st.image(np.asarray(r["image"]), clamp=True, width=180, caption="reconstruction")
        st.caption(r.get("note", ""))
    elif kind == "segmentation":
        st.success(f"###Foreground mask — {{r['foreground_ratio'] * 100:.1f}}% of pixels are foreground")
        st.image((np.asarray(r["mask"]) * 255).astype("uint8"), width=240, caption="mask")
    elif kind == "detection":
        st.success(f"### {{r['count']}} object(s) detected")
        names = r.get("class_names") or []
        for b in r["boxes"]:
            label = names[b["class"]] if 0 <= b["class"] < len(names) else str(b["class"])
            st.write(f"- **{{label}}** · confidence {{b['score']}} · box ({{b['x1']}}, {{b['y1']}}, "
                     f"{{b['x2']}}, {{b['y2']}})")
        if not r["boxes"]:
            st.info("No object above the confidence threshold. Try lowering it.")
    elif kind == "generation":
        st.success("### Predicted next tokens")
        st.markdown(f"**Next token:** `{{r['next_token_word']}}`")
        st.write("**Top candidates**")
        st.dataframe([{{"token": t["word"], "probability": t["prob"]}} for t in r["top_tokens"]],
                     use_container_width=True)
    elif kind == "pair":
        st.success(f"### {{r['label']}}   (squared-L2 distance {{r['distance']}}, threshold {{r['threshold']}})")
        st.metric("Confidence", f"{{r['confidence'] * 100:.1f}}%")
    elif kind == "denoise":
        st.success("### Denoised feature row")
        st.write("**Cleaned features (standardised space)**")
        st.dataframe({{"value": r["clean"]}}, use_container_width=True)
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
'''


_BODIES = {
    "tabular": '''
    st.subheader("Input features")
    pre = A.load(MODEL_ID)["pre"]
    names = pre.get("feature_names", [])
    presets = (samples or {}).get("features", {})
    if presets:
        choice = st.selectbox(A.SAMPLE_LABEL.get(MODEL_ID, "Preset row"), ["(custom)"] + list(presets))
        defaults = presets.get(choice, [0.0] * len(names)) if choice != "(custom)" else [0.0] * len(names)
    else:
        defaults = [0.0] * len(names)
    raw = st.text_area(
        f"{len(names)} comma-separated values (raw units)",
        ",".join(f"{v:.4f}" for v in defaults), height=120)
    if st.button(A.DEMO_LABEL.get(MODEL_ID, "Predict"), type="primary"):
        try:
            feats = [float(v) for v in raw.replace("\\n", ",").split(",") if v.strip()]
        except ValueError:
            st.error("Every value must be a number.")
            return
        if len(feats) != len(names):
            st.error(f"Expected {len(names)} values, received {len(feats)}.")
            return
        show_result(A.predict(MODEL_ID, {"features": feats}))
''',
    "image": '''
    st.subheader("Image input")
    t1, t2 = st.tabs(["Upload", "Sample digit"])
    img = None
    with t1:
        up = st.file_uploader("Upload a PNG/JPG", type=["png", "jpg", "jpeg", "webp"])
        if up:
            img = up
    with t2:
        i = st.slider("Sample index", 0, 5, 0)
        arr = _sample_image(i)
        if arr is not None:
            st.image(arr, clamp=True, width=140, caption="bundled real sample")
            if st.checkbox("Use this sample"):
                img = arr
    if img is not None:
        if not hasattr(img, "read"):
            st.image(img, clamp=True, width=140)
        if st.button(A.DEMO_LABEL.get(MODEL_ID, "Predict"), type="primary"):
            show_result(A.predict(MODEL_ID, {"image": img}))
''',
    "sequence": '''
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
''',
    "text": '''
    st.subheader("Text input")
    presets = (samples or {}).get("texts", [])
    default = presets[0] if presets else ""
    text = st.text_area("Enter a headline / short article", default, height=150)
    if st.button(A.DEMO_LABEL.get(MODEL_ID, "Predict"), type="primary"):
        if not text.strip():
            st.error("Please enter some text.")
            return
        show_result(A.predict(MODEL_ID, {"text": text}))
''',
    "text_gen": '''
    st.subheader("Prompt")
    presets = (samples or {}).get("texts", [])
    default = presets[0] if presets else "Machine learning is"
    text = st.text_area("Prompt to continue", default, height=120)
    if st.button(A.DEMO_LABEL.get(MODEL_ID, "Generate"), type="primary"):
        if not text.strip():
            st.error("Please enter a prompt.")
            return
        show_result(A.predict(MODEL_ID, {"text": text}))
''',
    "pair": '''
    st.subheader("Two digit images")
    i1 = st.slider("Sample A index", 0, 5, 0)
    i2 = st.slider("Sample B index", 0, 5, 0)
    a, b = _sample_image(i1), _sample_image(i2)
    c1, c2 = st.columns(2)
    with c1:
        up_a = st.file_uploader("Image A", type=["png", "jpg", "jpeg"])
        if up_a:
            st.image(up_a, width=140)
        elif a is not None:
            st.image(a, clamp=True, width=140, caption="sample A")
    with c2:
        up_b = st.file_uploader("Image B", type=["png", "jpg", "jpeg"])
        if up_b:
            st.image(up_b, width=140)
        elif b is not None:
            st.image(b, clamp=True, width=140, caption="sample B")
    if st.button(A.DEMO_LABEL.get(MODEL_ID, "Compare"), type="primary"):
        payload = {}
        payload["image_a"] = up_a if up_a is not None else a
        payload["image_b"] = up_b if up_b is not None else b
        if payload["image_a"] is None or payload["image_b"] is None:
            st.error("Provide two images (upload or use the bundled samples).")
            return
        show_result(A.predict(MODEL_ID, payload))
''',
    "segmentation": '''
    st.subheader("Image to segment")
    t1, t2 = st.tabs(["Upload", "Sample"])
    img = None
    with t1:
        up = st.file_uploader("Upload a PNG/JPG", type=["png", "jpg", "jpeg"])
        if up:
            img = up
    with t2:
        i = st.slider("Sample index", 0, 2, 0)
        arr = _sample_image(i)
        if arr is not None:
            st.image(arr, clamp=True, width=180, caption="bundled real sample")
            if st.checkbox("Use this sample"):
                img = arr
    if img is not None:
        if not hasattr(img, "read"):
            st.image(img, clamp=True, width=180)
        if st.button(A.DEMO_LABEL.get(MODEL_ID, "Segment"), type="primary"):
            show_result(A.predict(MODEL_ID, {"image": img}))
''',
    "detection": '''
    st.subheader("Image to run detection on")
    t1, t2 = st.tabs(["Upload", "Sample"])
    img = None
    with t1:
        up = st.file_uploader("Upload a PNG/JPG", type=["png", "jpg", "jpeg"])
        if up:
            img = up
    with t2:
        i = st.slider("Sample index", 0, 2, 0)
        arr = _sample_image(i)
        if arr is not None:
            st.image(arr, clamp=True, width=180, caption="bundled real sample")
            if st.checkbox("Use this sample"):
                img = arr
    conf = st.slider("Confidence threshold", 0.05, 0.95, 0.5, 0.05)
    if img is not None:
        if not hasattr(img, "read"):
            st.image(img, clamp=True, width=180)
        if st.button(A.DEMO_LABEL.get(MODEL_ID, "Detect"), type="primary"):
            show_result(A.predict(MODEL_ID, {"image": img, "conf": conf}))
''',
    "graph": '''
    st.subheader("Node feature vector")
    st.caption("Cora nodes are described by 1433 binary bag-of-words features.")
    presets = (samples or {}).get("rows", [])
    default = presets[0] if presets else [0.0] * 1433
    raw = st.text_area("1433 comma-separated values", ",".join(str(int(v)) for v in default), height=140)
    if st.button(A.DEMO_LABEL.get(MODEL_ID, "Classify node"), type="primary"):
        try:
            feats = [float(v) for v in raw.replace("\\n", ",").split(",") if v.strip()]
        except ValueError:
            st.error("Every value must be a number.")
            return
        if len(feats) != 1433:
            st.error(f"Expected 1433 values, received {len(feats)}.")
            return
        show_result(A.predict(MODEL_ID, {"features": feats}))
''',
}

_FOOT = '''

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
'''


def _frontend(model_id: int, info: dict, meta: dict) -> str:
    body = _BODIES[info["input_kind"]] if "input_kind" in info else None
    return None


KIND = {
    1: "tabular", 2: "image", 3: "sequence", 4: "sequence", 5: "sequence",
    6: "image", 7: "image", 8: "image", 9: "text", 10: "text", 11: "text_gen",
    12: "image", 13: "image", 14: "image", 15: "segmentation", 16: "detection",
    17: "pair", 18: "text_gen", 19: "tabular", 20: "graph",
}


def frontend_text(model_id: int, info: dict) -> str:
    head = (_HEAD.format(name=info["name"], dataset=info["dataset"], task=info["task"],
                         input=info["input"], mid=model_id))
    body = _BODIES[KIND[model_id]]
    return head + body + _FOOT


def generate_all() -> None:
    total_c = total_u = 0
    for mid in sorted(MODELS):
        info = MODELS[mid]
        folder = ROOT / info["folder"]
        meta = _load_meta(folder)
        created, updated = [], []
        write(folder / "backend" / "predictor.py", T.render(T.PREDICTOR, mid), created, updated)
        raw_pre = T.PREPROCESSING.replace("MODEL_ID_PLACEHOLDER,  # replaced at generation time\n", "")
        write(folder / "backend" / "preprocessing.py", T.render(raw_pre, mid), created, updated)
        write(folder / "backend" / "main.py", T.render(T.FASTAPI_MAIN, mid), created, updated)
        write(folder / "backend" / "requirements.txt", T.REQ_BACKEND, created, updated)
        write(folder / "frontend" / "app.py", frontend_text(mid, info), created, updated)
        write(folder / "app.py", T.render(T.ROOT_APP, mid), created, updated)
        write(folder / "requirements.txt", T.REQ_ROOT, created, updated)
        write(folder / "README_APP.md", _readme(mid, info, meta), created, updated)
        total_c += len(created)
        total_u += len(updated)
        print(f"[{mid:02d}] {info['folder']:16s} created={len(created)} updated={len(updated)}")
    print(f"TOTAL created={total_c} updated={total_u}")


if __name__ == "__main__":
    generate_all()
