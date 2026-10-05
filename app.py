from pathlib import Path
import sys

import numpy as np
import streamlit as st
from PIL import Image, ImageOps
import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from model_zoo import MODEL_SPECS, SPEC_BY_ID, train_one

st.set_page_config(page_title="Deep Learning Models | Prediction Lab", page_icon="🧠", layout="wide")
GITHUB = "https://github.com/deepvisionkararhaider-crypto/deep-learning-models"

FOLDERS = {
    1:"ann", 2:"cnn", 3:"rnn", 4:"lstm", 5:"gru", 6:"autoencoder", 7:"vae", 8:"gan",
    9:"transformer", 10:"bert", 11:"gpt", 12:"resnet", 13:"densenet", 14:"vit", 15:"unet",
    16:"yolo", 17:"siamese", 18:"seq2seq", 19:"diffusion", 20:"gnn"
}
IMAGE_IDS = {2, 12, 13, 14, 15, 16}
SEQUENCE_IDS = {3, 4, 5, 9, 10, 11, 18}
TABULAR_IDS = {1, 6, 7, 8, 19}

@st.cache_resource(show_spinner=False)
def trained_model(model_id, epochs=6, lr=0.001):
    model, losses, acc = train_one(model_id, epochs=epochs, lr=lr)
    model.eval()
    return model, losses, acc

@st.cache_data(show_spinner=False)
def training_result(model_id, epochs, lr):
    _, losses, acc = train_one(model_id, epochs=epochs, lr=lr)
    return losses, acc

def parse_values(text, count):
    try:
        values = [float(x.strip()) for x in text.replace("\n", ",").split(",") if x.strip()]
    except ValueError:
        return None, f"Please enter numbers separated by commas. Expected {count} values."
    if len(values) != count:
        return None, f"Expected exactly {count} values, but received {len(values)}."
    return np.asarray(values, dtype=np.float32), None

def example_values(count):
    return ", ".join(f"{x:.3f}" for x in np.linspace(-1, 1, count))

def prepare_input(model_id, kind):
    if model_id in TABULAR_IDS:
        st.markdown("**Enter 20 numeric features.** The demo classifier is trained on the same 20-feature synthetic schema used by the model lab.")
        text = st.text_area("20 feature values (comma-separated)", value=example_values(20), height=100)
        x, err = parse_values(text, 20)
        if err:
            st.error(err); return None
        return torch.tensor(x).reshape(1, 20)

    if model_id in IMAGE_IDS:
        uploaded = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg", "webp"], key=f"pred_image_{model_id}")
        if uploaded is None:
            st.info("Upload an image to run a prediction.")
            return None
        image = Image.open(uploaded).convert("L")
        image = ImageOps.fit(image, (28, 28))
        arr = np.asarray(image, dtype=np.float32) / 255.0
        st.image(image, caption="Model input: 28×28 grayscale", width=220)
        return torch.tensor(arr).unsqueeze(0).unsqueeze(0)

    if model_id in SEQUENCE_IDS:
        st.markdown("**Enter 12 sequence values.** Each value is expanded into the 8-feature demo sequence expected by the selected architecture.")
        text = st.text_area("12 sequence values (comma-separated)", value=example_values(12), height=100)
        values, err = parse_values(text, 12)
        if err:
            st.error(err); return None
        return torch.tensor(np.repeat(values[:, None], 8, axis=1)).unsqueeze(0)

    if model_id == 17:
        st.markdown("**Enter two 20-feature vectors.** The Siamese network compares the two representations.")
        a_text = st.text_area("Vector A — 20 values", value=example_values(20), height=90)
        b_text = st.text_area("Vector B — 20 values", value=example_values(20), height=90)
        a, err_a = parse_values(a_text, 20)
        b, err_b = parse_values(b_text, 20)
        if err_a:
            st.error("Vector A: " + err_a); return None
        if err_b:
            st.error("Vector B: " + err_b); return None
        return torch.tensor(np.stack([a, b], axis=0))

    if model_id == 20:
        st.markdown("**Enter 24 graph features = 6 nodes × 4 features.** Rows are nodes.")
        text = st.text_area("6×4 graph matrix (24 comma-separated values)", value=example_values(24), height=110)
        values, err = parse_values(text, 24)
        if err:
            st.error(err); return None
        return torch.tensor(values.reshape(1, 6, 4))

    return None

def render_prediction():
    st.title("🔮 Prediction Studio")
    st.write("Give the selected model actual input data, run a lightweight training pass, and then get a live prediction from the trained PyTorch model.")
    st.warning("These are portfolio/demo predictions on lightweight synthetic training tasks. They are not medical, production, or benchmark predictions.")

    names = {n: f"{n:02d} · {name}" for n, name, _, _, _ in MODEL_SPECS}
    selected = st.selectbox("Select architecture", list(names), format_func=lambda n: names[n])
    _, name, _, kind, desc = SPEC_BY_ID[selected]
    st.caption(f"{desc} · Input type: {kind}")

    left, right = st.columns([1.5, 1])
    with left:
        x = prepare_input(selected, kind)
    with right:
        epochs = st.slider("Training epochs", 1, 15, 6)
        lr = st.select_slider("Learning rate", [0.0003, 0.001, 0.003, 0.01], value=0.001, format_func=lambda v: f"{v:g}")
        st.caption("The model is trained in-memory on the app's deterministic demo dataset before prediction.")

    if st.button("🚀 Train + Predict", type="primary", use_container_width=True):
        if x is None:
            st.error("Provide valid input data first.")
            return
        with st.spinner(f"Training {name} and generating prediction..."):
            model, losses, train_acc = trained_model(selected, epochs, lr)
            with torch.no_grad():
                logits = model(x)
                probs = torch.softmax(logits, dim=1)[0]
                pred = int(torch.argmax(probs).item())
        st.success(f"Prediction completed for {name}.")
        a, b, c = st.columns(3)
        a.metric("Predicted class", str(pred))
        b.metric("Confidence", f"{float(probs[pred]):.1%}")
        c.metric("Demo training accuracy", f"{train_acc:.1%}")
        st.subheader("Prediction probabilities")
        st.bar_chart({"Class 0": [float(probs[0])], "Class 1": [float(probs[1])]})
        with st.expander("Training details"):
            st.write(f"Final training loss: **{losses[-1]:.4f}**")
            st.line_chart({"loss": losses})
            st.caption("Prediction uses the actual trained PyTorch model and the input you supplied. The training dataset is synthetic and deterministic.")

def render_training():
    st.title("🎛️ Training Lab")
    st.write("Train any of the 20 architectures and inspect its learning curve.")
    names = {n: f"{n:02d} · {name}" for n, name, _, _, _ in MODEL_SPECS}
    selected = st.selectbox("Model", list(names), format_func=lambda n: names[n], key="train_model")
    _, name, _, kind, desc = SPEC_BY_ID[selected]
    st.caption(desc)
    a, b = st.columns(2)
    epochs = a.slider("Epochs", 1, 30, 8, key="train_epochs")
    lr = b.select_slider("Learning rate", [0.0001, 0.0003, 0.001, 0.003, 0.01], value=0.001, format_func=lambda x: f"{x:g}", key="train_lr")
    if st.button("🚀 Train selected model", type="primary", use_container_width=True):
        with st.spinner(f"Training {name}..."):
            losses, acc = training_result(selected, epochs, lr)
        st.success(f"{name} trained successfully.")
        m1, m2 = st.columns(2)
        m1.metric("Final loss", f"{losses[-1]:.4f}")
        m2.metric("Training accuracy", f"{acc:.1%}")
        st.line_chart({"loss": losses})

def render_overview():
    st.title("🧠 Deep Learning Models Lab")
    st.caption("20 trainable architectures with real user-input prediction, live training, and a public Streamlit interface.")
    a, b, c, d = st.columns(4)
    a.metric("Architectures", "20")
    b.metric("Prediction Studio", "Live")
    c.metric("Training", "PyTorch")
    d.metric("Deployment", "Streamlit Cloud")
    st.divider()
    st.subheader("Start here")
    st.info("Go to **🔮 Prediction Studio**, select a model, enter/upload your data, then click **Train + Predict**. Image models accept image uploads; tabular, sequence, pair, and graph models accept numeric data.")
    st.subheader("Model families")
    for start in range(0, len(MODEL_SPECS), 4):
        cols = st.columns(4)
        for col, (n, name, _, kind, desc) in zip(cols, MODEL_SPECS[start:start + 4]):
            with col:
                st.markdown(f"**{n:02d} · {name}**")
                st.caption(kind.title())
                st.write(desc)

def render_catalog():
    st.title("📦 20-Model Catalog")
    for n, name, cls, kind, desc in MODEL_SPECS:
        with st.expander(f"{n:02d} · {name} — {kind.title()}"):
            st.write(desc)
            st.write(f"**Python class:** `{cls.__name__}`")
            st.write(f"**Repository folder:** `{n:02d}_{FOLDERS[n]}/`")
            st.link_button("Open source", f"{GITHUB}/tree/main/{n:02d}_{FOLDERS[n]}", use_container_width=True)

def render_image_playground():
    st.title("🖼️ Image Playground")
    uploaded = st.file_uploader("Upload PNG/JPG/JPEG", type=["png", "jpg", "jpeg", "webp"])
    if uploaded is None:
        st.info("Upload an image to inspect the exact preprocessing used by the image prediction models.")
        return
    image = Image.open(uploaded).convert("RGB")
    processed = ImageOps.fit(image, (28, 28)).convert("L")
    a, b = st.columns(2)
    a.image(image, caption="Original", use_container_width=True)
    b.image(processed, caption="28×28 grayscale model input", width=220)
    arr = np.asarray(processed, dtype=np.float32) / 255.0
    st.json({"shape": list(arr.shape), "min": float(arr.min()), "max": float(arr.max()), "mean": float(arr.mean())})

def render_about():
    st.title("📖 About")
    st.markdown("This is a single deep-learning portfolio containing 20 trainable architectures. Each model has its own numbered subfolder. The public app now provides both training and actual user-input inference rather than only a catalog.")
    st.code("""deep-learning-models/
├── app.py
├── model_zoo.py
├── requirements.txt
├── 01_ann/ ... 16_yolo/
├── 17_siamese/
├── 18_seq2seq/
├── 19_diffusion/
└── 20_gnn/""")
    st.link_button("GitHub repository", GITHUB)

def main():
    with st.sidebar:
        st.markdown("# 🧠 Deep Learning Lab")
        page = st.radio("Navigate", ["Overview", "🔮 Prediction Studio", "🎛️ Training Lab", "📦 20-Model Catalog", "🖼️ Image Playground", "About"])
        st.divider()
        st.link_button("⭐ GitHub", GITHUB, use_container_width=True)
    if page == "Overview": render_overview()
    elif page == "🔮 Prediction Studio": render_prediction()
    elif page == "🎛️ Training Lab": render_training()
    elif page == "📦 20-Model Catalog": render_catalog()
    elif page == "🖼️ Image Playground": render_image_playground()
    else: render_about()
    st.divider()
    st.caption("Deep Learning Models · 20 trainable architectures · user-input prediction · Streamlit")

if __name__ == "__main__":
    main()
