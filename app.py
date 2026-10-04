import math
from pathlib import Path

import numpy as np
import streamlit as st
from PIL import Image, ImageOps
from sklearn.datasets import load_breast_cancer
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from yolo_page import render_yolo

st.set_page_config(
    page_title="Deep Learning Models | Interactive Lab",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

ROOT = Path(__file__).resolve().parent

MODELS = [
    (1, "ANN (MLP)", "Binary Classification", "Breast Cancer Wisconsin", "01_ann", "Feed-forward neural network with dropout concept."),
    (2, "CNN", "Image Classification", "MNIST", "02_cnn", "Convolutional feature extraction for visual patterns."),
    (3, "RNN", "Sentiment Classification", "IMDB", "03_rnn", "Sequence modeling with recurrent hidden states."),
    (4, "LSTM", "Sentiment Classification", "IMDB", "04_lstm", "Long-range sequence modeling with gated memory."),
    (5, "GRU", "Sentiment Classification", "IMDB", "05_gru", "Efficient gated recurrent sequence modeling."),
    (6, "Autoencoder", "Reconstruction / Anomaly", "MNIST", "06_autoencoder", "Learns compact representations by reconstructing inputs."),
    (7, "VAE", "Generative Modeling", "MNIST", "07_vae", "Probabilistic latent-space representation and generation."),
    (8, "GAN", "Generative Modeling", "MNIST", "08_gan", "Adversarial generator/discriminator training."),
    (9, "Transformer", "Text Classification", "IMDB", "09_transformer", "Self-attention based sequence representation."),
    (10, "BERT", "Text Classification", "IMDB", "10_bert", "Bidirectional transformer language representation."),
    (11, "GPT", "Text Generation", "Custom Prompts", "11_gpt", "Autoregressive transformer text generation."),
    (12, "ResNet", "Image Classification", "CIFAR-10", "12_resnet", "Residual connections for deep visual networks."),
    (13, "DenseNet", "Image Classification", "CIFAR-10", "13_densenet", "Dense feature reuse through layer-to-layer connections."),
    (14, "ViT", "Image Classification", "CIFAR-10", "14_vit", "Image patches processed through transformer attention."),
    (15, "U-Net", "Image Segmentation", "Synthetic Circles", "15_unet", "Encoder-decoder architecture with skip connections."),
    (16, "YOLO", "Object Detection", "Synthetic Shapes", "16_yolo", "Single-stage detector: grid head + NMS, predicts boxes and classes in one pass."),
]

@st.cache_resource
def train_ann_demo():
    data = load_breast_cancer()
    model = make_pipeline(
        StandardScaler(),
        MLPClassifier(hidden_layer_sizes=(32, 16), max_iter=500, random_state=42),
    )
    model.fit(data.data, data.target)
    return model, data


def score_label(probability: float):
    return "High confidence" if probability >= 0.85 else "Moderate confidence" if probability >= 0.60 else "Low confidence"


def render_home():
    st.title("🧠 Deep Learning Models — Interactive Lab")
    st.caption("A professional Streamlit interface for exploring 15 deep-learning implementations.")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Models", "16")
    c2.metric("Model Families", "11+")
    c3.metric("Core Tasks", "5")
    c4.metric("Interface", "Streamlit")

    st.divider()
    st.subheader("What this project provides")
    a, b, c = st.columns(3)
    with a:
        st.markdown("### 📚 Model Library")
        st.write("Numbered, self-contained implementations covering classical neural networks, CNNs, sequence models, transformers, generative models, and segmentation.")
    with b:
        st.markdown("### 🎛️ Interactive Demo")
        st.write("Run a lightweight ANN demonstration directly in the browser without needing to train the full research models.")
    with c:
        st.markdown("### 🚀 Deployment Ready")
        st.write("Designed as a public Streamlit application with reproducible dependency and deployment configuration.")

    st.subheader("Architecture coverage")
    for start in range(0, len(MODELS), 3):
        cols = st.columns(3)
        for col, item in zip(cols, MODELS[start:start + 3]):
            n, name, task, dataset, folder, description = item
            with col:
                st.markdown(f"**{n:02d} · {name}**")
                st.caption(task)
                st.write(description)
                st.code(folder, language="text")


def render_models():
    st.title("📦 Model Catalog")
    st.write("Browse the complete model collection and jump directly to its source folder on GitHub.")
    for n, name, task, dataset, folder, description in MODELS:
        with st.expander(f"{n:02d} · {name} — {task}"):
            left, right = st.columns([2, 1])
            with left:
                st.write(description)
                st.markdown(f"**Dataset:** {dataset}")
                st.markdown(f"**Repository folder:** `{folder}/`")
            with right:
                st.link_button("Open source", f"https://github.com/deepvisionkararhaider-crypto/deep-learning-models/tree/main/{folder}", use_container_width=True)


def render_ann():
    st.title("🔬 Interactive ANN Demo")
    st.write("This browser demo trains a compact MLP on the same Breast Cancer Wisconsin dataset family used by the ANN project. The numbered model implementation remains in `01_ann/`.")

    model, data = train_ann_demo()
    sample_index = st.slider("Select a dataset sample", 0, len(data.data) - 1, 0)
    row = data.data[sample_index]
    probability = float(model.predict_proba([row])[0, 1])
    prediction = int(model.predict([row])[0])
    actual = int(data.target[sample_index])

    a, b, c = st.columns(3)
    a.metric("Predicted class", "Malignant" if prediction == 0 else "Benign")
    b.metric("Actual class", "Malignant" if actual == 0 else "Benign")
    c.metric("Confidence", f"{max(probability, 1 - probability):.1%}")

    st.progress(max(probability, 1 - probability), text=score_label(max(probability, 1 - probability)))
    st.info("This is an interactive lightweight demo; the full training scripts and original model implementations are preserved in the repository.")

    st.subheader("Input feature snapshot")
    names = data.feature_names
    values = row
    display = {name: float(value) for name, value in zip(names[:10], values[:10])}
    st.dataframe(display, use_container_width=True)


def render_image():
    st.title("🖼️ Image Playground")
    st.write("Upload an image to inspect the preprocessing pipeline used by computer-vision model families in this repository.")
    uploaded = st.file_uploader("Upload PNG/JPG/JPEG", type=["png", "jpg", "jpeg"])
    if not uploaded:
        st.info("Upload an image to begin.")
        return
    image = Image.open(uploaded).convert("RGB")
    size = st.slider("Resize", 64, 512, 224, step=32)
    processed = ImageOps.fit(image, (size, size))
    left, right = st.columns(2)
    with left:
        st.image(image, caption="Original", use_container_width=True)
    with right:
        st.image(processed, caption=f"Model-ready preview ({size}×{size})", use_container_width=True)
    arr = np.asarray(processed, dtype=np.float32) / 255.0
    st.write({"shape": list(arr.shape), "min": round(float(arr.min()), 4), "max": round(float(arr.max()), 4), "mean": round(float(arr.mean()), 4)})


def render_about():
    st.title("📖 About the Project")
    st.markdown("""
    **Deep Learning Models** is a structured collection of 15 runnable implementations spanning classification,
    sequence modeling, generative learning, image recognition, and segmentation.

    The Streamlit layer is intentionally separated from the numbered model implementations. This keeps the original
    educational/research code intact while providing a polished public-facing interface for demonstrations and discovery.
    """)
    st.subheader("Repository structure")
    st.code("""deep-learning-models/
├── app.py                    # Streamlit public application
├── requirements.txt          # Full model-development dependencies
├── requirements-streamlit.txt # Lightweight deployment dependencies
├── .streamlit/config.toml
├── 01_ann/
├── 02_cnn/
├── ...
├── 14_vit/
└── 15_unet/
""")
    st.subheader("Technology")
    st.write("Python · Streamlit · scikit-learn · NumPy · Pillow · TensorFlow/PyTorch model implementations")
    st.link_button("View GitHub repository", "https://github.com/deepvisionkararhaider-crypto/deep-learning-models")


def main():
    with st.sidebar:
        st.markdown("# 🧠 Deep Learning Lab")
        st.caption("15 architectures · interactive showcase")
        page = st.radio("Navigate", ["Overview", "Model Catalog", "ANN Demo", "YOLO Demo", "Image Playground", "About"], index=0)
        st.divider()
        st.markdown("**Source**")
        st.markdown("[GitHub repository](https://github.com/deepvisionkararhaider-crypto/deep-learning-models)")
        st.caption("Educational demonstration — verify model metrics in the individual implementation folders.")

    if page == "Overview":
        render_home()
    elif page == "Model Catalog":
        render_models()
    elif page == "ANN Demo":
        render_ann()
    elif page == "YOLO Demo":
        render_yolo()
    elif page == "Image Playground":
        render_image()
    else:
        render_about()

    st.divider()
    st.caption("Deep Learning Models · Streamlit Interactive Lab")


if __name__ == "__main__":
    main()
