"""Interactive Streamlit dashboard for the Deep Learning Models repository."""
from __future__ import annotations

import numpy as np
import streamlit as st

st.set_page_config(page_title="Deep Learning Models Lab", page_icon="🧠", layout="wide")

MODELS = {
    "ANN — Breast Cancer": ("01_ann", "Binary classification with a multilayer perceptron"),
    "CNN — MNIST": ("02_cnn", "Image classification with convolutional layers"),
    "RNN — IMDB": ("03_rnn", "Sequence classification with a recurrent network"),
    "LSTM — IMDB": ("04_lstm", "Sequence classification with LSTM memory"),
    "GRU — IMDB": ("05_gru", "Sequence classification with gated recurrent units"),
    "Autoencoder — MNIST": ("06_autoencoder", "Unsupervised reconstruction and anomaly detection"),
    "VAE — MNIST": ("07_vae", "Variational latent-space modeling"),
    "GAN — MNIST": ("08_gan", "Adversarial generative modeling"),
    "Transformer — IMDB": ("09_transformer", "Attention-based sequence classification"),
    "BERT — Sentiment": ("10_bert", "Transformer encoder fine-tuning / inference"),
    "GPT — Text Generation": ("11_gpt", "Autoregressive language generation"),
    "ResNet — CIFAR-10": ("12_resnet", "Residual convolutional image classification"),
    "DenseNet — CIFAR-10": ("13_densenet", "Dense connectivity for image classification"),
    "ViT — CIFAR-10": ("14_vit", "Vision Transformer image classification"),
    "U-Net — Segmentation": ("15_unet", "Encoder-decoder image segmentation"),
}

st.title("🧠 Deep Learning Models Lab")
st.caption("Interactive portfolio dashboard for 15 deep-learning architectures and their learning objectives.")

with st.sidebar:
    st.header("Explore Models")
    selected = st.selectbox("Architecture", list(MODELS))
    st.divider()
    st.markdown("**Repository**")
    st.markdown("[GitHub source](https://github.com/deepvisionkararhaider-crypto/deep-learning-models)")
    st.markdown("**Stack**")
    st.write("TensorFlow • PyTorch • Transformers • scikit-learn • Streamlit")

folder, description = MODELS[selected]
col1, col2, col3 = st.columns(3)
col1.metric("Models", "15")
col2.metric("Selected", folder.upper())
col3.metric("Interface", "Streamlit")

st.subheader(selected)
st.write(description)

st.info("The original repository remains the source of truth for each model implementation. This dashboard provides a consistent interactive entry point without claiming benchmark results that have not been measured in this deployed app.")

if folder == "02_cnn":
    st.markdown("### CNN image demo")
    st.write("Upload a small grayscale image or use the built-in MNIST sample pipeline.")
    uploaded = st.file_uploader("Upload PNG/JPG", type=["png", "jpg", "jpeg"])
    if uploaded:
        from PIL import Image
        image = Image.open(uploaded).convert("L")
        st.image(image, caption="Uploaded image", width=220)
        arr = np.asarray(image.resize((28, 28)), dtype=np.float32) / 255.0
        st.write("Preprocessed tensor shape:", (1, 28, 28, 1))
        st.write("Pixel range:", f"{arr.min():.3f} → {arr.max():.3f}")
    else:
        st.write("No image uploaded. The repository's CNN implementation can be run locally from `02_cnn/`.")
elif folder in {"03_rnn", "04_lstm", "05_gru", "09_transformer", "10_bert"}:
    text = st.text_area("Text input", "This model family is designed to understand sequential language.")
    st.write("Input characters:", len(text))
    st.write("Tokenization/model inference is kept in the corresponding repository folder so the dashboard stays lightweight.")
elif folder == "11_gpt":
    prompt = st.text_area("Generation prompt", "Artificial intelligence will")
    st.code(prompt)
    st.write("The GPT implementation is available in `11_gpt/`; connect a local/model runtime to run generation without embedding credentials in the app.")
else:
    st.markdown("### Model profile")
    profiles = {
        "01_ann": ("ANN / MLP", "Dense layers", "Tabular classification"),
        "05_gru": ("GRU", "Gated recurrence", "Sequence modeling"),
        "06_autoencoder": ("Autoencoder", "Encoder + decoder", "Reconstruction"),
        "07_vae": ("VAE", "Probabilistic latent space", "Generation"),
        "08_gan": ("GAN", "Generator + discriminator", "Generation"),
        "12_resnet": ("ResNet", "Residual blocks", "Vision"),
        "13_densenet": ("DenseNet", "Dense feature reuse", "Vision"),
        "14_vit": ("ViT", "Patch embeddings + attention", "Vision"),
        "15_unet": ("U-Net", "Encoder-decoder + skip connections", "Segmentation"),
    }
    name, architecture, task = profiles.get(folder, (selected, "Deep learning architecture", "Modeling"))
    a, b, c = st.columns(3)
    a.write("**Architecture**"); a.write(name)
    b.write("**Core idea**"); b.write(architecture)
    c.write("**Primary task**"); c.write(task)

st.divider()
st.subheader("Engineering notes")
st.markdown("""
- **Reproducibility:** model-specific scripts remain isolated under numbered directories.
- **No secrets:** the app does not require API keys.
- **Honest evaluation:** repository metrics are not presented as live deployment benchmarks.
- **Extensibility:** additional model-specific interactive panels can be added without coupling all architectures into one script.
""")
