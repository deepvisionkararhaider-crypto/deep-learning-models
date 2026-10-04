# Deep Learning Models — Interactive Streamlit Lab

A professional collection of **15 deep-learning model implementations** presented through a polished Streamlit interface. The original numbered model folders are preserved, while `app.py` provides a public-facing interactive experience for exploring the architectures and running lightweight browser demos.

> **Live demo:** deployment is configured, but a public URL has not been fabricated. The connected Render workspace currently requires payment information before it will create a public web service.

## 🚀 What changed

This repository is no longer only a model-code collection. It now includes:

- A production-oriented **Streamlit application** (`app.py`)
- Interactive model catalog covering all 15 implementations
- Lightweight **ANN demonstration** using the Breast Cancer dataset
- Image upload and preprocessing playground for computer-vision workflows
- Dedicated project/about pages and repository navigation
- Lightweight deployment dependency set in `requirements-streamlit.txt`
- Render deployment blueprint in `render.yaml`
- GitHub Actions Streamlit smoke test for syntax/import validation
- Existing numbered model implementations preserved

## 🧠 Model Catalog

| # | Model | Task | Dataset | Folder |
|---:|---|---|---|---|
| 01 | ANN (MLP) | Binary Classification | Breast Cancer Wisconsin | `01_ann/` |
| 02 | CNN | Image Classification | MNIST | `02_cnn/` |
| 03 | RNN | Sentiment Classification | IMDB | `03_rnn/` |
| 04 | LSTM | Sentiment Classification | IMDB | `04_lstm/` |
| 05 | GRU | Sentiment Classification | IMDB | `05_gru/` |
| 06 | Autoencoder | Reconstruction / Anomaly | MNIST | `06_autoencoder/` |
| 07 | VAE | Generative Modeling | MNIST | `07_vae/` |
| 08 | GAN | Generative Modeling | MNIST | `08_gan/` |
| 09 | Transformer | Text Classification | IMDB | `09_transformer/` |
| 10 | BERT | Text Classification | IMDB | `10_bert/` |
| 11 | GPT | Text Generation | Custom Prompts | `11_gpt/` |
| 12 | ResNet | Image Classification | CIFAR-10 | `12_resnet/` |
| 13 | DenseNet | Image Classification | CIFAR-10 | `13_densenet/` |
| 14 | ViT | Image Classification | CIFAR-10 | `14_vit/` |
| 15 | U-Net | Image Segmentation | Synthetic Circles | `15_unet/` |

## 🖥️ Streamlit Application

Run the lightweight public-facing application locally:

```bash
pip install -r requirements-streamlit.txt
streamlit run app.py
```

The interface contains:

1. **Overview** — project summary, architecture coverage, and key metrics
2. **Model Catalog** — all 15 model families with direct source-folder links
3. **ANN Demo** — browser-based MLP inference demonstration
4. **Image Playground** — image upload, resizing, normalization, and tensor-shape inspection
5. **About** — architecture, technology, and repository information

The application deliberately keeps the full training dependencies separate from the lightweight deployment stack. This prevents a public demo from unnecessarily installing the complete TensorFlow/PyTorch research environment.

## 📁 Project Structure

```text
deep-learning-models/
├── app.py
├── requirements.txt
├── requirements-streamlit.txt
├── render.yaml
├── README.md
├── .streamlit/
│   └── config.toml
├── .github/
│   └── workflows/
│       └── streamlit-smoke.yml
├── 01_ann/
├── 02_cnn/
├── 03_rnn/
├── 04_lstm/
├── 05_gru/
├── 06_autoencoder/
├── 07_vae/
├── 08_gan/
├── 09_transformer/
├── 10_bert/
├── 11_gpt/
├── 12_resnet/
├── 13_densenet/
├── 14_vit/
└── 15_unet/
```

## 📊 Evaluation Metrics

- **Accuracy** — fraction of correct predictions
- **Precision** — TP / (TP + FP)
- **F1 score** — harmonic mean of precision and recall
- **Confusion matrix** — true vs predicted class distribution
- **Reconstruction MSE** — used by reconstruction models such as autoencoders
- **Perplexity / generation loss** — relevant to language-generation models
- **IoU** — standard segmentation metric for U-Net-style tasks

## 🧪 Testing

The repository includes a GitHub Actions smoke test that installs the lightweight Streamlit dependency set and validates that `app.py` compiles and imports successfully.

Local validation:

```bash
python -m py_compile app.py
python -c "import app; print('Streamlit app import: OK')"
```

## ☁️ Deployment

### Render

A deployment blueprint is included in `render.yaml` and uses:

```text
Build: pip install -r requirements-streamlit.txt
Start: streamlit run app.py --server.port $PORT --server.address 0.0.0.0
```

The current connected Render workspace requires payment information before a public service can be created. Once billing is enabled, the blueprint can be deployed without changing the application code.

### Streamlit Community Cloud

The repository is also structured for deployment from the `main` branch with `app.py` as the application entry point. Select this repository and set the main file to `app.py` in the Streamlit deployment interface.

## 🔗 Repository

**GitHub:** https://github.com/deepvisionkararhaider-crypto/deep-learning-models

## 📚 Dataset Sources

| Dataset | Source |
|---|---|
| MNIST | http://yann.lecun.com/exdb/mnist/ |
| IMDB | https://ai.stanford.edu/~amaas/data/sentiment/ |
| CIFAR-10 | https://www.cs.toronto.edu/~kriz/cifar.html |
| Breast Cancer Wisconsin | https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+(Diagnostic) |
| GPT-2 | https://huggingface.co/gpt2 |
| DistilBERT | https://huggingface.co/distilbert-base-uncased-finetuned-sst-2-english |

## License

Educational/research project. Review individual model folders for implementation-specific notes and dataset licensing requirements.
