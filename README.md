# Deep Learning Models — 20 Trainable Architectures + Streamlit Prediction Lab

A professional, single-repository deep-learning portfolio with **20 trainable model families**, one numbered subfolder per model, and a public Streamlit application.

## 🚀 Highlights

- **20 trainable architectures** in one GitHub repository
- Dedicated numbered subfolder for every model
- **Prediction Studio** — provide numeric data or upload an image and receive a live prediction from a trained PyTorch model
- **Training Lab** — choose any architecture, epochs, and learning rate and inspect the loss curve
- Real PyTorch forward passes, automatic differentiation, Adam optimization, loss tracking, and prediction probabilities
- Image preprocessing playground
- **Streamlit Community Cloud ready** — no paid Render service required
- Lightweight runtime dependencies
- CPU-friendly deterministic demo data

## 🧠 Model Catalog

| # | Model | Family / task | Folder |
|---:|---|---|---|
| 01 | ANN / MLP | Feed-forward classification | `01_ann/` |
| 02 | CNN | Image classification | `02_cnn/` |
| 03 | RNN | Sequence classification | `03_rnn/` |
| 04 | LSTM | Sequence classification | `04_lstm/` |
| 05 | GRU | Sequence classification | `05_gru/` |
| 06 | Autoencoder | Representation learning | `06_autoencoder/` |
| 07 | VAE | Variational representation | `07_vae/` |
| 08 | GAN | Adversarial learning | `08_gan/` |
| 09 | Transformer | Self-attention sequence model | `09_transformer/` |
| 10 | BERT-style | Bidirectional transformer | `10_bert/` |
| 11 | GPT-style | Causal transformer | `11_gpt/` |
| 12 | ResNet | Residual vision model | `12_resnet/` |
| 13 | DenseNet | Dense-connectivity vision model | `13_densenet/` |
| 14 | ViT | Vision Transformer | `14_vit/` |
| 15 | U-Net | Encoder-decoder vision model | `15_unet/` |
| 16 | YOLO-style | Single-stage detection head | `16_yolo/` |
| 17 | Siamese Network | Metric learning | `17_siamese/` |
| 18 | Seq2Seq | Encoder-decoder sequence model | `18_seq2seq/` |
| 19 | Diffusion MLP | Noise-conditioned denoising | `19_diffusion/` |
| 20 | GNN | Graph message passing | `20_gnn/` |

## 🖥️ Streamlit Application

Run locally:

```bash
pip install -r requirements.txt
streamlit run app.py
```

### 🔮 Prediction Studio

The public application now accepts **real user input** instead of only displaying model information.

| Model family | User input |
|---|---|
| ANN / Autoencoder / VAE / GAN / Diffusion | 20 numeric features |
| CNN / ResNet / DenseNet / ViT / U-Net / YOLO-style | PNG/JPG/WEBP image upload |
| RNN / LSTM / GRU / Transformer / BERT / GPT / Seq2Seq | 12 sequence values |
| Siamese Network | Two 20-feature vectors |
| GNN | 6 nodes × 4 features = 24 values |

Workflow:

```text
user input
   ↓
architecture-specific preprocessing
   ↓
train lightweight PyTorch model
   ↓
forward pass on user data
   ↓
class prediction + confidence
```

The app trains on a deterministic synthetic demonstration dataset. Therefore the predictions demonstrate the complete ML inference workflow but **must not be presented as medical, production, or benchmark results**.

### 🎛️ Training Lab

The Training Lab lets you select any of the 20 architectures and configure epochs and learning rate. It performs real forward passes, backpropagation, Adam optimization, and reports loss/accuracy.

## 📁 Repository Structure

```text
deep-learning-models/
├── app.py
├── model_zoo.py
├── requirements.txt
├── requirements-streamlit.txt
├── requirements-full.txt
├── .streamlit/
├── .github/workflows/
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
├── 15_unet/
├── 16_yolo/
├── 17_siamese/
├── 18_seq2seq/
├── 19_diffusion/
└── 20_gnn/
```

## ☁️ Free Deployment — Streamlit Community Cloud

Use:

```text
Repository: deepvisionkararhaider-crypto/deep-learning-models
Branch: main
Main file: app.py
```

The root `requirements.txt` is intentionally lightweight for the public Streamlit deployment.

## ⚠️ Important

This portfolio uses synthetic deterministic data for the browser demo so all 20 architectures can run on CPU without downloading large datasets. The app is an **engineering/education demonstration of training and inference**, not a claim of real-world model performance.

## 🧪 Validation

```bash
python -m py_compile app.py model_zoo.py
python -c "import model_zoo; print(len(model_zoo.MODEL_SPECS), 'trainable models registered')"
```

## 🔗 Repository

https://github.com/deepvisionkararhaider-crypto/deep-learning-models

## License

Educational/research portfolio. Review individual model folders for implementation-specific dataset and licensing notes.
