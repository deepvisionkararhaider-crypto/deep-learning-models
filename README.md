# Deep Learning Models — 20 Trainable Architectures + Streamlit Lab

A professional, single-repository deep-learning portfolio with **20 trainable model families**, one subfolder per model, and a public-facing Streamlit application.

## 🚀 Highlights

- **20 trainable architectures** in one GitHub repository
- Dedicated numbered subfolder for every model
- Root `app.py` with an interactive model catalog
- **Interactive Training Lab**: choose any of the 20 models, set epochs and learning rate, and train it live in the browser
- Real PyTorch forward passes, automatic differentiation, Adam optimization, loss tracking, and accuracy reporting
- Image preprocessing playground
- Render deployment configuration included
- CPU-friendly synthetic demo data keeps the public app practical and avoids downloading large datasets at startup

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

The original model folders are preserved. Models 17–20 extend the collection with additional trainable architectures.

## 🖥️ Streamlit Application

Run locally:

```bash
pip install -r requirements-streamlit.txt
streamlit run app.py
```

The app includes:

1. **Overview** — architecture coverage and project summary
2. **20-Model Catalog** — model descriptions and GitHub navigation
3. **Interactive Training** — live training for all 20 models
4. **Image Playground** — upload, resize, normalize, and inspect image tensors
5. **About** — project structure and technology information

### Interactive training

The training page deliberately uses deterministic synthetic data rather than claiming benchmark performance. It demonstrates the engineering workflow end-to-end:

```text
synthetic data → model → forward pass → loss → backpropagation → Adam → metrics → loss curve
```

This makes the public demo fast, reproducible, and safe to run on CPU resources.

## 📁 Repository Structure

```text
deep-learning-models/
├── app.py
├── model_zoo.py
├── requirements.txt
├── requirements-streamlit.txt
├── render.yaml
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

## ☁️ Deployment

`render.yaml` is configured for a public Python web service:

```text
Build: pip install -r requirements-streamlit.txt
Start: streamlit run app.py --server.port $PORT --server.address 0.0.0.0
```

A live URL should only be published after the hosting provider reports the service as **Live**. The repository does not hard-code a fabricated deployment URL.

## 🧪 Validation

```bash
python -m py_compile app.py model_zoo.py
python -c "import model_zoo; print(len(model_zoo.MODEL_SPECS), 'trainable models registered')"
```

To smoke-test every interactive architecture:

```bash
python -c "import model_zoo; [model_zoo.train_one(i, epochs=1) for i, *_ in model_zoo.MODEL_SPECS]; print('20-model smoke test: OK')"
```

## 🛠️ Technology

Python · PyTorch · Streamlit · NumPy · scikit-learn · Pillow · OpenCV

## 🔗 Repository

https://github.com/deepvisionkararhaider-crypto/deep-learning-models

## License

Educational/research portfolio. Review individual model folders for implementation-specific dataset and licensing notes.
