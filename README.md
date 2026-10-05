# Deep Learning Models — 20 Trainable Architectures + Real-Data Streamlit Lab

A single professional deep-learning portfolio containing **20 trainable model families**, one numbered subfolder per architecture, and a free public Streamlit application.

## Highlights

- 20 trainable architectures
- Real public benchmark datasets mapped to every model
- Streamlit **Real Dataset Training Lab**
- Prediction/input-preprocessing studio
- CPU-sized cached samples for free Streamlit Community Cloud
- PyTorch training with autograd + Adam
- Dataset/task/source shown for every model
- Detailed dataset attribution in [`DATASETS.md`](DATASETS.md)

## 20-model real-data map

| # | Architecture | Real dataset | Task |
|---:|---|---|---|
| 01 | ANN / MLP | UCI Breast Cancer Wisconsin | Binary classification |
| 02 | CNN | UCI Handwritten Digits | 10-class image classification |
| 03 | RNN | UCI Smartphone HAR | Activity recognition |
| 04 | LSTM | UCI Smartphone HAR | Activity recognition |
| 05 | GRU | UCI Smartphone HAR | Activity recognition |
| 06 | Autoencoder | UCI Handwritten Digits | Image reconstruction |
| 07 | VAE | UCI Handwritten Digits | Variational representation |
| 08 | GAN | UCI Handwritten Digits | Adversarial digit generation |
| 09 | Transformer | 20 Newsgroups | Topic classification |
| 10 | BERT-style | 20 Newsgroups | Bidirectional topic classification |
| 11 | GPT-style | 20 Newsgroups | Causal language modeling |
| 12 | ResNet | UCI Handwritten Digits | Residual classification |
| 13 | DenseNet | UCI Handwritten Digits | Dense classification |
| 14 | ViT | UCI Handwritten Digits | Vision-transformer classification |
| 15 | U-Net | UCI Handwritten Digits | Foreground segmentation |
| 16 | YOLO-style | UCI Handwritten Digits | Localization + classification |
| 17 | Siamese Network | UCI Handwritten Digits | Metric learning |
| 18 | Seq2Seq | 20 Newsgroups | Noisy text reconstruction |
| 19 | Diffusion MLP | UCI Breast Cancer Wisconsin | Tabular denoising |
| 20 | GNN | Cora citation benchmark | Node classification |

See [`DATASETS.md`](DATASETS.md) for source links, attribution, and implementation notes.

## Streamlit app

Run locally:

```bash
pip install -r requirements.txt
streamlit run app.py
```

The public app has:

1. **Overview** — real-data model map
2. **Prediction Studio** — model-specific input/preprocessing interface
3. **Real Dataset Training Lab** — choose any of the 20 models and train it on its mapped benchmark
4. **20-Model Catalog** — architecture and source navigation
5. **About** — deployment and engineering notes

### Free deployment

Deploy to Streamlit Community Cloud with:

```text
Repository: deepvisionkararhaider-crypto/deep-learning-models
Branch: main
Main file: app.py
```

The app is intentionally CPU-sized. Larger datasets are downloaded only when their model is selected and are cached in the runtime.

## Repository structure

```text
deep-learning-models/
├── app.py
├── model_zoo.py
├── real_datasets.py
├── DATASETS.md
├── requirements.txt
├── requirements-streamlit.txt
├── requirements-full.txt
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

## Engineering note

The Streamlit registry is now the canonical public demo path and uses real benchmark data instead of synthetic placeholder data. Some architecture demos use deterministic annotations derived from real digit images (for example foreground masks/bounding boxes) so the free deployment remains lightweight. These are educational demonstrations and do not claim production or leaderboard performance.

## Validation

```bash
python -m py_compile app.py model_zoo.py real_datasets.py
```

The first training run for HAR, 20 Newsgroups, or Cora requires network access to download and cache the public dataset.

## License

Educational/research portfolio. Review individual model folders and upstream dataset sources for implementation-specific licensing and attribution requirements.
