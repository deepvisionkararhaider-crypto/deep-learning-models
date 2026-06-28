# Deep Learning Models

A comprehensive collection of **15 deep learning model implementations**, each self-contained with runnable code, real datasets, predictions, metrics, and plots.

## Models Overview Table

| # | Model | Task Type | Folder | Dataset | Accuracy | Precision | F1 | Notes |
|---|-------|-----------|--------|---------|----------|-----------|-----|-------|
| 1 | ANN (MLP) | Binary Classification | [01_ann](01_ann/) | [Breast Cancer](https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+(Diagnostic)) | ~0.974 | ~0.980 | ~0.980 | 3-layer, Dropout |
| 2 | CNN | Multi-class (10) | [02_cnn](02_cnn/) | [MNIST](http://yann.lecun.com/exdb/mnist/) | ~0.990 | ~0.990 | ~0.990 | 2 Conv layers |
| 3 | RNN | Binary Classification | [03_rnn](03_rnn/) | [IMDB](https://ai.stanford.edu/~amaas/data/sentiment/) | ~0.800 | ~0.800 | ~0.800 | SimpleRNN |
| 4 | LSTM | Binary Classification | [04_lstm](04_lstm/) | [IMDB](https://ai.stanford.edu/~amaas/data/sentiment/) | ~0.820 | ~0.820 | ~0.820 | Stacked LSTM |
| 5 | GRU | Binary Classification | [05_gru](05_gru/) | [IMDB](https://ai.stanford.edu/~amaas/data/sentiment/) | ~0.830 | ~0.830 | ~0.830 | Stacked GRU |
| 6 | Autoencoder | Reconstruction/Anomaly | [06_autoencoder](06_autoencoder/) | [MNIST](http://yann.lecun.com/exdb/mnist/) | N/A | N/A | N/A | Recon MSE ~0.015 |
| 7 | VAE | Generative | [07_vae](07_vae/) | [MNIST](http://yann.lecun.com/exdb/mnist/) | N/A | N/A | N/A | Latent dim=2 |
| 8 | GAN | Generative | [08_gan](08_gan/) | [MNIST](http://yann.lecun.com/exdb/mnist/) | N/A | N/A | N/A | D/G loss tracked |
| 9 | Transformer | Binary Classification | [09_transformer](09_transformer/) | [IMDB](https://ai.stanford.edu/~amaas/data/sentiment/) | ~0.840 | ~0.840 | ~0.840 | Multi-head attention |
| 10 | BERT | Binary Classification | [10_bert](10_bert/) | [IMDB (20 samples)](https://ai.stanford.edu/~amaas/data/sentiment/) | ~1.000 | ~1.000 | ~1.000 | DistilBERT SST-2 |
| 11 | GPT | Text Generation | [11_gpt](11_gpt/) | [Custom Prompts](https://huggingface.co/gpt2) | N/A | N/A | N/A | Perplexity ~9.9 |
| 12 | ResNet | Multi-class (10) | [12_resnet](12_resnet/) | [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html) | ~0.650 | ~0.650 | ~0.640 | Residual blocks |
| 13 | DenseNet | Multi-class (10) | [13_densenet](13_densenet/) | [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html) | ~0.640 | ~0.640 | ~0.630 | Dense connections |
| 14 | ViT | Multi-class (10) | [14_vit](14_vit/) | [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html) | ~0.420 | ~0.420 | ~0.410 | Patch=4, Heads=4 |
| 15 | U-Net | Segmentation | [15_unet](15_unet/) | [Synthetic Circles](https://arxiv.org/abs/1505.04597) | N/A | N/A | N/A | Mean IoU ~0.92 |

## Evaluation Metrics
- **Accuracy**: Fraction of correct predictions. For segmentation, pixel-level accuracy.
- **Precision**: TP/(TP+FP). For multi-class, weighted average.
- **F1 Score**: Harmonic mean of precision and recall.
- **Confusion Matrix**: True vs predicted label matrix for classification tasks.
- **For unsupervised/generative (AE, VAE, GAN)**: Reconstruction MSE, perplexity, discriminator loss — accuracy/precision/F1/confusion matrix not applicable.
- **For segmentation (U-Net)**: IoU (Intersection over Union) is the standard metric.

## Setup
```bash
pip install -r requirements.txt
cd 01_ann && python model.py
```
## Dataset Sources
| Dataset | Source |
|---------|--------|
| MNIST | http://yann.lecun.com/exdb/mnist/ |
| IMDB | https://ai.stanford.edu/~amaas/data/sentiment/ |
| CIFAR-10 | https://www.cs.toronto.edu/~kriz/cifar.html |
| Breast Cancer | https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+(Diagnostic) |
| GPT-2 | https://huggingface.co/gpt2 |
| DistilBERT | https://huggingface.co/distilbert-base-uncased-finetuned-sst-2-english |
