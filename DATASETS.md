# Real Dataset & Task Map

All 20 Streamlit training paths now use real public benchmark data rather than synthetic placeholder samples. The free public app uses CPU-sized subsets and caches downloads in the runtime.

| # | Model | Dataset | Task |
|---:|---|---|---|
| 01 | ANN / MLP | UCI Breast Cancer Wisconsin (Diagnostic) | Malignant vs benign classification |
| 02 | CNN | UCI Optical Recognition of Handwritten Digits | 10-class image classification |
| 03 | RNN | UCI Human Activity Recognition Using Smartphones | 6-class sensor-sequence classification |
| 04 | LSTM | UCI Human Activity Recognition Using Smartphones | 6-class sensor-sequence classification |
| 05 | GRU | UCI Human Activity Recognition Using Smartphones | 6-class sensor-sequence classification |
| 06 | Autoencoder | UCI Optical Recognition of Handwritten Digits | Image reconstruction |
| 07 | VAE | UCI Optical Recognition of Handwritten Digits | Variational image representation |
| 08 | GAN | UCI Optical Recognition of Handwritten Digits | Adversarial digit generation |
| 09 | Transformer | 20 Newsgroups | 4-topic text classification |
| 10 | BERT-style | 20 Newsgroups | Bidirectional text classification |
| 11 | GPT-style | 20 Newsgroups | Causal next-token language modeling |
| 12 | ResNet | UCI Optical Recognition of Handwritten Digits | Residual image classification |
| 13 | DenseNet | UCI Optical Recognition of Handwritten Digits | Dense image classification |
| 14 | ViT | UCI Optical Recognition of Handwritten Digits | Vision-transformer classification |
| 15 | U-Net | UCI Optical Recognition of Handwritten Digits | Foreground segmentation |
| 16 | YOLO-style | UCI Optical Recognition of Handwritten Digits | Digit localization + classification |
| 17 | Siamese Network | UCI Optical Recognition of Handwritten Digits | Same/different digit metric learning |
| 18 | Seq2Seq | 20 Newsgroups | Noisy text reconstruction |
| 19 | Diffusion MLP | UCI Breast Cancer Wisconsin (Diagnostic) | Tabular denoising |
| 20 | GNN | Cora citation benchmark (Planetoid) | 7-class node classification |

## Sources

- UCI Machine Learning Repository: https://archive.ics.uci.edu/
- Breast Cancer Wisconsin (Diagnostic): https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic
- Optical Recognition of Handwritten Digits: https://archive.ics.uci.edu/dataset/80/optical+recognition+of+handwritten+digits
- Human Activity Recognition Using Smartphones: https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones
- scikit-learn 20 Newsgroups: https://scikit-learn.org/stable/datasets/real_world.html
- Cora / Planetoid: https://github.com/kimiyoung/planetoid

## Important implementation note

For models whose canonical task needs richer annotations, the demo uses deterministic annotations derived from the real digit images (foreground masks and bounding boxes) so the public app remains small and dependency-light. These are educational demonstrations, not replacements for COCO/PASCAL/VOC-scale detection or medical segmentation benchmarks.

The public app intentionally trains a reproducible subset to remain practical on free Streamlit Community Cloud CPU resources. It does not claim benchmark-level accuracy.
