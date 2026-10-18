# Model Catalogue

**Generated from `models/manifest.json`.** Every model declares its input contract,
output contract, class list, preprocessing and checkpoint location. The Streamlit app
builds its input form from these declarations, so a model with an incomplete contract is
shown as not ready rather than being given invented fields.

Repository: **KararDeepVision/deep-learning-models** · pinned revision `aeab7749122941325aa52f916e942b01b5011dcd` · 32 models

## Medical imaging (6)

### `brain_cnn` — CNN (brain MRI)

- **Status**: `TRAINED_VERIFIED`
- **Task / modality**: classification / image
- **Architecture**: `cnn` — PyTorch (from scratch)
- **Input**: 224x224 RGB MRI slice — 224x224, 3 ch, mean [0.485, 0.456, 0.406], std [0.229, 0.224, 0.225]
- **Classes** (4): glioma, meningioma, no-tumor, pituitary
- **Output**: 4 class probabilities
- **Dataset**: Brain MRI tumor classification (4-class) (`brain_mri_cls`)
- **Splits**: {'test': 537, 'train': 3759, 'validation': 1073}
- **Training**: {'epochs': 3, 'lr': 0.001, 'batch_size': 8, 'split_seed': 42, 'backbone': 'from scratch', 'train_cap': 2000}
- **Metrics** (test n=537): accuracy=0.8585, precision_macro=0.858, recall_macro=0.8557, f1_macro=0.8555
- **Checkpoint**: `medical/brain_cnn/model.pt` (local `artifacts/medical/brain_cnn.pt`)
- **Limitations**: Trained from scratch; MRI is grayscale replicated to 3 channels.
- **Note**: remote artifact is published and marked verified

### `brain_densenet` — DenseNet121 (brain MRI)

- **Status**: `ARCH_ONLY`
- **Task / modality**: classification / image
- **Architecture**: `densenet` — PyTorch (transfer learning)
- **Input**: 224x224 RGB MRI slice — 224x224, 3 ch, mean [0.485, 0.456, 0.406], std [0.229, 0.224, 0.225]
- **Classes** (4): glioma, meningioma, no-tumor, pituitary
- **Output**: 4 class probabilities
- **Dataset**: Brain MRI tumor classification (4-class) (`brain_mri_cls`)
- **Checkpoint**: `medical/brain_densenet/model.pt` (local `None`)
- **Limitations**: ImageNet-pretrained backbone, fine-tuned classifier head.
- **Note**: architecture implemented; no trained weights in this checkout

### `brain_vit` — Vision Transformer (brain MRI)

- **Status**: `ARCH_ONLY`
- **Task / modality**: classification / image
- **Architecture**: `vit` — PyTorch (transfer learning)
- **Input**: 224x224 RGB MRI slice — 224x224, 3 ch, mean [0.485, 0.456, 0.406], std [0.229, 0.224, 0.225]
- **Classes** (4): glioma, meningioma, no-tumor, pituitary
- **Output**: 4 class probabilities
- **Dataset**: Brain MRI tumor classification (4-class) (`brain_mri_cls`)
- **Checkpoint**: `medical/brain_vit/model.pt` (local `None`)
- **Limitations**: Genuinely ImageNet-pretrained timm 'vit_tiny_patch16_224' (5.5M params, ~22MB artifact) fine-tuned end-to-end. Replaces the previous randomly-initialised CompactViT, which scored chance-level (0.0410) accuracy on 25-class plant disease.
- **Note**: architecture implemented; no trained weights in this checkout

### `brain_unet` — U-Net (brain tumor segmentation)

- **Status**: `ARCH_ONLY`
- **Task / modality**: segmentation / image
- **Architecture**: `unet` — PyTorch (from scratch)
- **Input**: 224x224 RGB MRI slice — 224x224, 3 ch, mean [0.485, 0.456, 0.406], std [0.229, 0.224, 0.225]
- **Classes** (2): background, tumor
- **Output**: 224x224 binary tumor mask
- **Dataset**: Brain tumor segmentation (instance polygons unioned to a binary mask) (`brain_seg`)
- **Checkpoint**: `medical/brain_unet/model.pt` (local `None`)
- **Limitations**: Binary target derived from the source dataset's COCO tumour polygons.
- **Note**: architecture implemented; no trained weights in this checkout

### `brain_yolo` — YOLO-style (brain tumor detection)

- **Status**: `ARCH_ONLY`
- **Task / modality**: detection / image
- **Architecture**: `yolo` — PyTorch (anchor-free, from scratch)
- **Input**: 448x448 RGB MRI slice — 448x448, 3 ch, mean [0.485, 0.456, 0.406], std [0.229, 0.224, 0.225]
- **Classes** (2): label_0, label_1
- **Output**: boxes + objectness
- **Dataset**: Brain tumor detection (COCO boxes) (`brain_det`)
- **Checkpoint**: `medical/brain_yolo/model.pt` (local `None`)
- **Limitations**: Source dataset ships two un-named numeric labels; reported as label_0/label_1.
- **Note**: architecture implemented; no trained weights in this checkout

### `brain_maskrcnn` — Mask R-CNN (brain tumor instances)

- **Status**: `DEFERRED`
- **Task / modality**: instance_segmentation / image
- **Architecture**: `maskrcnn` — PyTorch (torchvision)
- **Input**: variable-size MRI slice (resized to 256x256) — 256x256, 3 ch, mean [0.485, 0.456, 0.406], std [0.229, 0.224, 0.225]
- **Classes** (2): Tumor, 0
- **Output**: instances: box + label + mask
- **Dataset**: Brain tumor instance segmentation (COCO polygons) (`brain_instseg`)
- **Checkpoint**: `medical/brain_maskrcnn/model.pt` (local `None`)
- **Limitations**: —

## Agriculture (6)

### `plant_cnn` — CNN (plant leaf)

- **Status**: `ARCH_ONLY`
- **Task / modality**: classification / image
- **Architecture**: `cnn` — PyTorch (from scratch)
- **Input**: 224x224 RGB leaf image — 224x224, 3 ch, mean [0.485, 0.456, 0.406], std [0.229, 0.224, 0.225]
- **Classes** (25): Apple___Apple_scab, Apple___Black_rot, Apple___Cedar_apple_rust, Apple___healthy, Corn___Cercospora_leaf_spot, Corn___Common_rust, Corn___Northern_Leaf_Blight, Corn___healthy, Grape___Black_rot, Grape___Esca_(Black_Measles), Grape___Leaf_blight, Grape___healthy, Potato___Early_blight, Potato___Late_blight, Potato___healthy, Tomato___Bacterial_spot, Tomato___Early_blight, Tomato___Late_blight, Tomato___Leaf_Mold, Tomato___Septoria_leaf_spot, Tomato___Spider_mites, Tomato___Target_Spot, Tomato___Tomato_Yellow_Leaf_Curl_Virus, Tomato___Tomato_mosaic_virus, Tomato___healthy
- **Output**: 25 class probabilities
- **Dataset**: Plant leaf disease (25-class) (`plant_cls`)
- **Checkpoint**: `agriculture/plant_cnn/model.pt` (local `None`)
- **Limitations**: —
- **Note**: architecture implemented; no trained weights in this checkout

### `plant_densenet` — DenseNet121 (plant leaf)

- **Status**: `ARCH_ONLY`
- **Task / modality**: classification / image
- **Architecture**: `densenet` — PyTorch (transfer learning)
- **Input**: 224x224 RGB leaf image — 224x224, 3 ch, mean [0.485, 0.456, 0.406], std [0.229, 0.224, 0.225]
- **Classes** (25): Apple___Apple_scab, Apple___Black_rot, Apple___Cedar_apple_rust, Apple___healthy, Corn___Cercospora_leaf_spot, Corn___Common_rust, Corn___Northern_Leaf_Blight, Corn___healthy, Grape___Black_rot, Grape___Esca_(Black_Measles), Grape___Leaf_blight, Grape___healthy, Potato___Early_blight, Potato___Late_blight, Potato___healthy, Tomato___Bacterial_spot, Tomato___Early_blight, Tomato___Late_blight, Tomato___Leaf_Mold, Tomato___Septoria_leaf_spot, Tomato___Spider_mites, Tomato___Target_Spot, Tomato___Tomato_Yellow_Leaf_Curl_Virus, Tomato___Tomato_mosaic_virus, Tomato___healthy
- **Output**: 25 class probabilities
- **Dataset**: Plant leaf disease (25-class) (`plant_cls`)
- **Checkpoint**: `agriculture/plant_densenet/model.pt` (local `None`)
- **Limitations**: ImageNet-pretrained backbone, fine-tuned head.
- **Note**: architecture implemented; no trained weights in this checkout

### `plant_vit` — Vision Transformer (plant leaf)

- **Status**: `ARCH_ONLY`
- **Task / modality**: classification / image
- **Architecture**: `vit` — PyTorch (transfer learning)
- **Input**: 224x224 RGB leaf image — 224x224, 3 ch, mean [0.485, 0.456, 0.406], std [0.229, 0.224, 0.225]
- **Classes** (25): Apple___Apple_scab, Apple___Black_rot, Apple___Cedar_apple_rust, Apple___healthy, Corn___Cercospora_leaf_spot, Corn___Common_rust, Corn___Northern_Leaf_Blight, Corn___healthy, Grape___Black_rot, Grape___Esca_(Black_Measles), Grape___Leaf_blight, Grape___healthy, Potato___Early_blight, Potato___Late_blight, Potato___healthy, Tomato___Bacterial_spot, Tomato___Early_blight, Tomato___Late_blight, Tomato___Leaf_Mold, Tomato___Septoria_leaf_spot, Tomato___Spider_mites, Tomato___Target_Spot, Tomato___Tomato_Yellow_Leaf_Curl_Virus, Tomato___Tomato_mosaic_virus, Tomato___healthy
- **Output**: 25 class probabilities
- **Dataset**: Plant leaf disease (25-class) (`plant_cls`)
- **Checkpoint**: `agriculture/plant_vit/model.pt` (local `None`)
- **Limitations**: —
- **Note**: architecture implemented; no trained weights in this checkout

### `plant_unet` — U-Net (leaf disease segmentation)

- **Status**: `ARCH_ONLY`
- **Task / modality**: segmentation / image
- **Architecture**: `unet` — PyTorch (from scratch)
- **Input**: 224x224 RGB leaf image — 224x224, 3 ch, mean [0.485, 0.456, 0.406], std [0.229, 0.224, 0.225]
- **Classes** (2): background, disease
- **Output**: 224x224 binary disease mask
- **Dataset**: Leaf disease segmentation (binary masks) (`plant_seg`)
- **Checkpoint**: `agriculture/plant_unet/model.pt` (local `None`)
- **Limitations**: —
- **Note**: architecture implemented; no trained weights in this checkout

### `plant_yolo` — YOLO-style (cotton leaf detection)

- **Status**: `ARCH_ONLY`
- **Task / modality**: detection / image
- **Architecture**: `yolo` — PyTorch (anchor-free, from scratch)
- **Input**: 448x448 RGB leaf image — 448x448, 3 ch, mean [0.485, 0.456, 0.406], std [0.229, 0.224, 0.225]
- **Classes** (2): cotton-plant-disease, dc
- **Output**: boxes + objectness
- **Dataset**: Cotton leaf disease detection (COCO boxes) (`cotton_det`)
- **Checkpoint**: `agriculture/plant_yolo/model.pt` (local `None`)
- **Limitations**: —
- **Note**: architecture implemented; no trained weights in this checkout

### `plant_maskrcnn` — Mask R-CNN (leaf instances)

- **Status**: `BLOCKED`
- **Task / modality**: instance_segmentation / image
- **Architecture**: `maskrcnn` — PyTorch (torchvision)
- **Input**: variable-size leaf image (resized to 256x256) — 256x256, 3 ch, mean [0.485, 0.456, 0.406], std [0.229, 0.224, 0.225]
- **Classes** (2): leaf, disease
- **Output**: instances: box + label + mask
- **Dataset**: Leaf instance segmentation (COCO polygons) (`plant_instseg`)
- **Checkpoint**: `agriculture/plant_maskrcnn/model.pt` (local `None`)
- **Limitations**: No agriculture instance-segmentation dataset is available on the HuggingFace datasets-server parquet mirror. daiv05/corn-leaf-instance-segmentation exists but has no parquet conversion (0 rows, config-size failed), so it cannot be fetched offline. This model has no trained weights and is not selectable.
- **Note**: No agriculture instance-segmentation dataset is available on the HuggingFace datasets-server parquet mirror. daiv05/corn-leaf-instance-segmentation exists but has no parquet conversion (0 rows, config-size failed), so it cannot be fetched offline. This model has no trained weights and is not selectable.

## Deep-learning laboratories (20)

### `lab01` — 01 · ANN / MLP

- **Status**: `TRAINED_UNVERIFIED`
- **Task / modality**: tabular_classification / tabular
- **Architecture**: `lab:1` — PyTorch
- **Input**: 30 numeric cell-nucleus features
- **Classes** (2): malignant, benign
- **Output**: 2 class probabilities
- **Dataset**: Breast Cancer Wisconsin (Diagnostic) (`breast_cancer`)
- **Training**: {'epochs': 60, 'optimizer': 'adam'}
- **Metrics** (test n=—): accuracy=0.9561, f1_binary=0.9526
- **Checkpoint**: `laboratories/lab01/model.pt` (local `artifacts/lab/lab01.pt`)
- **Limitations**: —
- **Note**: weights present locally; fresh-download verification not recorded

### `lab02` — 02 · CNN

- **Status**: `TRAINED_UNVERIFIED`
- **Task / modality**: classification / image
- **Architecture**: `lab:2` — PyTorch
- **Input**: 28x28 grayscale image — 28x28, 1 ch, mean [0.5, 0.5, 0.5], std [0.5, 0.5, 0.5]
- **Classes** (10): 0, 1, 2, 3, 4, 5, 6, 7, 8, 9
- **Output**: 10 class probabilities
- **Dataset**: MNIST handwritten digits (`mnist`)
- **Training**: {'epochs': 15, 'batch_size': 128, 'train_n': 50000, 'test_n': 10000, 'seed': 42}
- **Metrics** (test n=—): accuracy=0.9324
- **Checkpoint**: `laboratories/lab02/model.pt` (local `artifacts/lab/lab02.pt`)
- **Limitations**: —
- **Note**: weights present locally; fresh-download verification not recorded

### `lab03` — 03 · RNN

- **Status**: `TRAINED_UNVERIFIED`
- **Task / modality**: sequence_classification / sequence
- **Architecture**: `lab:3` — PyTorch
- **Input**: 128x9 inertial sensor window
- **Classes** (6): walking, upstairs, downstairs, sitting, standing, laying
- **Output**: 6 class probabilities
- **Dataset**: UCI Smartphone HAR (`har`)
- **Training**: {'epochs': 12, 'batch_size': 128, 'train_n': 7352, 'test_n': 2947, 'split': 'official UCI train/test'}
- **Metrics** (test n=—): accuracy=0.6216
- **Checkpoint**: `laboratories/lab03/model.pt` (local `artifacts/lab/lab03.pt`)
- **Limitations**: Sequence model on raw inertial signals.
- **Note**: weights present locally; fresh-download verification not recorded

### `lab04` — 04 · LSTM

- **Status**: `TRAINED_UNVERIFIED`
- **Task / modality**: sequence_classification / sequence
- **Architecture**: `lab:4` — PyTorch
- **Input**: 128x9 inertial sensor window
- **Classes** (6): walking, upstairs, downstairs, sitting, standing, laying
- **Output**: 6 class probabilities
- **Dataset**: UCI Smartphone HAR (`har`)
- **Training**: {'epochs': 12, 'batch_size': 128, 'train_n': 7352, 'test_n': 2947, 'split': 'official UCI train/test'}
- **Metrics** (test n=—): accuracy=0.8775
- **Checkpoint**: `laboratories/lab04/model.pt` (local `artifacts/lab/lab04.pt`)
- **Limitations**: —
- **Note**: weights present locally; fresh-download verification not recorded

### `lab05` — 05 · GRU

- **Status**: `TRAINED_UNVERIFIED`
- **Task / modality**: sequence_classification / sequence
- **Architecture**: `lab:5` — PyTorch
- **Input**: 128x9 inertial sensor window
- **Classes** (6): walking, upstairs, downstairs, sitting, standing, laying
- **Output**: 6 class probabilities
- **Dataset**: UCI Smartphone HAR (`har`)
- **Training**: {'epochs': 12, 'batch_size': 128, 'train_n': 7352, 'test_n': 2947, 'split': 'official UCI train/test'}
- **Metrics** (test n=—): accuracy=0.9053
- **Checkpoint**: `laboratories/lab05/model.pt` (local `artifacts/lab/lab05.pt`)
- **Limitations**: —
- **Note**: weights present locally; fresh-download verification not recorded

### `lab06` — 06 · Autoencoder

- **Status**: `TRAINED_UNVERIFIED`
- **Task / modality**: reconstruction / image
- **Architecture**: `lab:6` — PyTorch
- **Input**: 28x28 grayscale image — 28x28, 1 ch, mean [0.5, 0.5, 0.5], std [0.5, 0.5, 0.5]
- **Output**: 28x28 reconstruction
- **Dataset**: MNIST handwritten digits (`mnist`)
- **Training**: {'epochs': 8, 'loss': 'mse'}
- **Metrics** (test n=—): recon_mse=0.02668
- **Checkpoint**: `laboratories/lab06/model.pt` (local `artifacts/lab/lab06.pt`)
- **Limitations**: —
- **Note**: weights present locally; fresh-download verification not recorded

### `lab07` — 07 · VAE

- **Status**: `TRAINED_UNVERIFIED`
- **Task / modality**: reconstruction / image
- **Architecture**: `lab:7` — PyTorch
- **Input**: 28x28 grayscale image — 28x28, 1 ch, mean [0.5, 0.5, 0.5], std [0.5, 0.5, 0.5]
- **Output**: 28x28 reconstruction + latent
- **Dataset**: MNIST handwritten digits (`mnist`)
- **Training**: {'epochs': 8, 'loss': 'bce + beta*kl'}
- **Metrics** (test n=—): recon_bce=0.14114, kl=5.4373
- **Checkpoint**: `laboratories/lab07/model.pt` (local `artifacts/lab/lab07.pt`)
- **Limitations**: —
- **Note**: weights present locally; fresh-download verification not recorded

### `lab08` — 08 · GAN (discriminator)

- **Status**: `TRAINED_UNVERIFIED`
- **Task / modality**: binary_classification / image
- **Architecture**: `lab:8` — PyTorch
- **Input**: 28x28 grayscale image — 28x28, 1 ch, mean [0.5, 0.5, 0.5], std [0.5, 0.5, 0.5]
- **Classes** (2): generated, real
- **Output**: real/fake score
- **Dataset**: MNIST handwritten digits (`mnist`)
- **Training**: {'epochs': 8, 'note': 'discriminator artifact; generator trained jointly'}
- **Metrics** (test n=—): accuracy=0.824
- **Checkpoint**: `laboratories/lab08/model.pt` (local `artifacts/lab/lab08.pt`)
- **Limitations**: —
- **Note**: weights present locally; fresh-download verification not recorded

### `lab09` — 09 · Transformer

- **Status**: `TRAINED_UNVERIFIED`
- **Task / modality**: text_classification / text
- **Architecture**: `lab:9` — PyTorch
- **Input**: text -> 64 tokens
- **Classes** (4): comp.graphics, sci.space, rec.sport.baseball, talk.politics.misc
- **Output**: 4 class probabilities
- **Dataset**: 20 Newsgroups (4 classes) (`newsgroups`)
- **Training**: {'epochs': 4, 'batch_size': 64, 'train_n': 0, 'length': 64, 'vocab': 4096}
- **Metrics** (test n=—): accuracy=0.0
- **Checkpoint**: `laboratories/lab09/model.pt` (local `artifacts/lab/lab09.pt`)
- **Limitations**: —
- **Note**: weights present locally; fresh-download verification not recorded

### `lab10` — 10 · BERT-style

- **Status**: `TRAINED_UNVERIFIED`
- **Task / modality**: text_classification / text
- **Architecture**: `lab:10` — PyTorch
- **Input**: text -> 64 tokens
- **Classes** (4): comp.graphics, sci.space, rec.sport.baseball, talk.politics.misc
- **Output**: 4 class probabilities
- **Dataset**: 20 Newsgroups (4 classes) (`newsgroups`)
- **Training**: {'epochs': 4, 'batch_size': 64, 'train_n': 0, 'length': 64, 'vocab': 4096}
- **Metrics** (test n=—): accuracy=0.0
- **Checkpoint**: `laboratories/lab10/model.pt` (local `artifacts/lab/lab10.pt`)
- **Limitations**: —
- **Note**: weights present locally; fresh-download verification not recorded

### `lab11` — 11 · GPT-style

- **Status**: `TRAINED_UNVERIFIED`
- **Task / modality**: language_modeling / text
- **Architecture**: `lab:11` — PyTorch
- **Input**: text prompt -> tokens
- **Output**: next-token logits
- **Dataset**: 20 Newsgroups corpus (`newsgroups`)
- **Training**: {'epochs': 4, 'batch_size': 64, 'length': 96}
- **Metrics** (test n=—): perplexity=11.36
- **Checkpoint**: `laboratories/lab11/model.pt` (local `artifacts/lab/lab11.pt`)
- **Limitations**: —
- **Note**: weights present locally; fresh-download verification not recorded

### `lab12` — 12 · ResNet

- **Status**: `ARCH_ONLY`
- **Task / modality**: classification / image
- **Architecture**: `lab:12` — PyTorch
- **Input**: 28x28 grayscale image — 28x28, 1 ch, mean [0.5, 0.5, 0.5], std [0.5, 0.5, 0.5]
- **Classes** (10): 0, 1, 2, 3, 4, 5, 6, 7, 8, 9
- **Output**: 10 class probabilities
- **Dataset**: MNIST handwritten digits (`mnist`)
- **Training**: {'epochs': 15, 'batch_size': 128, 'train_n': 50000, 'test_n': 10000, 'seed': 42}
- **Metrics** (test n=—): accuracy=0.852
- **Checkpoint**: `laboratories/lab12/model.pt` (local `None`)
- **Limitations**: —
- **Note**: architecture implemented; no trained weights in this checkout

### `lab13` — 13 · DenseNet

- **Status**: `ARCH_ONLY`
- **Task / modality**: classification / image
- **Architecture**: `lab:13` — PyTorch
- **Input**: 28x28 grayscale image — 28x28, 1 ch, mean [0.5, 0.5, 0.5], std [0.5, 0.5, 0.5]
- **Classes** (10): 0, 1, 2, 3, 4, 5, 6, 7, 8, 9
- **Output**: 10 class probabilities
- **Dataset**: MNIST handwritten digits (`mnist`)
- **Training**: {'epochs': 15, 'batch_size': 128, 'train_n': 50000, 'test_n': 10000, 'seed': 42}
- **Metrics** (test n=—): accuracy=0.7975
- **Checkpoint**: `laboratories/lab13/model.pt` (local `None`)
- **Limitations**: —
- **Note**: architecture implemented; no trained weights in this checkout

### `lab14` — 14 · ViT

- **Status**: `ARCH_ONLY`
- **Task / modality**: classification / image
- **Architecture**: `lab:14` — PyTorch
- **Input**: 28x28 grayscale image — 28x28, 1 ch, mean [0.5, 0.5, 0.5], std [0.5, 0.5, 0.5]
- **Classes** (10): 0, 1, 2, 3, 4, 5, 6, 7, 8, 9
- **Output**: 10 class probabilities
- **Dataset**: MNIST handwritten digits (`mnist`)
- **Training**: {'epochs': 15, 'batch_size': 128, 'train_n': 50000, 'test_n': 10000, 'seed': 42}
- **Metrics** (test n=—): accuracy=0.5877
- **Checkpoint**: `laboratories/lab14/model.pt` (local `None`)
- **Limitations**: —
- **Note**: architecture implemented; no trained weights in this checkout

### `lab15` — 15 · U-Net

- **Status**: `TRAINED_UNVERIFIED`
- **Task / modality**: segmentation / image
- **Architecture**: `lab:15` — PyTorch
- **Input**: 28x28 grayscale image — 28x28, 1 ch, mean [0.5, 0.5, 0.5], std [0.5, 0.5, 0.5]
- **Classes** (2): background, foreground
- **Output**: 28x28 binary mask
- **Dataset**: MNIST foreground segmentation (`mnist_seg`)
- **Training**: {'epochs': 6, 'loss': 'bce_dice'}
- **Metrics** (test n=—): dice=0.8628, iou=0.7587
- **Checkpoint**: `laboratories/lab15/model.pt` (local `artifacts/lab/lab15.pt`)
- **Limitations**: —
- **Note**: weights present locally; fresh-download verification not recorded

### `lab16` — 16 · YOLO-style

- **Status**: `ARCH_ONLY`
- **Task / modality**: detection / image
- **Architecture**: `lab:16` — PyTorch
- **Input**: 28x28 grayscale image — 28x28, 1 ch, mean [0.485, 0.456, 0.406], std [0.229, 0.224, 0.225]
- **Classes** (10): 0, 1, 2, 3, 4, 5, 6, 7, 8, 9
- **Output**: 7x7 grid detections
- **Dataset**: MNIST digit localisation (`mnist_det`)
- **Checkpoint**: `laboratories/lab16/model.pt` (local `None`)
- **Limitations**: —
- **Note**: architecture implemented; no trained weights in this checkout

### `lab17` — 17 · Siamese

- **Status**: `TRAINED_UNVERIFIED`
- **Task / modality**: metric_learning / image_pair
- **Architecture**: `lab:17` — PyTorch
- **Input**: two 28x28 grayscale images — 28x28, 1 ch, mean [0.5, 0.5, 0.5], std [0.5, 0.5, 0.5]
- **Output**: embedding distance
- **Dataset**: MNIST handwritten digits (`mnist`)
- **Training**: {'epochs': 6, 'loss': 'contrastive'}
- **Metrics** (test n=—): pair_accuracy=0.948
- **Checkpoint**: `laboratories/lab17/model.pt` (local `artifacts/lab/lab17.pt`)
- **Limitations**: —
- **Note**: weights present locally; fresh-download verification not recorded

### `lab18` — 18 · Seq2Seq

- **Status**: `TRAINED_UNVERIFIED`
- **Task / modality**: sequence_to_sequence / text
- **Architecture**: `lab:18` — PyTorch
- **Input**: noisy text -> tokens
- **Output**: reconstructed token logits
- **Dataset**: 20 Newsgroups corpus (`newsgroups`)
- **Training**: {'epochs': 4, 'batch_size': 64, 'length': 96}
- **Checkpoint**: `laboratories/lab18/model.pt` (local `artifacts/lab/lab18.pt`)
- **Limitations**: —
- **Note**: weights present locally; fresh-download verification not recorded

### `lab19` — 19 · Diffusion MLP

- **Status**: `TRAINED_UNVERIFIED`
- **Task / modality**: tabular_denoising / tabular
- **Architecture**: `lab:19` — PyTorch
- **Input**: 30 numeric features + noise level
- **Output**: predicted noise (30 dims)
- **Dataset**: Breast Cancer Wisconsin (Diagnostic) (`breast_cancer`)
- **Training**: {'steps': 400, 'loss': 'noise_prediction_mse'}
- **Metrics** (test n=—): denoise_mse=0.08571
- **Checkpoint**: `laboratories/lab19/model.pt` (local `artifacts/lab/lab19.pt`)
- **Limitations**: —
- **Note**: weights present locally; fresh-download verification not recorded

### `lab20` — 20 · GNN

- **Status**: `TRAINED_UNVERIFIED`
- **Task / modality**: graph_classification / graph
- **Architecture**: `lab:20` — PyTorch
- **Input**: 1433-dim node features + adjacency
- **Classes** (7): 0, 1, 2, 3, 4, 5, 6
- **Output**: 7 class probabilities
- **Dataset**: Cora citation network (`cora`)
- **Training**: {'epochs': 60, 'train_n': 1624, 'test_n': 542}
- **Metrics** (test n=—): accuracy=0.2269
- **Checkpoint**: `laboratories/lab20/model.pt` (local `artifacts/lab/lab20.pt`)
- **Limitations**: —
- **Note**: weights present locally; fresh-download verification not recorded

