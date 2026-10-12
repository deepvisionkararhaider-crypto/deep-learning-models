"""Model registry — the single source of truth for every model in the platform.

Two families live here:

* ``area == "lab"``  — the original 20 architectures (tabular / vision / sequence /
  NLP / generative / graph). Each has a *real* public dataset and a task-appropriate
  input/output contract.
* ``area in {"medical", "agriculture"}`` — the six computer-vision model types
  (CNN, DenseNet121, ViT, U-Net, YOLO-style, Mask R-CNN) instantiated for the
  medical and agricultural domains.

Nothing here claims a model is trained; that is decided by the presence of an
artifact on disk (see :mod:`src.artifacts`). The Streamlit UI reads this registry
plus the artifact metadata to report the *real* status of each model.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from . import models as M
from .models import lab20 as LAB20_MODS
from .datasets import loaders as L
from .imaging import IMAGENET_MEAN, IMAGENET_STD

# --------------------------------------------------------------------------- #
# registry entry
# --------------------------------------------------------------------------- #
@dataclass
class ModelSpec:
    id: str
    name: str
    area: str                       # lab | medical | agriculture
    task: str                       # classification | segmentation | detection | instance_segmentation | ...
    framework: str
    dataset_key: str                # key into src.datasets.loaders, or a lab dataset tag
    dataset_name: str
    classes: list[str] = field(default_factory=list)
    input_desc: str = ""
    input_size: int = 224
    in_ch: int = 3
    outputs: str = ""
    metric_names: list[str] = field(default_factory=list)
    builder: str = ""               # key resolved by build_model()
    build_kwargs: dict[str, Any] = field(default_factory=dict)
    notes: str = ""
    lab: bool = False
    blocked: str = ""               # non-empty => cannot be trained; the reason

    @property
    def is_cv(self) -> bool:
        return self.area in ("medical", "agriculture")

    @property
    def is_blocked(self) -> bool:
        return bool(self.blocked)


# --------------------------------------------------------------------------- #
# builders: id -> zero-argument constructor (weights loaded separately)
# --------------------------------------------------------------------------- #
def build_model(spec: ModelSpec) -> Callable[[], Any]:
    """Return a callable that constructs the *empty* architecture for a spec.

    Used by the runtime (offline, no pretrained download) and by the trainer's
    weight-loading step. ``pretrained`` is always False here on purpose.
    """
    b = spec.builder
    n = len(spec.classes)
    if b == "cnn":
        return lambda: M.build_cnn(n, in_ch=spec.in_ch, pretrained=False)
    if b == "densenet":
        return lambda: M.build_densenet(n, in_ch=spec.in_ch, pretrained=False)
    if b == "vit":
        return lambda: M.build_vit(n, in_ch=spec.in_ch, pretrained=False, image_size=spec.input_size)
    if b == "unet":
        return lambda: M.build_unet(n if n > 2 else 1, in_ch=spec.in_ch, pretrained=False)
    if b == "yolo":
        return lambda: M.build_yolo(n, in_ch=spec.in_ch, grid=spec.build_kwargs.get("grid", 14),
                                    pretrained=False)
    if b == "maskrcnn":
        return lambda: M.build_maskrcnn(n + 1, pretrained=False)   # +1 background
    if b.startswith("lab:"):
        cls = M.lab20.LAB20[int(b.split(":")[1])]
        return lambda: cls(**spec.build_kwargs)
    raise KeyError(f"unknown builder {b!r}")


# --------------------------------------------------------------------------- #
# the six CV model types, instantiated per domain
# --------------------------------------------------------------------------- #
_CNN_KW = {"grid": 14}

MEDICAL: list[ModelSpec] = [
    ModelSpec(
        id="brain_cnn", name="CNN (brain MRI)", area="medical", task="classification",
        framework="PyTorch (from scratch)", dataset_key="brain_mri_cls",
        dataset_name="Brain MRI tumor classification (4-class)",
        classes=L.BRAIN_MRI_CLASSES, input_desc="224x224 RGB MRI slice",
        input_size=224, in_ch=3, outputs="4 class probabilities",
        metric_names=["accuracy", "precision_macro", "recall_macro", "f1_macro"],
        builder="cnn", notes="Trained from scratch; MRI is grayscale replicated to 3 channels."),
    ModelSpec(
        id="brain_densenet", name="DenseNet121 (brain MRI)", area="medical", task="classification",
        framework="PyTorch (transfer learning)", dataset_key="brain_mri_cls",
        dataset_name="Brain MRI tumor classification (4-class)",
        classes=L.BRAIN_MRI_CLASSES, input_desc="224x224 RGB MRI slice",
        input_size=224, in_ch=3, outputs="4 class probabilities",
        metric_names=["accuracy", "precision_macro", "recall_macro", "f1_macro"],
        builder="densenet", notes="ImageNet-pretrained backbone, fine-tuned classifier head."),
    ModelSpec(
        id="brain_vit", name="Vision Transformer (brain MRI)", area="medical", task="classification",
        framework="PyTorch (transfer learning)", dataset_key="brain_mri_cls",
        dataset_name="Brain MRI tumor classification (4-class)",
        classes=L.BRAIN_MRI_CLASSES, input_desc="224x224 RGB MRI slice",
        input_size=224, in_ch=3, outputs="4 class probabilities",
        metric_names=["accuracy", "precision_macro", "recall_macro", "f1_macro"],
        builder="vit", notes="CompactViT trained from scratch (16x16 patches, dim 192). "
                             "torchvision ViT-B/16 (344 MB) exceeds the GitHub 100 MB file limit."),
    ModelSpec(
        id="brain_unet", name="U-Net (brain tumor segmentation)", area="medical",
        task="segmentation", framework="PyTorch (from scratch)", dataset_key="brain_seg",
        dataset_name="Brain tumor segmentation (instance polygons unioned to a binary mask)",
        classes=["background", "tumor"], input_desc="224x224 RGB MRI slice",
        input_size=224, in_ch=3, outputs="224x224 binary tumor mask",
        metric_names=["dice", "iou"], builder="unet",
        notes="Binary target derived from the source dataset's COCO tumour polygons."),
    ModelSpec(
        id="brain_yolo", name="YOLO-style (brain tumor detection)", area="medical",
        task="detection", framework="PyTorch (anchor-free, from scratch)",
        dataset_key="brain_det", dataset_name="Brain tumor detection (COCO boxes)",
        classes=L.BRAIN_DET_CLASSES, input_desc="448x448 RGB MRI slice", input_size=448, in_ch=3,
        outputs="boxes + objectness", metric_names=["mAP@50"], builder="yolo",
        build_kwargs={"grid": 14},
        notes="Source dataset ships two un-named numeric labels; reported as label_0/label_1."),
    ModelSpec(
        id="brain_maskrcnn", name="Mask R-CNN (brain tumor instances)", area="medical",
        task="instance_segmentation", framework="PyTorch (torchvision)",
        dataset_key="brain_instseg", dataset_name="Brain tumor instance segmentation (COCO polygons)",
        classes=L.BRAIN_INSTSEG_CLASSES, input_desc="variable-size MRI slice (resized to 256x256)",
        input_size=256, in_ch=3, outputs="instances: box + label + mask",
        metric_names=["mask_AP@50", "box_AP@50"], builder="maskrcnn"),
]

AGRI: list[ModelSpec] = [
    ModelSpec(
        id="plant_cnn", name="CNN (plant leaf)", area="agriculture", task="classification",
        framework="PyTorch (from scratch)", dataset_key="plant_cls",
        dataset_name="Plant leaf disease (25-class)", classes=L.PLANT_CLASSES,
        input_desc="224x224 RGB leaf image", input_size=224, in_ch=3,
        outputs="25 class probabilities",
        metric_names=["accuracy", "precision_macro", "recall_macro", "f1_macro"], builder="cnn"),
    ModelSpec(
        id="plant_densenet", name="DenseNet121 (plant leaf)", area="agriculture", task="classification",
        framework="PyTorch (transfer learning)", dataset_key="plant_cls",
        dataset_name="Plant leaf disease (25-class)", classes=L.PLANT_CLASSES,
        input_desc="224x224 RGB leaf image", input_size=224, in_ch=3,
        outputs="25 class probabilities",
        metric_names=["accuracy", "precision_macro", "recall_macro", "f1_macro"],
        builder="densenet", notes="ImageNet-pretrained backbone, fine-tuned head."),
    ModelSpec(
        id="plant_vit", name="Vision Transformer (plant leaf)", area="agriculture",
        task="classification", framework="PyTorch (transfer learning)", dataset_key="plant_cls",
        dataset_name="Plant leaf disease (25-class)", classes=L.PLANT_CLASSES,
        input_desc="224x224 RGB leaf image", input_size=224, in_ch=3,
        outputs="25 class probabilities",
        metric_names=["accuracy", "precision_macro", "recall_macro", "f1_macro"], builder="vit"),
    ModelSpec(
        id="plant_unet", name="U-Net (leaf disease segmentation)", area="agriculture",
        task="segmentation", framework="PyTorch (from scratch)", dataset_key="plant_seg",
        dataset_name="Leaf disease segmentation (binary masks)",
        classes=["background", "disease"], input_desc="224x224 RGB leaf image",
        input_size=224, in_ch=3, outputs="224x224 binary disease mask",
        metric_names=["dice", "iou"], builder="unet"),
    ModelSpec(
        id="plant_yolo", name="YOLO-style (cotton leaf detection)", area="agriculture",
        task="detection", framework="PyTorch (anchor-free, from scratch)",
        dataset_key="cotton_det", dataset_name="Cotton leaf disease detection (COCO boxes)",
        classes=L.COTTON_DET_CLASSES, input_desc="448x448 RGB leaf image", input_size=448, in_ch=3,
        outputs="boxes + objectness", metric_names=["mAP@50"], builder="yolo",
        build_kwargs={"grid": 14}),
    ModelSpec(
        id="plant_maskrcnn", name="Mask R-CNN (leaf instances)", area="agriculture",
        task="instance_segmentation", framework="PyTorch (torchvision)",
        dataset_key="plant_instseg", dataset_name="Leaf instance segmentation (COCO polygons)",
        classes=["leaf", "disease"], input_desc="variable-size leaf image (resized to 256x256)",
        input_size=256, in_ch=3, outputs="instances: box + label + mask",
        metric_names=["mask_AP@50", "box_AP@50"], builder="maskrcnn",
        blocked="No agriculture instance-segmentation dataset is available on the HuggingFace "
                "datasets-server parquet mirror. daiv05/corn-leaf-instance-segmentation exists "
                "but has no parquet conversion (0 rows, config-size failed), so it cannot be "
                "fetched offline. This model has no trained weights and is not selectable."),
]

# --------------------------------------------------------------------------- #
# the original 20-model laboratory
# --------------------------------------------------------------------------- #
def _lab(i: int, name: str, task: str, ds_key: str, ds_name: str, classes: list[str],
         inp: str, size: int, in_ch: int, out: str, metrics: list[str],
         kwargs: dict | None = None, notes: str = "") -> ModelSpec:
    return ModelSpec(id=f"lab{i:02d}", name=f"{i:02d} · {name}", area="lab", task=task,
                     framework="PyTorch", dataset_key=ds_key, dataset_name=ds_name,
                     classes=classes, input_desc=inp, input_size=size, in_ch=in_ch,
                     outputs=out, metric_names=metrics, builder=f"lab:{i}",
                     build_kwargs=kwargs or {}, notes=notes, lab=True)


LAB20: list[ModelSpec] = [
    _lab(1, "ANN / MLP", "tabular_classification", "breast_cancer", "Breast Cancer Wisconsin (Diagnostic)",
         ["malignant", "benign"], "30 numeric cell-nucleus features", 0, 30, "2 class probabilities",
         ["accuracy", "f1_binary"], {"in_features": 30, "n_classes": 2}),
    _lab(2, "CNN", "classification", "mnist", "MNIST handwritten digits", [str(d) for d in range(10)],
         "28x28 grayscale image", 28, 1, "10 class probabilities", ["accuracy"],
         {"n_classes": 10, "in_ch": 1}),
    _lab(3, "RNN", "sequence_classification", "har", "UCI Smartphone HAR",
         ["walking", "upstairs", "downstairs", "sitting", "standing", "laying"],
         "128x9 inertial sensor window", 0, 0, "6 class probabilities", ["accuracy"],
         {"n_classes": 6, "feat": 9}, notes="Sequence model on raw inertial signals."),
    _lab(4, "LSTM", "sequence_classification", "har", "UCI Smartphone HAR",
         ["walking", "upstairs", "downstairs", "sitting", "standing", "laying"],
         "128x9 inertial sensor window", 0, 0, "6 class probabilities", ["accuracy"],
         {"n_classes": 6, "feat": 9}),
    _lab(5, "GRU", "sequence_classification", "har", "UCI Smartphone HAR",
         ["walking", "upstairs", "downstairs", "sitting", "standing", "laying"],
         "128x9 inertial sensor window", 0, 0, "6 class probabilities", ["accuracy"],
         {"n_classes": 6, "feat": 9}),
    _lab(6, "Autoencoder", "reconstruction", "mnist", "MNIST handwritten digits", [],
         "28x28 grayscale image", 28, 1, "28x28 reconstruction", ["recon_mse"],
         {"latent": 32}),
    _lab(7, "VAE", "reconstruction", "mnist", "MNIST handwritten digits", [],
         "28x28 grayscale image", 28, 1, "28x28 reconstruction + latent", ["recon_bce", "kl"],
         {"latent": 32}),
    _lab(8, "GAN (discriminator)", "binary_classification", "mnist", "MNIST handwritten digits",
         ["generated", "real"], "28x28 grayscale image", 28, 1, "real/fake score", ["accuracy"],
         {}),
    _lab(9, "Transformer", "text_classification", "newsgroups", "20 Newsgroups (4 classes)",
         ["comp.graphics", "sci.space", "rec.sport.baseball", "talk.politics.misc"],
         "text -> 64 tokens", 0, 0, "4 class probabilities", ["accuracy"],
         {"vocab": 4096, "dim": 64, "n_classes": 4, "length": 64}),
    _lab(10, "BERT-style", "text_classification", "newsgroups", "20 Newsgroups (4 classes)",
         ["comp.graphics", "sci.space", "rec.sport.baseball", "talk.politics.misc"],
         "text -> 64 tokens", 0, 0, "4 class probabilities", ["accuracy"],
         {"vocab": 4096, "dim": 64, "n_classes": 4, "length": 64}),
    _lab(11, "GPT-style", "language_modeling", "newsgroups", "20 Newsgroups corpus", [],
         "text prompt -> tokens", 0, 0, "next-token logits", ["perplexity"],
         {"vocab": 96, "dim": 96}),
    _lab(12, "ResNet", "classification", "mnist", "MNIST handwritten digits", [str(d) for d in range(10)],
         "28x28 grayscale image", 28, 1, "10 class probabilities", ["accuracy"],
         {"n_classes": 10, "in_ch": 1}),
    _lab(13, "DenseNet", "classification", "mnist", "MNIST handwritten digits", [str(d) for d in range(10)],
         "28x28 grayscale image", 28, 1, "10 class probabilities", ["accuracy"],
         {"n_classes": 10, "in_ch": 1}),
    _lab(14, "ViT", "classification", "mnist", "MNIST handwritten digits", [str(d) for d in range(10)],
         "28x28 grayscale image", 28, 1, "10 class probabilities", ["accuracy"],
         {"n_classes": 10, "in_ch": 1}),
    _lab(15, "U-Net", "segmentation", "mnist_seg", "MNIST foreground segmentation",
         ["background", "foreground"], "28x28 grayscale image", 28, 1, "28x28 binary mask",
         ["dice", "iou"], {"in_ch": 1}),
    _lab(16, "YOLO-style", "detection", "mnist_det", "MNIST digit localisation",
         [str(d) for d in range(10)], "28x28 grayscale image", 28, 1, "7x7 grid detections",
         ["mAP@50"], {"n_classes": 10, "in_ch": 1, "grid": 7, "width": 16}),
    _lab(17, "Siamese", "metric_learning", "mnist", "MNIST handwritten digits", [],
         "two 28x28 grayscale images", 28, 1, "embedding distance", ["pair_accuracy"],
         {"in_ch": 1, "emb": 32}),
    _lab(18, "Seq2Seq", "sequence_to_sequence", "newsgroups", "20 Newsgroups corpus", [],
         "noisy text -> tokens", 0, 0, "reconstructed token logits", ["char_accuracy"],
         {"vocab": 96, "dim": 96}),
    _lab(19, "Diffusion MLP", "tabular_denoising", "breast_cancer", "Breast Cancer Wisconsin (Diagnostic)",
         [], "30 numeric features + noise level", 0, 30, "predicted noise (30 dims)", ["denoise_mse"],
         {"dim": 30}),
    _lab(20, "GNN", "graph_classification", "cora", "Cora citation network",
         ["0", "1", "2", "3", "4", "5", "6"], "1433-dim node features + adjacency", 0, 0,
         "7 class probabilities", ["accuracy"], {"in_features": 1433, "hidden": 64, "n_classes": 7}),
]

ALL_MODELS: list[ModelSpec] = MEDICAL + AGRI + LAB20
BY_ID: dict[str, ModelSpec] = {m.id: m for m in ALL_MODELS}
LAB_BY_INDEX: dict[int, ModelSpec] = {i: BY_ID[f"lab{i:02d}"] for i in range(1, 21)}
CV_MODELS: list[ModelSpec] = MEDICAL + AGRI


def normalisation_for(spec: ModelSpec) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    """All CV models use ImageNet statistics; lab vision models use [0,1]."""
    if spec.is_cv:
        return IMAGENET_MEAN, IMAGENET_STD
    return (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)
