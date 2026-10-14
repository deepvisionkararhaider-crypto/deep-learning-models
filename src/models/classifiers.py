"""Image classification architectures: CNN (from scratch), DenseNet121, ViT-B/16.

Each ``build_*`` returns a module with ``num_classes`` outputs. The ``pretrained``
flag is used **only by the offline trainer**; the Streamlit runtime always calls
these with ``pretrained=False`` so inference is fully offline.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class SmallCNN(nn.Module):
    """Compact VGG-style CNN trained from scratch (4 conv blocks)."""

    def __init__(self, n_classes: int, in_ch: int = 3, width: int = 32):
        super().__init__()
        w = width
        self.features = nn.Sequential(
            nn.Conv2d(in_ch, w, 3, padding=1), nn.BatchNorm2d(w), nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(w, w * 2, 3, padding=1), nn.BatchNorm2d(w * 2), nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(w * 2, w * 4, 3, padding=1), nn.BatchNorm2d(w * 4), nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(w * 4, w * 4, 3, padding=1), nn.BatchNorm2d(w * 4), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.fc = nn.Linear(w * 4, n_classes)

    def forward(self, x):
        return self.fc(self.features(x).flatten(1))


def build_cnn(n_classes: int, in_ch: int = 3, width: int = 32, pretrained: bool = False) -> nn.Module:
    return SmallCNN(n_classes, in_ch=in_ch, width=width)


def build_densenet(n_classes: int, in_ch: int = 3, pretrained: bool = False) -> nn.Module:
    from torchvision.models import densenet121, DenseNet121_Weights

    weights = DenseNet121_Weights.DEFAULT if pretrained else None
    m = densenet121(weights=weights)
    if in_ch != 3:
        m.features.conv0 = nn.Conv2d(in_ch, 64, kernel_size=7, stride=2, padding=3, bias=False)
    m.classifier = nn.Linear(m.classifier.in_features, n_classes)
    return m


class CompactViT(nn.Module):
    """A small from-scratch Vision Transformer (offline fallback only).

    This is the transformer that previously produced **chance-level accuracy**
    (plant_vit 0.0410): it was randomly initialised and trained for only a couple
    of epochs on CPU, which is not enough for a data-hungry ViT. It is kept only
    as a dependency-free fallback; the registry now ships a genuinely
    ImageNet-pretrained timm ViT — see :func:`build_vit`.
    """

    def __init__(self, n_classes: int, in_ch: int = 3, image_size: int = 224,
                 patch: int = 16, dim: int = 192, depth: int = 6, heads: int = 6):
        super().__init__()
        assert image_size % patch == 0, "image_size must be divisible by patch"
        self.patch = nn.Conv2d(in_ch, dim, kernel_size=patch, stride=patch)
        n_patches = (image_size // patch) ** 2
        self.cls = nn.Parameter(torch.zeros(1, 1, dim))
        self.pos = nn.Parameter(torch.zeros(1, n_patches + 1, dim))
        layer = nn.TransformerEncoderLayer(dim, heads, dim * 4, batch_first=True,
                                           dropout=0.1, activation="gelu")
        self.enc = nn.TransformerEncoder(layer, depth)
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, n_classes)

    def forward(self, x):
        t = self.patch(x).flatten(2).transpose(1, 2)          # (B, N, dim)
        cls = self.cls.expand(t.size(0), -1, -1)
        t = torch.cat([cls, t], dim=1) + self.pos
        t = self.norm(self.enc(t))
        return self.head(t[:, 0])


def build_vit(n_classes: int, in_ch: int = 3, pretrained: bool = False, image_size: int = 224,
              variant: str = "timm") -> nn.Module:
    """Vision Transformer.

    ``variant='timm'`` (default, shipped) — a genuinely **ImageNet-pretrained**
    ``vit_tiny_patch16_224`` from ``timm`` (5.5 M params, ~22 MB artifact). This
    replaces the previous randomly-initialised ``CompactViT``, which is the exact
    reason ``plant_vit`` scored chance-level (0.0410) accuracy: a from-scratch ViT
    cannot converge in 2-4 CPU epochs. The pretrained backbone fixes the root
    cause while staying well under GitHub's 100 MB file limit.

    ``variant='torchvision'`` — torchvision ``vit_b_16`` (86 M params, 344 MB
    artifact; over the GitHub limit) kept for completeness.

    ``pretrained`` is **only** True during offline training. At inference the
    runtime builds the same architecture with ``pretrained=False`` and loads the
    saved ``state_dict`` — so no weights are ever downloaded at serve time.
    """
    if variant == "torchvision":
        from torchvision.models import vit_b_16
        if pretrained:
            from torchvision.models import ViT_B_16_Weights
            m = vit_b_16(weights=ViT_B_16_Weights.DEFAULT)
        else:
            m = vit_b_16(weights=None)
        if in_ch != 3:
            m.conv_proj = nn.Conv2d(in_ch, m.conv_proj.out_channels,
                                    kernel_size=m.conv_proj.kernel_size, stride=m.conv_proj.stride)
        m.heads.head = nn.Linear(m.heads.head.in_features, n_classes)
        return m

    try:
        import timm
    except ImportError as exc:            # pragma: no cover - dependency is in requirements
        raise ImportError(
            "The ViT now uses the pretrained 'timm' backbone. Install it with "
            "`pip install timm>=1.0,<2` (it is listed in requirements.txt / "
            "requirements-train.txt)."
        ) from exc

    # dynamic_img_size lets the same checkpoint accept a different input size at
    # inference without breaking the stored positional embeddings.
    m = timm.create_model("vit_tiny_patch16_224", pretrained=pretrained,
                          num_classes=n_classes, in_chans=in_ch, dynamic_img_size=True)
    return m


CLASSIFIER_BUILDERS = {
    "cnn": build_cnn,
    "densenet": build_densenet,
    "vit": build_vit,
}
