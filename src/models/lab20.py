"""The original 20 deep-learning architectures, rebuilt with correct IO contracts.

Each class documents its exact input/output shape. The previous version fed a
30-feature ANN a 20-value vector (producing
``RuntimeError: mat1 and mat2 shapes cannot be multiplied (256x20 and 30x64)``);
here the feature count is a module constant and the trainer + runtime both read
it from the same registry entry, so the contract cannot drift.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

# ---- tabular -----------------------------------------------------------------
TABULAR_FEATURES = 30          # Breast Cancer Wisconsin (Diagnostic)


class MLP(nn.Module):
    """(N,30) -> (N,2)."""
    def __init__(self, in_features: int = TABULAR_FEATURES, n_classes: int = 2):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_features, 64), nn.ReLU(), nn.Dropout(0.2),
                                 nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, n_classes))

    def forward(self, x):
        return self.net(x)


# ---- vision (28x28 grayscale) -------------------------------------------------
class CNN(nn.Module):
    """(N,1,28,28) -> (N,10)."""
    def __init__(self, n_classes: int = 10, in_ch: int = 1):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_ch, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d(1))
        self.fc = nn.Linear(64, n_classes)

    def forward(self, x):
        return self.fc(self.features(x).flatten(1))


class ResNet(nn.Module):
    """(N,1,28,28) -> (N,10); one residual block."""
    def __init__(self, n_classes: int = 10, in_ch: int = 1):
        super().__init__()
        self.stem = nn.Conv2d(in_ch, 16, 3, padding=1)
        self.a = nn.Conv2d(16, 16, 3, padding=1)
        self.b = nn.Conv2d(16, 16, 3, padding=1)
        self.fc = nn.Linear(16, n_classes)

    def forward(self, x):
        z = F.relu(self.stem(x))
        z = F.relu(self.b(F.relu(self.a(z))) + z)
        return self.fc(F.adaptive_avg_pool2d(z, 1).flatten(1))


class DenseNet(nn.Module):
    """(N,1,28,28) -> (N,10); three densely connected layers."""
    def __init__(self, n_classes: int = 10, in_ch: int = 1):
        super().__init__()
        self.a = nn.Conv2d(in_ch, 12, 3, padding=1)
        self.b = nn.Conv2d(in_ch + 12, 12, 3, padding=1)
        self.c = nn.Conv2d(in_ch + 24, 12, 3, padding=1)
        self.fc = nn.Linear(36, n_classes)

    def forward(self, x):
        a = F.relu(self.a(x))
        b = F.relu(self.b(torch.cat([x, a], 1)))
        c = F.relu(self.c(torch.cat([x, a, b], 1)))
        return self.fc(F.adaptive_avg_pool2d(torch.cat([a, b, c], 1), 1).flatten(1))


class ViT(nn.Module):
    """(N,1,28,28) -> (N,10); 2x2 patches -> 196 tokens."""
    def __init__(self, n_classes: int = 10, in_ch: int = 1, dim: int = 64):
        super().__init__()
        self.patch = nn.Conv2d(in_ch, dim, 2, 2)
        self.cls = nn.Parameter(torch.zeros(1, 1, dim))
        layer = nn.TransformerEncoderLayer(dim, 4, dim * 2, batch_first=True, dropout=0.1)
        self.tr = nn.TransformerEncoder(layer, 2)
        self.fc = nn.Linear(dim, n_classes)

    def forward(self, x):
        t = self.patch(x).flatten(2).transpose(1, 2)
        cls = self.cls.expand(t.size(0), -1, -1)
        t = torch.cat([cls, t], 1)
        return self.fc(self.tr(t)[:, 0])


class UNetSmall(nn.Module):
    """(N,1,28,28) -> (N,1,28,28) logits; MNIST foreground segmentation."""
    def __init__(self, in_ch: int = 1):
        super().__init__()
        self.e1 = nn.Conv2d(in_ch, 8, 3, padding=1)
        self.e2 = nn.Conv2d(8, 16, 3, padding=1)
        self.out = nn.Conv2d(16, 1, 1)

    def forward(self, x):
        z = F.max_pool2d(F.relu(self.e1(x)), 2)
        return self.out(F.interpolate(F.relu(self.e2(z)), size=x.shape[-2:]))


class YOLOLite(nn.Module):
    """(N,1,28,28) -> (N,7,7,5+C); single digit localisation."""
    def __init__(self, n_classes: int = 10, in_ch: int = 1, grid: int = 7, width: int = 16):
        super().__init__()
        self.grid = grid
        self.back = nn.Sequential(
            nn.Conv2d(in_ch, width, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(width, width * 2, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d((grid, grid)))
        self.head = nn.Conv2d(width * 2, 5 + n_classes, 1)

    def forward(self, x):
        return self.head(self.back(x)).permute(0, 2, 3, 1).contiguous()


class Siamese(nn.Module):
    """Two (N,1,28,28) -> two (N,16) embeddings."""
    def __init__(self, in_ch: int = 1, emb: int = 32):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Conv2d(in_ch, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Flatten(), nn.Linear(32 * 7 * 7, emb))

    def forward(self, a, b):
        return self.enc(a), self.enc(b)


# ---- sequence (28 timesteps x 28 features) ------------------------------------
class _SeqClassifier(nn.Module):
    rnn_cls: type = nn.RNN

    def __init__(self, n_classes: int = 10, feat: int = 28, hidden: int = 64):
        super().__init__()
        self.rnn = self.rnn_cls(feat, hidden, batch_first=True)
        self.fc = nn.Linear(hidden, n_classes)

    def forward(self, x):
        out, _ = self.rnn(x)
        return self.fc(out[:, -1])


class RNN(_SeqClassifier):
    rnn_cls = nn.RNN


class LSTM(_SeqClassifier):
    rnn_cls = nn.LSTM


class GRU(_SeqClassifier):
    rnn_cls = nn.GRU


# ---- generative (28x28) -------------------------------------------------------
class Autoencoder(nn.Module):
    """(N,1,28,28) -> (N,1,28,28) reconstruction."""
    def __init__(self, latent: int = 32):
        super().__init__()
        self.enc = nn.Sequential(nn.Flatten(), nn.Linear(784, 128), nn.ReLU(), nn.Linear(128, latent))
        self.dec = nn.Sequential(nn.Linear(latent, 128), nn.ReLU(), nn.Linear(128, 784), nn.Sigmoid())

    def forward(self, x):
        return self.dec(self.enc(x)).view(-1, 1, 28, 28)


class VAE(nn.Module):
    """(N,1,28,28) -> (recon, mu, logvar)."""
    def __init__(self, latent: int = 32):
        super().__init__()
        self.h = nn.Linear(784, 128)
        self.mu = nn.Linear(128, latent)
        self.lv = nn.Linear(128, latent)
        self.dec = nn.Sequential(nn.Linear(latent, 128), nn.ReLU(), nn.Linear(128, 784), nn.Sigmoid())

    def forward(self, x):
        h = F.relu(self.h(x.flatten(1)))
        mu, lv = self.mu(h), self.lv(h)
        z = mu + torch.randn_like(mu) * torch.exp(0.5 * lv)
        return self.dec(z).view(-1, 1, 28, 28), mu, lv


class GANGenerator(nn.Module):
    def __init__(self, z: int = 64):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(z, 256), nn.ReLU(), nn.Linear(256, 784), nn.Sigmoid())

    def forward(self, z):
        return self.net(z).view(-1, 1, 28, 28)


class GANDiscriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Flatten(), nn.Linear(784, 256), nn.LeakyReLU(0.2), nn.Linear(256, 1))

    def forward(self, x):
        return self.net(x)


# ---- NLP ---------------------------------------------------------------------
class Transformer(nn.Module):
    """Token ids (N,L) -> (N,4)."""
    def __init__(self, vocab: int = 4096, dim: int = 64, n_classes: int = 4, length: int = 64):
        super().__init__()
        self.emb = nn.Embedding(vocab, dim)
        layer = nn.TransformerEncoderLayer(dim, 4, dim * 2, batch_first=True, dropout=0.1)
        self.enc = nn.TransformerEncoder(layer, 2)
        self.fc = nn.Linear(dim, n_classes)

    def forward(self, x):
        return self.fc(self.enc(self.emb(x)).mean(1))


class TinyBERT(Transformer):
    pass


class TinyGPT(nn.Module):
    """Token ids (N,L) -> (N,L,vocab) next-token logits."""
    def __init__(self, vocab: int = 96, dim: int = 96):
        super().__init__()
        self.emb = nn.Embedding(vocab, dim)
        self.pos = nn.Embedding(256, dim)
        layer = nn.TransformerEncoderLayer(dim, 4, dim * 2, batch_first=True, dropout=0.1)
        self.tr = nn.TransformerEncoder(layer, 2)
        self.fc = nn.Linear(dim, vocab)

    def forward(self, x):
        n = x.size(1)
        mask = torch.triu(torch.ones(n, n, device=x.device), 1).bool()
        h = self.emb(x) + self.pos(torch.arange(n, device=x.device))[None]
        return self.fc(self.tr(h, mask=mask))


class Seq2Seq(nn.Module):
    """Encoder-decoder char reconstruction; (N,L) ids -> (N,L,vocab) logits."""
    def __init__(self, vocab: int = 96, dim: int = 96):
        super().__init__()
        self.emb = nn.Embedding(vocab, dim)
        self.enc = nn.GRU(dim, dim, batch_first=True)
        self.dec = nn.GRU(dim, dim, batch_first=True)
        self.fc = nn.Linear(dim, vocab)

    def forward(self, x):
        h = self.enc(self.emb(x))[0]
        return self.fc(self.dec(h)[0])


# ---- tabular diffusion --------------------------------------------------------
class DiffusionMLP(nn.Module):
    """Noise predictor for a 30-dim tabular row: (N,30),(N,1) -> (N,30)."""
    def __init__(self, dim: int = TABULAR_FEATURES):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dim + 1, 128), nn.SiLU(), nn.Linear(128, 128),
                                 nn.SiLU(), nn.Linear(128, dim))

    def forward(self, x, t):
        return self.net(torch.cat([x, t], 1))


# ---- graph --------------------------------------------------------------------
class GNN(nn.Module):
    """(N,F),(N,N) -> (N,C); two-layer GCN."""
    def __init__(self, in_features: int = 1433, hidden: int = 64, n_classes: int = 7):
        super().__init__()
        self.w1 = nn.Linear(in_features, hidden)
        self.w2 = nn.Linear(hidden, hidden)
        self.fc = nn.Linear(hidden, n_classes)

    def forward(self, x, a):
        h = F.relu(self.w1(a @ x))
        return self.fc(F.relu(self.w2(a @ h)))


LAB20 = {
    1: MLP, 2: CNN, 3: RNN, 4: LSTM, 5: GRU, 6: Autoencoder, 7: VAE, 8: GANDiscriminator,
    9: Transformer, 10: TinyBERT, 11: TinyGPT, 12: ResNet, 13: DenseNet, 14: ViT,
    15: UNetSmall, 16: YOLOLite, 17: Siamese, 18: Seq2Seq, 19: DiffusionMLP, 20: GNN,
}
