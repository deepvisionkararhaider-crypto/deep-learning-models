"""Trainable noise-conditioned diffusion-style denoiser reference."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from model_zoo import DiffusionMLP, train_one
model = DiffusionMLP()
if __name__ == '__main__':
    _, losses, accuracy = train_one(19, epochs=10)
    print(f'Diffusion MLP trained: final_loss={losses[-1]:.4f}, accuracy={accuracy:.3f}')
