"""Trainable Siamese Network reference implementation."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from model_zoo import Siamese, train_one
model = Siamese()
if __name__ == '__main__':
    _, losses, accuracy = train_one(17, epochs=10)
    print(f'Siamese Network trained: final_loss={losses[-1]:.4f}, accuracy={accuracy:.3f}')
