"""Trainable message-passing Graph Neural Network reference implementation."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from model_zoo import GNN, train_one
model = GNN()
if __name__ == '__main__':
    _, losses, accuracy = train_one(20, epochs=10)
    print(f'GNN trained: final_loss={losses[-1]:.4f}, accuracy={accuracy:.3f}')
