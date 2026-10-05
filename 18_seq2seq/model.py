"""Trainable Seq2Seq encoder-decoder reference implementation."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from model_zoo import Seq2Seq, train_one
model = Seq2Seq()
if __name__ == '__main__':
    _, losses, accuracy = train_one(18, epochs=10)
    print(f'Seq2Seq trained: final_loss={losses[-1]:.4f}, accuracy={accuracy:.3f}')
