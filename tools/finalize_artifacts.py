"""Copy staged artifacts into each model folder and bundle small input samples.

Run after tools/build_artifacts.py:
    python tools/finalize_artifacts.py

For every model it writes:
    <folder>/model/model.pt          trained torch state_dict
    <folder>/model/preprocess.pkl    exact preprocessing objects
    <folder>/model/meta.json         human-readable metadata
    <folder>/model/samples.json      a few real example inputs for the UI
"""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

STAGE = ROOT / ".build" / "artifacts"
CACHE = ROOT / ".artifacts_cache"


def folder_of(model_id: int) -> str:
    from app_shared import MODELS
    return MODELS[model_id]["folder"]


def mnist_samples(n=8):
    d = np.load(CACHE / "mnist.npz")
    xtr, ytr = d["xtr"], d["ytr"]
    idx = [0, 1, 2, 3, 7, 12, 21, 42][:n]
    return {"images": [xtr[i].astype(int).tolist() for i in idx],
            "labels": [int(ytr[i]) for i in idx]}


def make_samples(model_id: int) -> dict:
    from tools.build_artifacts import (synth_segmentation, synth_detection, load_har_robust)
    import real_datasets as rd
    from sklearn.datasets import load_breast_cancer
    from sklearn.preprocessing import StandardScaler

    if model_id in (1, 19):
        d = load_breast_cancer(as_frame=True)
        scaler = StandardScaler().fit(d.data.values)
        ben = scaler.transform(d.data.values[d.target == 1][:1])[0]
        mal = scaler.transform(d.data.values[d.target == 0][:1])[0]
        return {"features": {"benign-like": [round(float(v), 5) for v in ben],
                             "malignant-like": [round(float(v), 5) for v in mal]},
                "feature_names": list(d.feature_names)}
    if model_id in (2, 12, 13, 14):
        return mnist_samples()
    if model_id in (3, 4, 5):
        x, y, names = load_har_robust()
        picks = [0, 1, 100]
        return {"sequences": [[round(float(v), 5) for v in x[i].reshape(-1)] for i in picks],
                "labels": [names[int(y[i])] for i in picks]}
    if model_id in (6, 7, 8):
        return mnist_samples()
    if model_id in (9, 10):
        texts, labels, cats = _text_examples()
        return {"texts": texts, "labels": [cats[l] for l in labels]}
    if model_id in (11, 18):
        texts, _, _ = _text_examples()
        return {"texts": texts}
    if model_id == 15:
        imgs, _ = synth_segmentation(n=4)
        return {"images": [imgs[i, :, :, 0].numpy().round(4).tolist() for i in range(3)]}
    if model_id == 16:
        imgs, _ = synth_detection(n=4)
        return {"images": [imgs[i, :, :, 0].numpy().round(4).tolist() for i in range(3)]}
    if model_id == 17:
        d = np.load(CACHE / "mnist.npz")
        xtr, ytr = d["xtr"], d["ytr"]
        return {"pairs": [[xtr[0].astype(int).tolist(), xtr[1].astype(int).tolist()],
                          [xtr[2].astype(int).tolist(), xtr[2].astype(int).tolist()]],
                "labels": [f"{int(ytr[0])} vs {int(ytr[1])}", f"{int(ytr[2])} vs {int(ytr[2])}"]}
    if model_id == 20:
        x, y, a = rd.cora()
        return {"rows": [x[i].tolist() for i in (0, 1)],
                "labels": [int(y[i]) for i in (0, 1)]}
    return {}


def _text_examples():
    try:
        from tools.build_artifacts import load_text_data
        texts, labels, cats = _load_text_raw()
        return texts[:3], labels[:3], cats
    except Exception:  # noqa: BLE001
        return ["The team won the championship final in overtime.",
                "Markets rallied as the central bank held interest rates steady.",
                "A new satellite will study the surface of Mars."], [1, 2, 3], \
               ["World", "Sports", "Business", "Sci/Tech"]


def _load_text_raw():
    from tools.build_artifacts import _load_text_corpus
    texts, labels, cats = _load_text_corpus()
    return texts[:3], labels[:3], cats


def write_npy_samples(model_id: int, dst: Path) -> list[str]:
    """Bundle small REAL input tensors so demo apps never touch the network."""
    from tools.build_artifacts import synth_segmentation, synth_detection
    written = []
    if model_id in (2, 6, 7, 8, 12, 13, 14, 17):
        d = np.load(CACHE / "mnist.npz")
        arr = d["xtr"][:6].astype("uint8")
        np.save(dst / "sample_images.npy", arr)
        np.save(dst / "sample_labels.npy", d["ytr"][:6].astype("int64"))
        written += ["sample_images.npy", "sample_labels.npy"]
    if model_id == 15:
        imgs, _ = synth_segmentation(n=3)
        np.save(dst / "sample_images.npy", (imgs[:, :, :, 0].numpy() * 255).astype("uint8"))
        written.append("sample_images.npy")
    if model_id == 16:
        imgs, _ = synth_detection(n=3)
        np.save(dst / "sample_images.npy", (imgs[:, :, :, 0].numpy() * 255).astype("uint8"))
        written.append("sample_images.npy")
    return written


def main() -> None:
    from app_shared import MODELS
    done = []
    for mid, info in sorted(MODELS.items()):
        src = STAGE / str(mid) / "model"
        if not (src / "model.pt").exists():
            print(f"[{mid:02d}] SKIP — no staged artifact")
            continue
        dst = ROOT / info["folder"] / "model"
        dst.mkdir(parents=True, exist_ok=True)
        for name in ("model.pt", "preprocess.pkl", "meta.json"):
            shutil.copy2(src / name, dst / name)
        try:
            (dst / "samples.json").write_text(json.dumps(make_samples(mid)), encoding="utf-8")
        except Exception as e:  # noqa: BLE001
            print(f"[{mid:02d}] samples warning: {type(e).__name__}: {e}")
            (dst / "samples.json").write_text("{}", encoding="utf-8")
        try:
            written = write_npy_samples(mid, dst)
            if written:
                print(f"[{mid:02d}] bundled sample tensors: {written}")
        except Exception as e:  # noqa: BLE001
            print(f"[{mid:02d}] npy samples warning: {type(e).__name__}: {e}")
        done.append(mid)
        print(f"[{mid:02d}] {info['folder']}/model  <- {sorted(p.name for p in dst.iterdir())}")
    print(f"finalized {len(done)}/{len(MODELS)} models")


if __name__ == "__main__":
    main()
