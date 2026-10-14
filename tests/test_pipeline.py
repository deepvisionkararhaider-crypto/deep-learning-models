"""Regression tests for the data pipeline, detection/segmentation math, artifacts,
split-leakage prevention and the training/verification status contract.

All fixtures here are **small synthetic tensors / temporary files** — they prove
the *code* is correct, never that a real model is accurate.
"""
from __future__ import annotations

import io
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import artifacts as A                       # noqa: E402
from src.datasets import loaders as L                # noqa: E402
from src.models import detection as DET              # noqa: E402
from src.models import segmentation as SEG           # noqa: E402
from src.registry import BY_ID, CV_MODELS, build_model  # noqa: E402


# --------------------------------------------------------------------------- #
# split routing / leakage
# --------------------------------------------------------------------------- #
def _touch_parquet(p: Path, text: bytes = b"not-a-real-parquet"):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(text)


def test_split_of_canonicalises_aliases():
    assert L.split_of(Path("valid-2.parquet")) == "validation"
    assert L.split_of(Path("val-0.parquet")) == "validation"
    assert L.split_of(Path("validation-1.parquet")) == "validation"
    assert L.split_of(Path("test-3.parquet")) == "test"
    assert L.split_of(Path("train-0.parquet")) == "train"


def test_files_for_never_returns_other_splits(tmp_path, monkeypatch):
    """Regression: reading every cached file regardless of split leaked test rows."""
    monkeypatch.setattr(L, "CACHE", tmp_path)
    key = "demo"
    for name in ("train-0.parquet", "validation-1.parquet", "test-2.parquet"):
        _touch_parquet(tmp_path / key / name)
    assert [p.name for p in L.files_for(key, "train")] == ["train-0.parquet"]
    assert [p.name for p in L.files_for(key, "test")] == ["test-2.parquet"]
    assert [p.name for p in L.files_for(key, "validation")] == ["validation-1.parquet"]
    assert len(L.files_for(key)) == 3
    assert L.split_counts(key) == {"train": 0, "validation": 0, "test": 0}  # fake files -> 0 rows


def test_missing_split_raises_not_silently_empty(tmp_path, monkeypatch):
    monkeypatch.setattr(L, "CACHE", tmp_path)
    _touch_parquet(tmp_path / "demo" / "train-0.parquet")
    with pytest.raises(FileNotFoundError):
        list(L._rows("demo", ["image"], split="test"))


def test_corrupt_parquet_raises_clear_error(tmp_path, monkeypatch):
    monkeypatch.setattr(L, "CACHE", tmp_path)
    _touch_parquet(tmp_path / "demo" / "train-0.parquet", b"this is not parquet")
    with pytest.raises(RuntimeError):
        list(L._rows("demo", ["image"], split="train"))


def test_stratified_split_is_disjoint_and_seeded():
    items = [(b"x", i % 4) for i in range(400)]
    a = L.stratified_split(items, seed=42)
    b = L.stratified_split(items, seed=42)
    ids = {k: [id(x) for x in v] for k, v in a.items()}
    assert all(len(v) > 0 for v in a.values())
    assert ids == {k: [id(x) for x in v] for k, v in b.items()}      # deterministic
    assert sum(len(v) for v in a.values()) == len(items)             # no row lost/duplicated
    assert a["train"] != a["test"]


# --------------------------------------------------------------------------- #
# detection math
# --------------------------------------------------------------------------- #
def test_box_iou_basic():
    a = torch.tensor([[0.0, 0.0, 10.0, 10.0]])
    assert float(DET.box_iou(a, a)[0, 0]) == pytest.approx(1.0)
    b = torch.tensor([[5.0, 5.0, 15.0, 15.0]])
    # intersection 5x5=25, union 100+100-25=175 -> 1/7
    assert float(DET.box_iou(a, b)[0, 0]) == pytest.approx(25.0 / 175.0, abs=1e-6)


def test_xyxy_xywh_roundtrip():
    b = torch.tensor([[1.0, 2.0, 11.0, 22.0]])
    assert torch.allclose(DET.xywh_to_xyxy(DET.xyxy_to_xywh(b)), b, atol=1e-6)


def test_encode_targets_places_box_in_expected_cell():
    # centre (0.5,0.5), size 0.2 -> grid 10 -> cell (5,5)
    box = torch.tensor([[0.4, 0.4, 0.6, 0.6]])
    tgt = DET.encode_targets([box], [torch.tensor([1])], grid=10, n_classes=2)
    assert tgt.shape == (1, 10, 10, 5 + 2)
    assert float(tgt[0, 5, 5, 0]) == 1.0
    assert int(tgt[0, 5, 5, 5:].argmax()) == 1
    assert float(tgt[0].sum()) > 0


def test_encode_targets_handles_empty_boxes():
    tgt = DET.encode_targets([torch.zeros((0, 4))], [torch.zeros((0,), dtype=torch.long)],
                             grid=7, n_classes=3)
    assert float(tgt.sum()) == 0.0


def test_yolo_loss_is_finite_and_has_gradients():
    model = DET.YoloLite(n_classes=2, grid=8)
    x = torch.rand(2, 3, 64, 64)
    box = torch.tensor([[0.1, 0.1, 0.5, 0.5]])
    tgt = DET.encode_targets([box, box], [torch.tensor([1]), torch.tensor([0])], grid=8, n_classes=2)
    loss = DET.yolo_loss(model(x), tgt)
    assert torch.isfinite(loss)
    loss.backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads)
    assert any(float(g.abs().sum()) > 0 for g in grads)


def test_yolo_loss_empty_target_is_finite():
    model = DET.YoloLite(n_classes=2, grid=4)
    tgt = torch.zeros(1, 4, 4, 7)
    loss = DET.yolo_loss(model(torch.rand(1, 3, 32, 32)), tgt)
    assert torch.isfinite(loss)


def test_decode_predictions_empty_when_low_confidence():
    pred = torch.full((7, 7, 5 + 2), -20.0)      # objectness sigmoid ~ 0
    assert DET.decode_predictions(pred, conf_thr=0.3) == []


def test_decode_predictions_returns_normalised_boxes():
    pred = torch.full((7, 7, 5 + 2), -20.0)
    pred[3, 3, 0] = 20.0                          # strong objectness
    pred[3, 3, 1:5] = torch.tensor([0.0, 0.0, 1.0, 1.0])   # centre-of-cell, full size
    pred[3, 3, 5] = 20.0                          # class 0
    out = DET.decode_predictions(pred, conf_thr=0.3)
    assert len(out) == 1
    b = out[0]
    assert 0.0 <= b["x1"] < b["x2"] <= 1.0 and 0.0 <= b["y1"] < b["y2"] <= 1.0
    assert b["class"] == 0


def test_average_precision_perfect_and_empty():
    gt = [torch.tensor([[0.0, 0.0, 0.5, 0.5]])]
    gl = [torch.tensor([0])]
    pb = [torch.tensor([[0.0, 0.0, 0.5, 0.5]])]
    ps = [torch.tensor([0.9])]
    pl = [torch.tensor([0])]
    assert DET.average_precision(pb, ps, pl, gt, gl, n_classes=1)["mAP@50"] == pytest.approx(1.0)
    # no predictions, but ground truth exists -> AP 0
    empty = DET.average_precision([torch.zeros((0, 4))], [torch.zeros((0,))],
                                  [torch.zeros((0,), dtype=torch.long)], gt, gl, n_classes=1)
    assert empty["mAP@50"] == 0.0
    # empty ground truth and no predictions -> that class is skipped, not a fake 0
    none = DET.average_precision([torch.zeros((0, 4))], [torch.zeros((0,))],
                                 [torch.zeros((0,), dtype=torch.long)],
                                 [torch.zeros((0, 4))], [torch.zeros((0,), dtype=torch.long)],
                                 n_classes=1)
    assert none["mAP@50"] == 0.0 and none["n_classes_scored"] == 0


def test_average_precision_wrong_class_scores_zero():
    gt = [torch.tensor([[0.0, 0.0, 0.5, 0.5]])]
    gl = [torch.tensor([0])]
    pb = [torch.tensor([[0.0, 0.0, 0.5, 0.5]])]
    ps = [torch.tensor([0.9])]
    pl = [torch.tensor([1])]                       # predicted the wrong class
    assert DET.average_precision(pb, ps, pl, gt, gl, n_classes=2)["mAP@50"] == pytest.approx(0.0)


def test_nms_suppresses_duplicates():
    boxes = torch.tensor([[0., 0., 10., 10.], [0.5, 0.5, 10.5, 10.5], [50., 50., 60., 60.]])
    scores = torch.tensor([0.9, 0.8, 0.7])
    assert len(DET.nms(boxes, scores, 0.5)) == 2


def test_objects_to_boxes_and_masks_labels():
    obj = {"bbox": [[10, 20, 30, 40]], "label": [1]}
    xyxy, labels = L._objects_to_boxes(obj, width=100, height=100)
    assert xyxy.tolist() == [[10.0, 20.0, 40.0, 60.0]]
    assert labels.tolist() == [1]
    xyxy, labels = L._objects_to_boxes({"bbox": []}, 10, 10)
    assert xyxy.shape == (0, 4) and labels.shape == (0,)


def test_polygons_to_mask():
    seg = [[0, 0, 10, 0, 10, 10, 0, 10]]
    m = L._polygons_to_mask(seg, 10, 10)
    assert m.shape == (10, 10) and m.max() == 1 and m.sum() > 0
    assert L._polygons_to_mask(None, 5, 5).max() == 0


# --------------------------------------------------------------------------- #
# segmentation math
# --------------------------------------------------------------------------- #
def test_dice_and_iou_perfect_prediction():
    logits = torch.full((1, 1, 8, 8), 20.0)          # sigmoid ~ 1
    target = torch.ones(1, 1, 8, 8)
    assert SEG.dice_coef(logits, target) == pytest.approx(1.0, abs=1e-4)
    assert SEG.iou_score(logits, target) == pytest.approx(1.0, abs=1e-4)


def test_dice_and_iou_empty_target_and_empty_prediction():
    logits = torch.full((1, 1, 8, 8), -20.0)         # sigmoid ~ 0
    target = torch.zeros(1, 1, 8, 8)
    assert SEG.dice_coef(logits, target) == pytest.approx(1.0, abs=1e-3)
    assert SEG.iou_score(logits, target) == pytest.approx(1.0, abs=1e-3)


def test_dice_loss_decreases_with_better_prediction():
    target = torch.zeros(1, 1, 8, 8); target[:, :, 2:6, 2:6] = 1
    good = torch.full((1, 1, 8, 8), -20.0); good[:, :, 2:6, 2:6] = 20.0
    bad = torch.full((1, 1, 8, 8), -20.0)
    assert float(SEG.dice_loss(good, target)) < float(SEG.dice_loss(bad, target))


def test_mask_array_binary_and_class_index():
    a = np.zeros((8, 8), np.uint8); a[2:6, 2:6] = 255
    buf = io.BytesIO(); Image.fromarray(a).save(buf, format="PNG")
    m = L.mask_array(buf.getvalue(), 8)
    assert m.shape == (8, 8) and m.max() == 1.0
    b = np.zeros((8, 8), np.uint8); b[1:3, 1:3] = 38     # class-index encoded
    buf2 = io.BytesIO(); Image.fromarray(b).save(buf2, format="PNG")
    assert L.mask_array(buf2.getvalue(), 8).max() == 1.0


# --------------------------------------------------------------------------- #
# ViT contract (the plant_vit / brain_vit root-cause fix)
# --------------------------------------------------------------------------- #
def test_vit_builds_with_pretrained_backbone_contract():
    from src.models.classifiers import build_vit
    m = build_vit(25, in_ch=3, pretrained=False, image_size=224)
    m.eval()
    out = m(torch.zeros(1, 3, 224, 224))
    assert out.shape == (1, 25)
    assert sum(p.numel() for p in m.parameters()) < 100_000_000


def test_vit_state_dict_roundtrip_strict():
    from src.models.classifiers import build_vit
    a = build_vit(4, in_ch=3, pretrained=False)
    b = build_vit(4, in_ch=3, pretrained=False)
    b.load_state_dict(a.state_dict(), strict=True)     # inference build must match train build


def test_registry_cv_vit_uses_timm_backbone():
    spec = BY_ID["plant_vit"]
    m = build_model(spec)()
    assert m(torch.zeros(1, 3, spec.input_size, spec.input_size)).shape == (1, len(spec.classes))


# --------------------------------------------------------------------------- #
# artifact lifecycle
# --------------------------------------------------------------------------- #
def test_checkpoint_save_load_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setattr(A, "ARTIFACTS", tmp_path)
    monkeypatch.setattr(A, "MEDICAL_DIR", tmp_path / "medical")
    monkeypatch.setattr(A, "META_DIR", tmp_path / "metadata")
    spec = BY_ID["brain_cnn"]
    model = build_model(spec)()
    meta = {"classes": spec.classes, "input": {"size": 224}, "trained": True, "task": "classification"}
    info = A.save_model(model, spec.id, spec.area, meta)
    assert info["weights_bytes"] > 0
    reloaded = A.load_model(spec.id, spec.area, build_model(spec))
    x = torch.zeros(1, 3, 224, 224)
    assert torch.allclose(model.eval()(x), reloaded(x), atol=1e-6)
    assert A.has_weights(spec.id, spec.area)


def test_load_model_missing_raises_artifact_missing(tmp_path, monkeypatch):
    monkeypatch.setattr(A, "ARTIFACTS", tmp_path)
    monkeypatch.setattr(A, "MEDICAL_DIR", tmp_path / "medical")
    monkeypatch.setattr(A, "META_DIR", tmp_path / "metadata")
    with pytest.raises(A.ArtifactMissing):
        A.load_model("brain_cnn", "medical", build_model(BY_ID["brain_cnn"]))


def test_load_model_corrupt_weights_raise_incompatible(tmp_path, monkeypatch):
    monkeypatch.setattr(A, "ARTIFACTS", tmp_path)
    monkeypatch.setattr(A, "MEDICAL_DIR", tmp_path / "medical")
    monkeypatch.setattr(A, "META_DIR", tmp_path / "metadata")
    p = A.weights_path("brain_cnn", "medical")
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(b"garbage-not-a-torch-file")
    with pytest.raises(A.ArtifactIncompatible):
        A.load_model("brain_cnn", "medical", build_model(BY_ID["brain_cnn"]))


def test_has_weights_enforces_mnist_gate(tmp_path, monkeypatch):
    monkeypatch.setattr(A, "ARTIFACTS", tmp_path)
    monkeypatch.setattr(A, "META_DIR", tmp_path / "metadata")
    monkeypatch.setattr(A, "AGRI_DIR", tmp_path / "agriculture")
    spec = BY_ID["lab02"]
    (tmp_path / "lab").mkdir(parents=True, exist_ok=True)
    (tmp_path / "metadata").mkdir(parents=True, exist_ok=True)
    (tmp_path / "lab" / "lab02.pt").write_bytes(b"x")
    (tmp_path / "metadata" / "lab02.json").write_text(json.dumps(
        {"model_id": "lab02", "trained": True, "dataset_key": "mnist",
         "task": "classification", "metrics": {"accuracy": 0.5}}))
    assert not A.has_weights("lab02", "lab")            # below the 0.90 gate
    (tmp_path / "metadata" / "lab02.json").write_text(json.dumps(
        {"model_id": "lab02", "trained": True, "dataset_key": "mnist",
         "task": "classification", "metrics": {"accuracy": 0.95}}))
    assert A.has_weights("lab02", "lab")


# --------------------------------------------------------------------------- #
# preprocessing parity
# --------------------------------------------------------------------------- #
def test_to_tensor_matches_make_transform():
    from src.imaging import IMAGENET_MEAN, IMAGENET_STD, make_transform, to_tensor
    img = Image.fromarray((np.random.rand(40, 30, 3) * 255).astype("uint8"))
    a = to_tensor(img, 32, IMAGENET_MEAN, IMAGENET_STD)
    b = make_transform(32, IMAGENET_MEAN, IMAGENET_STD, train=False)(img).unsqueeze(0)
    assert a.shape == b.shape == (1, 3, 32, 32)
    assert torch.allclose(a, b, atol=1e-5)


def test_load_image_rejects_unsupported_bytes():
    with pytest.raises(Exception):
        from src.imaging import load_image
        load_image(b"definitely not an image")


# --------------------------------------------------------------------------- #
# training / verification status contract
# --------------------------------------------------------------------------- #
def test_train_models_honest_exit_codes():
    """--defer must skip a model *and* record it; unknown ids must be rejected."""
    r = subprocess.run([sys.executable, "scripts/train_models.py", "--model", "brain_maskrcnn",
                        "--defer", "brain_maskrcnn"],
                       capture_output=True, text=True, cwd=str(ROOT), timeout=300)
    assert r.returncode == 0, r.stderr[-500:]
    assert "DEFERRED" in r.stdout
    bad = subprocess.run([sys.executable, "scripts/train_models.py", "--model", "brain_cnn",
                          "--defer", "not_a_model"],
                         capture_output=True, text=True, cwd=str(ROOT), timeout=120)
    assert bad.returncode != 0 and "unknown model ids" in (bad.stderr + bad.stdout)


def test_verify_models_require_none_never_fails_on_missing():
    r = subprocess.run([sys.executable, "scripts/verify_models.py", "--require", "none"],
                       capture_output=True, text=True, cwd=str(ROOT), timeout=600)
    assert r.returncode == 0, r.stdout[-800:] + r.stderr[-400:]
    assert "VERIFY PASSED" in r.stdout


def test_status_report_written():
    p = ROOT / "artifacts" / "status" / "verify.json"
    assert p.exists(), "verify_models.py must write artifacts/status/verify.json"
    data = json.loads(p.read_text())
    assert {"counts", "models", "required"} <= set(data)


def test_cv_registry_still_has_twelve_models():
    assert len(CV_MODELS) == 12
    assert {m.id for m in CV_MODELS} == {
        "brain_cnn", "brain_densenet", "brain_vit", "brain_unet", "brain_yolo", "brain_maskrcnn",
        "plant_cnn", "plant_densenet", "plant_vit", "plant_unet", "plant_yolo", "plant_maskrcnn"}
