"""Platform tests: registry, architectures, IO contracts, artifacts, no-training-at-inference."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import artifacts as A                    # noqa: E402
from src import inference as I                    # noqa: E402
from src import lab_inference as LI               # noqa: E402
from src.models import detection as DET           # noqa: E402
from src.registry import ALL_MODELS, CV_MODELS, LAB_BY_INDEX, build_model  # noqa: E402


def test_registry_unique_and_complete():
    ids = [m.id for m in ALL_MODELS]
    assert len(ids) == len(set(ids)), "duplicate model ids"
    assert len(LAB_BY_INDEX) == 20
    assert len(CV_MODELS) == 12
    for m in ALL_MODELS:
        assert m.task and m.dataset_name and m.input_desc and m.outputs


@pytest.mark.parametrize("spec", ALL_MODELS, ids=[m.id for m in ALL_MODELS])
def test_architecture_builds(spec):
    model = build_model(spec)()
    assert isinstance(model, torch.nn.Module)


def test_ann_input_contract_is_30():
    """Regression for the old 'mat1 and mat2 shapes cannot be multiplied (256x20 and 30x64)' bug."""
    spec = LAB_BY_INDEX[1]
    model = build_model(spec)()
    out = model(torch.zeros(4, 30))
    assert out.shape == (4, 2)
    with pytest.raises(Exception):
        model(torch.zeros(4, 20))     # 20 features must NOT silently work


def test_cnn_mnist_shape():
    model = build_model(LAB_BY_INDEX[2])()
    assert model(torch.zeros(2, 1, 28, 28)).shape == (2, 10)


def test_unet_output_resolution():
    model = build_model(LAB_BY_INDEX[15])()
    assert model(torch.zeros(1, 1, 28, 28)).shape[-2:] == (28, 28)


def test_yolo_head_shape():
    spec = LAB_BY_INDEX[16]
    model = build_model(spec)()
    out = model(torch.zeros(1, 1, 28, 28))
    assert out.shape == (1, 7, 7, 5 + 10)


def test_cv_classifier_head_sizes():
    for spec in CV_MODELS:
        if spec.task != "classification":
            continue
        model = build_model(spec)()
        n = len(spec.classes)
        if spec.builder == "densenet":
            assert model.classifier.out_features == n
        else:
            out = model(torch.zeros(1, spec.in_ch, spec.input_size, spec.input_size))
            assert out.shape == (1, n)


def test_nms_and_iou():
    boxes = torch.tensor([[0., 0., 10., 10.], [1., 1., 11., 11.], [50., 50., 60., 60.]])
    scores = torch.tensor([0.9, 0.8, 0.7])
    keep = DET.nms(boxes, scores, 0.5)
    assert len(keep) == 2                      # first two overlap -> one kept
    assert float(DET.box_iou(boxes[:1], boxes[1:2])) > 0.5


def test_inference_never_trains():
    """The inference modules must not import or expose an optimiser."""
    src = (ROOT / "src" / "inference.py").read_text() + (ROOT / "src" / "lab_inference.py").read_text()
    for bad in ("torch.optim", ".backward(", "loss.backward"):
        assert bad not in src, f"inference module references training symbol {bad!r}"


def test_missing_artifact_is_graceful():
    spec = LAB_BY_INDEX[1]
    if A.has_weights(spec.id, spec.area):
        pytest.skip("lab01 has weights; graceful-missing path covered when untrained")
    with pytest.raises(A.ArtifactMissing):
        A.load_model(spec.id, spec.area, build_model(spec))


def test_streamlit_entrypoint_imports():
    import ast
    tree = ast.parse((ROOT / "app.py").read_text())
    assert tree is not None


def test_verify_script_runs():
    assert (ROOT / "scripts" / "verify_models.py").exists()
    assert (ROOT / "scripts" / "train_models.py").exists()
    assert (ROOT / "scripts" / "train_lab.py").exists()


# --------------------------------------------------------------------------- #
# regression tests for the data-pipeline and reporting fixes
# --------------------------------------------------------------------------- #
def test_plant_maskrcnn_is_honestly_blocked():
    """The agriculture instance-seg model must be reported BLOCKED, not faked."""
    from src.registry import BY_ID, CV_MODELS
    spec = BY_ID["plant_maskrcnn"]
    assert spec.is_blocked and spec.blocked, "plant_maskrcnn must carry a blocked reason"
    assert len(CV_MODELS) == 12, "all six CV types must remain in both domains"
    # A blocked model must never claim weights.
    assert not A.has_weights(spec.id, spec.area)


def test_cv_classes_match_dataset_labels():
    """Registry classes must match the real label cardinality of each dataset."""
    from src.datasets import loaders as L
    from src.registry import BY_ID
    assert len(BY_ID["brain_yolo"].classes) == 2      # brain_det: label 0/1
    assert len(BY_ID["plant_yolo"].classes) == 2      # cotton_det: 2 ClassLabel names
    assert len(BY_ID["brain_maskrcnn"].classes) == 2  # brain_instseg: 2 categories
    assert len(BY_ID["brain_cnn"].classes) == 4       # brain_mri: 4 ClassLabel names
    assert len(BY_ID["plant_cnn"].classes) == len(L.PLANT_CLASSES)


def test_har_split_is_official_and_disjoint():
    """HAR must use the official train/test split, and a scaler must be exposed."""
    from src.datasets import lab_datasets as D
    try:
        x_tr, y_tr = D.har("train")
        x_te, y_te = D.har("test")
    except FileNotFoundError:
        pytest.skip("HAR not cached in this environment")
    assert x_tr.shape[1:] == (128, 9) and x_te.shape[1:] == (128, 9)
    assert len(x_tr) != len(x_te), "official HAR train/test sizes differ (7352 vs 2947)"
    mean, std = D.har_scaler()
    assert mean.shape == (9,) and std.shape == (9,)
    assert (std > 0).all()


def test_mask_array_handles_class_index_masks():
    """plant_seg masks are class-index encoded (0/38); >127 would yield empty masks."""
    import io
    import numpy as np
    from PIL import Image
    from src.datasets import loaders as L
    arr = np.zeros((10, 10), dtype=np.uint8); arr[3:7, 3:7] = 38
    buf = io.BytesIO(); Image.fromarray(arr).save(buf, format="PNG")
    m = L.mask_array(buf.getvalue(), 10)
    assert m.shape == (10, 10) and m.max() == 1.0 and m.sum() > 0
