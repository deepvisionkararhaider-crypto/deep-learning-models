#!/usr/bin/env python3
"""Report the REAL status of every model by loading each checkpoint and running inference.

    python scripts/verify_models.py                 # CV models are required
    python scripts/verify_models.py --require all    # labs required too
    python scripts/verify_models.py --require none   # report only, never fail on missing

States
------
READY          weights loaded in this process and an inference smoke test passed
INFER-FAILED   weights exist but load/inference raised (a real defect)
BELOW-GATE     weights load, but a recorded quality gate is not met (e.g. MNIST < 0.90)
MISSING        architecture builds, no trained weights on disk
BLOCKED        no usable dataset exists — cannot be trained
DEFERRED       the training run explicitly deferred it (recorded in status/summary.json)

Exit code
---------
0 only when every *required* model is READY and no model anywhere is INFER-FAILED.
"Required" is controlled by --require:
  cv    (default) the 12 medical/agriculture models, minus BLOCKED/DEFERRED ones
  all   every registered model, minus BLOCKED/DEFERRED ones
  none  nothing is required (report only)

Writes ``artifacts/status/verify.json`` with the full machine-readable result.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import artifacts as A           # noqa: E402
from src import inference as I           # noqa: E402
from src import lab_inference as LI      # noqa: E402
from src.registry import ALL_MODELS, CV_MODELS, build_model  # noqa: E402

STATUS_DIR = ROOT / "artifacts" / "status"

READY, INFER_FAILED, BELOW_GATE, MISSING, BLOCKED, DEFERRED = (
    "READY", "INFER-FAILED", "BELOW-GATE", "MISSING", "BLOCKED", "DEFERRED")


def _probe_image(size: int) -> Image.Image:
    return Image.fromarray((np.random.rand(size, size, 3) * 255).astype("uint8"))


def _deferred_from_summary() -> set[str]:
    p = STATUS_DIR / "summary.json"
    if not p.exists():
        return set()
    try:
        data = json.loads(p.read_text())
    except Exception:                       # noqa: BLE001
        return set()
    return {m["model_id"] for m in data.get("models", []) if m.get("status") == DEFERRED}


def verify_cv(spec) -> tuple[str, str]:
    if spec.is_blocked:
        return BLOCKED, spec.blocked[:200]
    if not A.has_weights(spec.id, spec.area):
        wpath = A.weights_path(spec.id, spec.area)
        if wpath.is_file():
            return BELOW_GATE, "weights exist but the recorded quality gate is not met"
        build_model(spec)()               # architecture must at least construct
        return MISSING, "no trained weights on disk"
    model, meta = I.load(spec)
    size = int((meta.get("input") or {}).get("size", spec.input_size))
    img = _probe_image(size)
    if spec.task == "classification":
        r = I.predict_classification(spec, img)
        assert 0 <= r["confidence"] <= 1 and r["predicted"] is not None
        assert len(r["probabilities"]) == len(spec.classes)
    elif spec.task == "segmentation":
        r = I.predict_segmentation(spec, img)
        assert r["mask"].shape == (size, size)
    elif spec.task == "detection":
        r = I.predict_detection(spec, img)
        assert "boxes" in r
    elif spec.task == "instance_segmentation":
        r = I.predict_instance_segmentation(spec, img)
        assert "instances" in r
    return READY, ""


def verify_lab(spec) -> tuple[str, str]:
    if not A.has_weights(spec.id, spec.area):
        wpath = A.weights_path(spec.id, spec.area)
        if wpath.is_file():
            meta = A.read_metadata(spec.id) or {}
            acc = (meta.get("metrics") or {}).get("accuracy")
            return BELOW_GATE, f"weights present but accuracy={acc} < serving gate 0.90"
        build_model(spec)()
        return MISSING, "no trained weights on disk"
    kind = LI.INPUT_KIND[spec.id]
    if kind == "tabular":
        LI.predict_tabular(spec, [0.0] * 30)
    elif kind == "tabular_denoise":
        LI.denoise_tabular(spec, [0.0] * 30)
    elif kind == "graph":
        LI.gnn_predict(spec, [0.0] * 1433)
    elif kind == "image":
        img = _probe_image(28)
        if spec.id in ("lab06", "lab07"):
            LI.reconstruct(spec, img)
        elif spec.id == "lab08":
            LI.gan_score(spec, img)
        elif spec.id == "lab15":
            LI.segmentation_28(spec, img)
        elif spec.id == "lab16":
            LI.detection_28(spec, img)
        else:
            LI.predict_image_class(spec, img)
    elif kind == "image_pair":
        LI.siamese_pair(spec, _probe_image(28), _probe_image(28))
    elif kind == "sequence":
        LI.predict_sequence(spec, [0.0] * (128 * 9))
    elif kind == "text":
        LI.text_classify(spec, "the team launched a rocket")
    elif kind == "text_prompt":
        LI.generate_text(spec, "the team launched")
    return READY, ""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--require", choices=["cv", "all", "none"], default="cv")
    args = ap.parse_args()

    deferred = _deferred_from_summary()
    rows, counts = [], {}
    for spec in ALL_MODELS:
        try:
            state, note = verify_lab(spec) if spec.area == "lab" else verify_cv(spec)
        except Exception as exc:                                  # noqa: BLE001
            state, note = INFER_FAILED, f"{type(exc).__name__}: {exc}"
            traceback.print_exc()
        if spec.id in deferred and state in (MISSING, BELOW_GATE):
            state, note = DEFERRED, "explicitly deferred in the training run"
        counts[state] = counts.get(state, 0) + 1
        rows.append({"model_id": spec.id, "name": spec.name, "area": spec.area,
                     "task": spec.task, "state": state, "note": note})

    print(f"\n{'model':18s} {'area':12s} {'task':22s} status")
    print("-" * 92)
    for r in rows:
        print(f"{r['model_id']:18s} {r['area']:12s} {r['task']:22s} {r['state']:12s} {r['note'][:34]}")
    print("-" * 92)
    print(f"READY={counts.get(READY,0)}  INFER-FAILED={counts.get(INFER_FAILED,0)}  "
          f"BELOW-GATE={counts.get(BELOW_GATE,0)}  MISSING={counts.get(MISSING,0)}  "
          f"BLOCKED={counts.get(BLOCKED,0)}  DEFERRED={counts.get(DEFERRED,0)}")

    if args.require == "none":
        required = set()
    elif args.require == "all":
        required = {s.id for s in ALL_MODELS}
    else:
        required = {s.id for s in CV_MODELS}
    required -= {r["model_id"] for r in rows if r["state"] in (BLOCKED, DEFERRED)}
    not_ready = sorted(r["model_id"] for r in rows
                       if r["model_id"] in required and r["state"] != READY)
    infer_failed = sorted(r["model_id"] for r in rows if r["state"] == INFER_FAILED)

    report = {"generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
              "require": args.require, "counts": counts,
              "required": sorted(required), "not_ready": not_ready,
              "infer_failed": infer_failed, "models": rows}
    STATUS_DIR.mkdir(parents=True, exist_ok=True)
    (STATUS_DIR / "verify.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    ok = not infer_failed and not not_ready
    if infer_failed:
        print(f"\nINFERENCE FAILURES ({len(infer_failed)}): {', '.join(infer_failed)}")
    if not_ready:
        print(f"\nREQUIRED MODELS NOT READY ({len(not_ready)}): {', '.join(not_ready)}")
    print(f"\nVERIFY {'PASSED' if ok else 'FAILED'} (require={args.require})")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
