#!/usr/bin/env python3
"""Report the REAL status of every model: does its artifact load and run inference?

    python scripts/verify_models.py

States
------
* READY          — weights loaded and an inference smoke test passed
* ARCH-ONLY      — architecture builds, but no trained weights on disk
* INFER-FAILED   — weights present but load/inference raised

Exit code is non-zero if any model with weights fails inference.
"""
from __future__ import annotations

import sys
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
from src.registry import ALL_MODELS, build_model  # noqa: E402


def _probe_image(size: int) -> Image.Image:
    return Image.fromarray((np.random.rand(size, size, 3) * 255).astype("uint8"))


def verify_cv(spec) -> str:
    if spec.is_blocked:
        # No dataset exists, so there is nothing to train or verify. Report it as
        # BLOCKED rather than ARCH-ONLY so it is never mistaken for "just untrained".
        return "BLOCKED"
    if not A.has_weights(spec.id, spec.area):
        build_model(spec)()          # architecture must at least construct
        return "ARCH-ONLY"
    model, meta = I.load(spec)
    size = int((meta.get("input") or {}).get("size", spec.input_size))
    img = _probe_image(size)
    if spec.task == "classification":
        r = I.predict_classification(spec, img)
        assert 0 <= r["confidence"] <= 1 and r["predicted"] is not None
    elif spec.task == "segmentation":
        r = I.predict_segmentation(spec, img)
        assert r["mask"].shape == (size, size)
    elif spec.task == "detection":
        I.predict_detection(spec, img)
    elif spec.task == "instance_segmentation":
        I.predict_instance_segmentation(spec, img)
    return "READY"


def verify_lab(spec) -> str:
    if not A.has_weights(spec.id, spec.area):
        build_model(spec)()
        return "ARCH-ONLY"
    kind = LI.INPUT_KIND[spec.id]
    if kind == "tabular":
        LI.predict_tabular(spec, [0.0] * 30)
    elif kind == "tabular_denoise":
        LI.denoise_tabular(spec, [0.0] * 30)
    elif kind == "graph":
        LI.gnn_predict(spec, [0.0] * 1433)
    elif kind in ("image",):
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
    return "READY"


def main() -> int:
    counts = {"READY": 0, "ARCH-ONLY": 0, "INFER-FAILED": 0, "BLOCKED": 0}
    rows = []
    for spec in ALL_MODELS:
        try:
            state = verify_lab(spec) if spec.area == "lab" else verify_cv(spec)
        except Exception as exc:  # noqa: BLE001
            state = "INFER-FAILED"
            traceback.print_exc()
            rows.append((spec.id, spec.name, state, f"{type(exc).__name__}: {exc}"))
            counts[state] += 1
            continue
        counts[state] += 1
        rows.append((spec.id, spec.name, state, ""))

    print(f"\n{'model':18s} {'name':34s} status")
    print("-" * 78)
    for mid, name, state, err in rows:
        print(f"{mid:18s} {name[:34]:34s} {state} {err[:40]}")
    print("-" * 78)
    print(f"READY={counts['READY']}  ARCH-ONLY={counts['ARCH-ONLY']}  "
          f"BLOCKED={counts['BLOCKED']}  INFER-FAILED={counts['INFER-FAILED']}")
    return 1 if counts["INFER-FAILED"] or counts["ARCH-ONLY"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
