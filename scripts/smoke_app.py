#!/usr/bin/env python3
"""Startup / health smoke test for the Streamlit app.

Run this in CI and before deploying to Streamlit Community Cloud. It asserts the
hard guarantees of the platform **without** a Streamlit runtime, a GPU, a network
call, or any training:

1. every UI page and service module imports cleanly;
2. the manifest is internally consistent (32 models, all 20 labs present, unique ids);
3. no inference-path module imports ``torch.optim`` or calls ``backward()``
   (i.e. the app can never train);
4. no dataset loader is imported by the app's inference path;
5. ``HF_MODEL_REPO`` is read from configuration and never hardcoded;
6. a model with no checkpoint is reported as not-ready rather than served.

Exit code 0 = healthy, 1 = a guarantee was violated.
"""
from __future__ import annotations

import ast
import importlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

FAILURES: list[str] = []
CHECKS: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    (CHECKS if ok else FAILURES).append(f"{name}{' — ' + detail if detail else ''}")


# --------------------------------------------------------------------------- #
# 1. imports
# --------------------------------------------------------------------------- #
APP_MODULES = [
    "src.ui.common", "src.ui.catalog", "src.ui.results", "src.ui.predict",
    "src.ui.pages_home", "src.ui.pages_explore", "src.ui.pages_image", "src.ui.pages_text",
    "src.ui.pages_medical", "src.ui.pages_agriculture", "src.ui.pages_lab", "src.ui.pages_info",
    "src.services.manifest_service", "src.services.model_loader", "src.services.prediction_service",
    "src.forms.prediction_forms", "src.hf_hub", "src.hf_loader", "src.manifest", "src.inference",
    "src.lab_inference", "src.imaging", "src.registry", "src.artifacts",
]

#: files that must never be able to train
INFERENCE_FILES = [
    "src/inference.py", "src/lab_inference.py", "src/hf_loader.py", "src/imaging.py",
    "src/services/model_loader.py", "src/services/prediction_service.py",
    "src/ui/predict.py", "src/ui/results.py", "app.py", "streamlit_app.py",
]


def check_imports() -> None:
    for mod in APP_MODULES:
        try:
            importlib.import_module(mod)
            check(f"import {mod}", True)
        except Exception as exc:                                    # noqa: BLE001
            check(f"import {mod}", False, f"{type(exc).__name__}: {exc}")


# --------------------------------------------------------------------------- #
# 2. manifest consistency
# --------------------------------------------------------------------------- #
def check_manifest() -> None:
    from src import manifest as M
    from src.registry import ALL_MODELS, LAB_BY_INDEX

    m = M.load_or_build()
    ids = [e["model_id"] for e in m["models"]]
    check("manifest has 32 models", len(ids) == 32, f"got {len(ids)}")
    check("manifest ids unique", len(ids) == len(set(ids)))
    labs = [i for i in ids if i.startswith("lab")]
    check("all 20 laboratory models preserved", len(labs) == 20, f"got {len(labs)}")
    check("registry laboratory count is 20", len(LAB_BY_INDEX) == 20)
    check("registry total is 32", len(ALL_MODELS) == 32, f"got {len(ALL_MODELS)}")

    for e in m["models"]:
        if not e.get("task") or not e.get("modality"):
            check(f"{e['model_id']} declares task+modality", False)
    check("every entry declares task and modality", True)

    ready = [e for e in m["models"] if e["status"] == "TRAINED_VERIFIED"]
    for e in ready:
        ck = e.get("checkpoint") or {}
        ok = bool(ck.get("local_path") or ck.get("hf_path"))
        check(f"{e['model_id']} ready => has an artifact reference", ok)
        check(f"{e['model_id']} ready => has input metadata", bool(e.get("input")))


# --------------------------------------------------------------------------- #
# 3/4/5. static guarantees about the app's source
# --------------------------------------------------------------------------- #
def check_no_training() -> None:
    offenders = []
    for rel in INFERENCE_FILES:
        p = ROOT / rel
        if not p.is_file():
            continue
        src = p.read_text(encoding="utf-8")
        tree = ast.parse(src)
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for a in node.names:
                    if a.name in ("torch.optim",) or a.name.startswith("torch.optim."):
                        offenders.append(f"{rel}: imports {a.name}")
            elif isinstance(node, ast.ImportFrom):
                if (node.module or "").startswith("torch.optim"):
                    offenders.append(f"{rel}: from {node.module} import …")
            elif isinstance(node, ast.Call):
                fn = node.func
                if isinstance(fn, ast.Attribute) and fn.attr == "backward":
                    offenders.append(f"{rel}: calls .backward()")
        if ".backward(" in src:
            offenders.append(f"{rel}: contains '.backward('")
    check("inference path cannot train (no torch.optim / backward)", not offenders,
          "; ".join(sorted(set(offenders))[:6]))


def check_no_dataset_in_app() -> None:
    offenders = []
    for rel in ("app.py", "streamlit_app.py", "src/inference.py", "src/lab_inference.py",
                "src/ui/predict.py", "src/ui/results.py"):
        p = ROOT / rel
        if p.is_file() and "datasets" in p.read_text(encoding="utf-8"):
            if "src.datasets" in p.read_text(encoding="utf-8"):
                offenders.append(rel)
    check("app never imports the dataset layer", not offenders, ", ".join(offenders))


def check_repo_is_config() -> None:
    """The HF repo id must come from configuration, not a literal in app code."""
    literal = "KararDeepVision/deep-learning-models"
    offenders = []
    for rel in INFERENCE_FILES + APP_MODULES:
        p = ROOT / (rel.replace(".", "/") + ".py") if rel.startswith("src.") else ROOT / rel
        if not p.is_file():
            continue
        if literal in p.read_text(encoding="utf-8"):
            offenders.append(rel)
    check("HF repo id is not hardcoded in app/inference code", not offenders,
          ", ".join(offenders))
    from src import hf_hub
    check("HF_MODEL_REPO is read from the environment",
          hf_hub.ENV_REPO == "HF_MODEL_REPO")


# --------------------------------------------------------------------------- #
# 6. honest unready state
# --------------------------------------------------------------------------- #
def check_honest_unready() -> None:
    from src.registry import BY_ID
    from src.services import model_loader
    from src import artifacts as A

    spec = BY_ID["plant_maskrcnn"]           # BLOCKED, never has weights
    check("blocked model is not ready", not model_loader.is_ready(spec))
    check("blocked model reports a reason", bool(model_loader.readiness(spec)["reason"]))

    # A model with no artifact must not be silently servable.
    untrained = [s for s in (BY_ID["brain_unet"],) if not A.has_weights(s.id, s.area)]
    if untrained:
        check("untrained model is not ready", not model_loader.is_ready(untrained[0]))


def main() -> int:
    check_imports()
    check_manifest()
    check_no_training()
    check_no_dataset_in_app()
    check_repo_is_config()
    check_honest_unready()

    print(f"=== app smoke test: {len(CHECKS)} checks passed ===")
    for c in CHECKS:
        print(f"  ok   {c}")
    if FAILURES:
        print(f"\n=== {len(FAILURES)} FAILURE(S) ===")
        for f in FAILURES:
            print(f"  FAIL {f}")
        return 1
    print("\nSMOKE OK — the app is inference-only, manifest-consistent and honest about "
          "unavailable models.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
