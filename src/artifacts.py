"""Artifact (trained-weight + metadata) persistence with graceful degradation.

Every trained model is written as::

    artifacts/<area>/<model_id>.pt          # torch state_dict (CPU)
    artifacts/metadata/<model_id>.json      # contract + metrics + provenance

The runtime never trains. It calls :func:`load_model` which either returns a
loaded, eval-mode model or raises :class:`ArtifactMissing` so the UI can show an
honest "training required" state instead of pretending a random model is real.
"""
from __future__ import annotations

import json
import platform
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

REPO = Path(__file__).resolve().parents[1]
ARTIFACTS = REPO / "artifacts"
MEDICAL_DIR = ARTIFACTS / "medical"
AGRI_DIR = ARTIFACTS / "agriculture"
META_DIR = ARTIFACTS / "metadata"

ARTIFACT_VERSION = "2.0.0"

# Machine-readable per-model status values written by scripts/train_models.py and
# read by scripts/verify_models.py + the Streamlit UI.
STATUS_VALUES = ("TRAINED_VERIFIED", "TRAINED_UNVERIFIED", "ARCH_ONLY",
                 "BLOCKED", "DEFERRED", "FAILED")
STATUS_DIR = ARTIFACTS / "status"


def _rel(p: Path) -> str:
    """Repo-relative path when possible, else the absolute path.

    ``Path.relative_to`` raises when the artifacts directory lives outside the
    repo (e.g. an alternate cache/volume), so callers must not assume it succeeds.
    """
    try:
        return str(p.relative_to(REPO))
    except ValueError:
        return str(p)


def read_status(model_id: str) -> dict[str, Any] | None:
    """The structured training record for one model, if it was written."""
    p = STATUS_DIR / f"{model_id}.json"
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:                                  # noqa: BLE001
        return None


def read_run_summary() -> dict[str, Any] | None:
    """The summary.json of the most recent training run (or None)."""
    p = STATUS_DIR / "summary.json"
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:                                  # noqa: BLE001
        return None


class ArtifactMissing(FileNotFoundError):
    """Raised when a model has no trained weights on disk."""


class ArtifactIncompatible(RuntimeError):
    """Raised when stored weights do not match the architecture contract."""


def _dir_for(area: str) -> Path:
    return {"medical": MEDICAL_DIR, "agriculture": AGRI_DIR}.get(area, ARTIFACTS / area)


def weights_path(model_id: str, area: str) -> Path:
    return _dir_for(area) / f"{model_id}.pt"


def metadata_path(model_id: str) -> Path:
    return META_DIR / f"{model_id}.json"


def save_model(model: torch.nn.Module, model_id: str, area: str, metadata: dict[str, Any]) -> dict[str, Any]:
    """Persist weights + metadata. Returns the metadata actually written."""
    wpath = weights_path(model_id, area)
    wpath.parent.mkdir(parents=True, exist_ok=True)
    META_DIR.mkdir(parents=True, exist_ok=True)

    model = model.cpu().eval()
    torch.save(model.state_dict(), wpath)

    meta = dict(metadata)
    meta.setdefault("artifact_version", ARTIFACT_VERSION)
    meta["model_id"] = model_id
    meta["area"] = area
    meta["weights_file"] = _rel(wpath)
    meta["weights_bytes"] = wpath.stat().st_size
    meta["saved_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    meta["python"] = sys.version.split()[0]
    meta["torch"] = torch.__version__
    meta["platform"] = platform.platform()
    try:
        import subprocess
        meta["git_commit"] = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        meta.setdefault("git_commit", None)
    metadata_path(model_id).write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return meta


def has_weights(model_id: str, area: str) -> bool:
    """Return True only when an artifact exists and passes basic integrity gates.

    A file on disk is not proof that a classifier is useful. In particular,
    the MNIST lab classifiers are not exposed in the prediction selector when
    their recorded held-out accuracy is below 90%; retrain them before serving.
    Other tasks still require both weights and parseable metadata marked trained.
    """
    wpath = weights_path(model_id, area)
    if not wpath.is_file() or wpath.stat().st_size == 0:
        return False
    meta = read_metadata(model_id)
    if not meta or meta.get("model_id", model_id) != model_id:
        return False
    if meta.get("trained") is False:
        return False
    if area == "lab" and meta.get("dataset_key") == "mnist" and meta.get("task") == "classification":
        accuracy = (meta.get("metrics") or {}).get("accuracy")
        if not isinstance(accuracy, (int, float)) or accuracy < 0.90:
            return False
    return True


def read_metadata(model_id: str) -> dict[str, Any] | None:
    p = metadata_path(model_id)
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None


def load_model(model_id: str, area: str, build_fn) -> torch.nn.Module:
    """Build the architecture (no pretrained download) and load stored weights.

    ``build_fn`` must return an *empty* architecture whose ``state_dict`` keys
    match what was saved. This keeps inference offline: we never fetch pretrained
    backbones at runtime.
    """
    wpath = weights_path(model_id, area)
    if not wpath.exists():
        raise ArtifactMissing(
            f"No trained weights for '{model_id}' (expected {_rel(wpath)}). "
            f"Train it with: python scripts/train_models.py --model {model_id}"
        )
    model = build_fn()
    try:
        state = torch.load(wpath, map_location="cpu", weights_only=True)
        model.load_state_dict(state, strict=True)
    except Exception as exc:  # noqa: BLE001
        raise ArtifactIncompatible(
            f"Stored weights for '{model_id}' are incompatible with the current "
            f"architecture ({type(exc).__name__}: {exc}). Retrain the model."
        ) from exc
    model.eval()
    return model


@dataclass
class LoadedModel:
    model: torch.nn.Module
    metadata: dict[str, Any]

    @property
    def classes(self) -> list[str]:
        return list(self.metadata.get("classes") or [])

    @property
    def input_size(self) -> int:
        return int((self.metadata.get("input") or {}).get("size", 224))


def load(model_id: str, area: str, build_fn) -> LoadedModel:
    meta = read_metadata(model_id) or {}
    return LoadedModel(model=load_model(model_id, area, build_fn), metadata=meta)


def registry_summary() -> list[dict[str, Any]]:
    """Every metadata file on disk, for the dashboard / evaluation pages."""
    out = []
    if META_DIR.exists():
        for p in sorted(META_DIR.glob("*.json")):
            try:
                out.append(json.loads(p.read_text(encoding="utf-8")))
            except Exception:
                continue
    return out
