"""The machine-readable model manifest — the platform's single source of truth.

``models/manifest.json`` is generated (never hand-edited) from three inputs:

1. :mod:`src.registry` — the architecture/contract definitions (32 models:
   20 laboratory + 6 medical + 6 agriculture).
2. Local artifact metadata — what has actually been trained in this checkout.
3. The published Hub manifest — what is actually available remotely, with the
   pinned revision and per-file SHA-256.

Status is therefore *derived evidence*, not a claim:

===============  ============================================================
status           meaning
===============  ============================================================
TRAINED_VERIFIED weights exist, inference smoke test passed, and (when a Hub
                 artifact is configured) the remote copy downloaded and loaded
                 successfully with a matching checksum
TRAINED_UNVERIFIED weights exist and load, but fresh-download verification has
                 not been recorded
ARCH_ONLY        architecture implemented, no trained weights
BLOCKED          cannot be trained (no dataset / no schema); reason recorded
DEFERRED         intentionally postponed (e.g. too heavy for the current runner)
FAILED           the last training/verification attempt failed; see ``error``
===============  ============================================================

Nothing in this module invents metrics, URLs or availability.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from . import artifacts as A
from . import hf_hub
from .registry import ALL_MODELS, BY_ID, ModelSpec

REPO = Path(__file__).resolve().parents[1]
MANIFEST_PATH = REPO / "models" / "manifest.json"

#: Tasks whose input is an image (drives the dynamic form generator).
IMAGE_TASKS = {"classification", "segmentation", "detection", "instance_segmentation"}

MODALITY_BY_TASK = {
    "tabular_classification": "tabular",
    "tabular_denoising": "tabular",
    "classification": "image",
    "segmentation": "image",
    "detection": "image",
    "instance_segmentation": "image",
    "sequence_classification": "sequence",
    "text_classification": "text",
    "language_modeling": "text",
    "sequence_to_sequence": "text",
    "graph_classification": "graph",
    "reconstruction": "image",
    "binary_classification": "image",
    "metric_learning": "image_pair",
}


def modality_for(spec: ModelSpec) -> str:
    """Input modality, derived from the task (not guessed per model)."""
    if spec.task in MODALITY_BY_TASK:
        return MODALITY_BY_TASK[spec.task]
    # lab vision models are 28x28 grayscale images; everything else falls back
    if spec.in_ch:
        return "image"
    return "tabular"


def _status_from_local(spec: ModelSpec, record: dict[str, Any] | None) -> tuple[str, str]:
    """Local (checkout) status + a human reason. Never claims more than it can show."""
    if spec.is_blocked:
        return "BLOCKED", spec.blocked
    if record and record.get("status") in A.STATUS_VALUES:
        status = record["status"]
        reason = record.get("error") or record.get("note") or ""
        if status == "TRAINED_VERIFIED" and not A.has_weights(spec.id, spec.area):
            return "ARCH_ONLY", "status record claims verified but no weights are present locally"
        return status, reason
    if A.has_weights(spec.id, spec.area):
        return "TRAINED_UNVERIFIED", "weights present locally; fresh-download verification not recorded"
    return "ARCH_ONLY", "architecture implemented; no trained weights in this checkout"


def build_entry(spec: ModelSpec, remote: dict[str, Any] | None = None,
                *, verified: dict[str, Any] | None = None) -> dict[str, Any]:
    """One manifest entry: contract + honest status + artifact location."""
    meta = A.read_metadata(spec.id) or {}
    record = A.read_status(spec.id)
    status, reason = _status_from_local(spec, record)

    remote_entry = ((remote or {}).get("models") or {}).get(spec.id) or {}
    remote_status = remote_entry.get("status")
    weights_sha = remote_entry.get("weights_sha256")

    # A remote artifact that has been verified end-to-end upgrades an unverified
    # local status; it never invents metrics.
    if remote_entry and verified and verified.get("model_id") == spec.id:
        status = "TRAINED_VERIFIED"
        reason = "remote artifact downloaded from a pinned revision and passed an inference smoke test"
    elif status == "TRAINED_UNVERIFIED" and remote_status == "TRAINED_VERIFIED":
        status = "TRAINED_VERIFIED"
        reason = "remote artifact is published and marked verified"

    classes = meta.get("classes") or spec.classes
    inp = meta.get("input") or {}
    metrics = meta.get("metrics") or remote_entry.get("metrics") or {}

    return {
        "model_id": spec.id,
        "display_name": spec.name,
        "area": spec.area,
        "domain": {"medical": "medical", "agriculture": "agriculture", "lab": "laboratory"}[spec.area],
        "task": spec.task,
        "modality": modality_for(spec),
        "framework": spec.framework,
        "architecture": spec.builder,
        "builder": spec.builder,
        "build_kwargs": dict(spec.build_kwargs or {}),
        "input": {
            "modality": modality_for(spec),
            "size": int(inp.get("size", spec.input_size)),
            "channels": int(inp.get("channels", spec.in_ch)),
            "mean": list(inp.get("mean", (0.485, 0.456, 0.406))),
            "std": list(inp.get("std", (0.229, 0.224, 0.225))),
            "description": spec.input_desc,
        },
        "outputs": spec.outputs,
        "classes": list(classes),
        "n_classes": len(classes),
        "dataset": meta.get("dataset", spec.dataset_name),
        "dataset_key": meta.get("dataset_key", spec.dataset_key),
        "metrics": metrics,
        "metric_names": list(spec.metric_names),
        "test_set_size": meta.get("test_set_size"),
        "split_manifest": meta.get("split_manifest") or {},
        "training": meta.get("training") or {},
        "weights_bytes": meta.get("weights_bytes") or remote_entry.get("weights_bytes"),
        "weights_sha256": meta.get("weights_sha256") or weights_sha,
        "checkpoint": {
            "kind": "huggingface" if hf_hub.is_configured() else "local",
            "local_path": A._rel(A.weights_path(spec.id, spec.area)) if A.has_weights(spec.id, spec.area) else None,
            "hf_repo": (remote or {}).get("hf_repo") or (hf_hub.model_repo() if hf_hub.is_configured() else None),
            "hf_path": hf_hub.artifact_paths(spec.id, spec.area)["weights"] if hf_hub.is_configured() else None,
            "hf_revision": (remote or {}).get("hf_revision") or remote_entry.get("hf_revision"),
        },
        "status": status,
        "verification": (verified or {}).get("state") if verified else None,
        "limitations": meta.get("limitations") or spec.notes or spec.blocked,
        "note": reason,
        "is_cv": spec.is_cv,
        "lab": spec.lab,
    }


def build_manifest(*, remote: dict[str, Any] | None = None,
                   verified: dict[str, dict[str, Any]] | None = None) -> dict[str, Any]:
    """Build the complete manifest for every registry model."""
    verified = verified or {}
    entries = [build_entry(s, remote, verified=verified.get(s.id)) for s in ALL_MODELS]
    counts: dict[str, int] = {}
    for e in entries:
        counts[e["status"]] = counts.get(e["status"], 0) + 1

    repo = hf_hub.model_repo() if hf_hub.is_configured() else None
    revision = (remote or {}).get("hf_revision")
    return {
        "schema_version": 2,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "hf_repo": repo,
        "hf_revision": revision,
        "hub_manifest_present": bool(remote),
        "totals": {
            "models": len(entries),
            "laboratory": sum(1 for e in entries if e["area"] == "lab"),
            "medical": sum(1 for e in entries if e["area"] == "medical"),
            "agriculture": sum(1 for e in entries if e["area"] == "agriculture"),
        },
        "status_counts": counts,
        "ready_models": sorted(e["model_id"] for e in entries if e["status"] == "TRAINED_VERIFIED"),
        "models": entries,
    }


def write_manifest(manifest: dict[str, Any], path: str | Path = MANIFEST_PATH) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return p


def read_manifest(path: str | Path = MANIFEST_PATH) -> dict[str, Any] | None:
    p = Path(path)
    if not p.is_file():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:                                              # noqa: BLE001
        return None


def entries_by_id(manifest: dict[str, Any] | None = None) -> dict[str, dict[str, Any]]:
    m = manifest if manifest is not None else (read_manifest() or {})
    return {e["model_id"]: e for e in (m.get("models") or [])}


def ready_entries(manifest: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    return [e for e in (manifest or read_manifest() or {}).get("models", [])
            if e["status"] == "TRAINED_VERIFIED"]


def fallback_entry(spec: ModelSpec) -> dict[str, Any]:
    """Entry used when no manifest has been generated yet (first clone).

    Deliberately conservative: architecture-only, no metrics, no remote claim.
    """
    return build_entry(spec, None, verified=None)


def load_or_build() -> dict[str, Any]:
    """Manifest from disk, or built on the fly when the file is absent."""
    return read_manifest() or build_manifest()


__all__ = [
    "MANIFEST_PATH", "IMAGE_TASKS", "MODALITY_BY_TASK", "modality_for",
    "build_entry", "build_manifest", "write_manifest", "read_manifest",
    "entries_by_id", "ready_entries", "fallback_entry", "load_or_build",
]
