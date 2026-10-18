"""Hugging-Face-backed model loader with architecture-specific adapters.

The app is **inference-only** and **Hub-first**:

* If a model's weights exist locally (a training checkout), they are used.
* Otherwise the artifact is downloaded from ``HF_MODEL_REPO`` at the *pinned
  revision* recorded in the manifest, checksum-verified, and loaded.

Nothing here trains. Nothing here downloads at import time — every download is
triggered by an explicit user selection, so a Streamlit cold start stays fast and
memory-light (see :func:`load_model` and the ``st.cache_resource`` wrapper in
:mod:`src.services.model_loader`).

Loading is deliberately **not** one generic ``load_state_dict``: the empty
architecture comes from :func:`src.registry.build_model`, which knows the exact
constructor for every architecture (CNN / DenseNet / timm-ViT / U-Net / YOLO-style
/ Mask R-CNN / each of the 20 lab models). That is the adapter.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch

from . import artifacts as A
from . import hf_hub, manifest as M
from .registry import ModelSpec, build_model


class NotPublished(RuntimeError):
    """The model has no usable checkpoint locally or on the Hub."""


def entry_for(spec: ModelSpec) -> dict[str, Any]:
    """Manifest entry for a spec (falls back to a conservative local build)."""
    return M.entries_by_id().get(spec.id) or M.fallback_entry(spec)


def is_ready(spec: ModelSpec) -> bool:
    """Ready == a verified checkpoint is reachable.

    Local weights count only when the integrity gate in :func:`src.artifacts.has_weights`
    passes; a remote entry counts only when it is marked ``TRAINED_VERIFIED`` and
    carries a weights sha256 (so we can verify what we download).
    """
    if spec.is_blocked:
        return False
    if A.has_weights(spec.id, spec.area):
        return True
    e = entry_for(spec)
    return (e.get("status") == "TRAINED_VERIFIED"
            and bool((e.get("checkpoint") or {}).get("hf_path") or e.get("weights_sha256")))


def readiness_reason(spec: ModelSpec) -> str:
    if spec.is_blocked:
        return spec.blocked
    if A.has_weights(spec.id, spec.area):
        return "weights present in this checkout"
    e = entry_for(spec)
    if e.get("status") == "TRAINED_VERIFIED":
        return "published and verified on the Hugging Face Hub"
    return e.get("note") or e.get("limitations") or "no trained checkpoint available"


def resolve_checkpoint(spec: ModelSpec, *, allow_download: bool = True) -> tuple[Path, dict[str, Any]]:
    """Return ``(local_weights_path, config)`` for a model, downloading if needed.

    Raises :class:`NotPublished` when neither a local nor a remote checkpoint is
    available, so the UI can render an honest empty state.
    """
    # 1) local artifact (a training/CI checkout, or a previously cached download)
    local_w = A.weights_path(spec.id, spec.area)
    if local_w.is_file() and local_w.stat().st_size > 0:
        cfg = A.read_metadata(spec.id) or {}
        return local_w, cfg

    if not allow_download:
        raise NotPublished(f"'{spec.id}' has no local weights and downloads are disabled")

    if not hf_hub.is_configured():
        raise NotPublished(
            f"'{spec.id}' has no local weights and {hf_hub.ENV_REPO} is not configured."
        )

    e = entry_for(spec)
    ck = e.get("checkpoint") or {}
    revision = ck.get("hf_revision") or e.get("hf_revision")
    expect = e.get("weights_sha256")
    try:
        files = hf_hub.download_artifact(spec.id, spec.area, revision=revision,
                                         expect_sha256=expect)
    except hf_hub.HubError as exc:
        raise NotPublished(f"'{spec.id}' could not be downloaded: {exc}") from exc

    cfg_path = Path(files["config"])
    cfg = json.loads(cfg_path.read_text(encoding="utf-8")) if cfg_path.is_file() else {}
    return Path(files["weights"]), cfg


def load_model(spec: ModelSpec, *, allow_download: bool = True) -> tuple[torch.nn.Module, dict[str, Any]]:
    """Build the architecture, load the checkpoint, and return ``(model, config)``.

    ``model.eval()`` is always applied; callers wrap the forward pass in
    ``torch.inference_mode()``.
    """
    weights, cfg = resolve_checkpoint(spec, allow_download=allow_download)
    model = build_model(spec)()
    try:
        state = _load_state(weights)
        model.load_state_dict(state, strict=True)
    except Exception as exc:                                        # noqa: BLE001
        raise A.ArtifactIncompatible(
            f"Checkpoint for '{spec.id}' is incompatible with the current architecture "
            f"({type(exc).__name__}: {exc})."
        ) from exc
    model.eval()
    return model, cfg


def _load_state(path: Path) -> dict[str, Any]:
    """Load a state dict from safetensors when available, else a torch pickle."""
    if path.suffix == ".safetensors":
        from safetensors.torch import load_file
        return load_file(str(path))
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except Exception:                                               # noqa: BLE001
        # Older checkpoints may store a wrapper dict; unwrap conservatively.
        obj = torch.load(path, map_location="cpu", weights_only=False)
        if isinstance(obj, dict) and "state_dict" in obj:
            return obj["state_dict"]
        raise


def input_contract(spec: ModelSpec, cfg: dict[str, Any] | None = None) -> dict[str, Any]:
    """Preprocessing contract actually used for inference.

    Config (published with the weights) wins; the registry is the fallback, so a
    model whose artifact predates ``config.json`` still preprocesses correctly.
    """
    cfg = cfg or {}
    inp = cfg.get("input") or {}
    e = entry_for(spec)
    einp = e.get("input") or {}
    return {
        "modality": inp.get("modality") or einp.get("modality") or M.modality_for(spec),
        "size": int(inp.get("size") or einp.get("size") or spec.input_size or 224),
        "channels": int(inp.get("channels") or einp.get("channels") or spec.in_ch or 3),
        "mean": tuple(inp.get("mean") or einp.get("mean") or (0.485, 0.456, 0.406)),
        "std": tuple(inp.get("std") or einp.get("std") or (0.229, 0.224, 0.225)),
    }


def classes_for(spec: ModelSpec, cfg: dict[str, Any] | None = None) -> list[str]:
    cfg = cfg or {}
    return list(cfg.get("classes") or entry_for(spec).get("classes") or spec.classes)
