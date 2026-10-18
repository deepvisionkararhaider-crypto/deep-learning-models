"""Resumable training state: optimizer / scheduler / epoch / step / RNG.

A long CPU training run can be interrupted (runner timeout, OOM, cancellation).
Without full training state the only options are "start over" or "pretend it
finished" — both unacceptable. This module persists everything needed to continue
*exactly* where the run stopped:

* model ``state_dict``
* optimizer ``state_dict`` (AdamW moments)
* scheduler ``state_dict`` (LR position)
* completed epoch index and global step
* best-so-far metric (so an early-stopping decision survives a resume)
* Python / NumPy / Torch RNG states

Writes are **atomic**: state goes to ``<name>.tmp`` and is then ``os.replace``-d
into place, so a crash mid-write can never leave a half-written checkpoint that a
later resume would trust. A resumed checkpoint is validated (model + optimizer +
epoch present, epoch < total) before it is used; anything malformed raises
:class:`InvalidCheckpoint` and the caller falls back to a fresh run rather than
silently training from garbage.

Training state lives under ``artifacts/checkpoints/<model_id>/`` and is gitignored
— it is a scratch working file, not a published artifact.
"""
from __future__ import annotations

import json
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
CHECKPOINT_DIR = REPO / "artifacts" / "checkpoints"

STATE_NAME = "state.pt"
INFO_NAME = "state.json"


class InvalidCheckpoint(RuntimeError):
    """The resume state exists but is unusable (corrupt, truncated, mismatched)."""


@dataclass
class ResumeState:
    """Everything needed to continue a run."""

    model_id: str
    start_epoch: int
    global_step: int
    best_metric: float | None
    history: list[float]
    loaded: bool
    reason: str = ""


def state_dir(model_id: str) -> Path:
    return CHECKPOINT_DIR / model_id


def state_path(model_id: str) -> Path:
    return state_dir(model_id) / STATE_NAME


def info_path(model_id: str) -> Path:
    return state_dir(model_id) / INFO_NAME


def _rng_state() -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }


def _set_rng_state(state: dict[str, Any]) -> None:
    try:
        random.setstate(state["python"])
        np.random.set_state(state["numpy"])
        torch.set_rng_state(state["torch"])
    except Exception:                                              # noqa: BLE001
        # RNG state is a best-effort reproducibility aid; never fail a resume on it.
        pass


def save_state(model_id: str, *, model: torch.nn.Module, optimizer: torch.optim.Optimizer,
               scheduler: Any = None, epoch: int, global_step: int,
               best_metric: float | None = None, history: list[float] | None = None,
               extra: dict[str, Any] | None = None) -> Path:
    """Atomically persist full training state after an epoch completes."""
    d = state_dir(model_id)
    d.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_id": model_id,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict() if optimizer is not None else None,
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "epoch": int(epoch),
        "global_step": int(global_step),
        "best_metric": best_metric,
        "history": list(history or []),
        "rng": _rng_state(),
        "saved_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "torch": torch.__version__,
        "extra": extra or {},
    }
    target = state_path(model_id)
    tmp = target.with_suffix(".tmp")
    torch.save(payload, tmp)
    os.replace(tmp, target)                    # atomic on POSIX

    info = {k: v for k, v in payload.items() if k not in ("model", "optimizer", "scheduler", "rng")}
    info["state_bytes"] = target.stat().st_size
    tmpj = info_path(model_id).with_suffix(".json.tmp")
    tmpj.write_text(json.dumps(info, indent=2, default=str), encoding="utf-8")
    os.replace(tmpj, info_path(model_id))
    return target


def load_state(model_id: str, *, model: torch.nn.Module,
               optimizer: torch.optim.Optimizer | None = None,
               scheduler: Any = None, total_epochs: int | None = None) -> ResumeState:
    """Restore training state into live objects.

    Raises :class:`InvalidCheckpoint` when the file is missing, truncated, belongs
    to a different architecture, or records an epoch >= ``total_epochs`` (already
    finished) — the caller then trains from scratch instead of trusting it.
    """
    p = state_path(model_id)
    if not p.is_file() or p.stat().st_size == 0:
        raise InvalidCheckpoint(f"no resume state at {p}")
    try:
        payload = torch.load(p, map_location="cpu", weights_only=False)
    except Exception as exc:                                        # noqa: BLE001
        raise InvalidCheckpoint(f"unreadable state file ({type(exc).__name__}: {exc})") from exc
    if not isinstance(payload, dict) or "model" not in payload or "epoch" not in payload:
        raise InvalidCheckpoint("state file is missing required keys (model/epoch)")

    try:
        model.load_state_dict(payload["model"], strict=True)
    except Exception as exc:                                        # noqa: BLE001
        raise InvalidCheckpoint(f"state model does not match the architecture: {exc}") from exc

    if optimizer is not None and payload.get("optimizer"):
        try:
            optimizer.load_state_dict(payload["optimizer"])
        except Exception as exc:                                    # noqa: BLE001
            raise InvalidCheckpoint(f"optimizer state incompatible: {exc}") from exc
    if scheduler is not None and payload.get("scheduler"):
        try:
            scheduler.load_state_dict(payload["scheduler"])
        except Exception:                                           # noqa: BLE001
            pass

    done = int(payload["epoch"])
    if total_epochs is not None and done >= total_epochs:
        raise InvalidCheckpoint(f"state already completed {done}/{total_epochs} epochs")

    if payload.get("rng"):
        _set_rng_state(payload["rng"])

    return ResumeState(model_id=model_id, start_epoch=done, global_step=int(payload.get("global_step", 0)),
                       best_metric=payload.get("best_metric"), history=list(payload.get("history") or []),
                       loaded=True, reason=f"resumed at epoch {done}")


def has_state(model_id: str) -> bool:
    p = state_path(model_id)
    return p.is_file() and p.stat().st_size > 0


def describe(model_id: str) -> dict[str, Any] | None:
    p = info_path(model_id)
    if not p.is_file():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:                                               # noqa: BLE001
        return None


def clear(model_id: str) -> None:
    """Remove the scratch state after a successful, saved run."""
    d = state_dir(model_id)
    if not d.exists():
        return
    for f in d.glob("state.*"):
        f.unlink(missing_ok=True)
