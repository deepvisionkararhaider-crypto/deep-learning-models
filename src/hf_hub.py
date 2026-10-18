"""Hugging Face Hub artifact management.

Trained checkpoints live on the Hugging Face Hub — **not** in this git repository.
This module is the single place that knows the remote layout, so training,
publishing, verification and inference all agree.

Remote layout (``HF_MODEL_REPO``)::

    medical/<model_id>/model.pt          # torch state_dict (CPU)
    medical/<model_id>/config.json       # architecture + preprocessing contract
    medical/<model_id>/metrics.json      # measured metrics, status, checksums
    medical/<model_id>/model_card.md     # human-readable card
    agriculture/<model_id>/...
    laboratories/lab01/...
    manifest.json                        # index of every published artifact
    README.md

Design rules enforced here:

* **No secrets in code.** The token is read from ``HF_TOKEN`` (environment) or an
  explicit argument; it is never logged, echoed or written to a file by this module.
* **Revision pinning.** Every download/upload can be pinned to an immutable commit
  sha; :func:`resolve_revision` turns a branch name into a concrete sha.
* **Integrity.** Every artifact records a SHA-256 per file; downloads verify it
  before the file is handed to the loader.
* **Reads are anonymous-friendly.** Public repos download without a token, so the
  Streamlit app never needs write credentials.
* **Retries with backoff** for transient network errors; a hard failure raises
  :class:`HubUnavailable` rather than silently returning a broken path.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable

# The repo id is configuration, never a hard-coded guess. ``HF_MODEL_REPO`` is set
# as a GitHub Actions *variable* and (optionally) as a Streamlit secret.
ENV_REPO = "HF_MODEL_REPO"
ENV_TOKEN = "HF_TOKEN"

#: Files that make up one published artifact (relative to the artifact directory).
ARTIFACT_FILES = ("config.json", "metrics.json", "model.pt", "model_card.md")
WEIGHTS_NAME = "model.pt"
CONFIG_NAME = "config.json"
METRICS_NAME = "metrics.json"
CARD_NAME = "model_card.md"
MANIFEST_NAME = "manifest.json"

#: ``area`` -> remote directory prefix.
AREA_PREFIX = {
    "medical": "medical",
    "agriculture": "agriculture",
    "lab": "laboratories",
}

DEFAULT_CACHE = Path(os.environ.get("HF_ARTIFACT_CACHE") or (Path.home() / ".cache" / "dl-models-hf"))


class HubError(RuntimeError):
    """Base class for Hub problems surfaced to the UI/CLI."""


class HubUnavailable(HubError):
    """Network/HF failure after retries (offline, rate-limited, bad revision)."""


class HubAuthError(HubError):
    """Missing or invalid token for a write operation."""


class ChecksumMismatch(HubError):
    """Downloaded bytes do not match the recorded SHA-256."""


# --------------------------------------------------------------------------- #
# configuration
# --------------------------------------------------------------------------- #
def model_repo(explicit: str | None = None) -> str:
    """Return the configured model repository id.

    Raises a clear error rather than inventing a value: the master spec requires
    that we never fabricate a repository id.
    """
    repo = (explicit or os.environ.get(ENV_REPO) or "").strip()
    if not repo or "/" not in repo:
        raise HubError(
            f"No Hugging Face model repository configured. Set the {ENV_REPO} "
            f"environment variable (GitHub: Settings → Secrets and variables → "
            f"Actions → Variables) to e.g. '<user>/<repo>'."
        )
    return repo


def is_configured() -> bool:
    """True when a model repository id is available (no network call)."""
    try:
        model_repo()
        return True
    except HubError:
        return False


def token(explicit: str | None = None) -> str | None:
    """The HF token, or None. Never printed by this module."""
    if explicit:
        return explicit
    return os.environ.get(ENV_TOKEN) or None


def has_write_token() -> bool:
    return bool(token())


# --------------------------------------------------------------------------- #
# paths
# --------------------------------------------------------------------------- #
def area_prefix(area: str) -> str:
    try:
        return AREA_PREFIX[area]
    except KeyError as exc:
        raise HubError(f"Unknown area {area!r}; expected one of {sorted(AREA_PREFIX)}") from exc


def artifact_dir(model_id: str, area: str) -> str:
    """Remote directory for one model, e.g. ``medical/brain_cnn``."""
    return f"{area_prefix(area)}/{model_id}"


def artifact_paths(model_id: str, area: str) -> dict[str, str]:
    """Remote path of each artifact file (weights, config, metrics, card)."""
    d = artifact_dir(model_id, area)
    return {
        "weights": f"{d}/{WEIGHTS_NAME}",
        "config": f"{d}/{CONFIG_NAME}",
        "metrics": f"{d}/{METRICS_NAME}",
        "card": f"{d}/{CARD_NAME}",
    }


# --------------------------------------------------------------------------- #
# hashing
# --------------------------------------------------------------------------- #
def sha256_file(path: str | Path, chunk: int = 1 << 20) -> str:
    """Streaming SHA-256 of a file (constant memory)."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


# --------------------------------------------------------------------------- #
# retries
# --------------------------------------------------------------------------- #
def with_retries(fn: Callable[[], Any], *, attempts: int = 3, base_delay: float = 1.5,
                 label: str = "huggingface call") -> Any:
    """Run ``fn`` with exponential backoff; raise :class:`HubUnavailable` at the end."""
    last: Exception | None = None
    for i in range(attempts):
        try:
            return fn()
        except Exception as exc:                                    # noqa: BLE001
            last = exc
            msg = str(exc).lower()
            if any(k in msg for k in ("401", "403", "unauthorized", "forbidden", "invalid token")):
                raise HubAuthError(f"{label} was rejected by Hugging Face: {exc}") from exc
            if any(k in msg for k in ("404", "not found", "revision not found", "no such")):
                raise HubUnavailable(f"{label} not found on Hugging Face: {exc}") from exc
            if i < attempts - 1:
                time.sleep(base_delay * (2 ** i))
    raise HubUnavailable(f"{label} failed after {attempts} attempts: {last}") from last


# --------------------------------------------------------------------------- #
# local artifact assembly
# --------------------------------------------------------------------------- #
def build_config(spec, metadata: dict[str, Any]) -> dict[str, Any]:
    """Architecture + preprocessing contract stored next to the weights.

    ``config.json`` is what lets a fresh clone rebuild the exact architecture and
    reproduce the exact preprocessing without reading the training code.
    """
    inp = metadata.get("input") or {}
    return {
        "model_id": spec.id,
        "display_name": spec.name,
        "area": spec.area,
        "task": spec.task,
        "framework": spec.framework,
        "builder": spec.builder,
        "build_kwargs": dict(spec.build_kwargs or {}),
        "classes": list(metadata.get("classes") or spec.classes),
        "n_classes": len(metadata.get("classes") or spec.classes),
        "input": {
            "modality": metadata.get("modality") or ("image" if spec.in_ch else "features"),
            "size": int(inp.get("size", spec.input_size)),
            "channels": int(inp.get("channels", spec.in_ch)),
            "mean": list(inp.get("mean", (0.485, 0.456, 0.406))),
            "std": list(inp.get("std", (0.229, 0.224, 0.225))),
        },
        "outputs": metadata.get("outputs", spec.outputs),
        "weights_file": WEIGHTS_NAME,
        "weights_format": "torch-state-dict",
        "artifact_version": metadata.get("artifact_version"),
        "dataset": metadata.get("dataset", spec.dataset_name),
        "dataset_key": metadata.get("dataset_key", spec.dataset_key),
    }


def build_metrics_record(spec, metadata: dict[str, Any], *, weights_sha256: str,
                         status: str, revision: str | None = None,
                         repo: str | None = None) -> dict[str, Any]:
    """The published ``metrics.json``: measured numbers + integrity + status."""
    return {
        "model_id": spec.id,
        "area": spec.area,
        "task": spec.task,
        "status": status,
        "metrics": metadata.get("metrics") or {},
        "test_set_size": metadata.get("test_set_size"),
        "split_manifest": metadata.get("split_manifest") or {},
        "dataset": metadata.get("dataset", spec.dataset_name),
        "dataset_key": metadata.get("dataset_key", spec.dataset_key),
        "trained_at": metadata.get("saved_at"),
        "training": metadata.get("training") or {},
        "weights_bytes": metadata.get("weights_bytes"),
        "weights_sha256": weights_sha256,
        "artifact_version": metadata.get("artifact_version"),
        "git_commit": metadata.get("git_commit"),
        "hf_repo": repo,
        "hf_revision": revision,
        "limitations": metadata.get("limitations") or spec.notes,
    }


def render_model_card(spec, metrics: dict[str, Any]) -> str:
    """A small, honest model card (no fabricated benchmarks)."""
    lines = [
        f"# {spec.name} (`{spec.id}`)",
        "",
        f"- **Area**: {spec.area}",
        f"- **Task**: `{spec.task}`",
        f"- **Framework**: {spec.framework}",
        f"- **Dataset**: {spec.dataset_name} (`{spec.dataset_key}`)",
        f"- **Status**: `{metrics.get('status', 'UNKNOWN')}`",
        "",
        "## Input / output",
        "",
        f"- Input: {spec.input_desc or 'see config.json'}",
        f"- Output: {spec.outputs or 'see config.json'}",
        f"- Classes: {', '.join(spec.classes) if spec.classes else 'n/a'}",
        "",
        "## Measured metrics (held-out test split)",
        "",
    ]
    measured = metrics.get("metrics") or {}
    if measured:
        lines += ["| metric | value |", "|---|---|"]
        lines += [f"| {k} | {v} |" for k, v in measured.items()
                  if isinstance(v, (int, float))]
    else:
        lines.append("_No metrics recorded yet._")
    if metrics.get("test_set_size") is not None:
        lines += ["", f"Test-set size: **{metrics['test_set_size']}** examples."]
    lines += [
        "",
        "## Limitations",
        "",
        metrics.get("limitations") or spec.notes or "See the repository documentation.",
        "",
        "## Integrity",
        "",
        f"- weights sha256: `{metrics.get('weights_sha256')}`",
        f"- weights bytes: `{metrics.get('weights_bytes')}`",
        f"- repository: `{metrics.get('hf_repo')}`",
        f"- revision: `{metrics.get('hf_revision')}`",
        "",
        "_Predictions are statistical estimates and can be wrong. Not a medical "
        "device; not a substitute for professional assessment._",
        "",
    ]
    return "\n".join(lines)


@dataclass
class LocalArtifact:
    """A fully assembled, ready-to-upload artifact directory."""

    model_id: str
    area: str
    directory: Path
    config: dict[str, Any]
    metrics: dict[str, Any]
    weights_sha256: str

    @property
    def weights_path(self) -> Path:
        return self.directory / WEIGHTS_NAME

    def files(self) -> list[Path]:
        return [self.directory / n for n in ARTIFACT_FILES if (self.directory / n).is_file()]


def assemble_artifact(spec, weights_path: str | Path, metadata: dict[str, Any],
                      out_dir: str | Path, *, status: str = "TRAINED_UNVERIFIED",
                      revision: str | None = None, repo: str | None = None) -> LocalArtifact:
    """Copy weights and write config/metrics/card into ``out_dir``.

    The result is exactly the directory that gets uploaded, so what is verified
    locally is byte-identical to what is served remotely.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    wsrc = Path(weights_path)
    if not wsrc.is_file() or wsrc.stat().st_size == 0:
        raise HubError(f"Refusing to publish '{spec.id}': weights file missing or empty ({wsrc})")
    dst = out / WEIGHTS_NAME
    if wsrc.resolve() != dst.resolve():
        shutil.copy2(wsrc, dst)

    digest = sha256_file(dst)
    config = build_config(spec, metadata)
    metrics = build_metrics_record(spec, metadata, weights_sha256=digest, status=status,
                                   revision=revision, repo=repo)
    metrics["weights_bytes"] = dst.stat().st_size
    (out / CONFIG_NAME).write_text(json.dumps(config, indent=2), encoding="utf-8")
    (out / METRICS_NAME).write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (out / CARD_NAME).write_text(render_model_card(spec, metrics), encoding="utf-8")
    return LocalArtifact(spec.id, spec.area, out, config, metrics, digest)


# --------------------------------------------------------------------------- #
# remote operations
# --------------------------------------------------------------------------- #
def _api():
    from huggingface_hub import HfApi
    return HfApi()


def ensure_repo(repo: str | None = None, *, private: bool = False, tok: str | None = None) -> str:
    """Create the model repository if it does not exist. Returns the repo id."""
    repo = model_repo(repo)
    if not token(tok):
        raise HubAuthError(
            "Creating a Hugging Face repository requires a write token. Set the "
            "HF_TOKEN secret (GitHub: Settings → Secrets and variables → Actions)."
        )
    api = _api()

    def _create():
        return api.create_repo(repo_id=repo, repo_type="model", private=private, exist_ok=True,
                               token=token(tok))

    with_retries(_create, label=f"create_repo({repo})")
    return repo


def upload_artifact(art: LocalArtifact, *, repo: str | None = None, tok: str | None = None,
                    commit_message: str | None = None, revision: str | None = None) -> dict[str, Any]:
    """Upload every file of an assembled artifact. Returns the new commit sha.

    Uploads are per-file so a failure is attributable and resumable; the returned
    sha is the repository revision *after* the last upload.
    """
    repo = model_repo(repo)
    tok = token(tok)
    if not tok:
        raise HubAuthError(
            "Publishing to Hugging Face requires HF_TOKEN. Add it under "
            "Settings → Secrets and variables → Actions → Secrets."
        )
    api = _api()
    commit = commit_message or f"Publish {art.model_id} ({art.metrics.get('status')})"
    last: Any = None
    for name in ARTIFACT_FILES:
        p = art.directory / name
        if not p.is_file():
            continue
        target = f"{artifact_dir(art.model_id, art.area)}/{name}"
        last = with_retries(
            lambda p=p, target=target: api.upload_file(
                path_or_fileobj=str(p), path_in_repo=target, repo_id=repo,
                repo_type="model", revision=revision, commit_message=commit, token=tok,
            ),
            label=f"upload {target}",
        )
    return {"commit": getattr(last, "oid", None), "repo": repo,
            "dir": artifact_dir(art.model_id, art.area)}


def upload_manifest(path: str | Path, *, repo: str | None = None, tok: str | None = None,
                    commit_message: str = "Update artifact manifest") -> dict[str, Any]:
    repo = model_repo(repo)
    tok = token(tok)
    if not tok:
        raise HubAuthError("Publishing the manifest requires HF_TOKEN.")
    api = _api()
    info = with_retries(
        lambda: api.upload_file(path_or_fileobj=str(path), path_in_repo=MANIFEST_NAME,
                                repo_id=repo, repo_type="model",
                                commit_message=commit_message, token=tok),
        label="upload manifest.json",
    )
    return {"commit": getattr(info, "oid", None), "repo": repo}


def resolve_revision(repo: str | None = None, revision: str | None = None,
                     tok: str | None = None) -> str | None:
    """Turn a branch/tag into an immutable commit sha (best effort, None offline)."""
    repo = model_repo(repo)
    try:
        info = with_retries(lambda: _api().repo_info(repo_id=repo, repo_type="model",
                                                     revision=revision, token=token(tok)),
                            attempts=2, label=f"repo_info({repo})")
        return getattr(info, "sha", None)
    except HubError:
        return None


def remote_files(repo: str | None = None, revision: str | None = None,
                 tok: str | None = None) -> list[str]:
    repo = model_repo(repo)
    return list(with_retries(
        lambda: _api().list_repo_files(repo_id=repo, repo_type="model",
                                       revision=revision, token=token(tok)),
        label=f"list_repo_files({repo})",
    ))


def remote_manifest(repo: str | None = None, revision: str | None = None,
                    tok: str | None = None) -> dict[str, Any] | None:
    """Fetch ``manifest.json`` from the Hub (None when it does not exist yet)."""
    try:
        p = download_file(MANIFEST_NAME, repo=repo, revision=revision, tok=tok)
    except HubUnavailable:
        return None
    try:
        return json.loads(Path(p).read_text(encoding="utf-8"))
    except Exception:                                              # noqa: BLE001
        return None


def download_file(filename: str, *, repo: str | None = None, revision: str | None = None,
                  tok: str | None = None, cache_dir: str | Path | None = None) -> str:
    """Download one file with the HF cache; returns the local path.

    Anonymous download is used when no token is configured, which is how the
    Streamlit app reads public artifacts without any credentials.
    """
    from huggingface_hub import hf_hub_download

    repo = model_repo(repo)
    cache = str(cache_dir or DEFAULT_CACHE)
    return with_retries(
        lambda: hf_hub_download(repo_id=repo, filename=filename, repo_type="model",
                                revision=revision, cache_dir=cache, token=token(tok)),
        label=f"download {repo}/{filename}",
    )


def download_artifact(model_id: str, area: str, *, repo: str | None = None,
                      revision: str | None = None, tok: str | None = None,
                      cache_dir: str | Path | None = None,
                      expect_sha256: str | None = None) -> dict[str, Any]:
    """Download the four artifact files into one local directory.

    Verifies the weights SHA-256 when an expected digest is known (from the
    manifest), so a truncated/corrupt download fails loudly instead of silently
    loading bad weights.
    """
    paths = artifact_paths(model_id, area)
    local: dict[str, str] = {}
    for key in ("config", "metrics", "weights", "card"):
        try:
            local[key] = download_file(paths[key], repo=repo, revision=revision, tok=tok,
                                       cache_dir=cache_dir)
        except HubUnavailable as exc:
            if key == "card":            # card is informational; the rest are required
                continue
            raise HubUnavailable(
                f"Artifact '{model_id}' is not fully published: missing {paths[key]} ({exc})"
            ) from exc
    if expect_sha256:
        got = sha256_file(local["weights"])
        if got != expect_sha256:
            raise ChecksumMismatch(
                f"Checksum mismatch for '{model_id}': manifest says {expect_sha256[:12]}…, "
                f"downloaded file hashes to {got[:12]}… — refusing to load."
            )
    return local


def verify_remote(model_id: str, area: str, *, repo: str | None = None,
                  revision: str | None = None, tok: str | None = None,
                  expect_sha256: str | None = None) -> dict[str, Any]:
    """Confirm the remote artifact exists, is complete, and matches its checksum."""
    paths = artifact_paths(model_id, area)
    present = set(remote_files(repo=repo, revision=revision, tok=tok))
    missing = [p for p in paths.values() if p not in present and not p.endswith(CARD_NAME)]
    if missing:
        raise HubUnavailable(f"Remote artifact '{model_id}' is incomplete; missing: {missing}")
    local = download_artifact(model_id, area, repo=repo, revision=revision, tok=tok,
                             expect_sha256=expect_sha256)
    return {
        "model_id": model_id,
        "area": area,
        "files": paths,
        "local": local,
        "weights_sha256": sha256_file(local["weights"]),
        "weights_bytes": Path(local["weights"]).stat().st_size,
    }


# --------------------------------------------------------------------------- #
# manifest
# --------------------------------------------------------------------------- #
def build_manifest(entries: Iterable[dict[str, Any]], *, repo: str | None = None,
                   revision: str | None = None, generated_at: str | None = None) -> dict[str, Any]:
    """Assemble the repository-level ``manifest.json`` from per-model entries."""
    items = {e["model_id"]: e for e in entries}
    ready = sorted(k for k, e in items.items() if e.get("status") == "TRAINED_VERIFIED")
    return {
        "schema_version": 1,
        "generated_at": generated_at or time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "hf_repo": repo,
        "hf_revision": revision,
        "count": len(items),
        "ready_count": len(ready),
        "ready_models": ready,
        "models": items,
    }


def write_manifest(manifest: dict[str, Any], path: str | Path) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return p


def read_manifest(path: str | Path) -> dict[str, Any] | None:
    p = Path(path)
    if not p.is_file():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:                                              # noqa: BLE001
        return None
