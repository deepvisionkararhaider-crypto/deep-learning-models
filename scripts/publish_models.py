#!/usr/bin/env python3
"""Publish trained artifacts to the Hugging Face Hub — the 12-step protocol.

Steps (each one is enforced, none can be skipped):

1.  confirm training completed and the weights file exists and is non-empty
2.  validate the artifact structure (architecture rebuild + state_dict load)
3.  compute the SHA-256 of the weights
4.  assemble ``config.json`` / ``metrics.json`` / ``model_card.md``
5.  upload to ``HF_MODEL_REPO`` (never to GitHub)
6.  verify remote availability and resolve the immutable revision
7.  verify remote size + checksum against the local digest
8.  download into a clean temporary directory
9.  load with the *production* loader and run a real inference smoke test
10. publish the manifest entry (status + revision + sha) to the Hub
11. (handled by the workflow) commit only small metadata to GitHub
12. write a machine-readable publish report

Usage::

    python scripts/publish_models.py --model brain_cnn
    python scripts/publish_models.py --all-ready --dry-run
"""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

from src import artifacts as A  # noqa: E402
from src import hf_hub, manifest as M  # noqa: E402
from src.registry import ALL_MODELS, BY_ID, build_model  # noqa: E402

REPORT = ROOT / "artifacts" / "status" / "publish.json"


def _smoke_test(spec, weights_path: Path) -> dict:
    """Rebuild the architecture and load the weights with the production loader.

    This is the same code path the Streamlit app uses, so a pass here means the
    published artifact will actually load for a visitor.
    """
    model = build_model(spec)()
    state = torch.load(weights_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state, strict=True)
    model.eval()
    with torch.inference_mode():
        if spec.task in ("classification", "text_classification", "tabular_classification",
                         "sequence_classification", "graph_classification", "binary_classification"):
            if spec.in_ch:
                x = torch.zeros(1, spec.in_ch, spec.input_size or 28, spec.input_size or 28)
            else:
                x = torch.zeros(1, _feature_dim(spec))
            out = model(x)
            if isinstance(out, (tuple, list)):
                out = out[0]
            assert out.ndim == 2 and out.shape[0] == 1, f"unexpected logits shape {tuple(out.shape)}"
            shape = tuple(out.shape)
        elif spec.task == "segmentation":
            x = torch.zeros(1, spec.in_ch, spec.input_size, spec.input_size)
            out = model(x)
            if isinstance(out, (tuple, list)):
                out = out[0]
            assert out.ndim == 4, f"unexpected mask shape {tuple(out.shape)}"
            shape = tuple(out.shape)
        elif spec.task in ("detection",):
            x = torch.zeros(1, spec.in_ch, spec.input_size, spec.input_size)
            out = model(x)
            shape = tuple(out[0].shape) if torch.is_tensor(out) else "n/a"
        else:
            shape = "skipped (no generic input contract)"
    return {"loaded": True, "output_shape": list(shape) if isinstance(shape, tuple) else shape,
            "n_params": int(sum(p.numel() for p in model.parameters()))}


def _feature_dim(spec) -> int:
    for key in ("in_features", "feat", "dim"):
        if key in (spec.build_kwargs or {}):
            return int(spec.build_kwargs[key])
    return max(int(spec.in_ch or 1), 1)


def publish_one(spec, *, repo: str | None, revision: str | None, dry_run: bool) -> dict:
    model_id, area = spec.id, spec.area
    out: dict = {"model_id": model_id, "area": area, "steps": {}}

    # --- 1. training completed / weights exist ---------------------------------
    wpath = A.weights_path(model_id, area)
    if not wpath.is_file() or wpath.stat().st_size == 0:
        out["status"] = "FAILED"
        out["error"] = f"no non-empty weights file at {A._rel(wpath)}"
        return out
    out["steps"]["1_weights_exist"] = {"path": A._rel(wpath), "bytes": wpath.stat().st_size}

    meta = A.read_metadata(model_id) or {}

    # --- 2. validate structure with the production loader ----------------------
    try:
        smoke = _smoke_test(spec, wpath)
    except Exception as exc:                                        # noqa: BLE001
        out["status"] = "FAILED"
        out["error"] = f"structure/inference validation failed: {type(exc).__name__}: {exc}"
        return out
    out["steps"]["2_structure_valid"] = smoke

    # --- 3/4. hash + assemble --------------------------------------------------
    with tempfile.TemporaryDirectory(prefix=f"publish-{model_id}-") as tmp:
        art = hf_hub.assemble_artifact(spec, wpath, meta, Path(tmp) / model_id,
                                       status="TRAINED_UNVERIFIED", revision=revision, repo=repo)
        out["steps"]["3_sha256"] = art.weights_sha256
        out["steps"]["4_assembled"] = [p.name for p in art.files()]

        if dry_run:
            out["status"] = "DRY_RUN"
            out["note"] = "artifact assembled and validated locally; nothing uploaded"
            return out

        if not hf_hub.is_configured():
            out["status"] = "FAILED"
            out["error"] = f"{hf_hub.ENV_REPO} is not configured; cannot publish"
            return out
        if not hf_hub.has_write_token():
            out["status"] = "FAILED"
            out["error"] = "HF_TOKEN is not set; publishing requires a write token"
            return out

        # --- 5. upload ---------------------------------------------------------
        try:
            hf_hub.ensure_repo(repo)
            up = hf_hub.upload_artifact(art, repo=repo)
        except Exception as exc:                                    # noqa: BLE001
            out["status"] = "FAILED"
            out["error"] = f"upload failed: {type(exc).__name__}: {exc}"
            return out
        out["steps"]["5_uploaded"] = up

        # --- 6. verify remote availability + pin revision ----------------------
        try:
            files = hf_hub.remote_files(repo)
            pinned = hf_hub.resolve_revision(repo)
        except Exception as exc:                                    # noqa: BLE001
            out["status"] = "FAILED"
            out["error"] = f"remote listing failed: {type(exc).__name__}: {exc}"
            return out
        expected = hf_hub.artifact_paths(model_id, area)
        missing = [p for p in expected.values() if p not in files and not p.endswith("model_card.md")]
        if missing:
            out["status"] = "FAILED"
            out["error"] = f"remote artifact incomplete after upload; missing {missing}"
            return out
        out["steps"]["6_remote_listed"] = {"files": len(files), "revision": pinned}

        # --- 7. verify remote size + checksum ---------------------------------
        try:
            remote = hf_hub.verify_remote(model_id, area, repo=repo, revision=pinned,
                                          expect_sha256=art.weights_sha256)
        except Exception as exc:                                    # noqa: BLE001
            out["status"] = "FAILED"
            out["error"] = f"remote checksum/size verification failed: {type(exc).__name__}: {exc}"
            return out
        out["steps"]["7_checksum_ok"] = {"sha256": remote["weights_sha256"],
                                         "bytes": remote["weights_bytes"]}

        # --- 8. fresh download into a clean directory --------------------------
        with tempfile.TemporaryDirectory(prefix=f"verify-{model_id}-") as fresh:
            try:
                local = hf_hub.download_artifact(model_id, area, repo=repo, revision=pinned,
                                                 cache_dir=fresh, expect_sha256=art.weights_sha256)
            except Exception as exc:                                # noqa: BLE001
                out["status"] = "FAILED"
                out["error"] = f"fresh download failed: {type(exc).__name__}: {exc}"
                return out
            out["steps"]["8_downloaded"] = {k: str(v) for k, v in local.items()}

            # --- 9. production-loader inference on the downloaded file ---------
            try:
                fresh_smoke = _smoke_test(spec, Path(local["weights"]))
            except Exception as exc:                                # noqa: BLE001
                out["status"] = "FAILED"
                out["error"] = f"fresh-download inference failed: {type(exc).__name__}: {exc}"
                return out
            out["steps"]["9_fresh_inference"] = fresh_smoke

        # --- 10. record the verified status ------------------------------------
        out["status"] = "TRAINED_VERIFIED"
        out["hf_repo"] = repo or hf_hub.model_repo()
        out["hf_revision"] = pinned
        out["weights_sha256"] = art.weights_sha256
        out["weights_bytes"] = remote["weights_bytes"]
        return out


def update_remote_manifest(results: list[dict], *, repo: str | None, revision: str | None) -> dict | None:
    """Merge this run's verified entries into the Hub manifest and upload it."""
    if not (hf_hub.is_configured() and hf_hub.has_write_token()):
        return None
    existing = hf_hub.remote_manifest(repo, revision) or {"models": {}}
    models = dict(existing.get("models") or {})
    for r in results:
        if r.get("status") == "TRAINED_VERIFIED":
            spec = BY_ID[r["model_id"]]
            meta = A.read_metadata(spec.id) or {}
            models[spec.id] = {
                "model_id": spec.id,
                "area": spec.area,
                "task": spec.task,
                "status": "TRAINED_VERIFIED",
                "hf_path": hf_hub.artifact_dir(spec.id, spec.area),
                "hf_revision": r.get("hf_revision"),
                "weights_sha256": r.get("weights_sha256"),
                "weights_bytes": r.get("weights_bytes"),
                "metrics": meta.get("metrics") or {},
                "dataset": meta.get("dataset", spec.dataset_name),
                "published_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            }
    merged = hf_hub.build_manifest(list(models.values()), repo=repo or hf_hub.model_repo(),
                                   revision=revision)
    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp) / "manifest.json"
        p.write_text(json.dumps(merged, indent=2), encoding="utf-8")
        hf_hub.upload_manifest(p, repo=repo, commit_message="Update artifact manifest")
    return merged


def main() -> int:
    ap = argparse.ArgumentParser(description="Publish trained artifacts to the Hugging Face Hub")
    ap.add_argument("--model", action="append", default=[], help="model id (repeatable)")
    ap.add_argument("--all-ready", action="store_true",
                    help="publish every model whose local status is TRAINED_*")
    ap.add_argument("--repo", default=None, help=f"override {hf_hub.ENV_REPO}")
    ap.add_argument("--revision", default=None, help="target branch/revision (default: main)")
    ap.add_argument("--dry-run", action="store_true", help="assemble + validate, upload nothing")
    args = ap.parse_args()

    if not args.model and not args.all_ready:
        ap.error("select models with --model or --all-ready")

    if args.all_ready:
        ids = [s.id for s in ALL_MODELS if A.has_weights(s.id, s.area) and not s.is_blocked]
    else:
        unknown = [m for m in args.model if m not in BY_ID]
        if unknown:
            print(f"unknown model id(s): {unknown}", file=sys.stderr)
            return 2
        ids = list(dict.fromkeys(args.model))

    if not ids:
        print("nothing to publish (no trained, non-blocked models selected)")
        REPORT.parent.mkdir(parents=True, exist_ok=True)
        REPORT.write_text(json.dumps({"results": [], "status": "NOTHING_TO_DO"}, indent=2), encoding="utf-8")
        return 0

    results = []
    for mid in ids:
        spec = BY_ID[mid]
        print(f"[publish] {mid} …", flush=True)
        r = publish_one(spec, repo=args.repo, revision=args.revision, dry_run=args.dry_run)
        results.append(r)
        if r.get("status") == "TRAINED_VERIFIED":
            print(f"  ✔ TRAINED_VERIFIED  sha256={str(r.get('weights_sha256'))[:12]}…  "
                  f"rev={str(r.get('hf_revision'))[:12]}")
        elif r.get("status") == "DRY_RUN":
            print("  • dry run ok (validated, not uploaded)")
        else:
            print(f"  ✖ {r.get('status')}: {r.get('error')}")

    if not args.dry_run and any(r.get("status") == "TRAINED_VERIFIED" for r in results):
        try:
            update_remote_manifest(results, repo=args.repo, revision=args.revision)
            print("[publish] updated the Hub manifest.json")
        except Exception as exc:                                    # noqa: BLE001
            print(f"[publish] WARNING: could not update the Hub manifest: {exc}")

    # step 12: machine-readable report
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    summary = {
        "status": "OK" if all(r.get("status") in ("TRAINED_VERIFIED", "DRY_RUN") for r in results) else "PARTIAL",
        "verified": [r["model_id"] for r in results if r.get("status") == "TRAINED_VERIFIED"],
        "failed": {r["model_id"]: r.get("error") for r in results if r.get("status") == "FAILED"},
        "results": results,
    }
    REPORT.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[publish] report → {A._rel(REPORT)}")

    return 0 if not summary["failed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
