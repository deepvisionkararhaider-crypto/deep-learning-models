#!/usr/bin/env python3
"""Fetch published artifacts from the Hugging Face Hub into a local cache.

Useful for offline development and for pre-seeding a Streamlit deployment's disk
cache. The Streamlit app does **not** need this script — it downloads lazily and
caches via ``huggingface_hub``.

Usage::

    python scripts/fetch_models.py --model brain_cnn
    python scripts/fetch_models.py --all-ready
    python scripts/fetch_models.py --list          # show the remote manifest
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import hf_hub, manifest as M  # noqa: E402
from src.registry import BY_ID  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description="Download published artifacts from the HF Hub")
    ap.add_argument("--model", action="append", default=[], help="model id (repeatable)")
    ap.add_argument("--all-ready", action="store_true", help="every model marked TRAINED_VERIFIED")
    ap.add_argument("--repo", default=None)
    ap.add_argument("--revision", default=None, help="pin to an immutable revision")
    ap.add_argument("--list", action="store_true", help="print the remote manifest and exit")
    ap.add_argument("--out", default=None, help="copy artifacts into this directory")
    args = ap.parse_args()

    if not hf_hub.is_configured():
        print(f"error: {hf_hub.ENV_REPO} is not set", file=sys.stderr)
        return 2
    repo = hf_hub.model_repo(args.repo)

    if args.list:
        revision = hf_hub.resolve_revision(repo, args.revision)
        remote = hf_hub.remote_manifest(repo, revision)
        if not remote:
            print(f"no manifest.json published in {repo} yet")
            return 1
        print(f"{repo}@{str(revision)[:12]} — {remote.get('count', 0)} published artifacts")
        for mid, e in sorted((remote.get("models") or {}).items()):
            print(f"  {mid:<20} {e.get('status', '?'):<18} "
                  f"sha={str(e.get('weights_sha256'))[:10]}… rev={str(e.get('hf_revision'))[:10]}")
        return 0

    if args.all_ready:
        revision = hf_hub.resolve_revision(repo, args.revision)
        remote = hf_hub.remote_manifest(repo, revision) or {}
        ids = [k for k, v in (remote.get("models") or {}).items()
               if v.get("status") == "TRAINED_VERIFIED"]
    else:
        ids = list(dict.fromkeys(args.model))

    if not ids:
        print("nothing selected; use --model <id> or --all-ready", file=sys.stderr)
        return 2

    unknown = [i for i in ids if i not in BY_ID]
    if unknown:
        print(f"unknown model id(s): {unknown}", file=sys.stderr)
        return 2

    failures = []
    for mid in ids:
        spec = BY_ID[mid]
        print(f"[fetch] {mid} …", flush=True)
        try:
            local = hf_hub.download_artifact(mid, spec.area, repo=repo, revision=args.revision,
                                             expect_sha256=None)
        except Exception as exc:                                    # noqa: BLE001
            print(f"  ✖ {type(exc).__name__}: {exc}")
            failures.append(mid)
            continue
        metrics = json.loads(Path(local["metrics"]).read_text(encoding="utf-8"))
        print(f"  ✔ {metrics.get('status')}  {Path(local['weights']).stat().st_size / 1e6:.2f} MB")
        if args.out:
            dest = Path(args.out) / mid
            dest.mkdir(parents=True, exist_ok=True)
            for key, src in local.items():
                (dest / Path(src).name).write_bytes(Path(src).read_bytes())
            print(f"  → copied to {dest}")

    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
