#!/usr/bin/env python3
"""Generate ``models/manifest.json`` — the platform's single source of truth.

Sources combined (in priority order):

1. :mod:`src.registry` — architecture + contract for all 32 models.
2. Local artifacts (``artifacts/metadata/*.json``, ``artifacts/status/*.json``).
3. The published Hub ``manifest.json`` (remote availability, revision, SHA-256).

Usage::

    python scripts/build_manifest.py                 # local evidence only
    python scripts/build_manifest.py --remote        # merge the Hub manifest
    python scripts/build_manifest.py --check         # exit 1 if the file is stale
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


def main() -> int:
    ap = argparse.ArgumentParser(description="Build models/manifest.json")
    ap.add_argument("--remote", action="store_true",
                    help="merge the published Hugging Face manifest (needs HF_MODEL_REPO)")
    ap.add_argument("--out", default=str(M.MANIFEST_PATH))
    ap.add_argument("--check", action="store_true",
                    help="do not write; fail if the existing manifest is out of date")
    args = ap.parse_args()

    remote = None
    if args.remote:
        if not hf_hub.is_configured():
            print(f"[manifest] {hf_hub.ENV_REPO} is not set — skipping remote merge")
        else:
            repo = hf_hub.model_repo()
            revision = hf_hub.resolve_revision(repo)
            remote = hf_hub.remote_manifest(repo, revision)
            if remote is None:
                print(f"[manifest] no manifest.json found in {repo} yet")
            else:
                remote = dict(remote)
                remote.setdefault("hf_repo", repo)
                remote["hf_revision"] = revision
                print(f"[manifest] merged remote manifest from {repo}@{str(revision)[:12]}")

    built = M.build_manifest(remote=remote)
    text = json.dumps(built, indent=2)

    if args.check:
        current = Path(args.out)
        # `generated_at` is a build timestamp, so it can never match on a rebuild.
        # Compare everything else; a timestamp-only difference is not staleness.
        def _payload(raw: str) -> str:
            try:
                data = json.loads(raw)
            except Exception:
                return raw.strip()
            if isinstance(data, dict):
                data.pop("generated_at", None)
            return json.dumps(data, indent=2, sort_keys=True)

        same = current.is_file() and _payload(current.read_text(encoding="utf-8")) == _payload(text)
        if not same:
            print("[manifest] STALE — models/manifest.json does not match the repository state")
            return 1
        print("[manifest] up to date")
        return 0

    p = Path(args.out)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text, encoding="utf-8")

    counts = built["status_counts"]
    print(f"[manifest] wrote {p.relative_to(ROOT)} — {built['totals']['models']} models")
    for status in ("TRAINED_VERIFIED", "TRAINED_UNVERIFIED", "ARCH_ONLY", "BLOCKED", "DEFERRED", "FAILED"):
        if counts.get(status):
            print(f"  {status:<18} {counts[status]}")
    ready = built["ready_models"]
    print(f"  ready (TRAINED_VERIFIED): {len(ready)}" + (f" → {', '.join(ready)}" if ready else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
