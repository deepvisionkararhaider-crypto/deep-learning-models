"""Resumable downloader for the real HuggingFace datasets used by training.

Bandwidth in this environment is ~180 kB/s, so downloads are the long pole.
Files are cached under .cache/parquet/<key>/<split>.parquet and skipped if present.
"""
import json, os, sys, time, urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / ".cache" / "parquet"
MANIFEST = json.loads((ROOT / ".cache" / "parquet_manifest.json").read_text())

def fetch(url, dest: Path):
    if dest.exists() and dest.stat().st_size > 0:
        print(f"  skip {dest.name} ({dest.stat().st_size/1e6:.1f}MB)", flush=True)
        return True
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(".part")
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=60) as r, open(tmp, "wb") as f:
            total = int(r.headers.get("Content-Length") or 0)
            got = 0
            while True:
                chunk = r.read(1 << 16)
                if not chunk:
                    break
                f.write(chunk); got += len(chunk)
                if total and got % (5 << 20) < (1 << 16):
                    el = time.time() - t0
                    print(f"    {dest.name} {got/1e6:.0f}/{total/1e6:.0f}MB {got/el/1e3:.0f}kB/s", flush=True)
        tmp.rename(dest)
        print(f"  ok {dest.name} ({dest.stat().st_size/1e6:.1f}MB, {time.time()-t0:.0f}s)", flush=True)
        return True
    except Exception as e:
        print(f"  FAIL {dest.name}: {type(e).__name__}: {str(e)[:120]}", flush=True)
        return False

def main(keys):
    failures = []
    for key in keys:
        info = MANIFEST.get(key)
        if not info:
            failures.append(f"{key}: no manifest entry")
            print(f"FAIL: no manifest entry for {key}", flush=True)
            continue
        print(f"=== {key}: {info['dataset']}", flush=True)
        for i, f in enumerate(info["files"]):
            dest = CACHE / key / f"{f['split']}-{i}.parquet"
            if not fetch(f["url"], dest):
                failures.append(f"{key}/{dest.name}")
    if failures:
        print("Dataset fetch failures: " + ", ".join(failures), flush=True)
        raise SystemExit(1)
    print("DONE: all requested dataset files are available", flush=True)

if __name__ == "__main__":
    keys = sys.argv[1:] or list(MANIFEST)
    main(keys)
