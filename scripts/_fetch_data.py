"""Resumable downloader for public Hugging Face dataset-server parquet files.

The manifest is defined in Python rather than under .cache/: .cache is intentionally
ignored by git and therefore must never contain required source configuration.
Datasets are cached under .cache/parquet/<key>/<split>-<index>.parquet.
"""
import json
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / ".cache" / "parquet"
DATASETS = {
    "brain_mri_cls": "PranomVignesh/MRI-Images-of-Brain-Tumor",
    "plant_cls": "Dipra444/plant-leaf-disease-jharkhand",
    "brain_det": "mmenendezg/brain-tumor-object-detection",
    "cotton_det": "Francesco/cotton-plant-disease",
    "plant_seg": "Mauregato/leaf_disease_segmentation",
    "brain_instseg": "dwb2023/brain-tumor-image-dataset-semantic-segmentation",
}


def parquet_manifest(key: str) -> dict:
    """Ask HF datasets-server for current parquet URLs; no checked-in/cache manifest."""
    dataset = DATASETS.get(key)
    if not dataset:
        raise KeyError(f"Unknown dataset key {key!r}; known keys: {', '.join(DATASETS)}")
    query = urllib.parse.urlencode({"dataset": dataset})
    url = f"https://datasets-server.huggingface.co/parquet?{query}"
    req = urllib.request.Request(url, headers={"User-Agent": "deep-learning-models-dataset-fetcher/1.0"})
    with urllib.request.urlopen(req, timeout=60) as response:
        payload = json.load(response)
    files = payload.get("parquet_files", [])
    if not files:
        raise RuntimeError(f"Hugging Face returned no parquet files for {dataset}; response keys={list(payload)}")
    return {
        "dataset": dataset,
        "files": [
            {"split": item.get("split", "data"), "url": item["url"]}
            for item in files if item.get("url")
        ],
    }


def fetch(url: str, dest: Path) -> bool:
    if dest.exists() and dest.stat().st_size > 0:
        print(f"  skip {dest.name} ({dest.stat().st_size / 1e6:.1f}MB)", flush=True)
        return True
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=90) as response, open(tmp, "wb") as f:
            total = int(response.headers.get("Content-Length") or 0)
            got = 0
            while True:
                chunk = response.read(1 << 16)
                if not chunk:
                    break
                f.write(chunk)
                got += len(chunk)
                if total and got % (5 << 20) < (1 << 16):
                    elapsed = max(time.time() - t0, 0.001)
                    print(f"    {dest.name} {got / 1e6:.0f}/{total / 1e6:.0f}MB {got / elapsed / 1e3:.0f}kB/s", flush=True)
        if tmp.stat().st_size == 0:
            raise RuntimeError("downloaded file is empty")
        tmp.replace(dest)
        print(f"  ok {dest.name} ({dest.stat().st_size / 1e6:.1f}MB, {time.time() - t0:.0f}s)", flush=True)
        return True
    except Exception as exc:
        tmp.unlink(missing_ok=True)
        print(f"  FAIL {dest.name}: {type(exc).__name__}: {str(exc)[:180]}", flush=True)
        return False


def main(keys):
    failures = []
    for key in keys:
        try:
            info = parquet_manifest(key)
        except Exception as exc:
            failures.append(f"{key}: manifest lookup failed ({type(exc).__name__}: {exc})")
            print(f"FAIL {failures[-1]}", flush=True)
            continue
        print(f"=== {key}: {info['dataset']}", flush=True)
        for i, item in enumerate(info["files"]):
            split = str(item["split"]).replace("/", "_")
            dest = CACHE / key / f"{split}-{i}.parquet"
            if not fetch(item["url"], dest):
                failures.append(f"{key}/{dest.name}")
    if failures:
        print("Dataset fetch failures: " + ", ".join(failures), flush=True)
        raise SystemExit(1)
    print("DONE: all requested dataset files are available", flush=True)


if __name__ == "__main__":
    keys = sys.argv[1:] or list(DATASETS)
    main(keys)
