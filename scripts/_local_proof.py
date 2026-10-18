"""Local end-to-end proof: download brain_mri_cls, train brain_cnn (from scratch),
save the artifact, then reload it in a fresh process and run inference.

This exists because the sandbox cannot trigger GitHub Actions (the App token has
no actions:write). It produces ONE real, honestly-labelled checkpoint so the full
pipeline (fetch -> train -> evaluate -> save -> fresh-load verify) is demonstrated
with evidence rather than claimed.
"""
import sys
import time

sys.path.insert(0, ".")

from scripts import _fetch_data           # noqa: E402
from scripts import train_models as T     # noqa: E402
from src.registry import BY_ID            # noqa: E402

t0 = time.time()
_fetch_data.main(["brain_mri_cls"])
print(f"[fetch] {time.time() - t0:.0f}s", flush=True)

T.TRAIN_CAPS["brain_mri_cls"] = 2000       # cap for sandbox speed; recorded in metadata
spec = BY_ID["brain_cnn"]
t1 = time.time()
res = T.train_classifier(spec, epochs=3, batch_size=8, lr=1e-3)
print(f"[train] {time.time() - t1:.0f}s metrics={res['metrics'].get('accuracy')}", flush=True)

info = T.save(spec, res)
print(f"[save] {info['weights_file']} {info['weights_bytes']} bytes", flush=True)

ok, why = T.fresh_load_check(spec)
print(f"[fresh-load] ok={ok} why={why}", flush=True)
print("DONE" if ok else "FRESH_LOAD_FAILED", flush=True)
