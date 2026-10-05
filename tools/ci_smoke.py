"""CI smoke test: verify every model's app renders and every backend responds.

Run locally or in CI (after ``pip install -r requirements.txt`` + fastapi/httpx):

    python tools/ci_smoke.py

Checks
------
1. every model has a committed artifact (model.pt + preprocess.pkl + meta.json)
2. every Streamlit app renders without an exception (streamlit AppTest)
3. every FastAPI backend answers /health and /predict and rejects bad input,
   each in its **own** interpreter (every backend defines a module named
   ``predictor``, so loading them all in one process would alias them).

Exits non-zero if anything fails, so it is safe to use as a CI gate.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import app_shared as A  # noqa: E402

_BACKEND_RUNNER = r'''
import base64, importlib.util, io, json, sys
import numpy as np
from PIL import Image
mid, folder = int(sys.argv[1]), sys.argv[2]
sys.path.insert(0, f"{folder}/backend")
spec = importlib.util.spec_from_file_location("backend_main", f"{folder}/backend/main.py")
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
from fastapi.testclient import TestClient
c = TestClient(mod.app)
KIND = {1:"tab",2:"img",3:"seq",4:"seq",5:"seq",6:"img",7:"img",8:"img",9:"text",
        10:"text",11:"text",12:"img",13:"img",14:"img",15:"img64",16:"img64",
        17:"pair",18:"text",19:"tab",20:"tab"}
assert c.get("/health").json()["status"] == "ok"
k = KIND[mid]
if k == "tab":
    s = json.load(open(f"{folder}/model/samples.json"))
    feats = (s.get("features") or {}).get("benign-like") or (s.get("rows") or [[0.0]*30])[0]
    assert c.post("/predict", json={"features": feats}).status_code == 200
    assert c.post("/predict", json={"features": [1, 2, 3]}).status_code == 400
    assert c.post("/predict", json={}).status_code == 422
elif k == "text":
    r = c.post("/predict/text", json={"text": "The team won the final."})
    assert r.status_code == 200, r.text
    assert c.post("/predict/text", json={}).status_code == 422
elif k in ("img", "img64"):
    arr = np.load(f"{folder}/model/sample_images.npy")[0]
    buf = io.BytesIO(); Image.fromarray(arr).save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode()
    assert c.post("/predict/image-base64", json={"image_base64": b64}).status_code == 200
    assert c.post("/predict/image-base64", json={"image_base64": "@@@"}).status_code == 400
print("OK")
'''


def check_artifacts() -> list[str]:
    missing = []
    for mid, info in sorted(A.MODELS.items()):
        d = ROOT / info["folder"] / "model"
        for name in ("model.pt", "preprocess.pkl", "meta.json"):
            if not (d / name).exists():
                missing.append(f"{info['folder']}/{name}")
    return missing


def check_apps() -> list[str]:
    from streamlit.testing.v1 import AppTest
    failures = []
    targets = [(mid, f"{A.MODELS[mid]['folder']}/frontend/app.py") for mid in sorted(A.MODELS)]
    targets.append((0, "dashboard.py"))
    for mid, target in targets:
        try:
            at = AppTest.from_file(str(ROOT / target), default_timeout=180)
            at.run()
            errs = [e.value for e in at.exception]
            if errs:
                failures.append(f"{target}: {errs[0][:200]}")
        except Exception as exc:  # noqa: BLE001
            failures.append(f"{target}: {type(exc).__name__}: {exc}")
    return failures


def check_backends() -> list[str]:
    runner = ROOT / ".build" / "_backend_runner.py"
    runner.parent.mkdir(parents=True, exist_ok=True)
    runner.write_text(_BACKEND_RUNNER, encoding="utf-8")
    failures = []
    for mid, info in sorted(A.MODELS.items()):
        folder = info["folder"]
        proc = subprocess.run([sys.executable, str(runner), str(mid), folder],
                              cwd=str(ROOT), capture_output=True, text=True, timeout=180)
        if proc.returncode != 0 or "OK" not in proc.stdout:
            lines = (proc.stderr or proc.stdout).strip().splitlines() or ["unknown error"]
            failures.append(f"{folder}: {lines[-1][:200]}")
    return failures


def main() -> int:
    print("1) artifacts ...")
    miss = check_artifacts()
    print(f"   missing: {miss or 'none'}")
    print("2) streamlit apps ...")
    app_fail = check_apps()
    print(f"   failures: {len(app_fail)}")
    for f in app_fail:
        print("   -", f)
    print("3) fastapi backends ...")
    be_fail = check_backends()
    print(f"   failures: {len(be_fail)}")
    for f in be_fail:
        print("   -", f)
    ok = not (miss or app_fail or be_fail)
    print("\nSMOKE:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
