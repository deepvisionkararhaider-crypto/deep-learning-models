"""Tests for the Hugging Face artifact layer, manifest, dynamic forms and resume.

All network access is mocked or avoided: unit tests must never download a model.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import hf_hub, manifest as M, training_state as TS  # noqa: E402
from src.registry import ALL_MODELS, BY_ID, CV_MODELS, LAB_BY_INDEX  # noqa: E402


# --------------------------------------------------------------------------- #
# 1. artifact path construction + revision pinning
# --------------------------------------------------------------------------- #
def test_artifact_paths_follow_the_documented_layout():
    p = hf_hub.artifact_paths("brain_cnn", "medical")
    assert p["weights"] == "medical/brain_cnn/model.pt"
    assert p["config"] == "medical/brain_cnn/config.json"
    assert p["metrics"] == "medical/brain_cnn/metrics.json"
    assert p["card"] == "medical/brain_cnn/model_card.md"
    assert hf_hub.artifact_paths("plant_unet", "agriculture")["weights"] == \
        "agriculture/plant_unet/model.pt"
    assert hf_hub.artifact_paths("lab01", "lab")["weights"] == \
        "laboratories/lab01/model.pt"


def test_unknown_area_is_rejected():
    with pytest.raises(hf_hub.HubError):
        hf_hub.artifact_dir("x", "nope")


def test_model_repo_requires_configuration(monkeypatch):
    monkeypatch.delenv(hf_hub.ENV_REPO, raising=False)
    with pytest.raises(hf_hub.HubError):
        hf_hub.model_repo()
    assert not hf_hub.is_configured()


def test_model_repo_reads_the_env_var(monkeypatch):
    monkeypatch.setenv(hf_hub.ENV_REPO, "someone/some-models")
    assert hf_hub.model_repo() == "someone/some-models"
    assert hf_hub.is_configured()


def test_token_is_never_required_for_reading(monkeypatch):
    monkeypatch.delenv(hf_hub.ENV_TOKEN, raising=False)
    assert hf_hub.token() is None
    assert not hf_hub.has_write_token()          # reads still work anonymously


def test_upload_without_token_raises_auth_error(monkeypatch, tmp_path):
    monkeypatch.setenv(hf_hub.ENV_REPO, "someone/some-models")
    monkeypatch.delenv(hf_hub.ENV_TOKEN, raising=False)
    art = hf_hub.LocalArtifact("x", "medical", tmp_path, {}, {}, "0" * 64)
    with pytest.raises(hf_hub.HubAuthError):
        hf_hub.upload_artifact(art)


def test_ensure_repo_without_token_raises(monkeypatch):
    monkeypatch.setenv(hf_hub.ENV_REPO, "someone/some-models")
    monkeypatch.delenv(hf_hub.ENV_TOKEN, raising=False)
    with pytest.raises(hf_hub.HubAuthError):
        hf_hub.ensure_repo()


# --------------------------------------------------------------------------- #
# 2. hashing / checksum validation
# --------------------------------------------------------------------------- #
def test_sha256_matches_the_known_digest(tmp_path):
    p = tmp_path / "f.bin"
    p.write_bytes(b"hello world")
    assert hf_hub.sha256_file(p) == \
        "b94d27b9934d3e08a52e52d7da7dabfac484efe37a5380ee9088f7ace2efcde9"


def test_checksum_mismatch_is_detected(monkeypatch, tmp_path):
    """A truncated/corrupt download must fail loudly, not load silently."""
    good = tmp_path / "model.pt"
    torch.save({"w": torch.zeros(2, 2)}, good)
    wrong = "0" * 64

    def fake_download(filename, **kwargs):
        f = tmp_path / Path(filename).name
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_bytes(good.read_bytes())
        return str(f)

    monkeypatch.setenv(hf_hub.ENV_REPO, "someone/some-models")
    monkeypatch.setattr(hf_hub, "download_file", fake_download)
    with pytest.raises(hf_hub.ChecksumMismatch):
        hf_hub.download_artifact("brain_cnn", "medical", expect_sha256=wrong)


def test_download_artifact_accepts_a_matching_checksum(monkeypatch, tmp_path):
    good = tmp_path / "model.pt"
    torch.save({"w": torch.zeros(2, 2)}, good)
    digest = hf_hub.sha256_file(good)

    def fake_download(filename, **kwargs):
        f = tmp_path / Path(filename).name
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_bytes(good.read_bytes())
        return str(f)

    monkeypatch.setenv(hf_hub.ENV_REPO, "someone/some-models")
    monkeypatch.setattr(hf_hub, "download_file", fake_download)
    out = hf_hub.download_artifact("brain_cnn", "medical", expect_sha256=digest)
    assert Path(out["weights"]).is_file()


def test_missing_remote_artifact_reports_clearly(monkeypatch):
    def boom(*a, **k):
        raise hf_hub.HubUnavailable("404 not found")

    monkeypatch.setenv(hf_hub.ENV_REPO, "someone/some-models")
    monkeypatch.setattr(hf_hub, "download_file", boom)
    with pytest.raises(hf_hub.HubUnavailable) as e:
        hf_hub.download_artifact("brain_cnn", "medical")
    assert "not fully published" in str(e.value)


# --------------------------------------------------------------------------- #
# 3. retries / auth failure handling
# --------------------------------------------------------------------------- #
def test_retries_eventually_succeed():
    calls = {"n": 0}

    def flaky():
        calls["n"] += 1
        if calls["n"] < 3:
            raise ConnectionError("transient")
        return "ok"

    assert hf_hub.with_retries(flaky, attempts=4, base_delay=0.01) == "ok"
    assert calls["n"] == 3


def test_auth_failure_is_not_retried():
    calls = {"n": 0}

    def unauthorized():
        calls["n"] += 1
        raise RuntimeError("401 Unauthorized")

    with pytest.raises(hf_hub.HubAuthError):
        hf_hub.with_retries(unauthorized, attempts=3, base_delay=0.01)
    assert calls["n"] == 1, "an auth error must fail fast, not retry"


def test_not_found_is_not_retried():
    calls = {"n": 0}

    def missing():
        calls["n"] += 1
        raise RuntimeError("404 Not Found")

    with pytest.raises(hf_hub.HubUnavailable):
        hf_hub.with_retries(missing, attempts=3, base_delay=0.01)
    assert calls["n"] == 1


def test_persistent_failure_raises_hub_unavailable():
    def always_fails():
        raise ConnectionError("nope")

    with pytest.raises(hf_hub.HubUnavailable):
        hf_hub.with_retries(always_fails, attempts=2, base_delay=0.01)


# --------------------------------------------------------------------------- #
# 4. artifact assembly (the thing that actually gets uploaded)
# --------------------------------------------------------------------------- #
def _tiny_model_and_meta():
    model = nn.Linear(4, 2)
    meta = {
        "classes": ["a", "b"], "input": {"size": 32, "channels": 3,
                                         "mean": [0.5, 0.5, 0.5], "std": [0.5, 0.5, 0.5]},
        "metrics": {"accuracy": 0.9}, "test_set_size": 10,
        "split_manifest": {"train": 50, "validation": 10, "test": 10},
        "dataset": "toy", "dataset_key": "toy", "artifact_version": "2.0.0",
        "saved_at": "2026-01-01T00:00:00Z",
    }
    return model, meta


def test_assemble_artifact_writes_all_four_files(tmp_path):
    spec = BY_ID["brain_cnn"]
    model, meta = _tiny_model_and_meta()
    w = tmp_path / "w.pt"
    torch.save(model.state_dict(), w)
    art = hf_hub.assemble_artifact(spec, w, meta, tmp_path / "out")
    names = {p.name for p in art.files()}
    assert names == {"config.json", "metrics.json", "model.pt", "model_card.md"}
    cfg = json.loads((art.directory / "config.json").read_text())
    assert cfg["model_id"] == "brain_cnn"
    assert cfg["classes"] == ["a", "b"]
    assert cfg["weights_file"] == "model.pt"
    met = json.loads((art.directory / "metrics.json").read_text())
    assert met["weights_sha256"] == art.weights_sha256
    assert met["status"] == "TRAINED_UNVERIFIED"
    assert art.weights_sha256 == hf_hub.sha256_file(art.directory / "model.pt")


def test_assemble_refuses_empty_weights(tmp_path):
    spec = BY_ID["brain_cnn"]
    _, meta = _tiny_model_and_meta()
    empty = tmp_path / "empty.pt"
    empty.write_bytes(b"")
    with pytest.raises(hf_hub.HubError):
        hf_hub.assemble_artifact(spec, empty, meta, tmp_path / "out")


def test_model_card_has_no_fabricated_claims(tmp_path):
    spec = BY_ID["brain_cnn"]
    model, meta = _tiny_model_and_meta()
    w = tmp_path / "w.pt"
    torch.save(model.state_dict(), w)
    art = hf_hub.assemble_artifact(spec, w, meta, tmp_path / "out")
    card = (art.directory / "model_card.md").read_text()
    assert "not a medical device" in card.lower()
    assert "0.9" in card                      # the real measured metric
    assert "100%" not in card                 # no invented benchmark


# --------------------------------------------------------------------------- #
# 5. manifest correctness
# --------------------------------------------------------------------------- #
def test_manifest_covers_every_registry_model():
    m = M.build_manifest()
    assert m["totals"]["models"] == 32
    assert m["totals"]["laboratory"] == 20
    assert m["totals"]["medical"] == 6
    assert m["totals"]["agriculture"] == 6
    ids = [e["model_id"] for e in m["models"]]
    assert len(ids) == len(set(ids)), "model ids must be unique"
    assert ids == [s.id for s in ALL_MODELS], "manifest must mirror the registry"


def test_manifest_preserves_all_20_lab_ids():
    m = M.build_manifest()
    labs = sorted(e["model_id"] for e in m["models"] if e["area"] == "lab")
    assert labs == [f"lab{i:02d}" for i in range(1, 21)]


def test_every_manifest_entry_declares_task_and_modality():
    for e in M.build_manifest()["models"]:
        assert e["task"], e["model_id"]
        assert e["modality"], e["model_id"]
        assert e["status"] in M.A.STATUS_VALUES, e["model_id"]


def test_ready_entries_always_carry_artifact_and_input_metadata():
    for e in M.build_manifest()["models"]:
        if e["status"] == "TRAINED_VERIFIED":
            ck = e["checkpoint"]
            assert ck.get("local_path") or ck.get("hf_path"), e["model_id"]
            assert e["input"]["size"], e["model_id"]


def test_blocked_model_is_reported_blocked_with_a_reason():
    e = M.entries_by_id(M.build_manifest())["plant_maskrcnn"]
    assert e["status"] == "BLOCKED"
    assert "parquet" in (e["limitations"] or "").lower() or e["limitations"]


def test_modality_mapping_is_task_driven():
    assert M.modality_for(BY_ID["brain_cnn"]) == "image"
    assert M.modality_for(BY_ID["lab01"]) == "tabular"
    assert M.modality_for(BY_ID["lab03"]) == "sequence"
    assert M.modality_for(BY_ID["lab09"]) == "text"
    assert M.modality_for(BY_ID["lab20"]) == "graph"


def test_manifest_status_is_derived_not_claimed(tmp_path, monkeypatch):
    """A status record claiming VERIFIED with no weights must be downgraded."""
    monkeypatch.setattr(M.A, "STATUS_DIR", tmp_path / "status")
    monkeypatch.setattr(M.A, "META_DIR", tmp_path / "meta")
    monkeypatch.setattr(M.A, "MEDICAL_DIR", tmp_path / "medical")
    monkeypatch.setattr(M.A, "AGRI_DIR", tmp_path / "agri")
    (tmp_path / "status").mkdir(parents=True, exist_ok=True)
    (tmp_path / "status" / "brain_cnn.json").write_text(
        json.dumps({"model_id": "brain_cnn", "status": "TRAINED_VERIFIED"}))
    e = M.build_entry(BY_ID["brain_cnn"])
    assert e["status"] == "ARCH_ONLY"
    assert "no weights" in e["note"]


def test_manifest_roundtrips_through_disk(tmp_path):
    m = M.build_manifest()
    p = M.write_manifest(m, tmp_path / "manifest.json")
    assert M.read_manifest(p)["totals"]["models"] == 32
    assert M.read_manifest(tmp_path / "missing.json") is None


def test_fallback_entry_is_conservative():
    e = M.fallback_entry(BY_ID["brain_unet"])
    assert e["status"] in ("ARCH_ONLY", "TRAINED_UNVERIFIED", "BLOCKED")
    assert e["metrics"] == {} or isinstance(e["metrics"], dict)


# --------------------------------------------------------------------------- #
# 6. resume-capable training state
# --------------------------------------------------------------------------- #
def _fresh_state_dir(monkeypatch, tmp_path):
    monkeypatch.setattr(TS, "CHECKPOINT_DIR", tmp_path / "ckpt")


def test_resume_state_roundtrip(monkeypatch, tmp_path):
    _fresh_state_dir(monkeypatch, tmp_path)
    m = nn.Linear(4, 3)
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=4)
    for ep in range(2):
        m(torch.randn(2, 4)).sum().backward()
        opt.step(); opt.zero_grad(); sched.step()
        TS.save_state("demo", model=m, optimizer=opt, scheduler=sched,
                      epoch=ep + 1, global_step=(ep + 1) * 5, history=[1.0, 0.7])

    m2 = nn.Linear(4, 3)
    opt2 = torch.optim.AdamW(m2.parameters(), lr=1e-3)
    sched2 = torch.optim.lr_scheduler.CosineAnnealingLR(opt2, T_max=4)
    st = TS.load_state("demo", model=m2, optimizer=opt2, scheduler=sched2, total_epochs=6)
    assert st.loaded and st.start_epoch == 2 and st.global_step == 10
    assert st.history == [1.0, 0.7]
    assert torch.allclose(m.weight, m2.weight)
    assert len(opt2.state_dict()["state"]) == len(opt.state_dict()["state"])


def test_finished_state_is_rejected(monkeypatch, tmp_path):
    _fresh_state_dir(monkeypatch, tmp_path)
    m = nn.Linear(3, 2)
    opt = torch.optim.AdamW(m.parameters())
    TS.save_state("demo", model=m, optimizer=opt, epoch=3, global_step=3)
    with pytest.raises(TS.InvalidCheckpoint):
        TS.load_state("demo", model=m, total_epochs=3)


def test_corrupt_state_is_rejected(monkeypatch, tmp_path):
    _fresh_state_dir(monkeypatch, tmp_path)
    p = TS.state_path("demo")
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(b"not a checkpoint at all")
    with pytest.raises(TS.InvalidCheckpoint):
        TS.load_state("demo", model=nn.Linear(2, 2), total_epochs=5)


def test_state_for_a_different_architecture_is_rejected(monkeypatch, tmp_path):
    _fresh_state_dir(monkeypatch, tmp_path)
    TS.save_state("demo", model=nn.Linear(4, 4), optimizer=None, epoch=1, global_step=1)
    with pytest.raises(TS.InvalidCheckpoint):
        TS.load_state("demo", model=nn.Linear(7, 7), total_epochs=5)


def test_missing_state_raises(monkeypatch, tmp_path):
    _fresh_state_dir(monkeypatch, tmp_path)
    with pytest.raises(TS.InvalidCheckpoint):
        TS.load_state("nope", model=nn.Linear(2, 2), total_epochs=1)


def test_state_write_is_atomic_and_clearable(monkeypatch, tmp_path):
    _fresh_state_dir(monkeypatch, tmp_path)
    TS.save_state("demo", model=nn.Linear(2, 2), optimizer=None, epoch=1, global_step=1)
    assert TS.has_state("demo")
    assert not list((tmp_path / "ckpt" / "demo").glob("*.tmp")), "no .tmp left behind"
    TS.clear("demo")
    assert not TS.has_state("demo")


def test_trainer_exposes_a_resume_flag():
    r = subprocess.run([sys.executable, "scripts/train_models.py", "--help"],
                       capture_output=True, text=True, cwd=ROOT)
    assert r.returncode == 0
    assert "--resume" in r.stdout


# --------------------------------------------------------------------------- #
# 7. dynamic forms
# --------------------------------------------------------------------------- #
def test_form_kind_matches_the_declared_contract():
    from src.forms import prediction_forms as PF
    e = M.entries_by_id(M.build_manifest())
    assert PF.form_kind(BY_ID["brain_cnn"], e["brain_cnn"]) == "image"
    assert PF.form_kind(BY_ID["brain_unet"], e["brain_unet"]) == "image"
    assert PF.form_kind(BY_ID["brain_yolo"], e["brain_yolo"]) == "image"
    assert PF.form_kind(BY_ID["lab01"], e["lab01"]) == "tabular"
    assert PF.form_kind(BY_ID["lab03"], e["lab03"]) == "sequence"
    assert PF.form_kind(BY_ID["lab09"], e["lab09"]) == "text"
    assert PF.form_kind(BY_ID["lab11"], e["lab11"]) == "text_prompt"
    assert PF.form_kind(BY_ID["lab17"], e["lab17"]) == "image_pair"
    assert PF.form_kind(BY_ID["lab19"], e["lab19"]) == "tabular_denoise"
    assert PF.form_kind(BY_ID["lab20"], e["lab20"]) == "graph"


def test_form_schema_exposes_limits_and_classes():
    from src.forms import prediction_forms as PF
    e = M.entries_by_id(M.build_manifest())
    s = PF.schema(BY_ID["brain_cnn"], e["brain_cnn"])
    assert s["modality"] == "image"
    assert s["max_image_mb"] > 0
    assert s["max_text_chars"] > 0
    assert s["classes"]


def test_an_image_model_is_never_given_a_text_form():
    """The generator must follow the contract, not a generic default."""
    from src.forms import prediction_forms as PF
    e = M.entries_by_id(M.build_manifest())
    for mid in ("brain_cnn", "brain_unet", "brain_yolo", "plant_cnn"):
        assert PF.form_kind(BY_ID[mid], e[mid]) == "image"
    for mid in ("lab09", "lab10"):
        assert PF.form_kind(BY_ID[mid], e[mid]) in ("text", "text_prompt")


def test_image_validation_rejects_non_images():
    from src.services.prediction_service import image_bytes_ok
    ok, err = image_bytes_ok(b"definitely not an image")
    assert not ok and err
    ok, err = image_bytes_ok(b"")
    assert not ok


def test_image_validation_rejects_oversized_payloads():
    from src.services.prediction_service import image_bytes_ok
    ok, err = image_bytes_ok(b"x" * (16 * 1024 * 1024), max_mb=15.0)
    assert not ok and "larger" in err


# --------------------------------------------------------------------------- #
# 8. the app's honesty + inference-only guarantees
# --------------------------------------------------------------------------- #
def test_smoke_script_passes():
    r = subprocess.run([sys.executable, "scripts/smoke_app.py"],
                       capture_output=True, text=True, cwd=ROOT)
    assert r.returncode == 0, r.stdout + r.stderr


def test_app_entry_points_exist_and_parse():
    import ast
    for rel in ("app.py", "streamlit_app.py"):
        p = ROOT / rel
        assert p.is_file(), rel
        ast.parse(p.read_text(encoding="utf-8"))


def test_no_inference_module_imports_an_optimiser():
    for rel in ("src/inference.py", "src/lab_inference.py", "src/hf_loader.py",
                "src/services/model_loader.py", "src/services/prediction_service.py"):
        src = (ROOT / rel).read_text(encoding="utf-8")
        assert "torch.optim" not in src, rel
        assert ".backward(" not in src, rel


def test_cv_registry_still_has_twelve_models():
    assert len(CV_MODELS) == 12
    assert len(LAB_BY_INDEX) == 20


def test_manifest_service_reports_no_hardcoded_counts():
    src = (ROOT / "src/services/manifest_service.py").read_text(encoding="utf-8")
    assert "KararDeepVision" not in src
    assert "MODEL_COUNT = len(ALL_MODELS)" in src
