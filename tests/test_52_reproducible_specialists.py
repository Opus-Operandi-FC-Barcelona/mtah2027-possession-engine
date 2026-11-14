"""Issue #52: the submission depends only on the repository (RULES.md, official issue #11).

``--specialists auto`` must load the committed ``models/specialists.json`` even when the external benchmark
cache sits next to the repository, the committed file must agree with ``configs/default.yaml``, and the
pipeline diagnostics must name the emissions that were actually used.
"""
from __future__ import annotations

import json

import pytest

from mtah2027.config import REPO_ROOT, load_section
from mtah2027.models import specialist_fusion as sf

MODEL = REPO_ROOT / "models" / "specialists.json"


def test_auto_loads_the_committed_file_even_with_external_matches_on_disk(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(sf, "find_bench_dir", lambda explicit=None: tmp_path)
    monkeypatch.setattr(sf, "train_specialists", lambda *a, **k: calls.append("train") or "trained")
    monkeypatch.setattr(sf, "load_specialists", lambda path, cfg=None: calls.append("load") or "loaded")
    model = tmp_path / "specialists.json"
    model.write_text("{}", encoding="utf-8")
    assert sf.resolve_specialists("auto", model_path=model) == "loaded"
    assert calls == ["load"]
    assert sf.resolve_specialists("train", model_path=model) == "trained"
    assert sf.resolve_specialists("auto", model_path=model, prefer_file=False) == "trained"


def test_auto_trains_only_when_no_committed_file(monkeypatch, tmp_path):
    monkeypatch.setattr(sf, "find_bench_dir", lambda explicit=None: tmp_path)
    monkeypatch.setattr(sf, "train_specialists", lambda *a, **k: "trained")
    assert sf.resolve_specialists("auto", model_path=tmp_path / "missing.json") == "trained"
    # synthetic CI runs: auto never trains
    assert sf.resolve_specialists("auto", model_path=tmp_path / "missing.json", train_fallback=False) is None


def test_generate_submission_does_not_force_training_order():
    source = (REPO_ROOT / "scripts" / "generate_submission.py").read_text(encoding="utf-8")
    assert "prefer_file=args.synthetic" not in source


@pytest.mark.skipif(not MODEL.is_file(), reason="no committed specialists file")
def test_committed_model_matches_the_training_configuration():
    training = json.loads(MODEL.read_text(encoding="utf-8"))["training"]
    cfg = load_section("specialist_data")
    extra = list(cfg.get("extra_sources") or [])
    sources = training.get("sources") or {}
    weights = training.get("source_weights") or {}
    if any(str(s).startswith("pff") for s in extra):
        n_pff = len(sources.get("pff", []))
        cap = int(cfg.get("pff_max_matches", 0) or 0)
        assert n_pff > 0 and (cap == 0 or n_pff == cap)
        assert weights.get("pff") == pytest.approx(float(cfg.get("pff_weight", 1.0)))
    else:
        assert not sources.get("pff")
    if not any(str(s).startswith("skillcorner") for s in extra):
        assert not sources.get("skillcorner")


def test_target_match_adaptation_stays_off():
    # official issue #2: nothing is trained or adapted on the provided game
    fusion = load_section("specialist_fusion")
    assert "use_pseudo_labels" in fusion and fusion["use_pseudo_labels"] is False
