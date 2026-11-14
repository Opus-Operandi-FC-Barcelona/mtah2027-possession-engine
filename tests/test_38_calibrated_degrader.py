"""Issue #38: degrader calibrated to the label-free gap statistics of the competition ATD."""
from __future__ import annotations

import dataclasses
import importlib
import sys
from pathlib import Path

import numpy as np
import pytest

from mtah2027.config import load_section
from mtah2027.models.external_degrader import DegraderParams, degrade_match

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "tests"))
from test_19_external_degrader import FPS, _match  # noqa: E402


@pytest.fixture(scope="module")
def match():
    return _match(n=120000)


def _runs(mask):
    d = np.diff(np.concatenate([[0], mask.astype(np.int8), [0]]))
    return np.flatnonzero(d == 1), np.flatnonzero(d == -1)


def test_config_default_and_target_section():
    assert DegraderParams.from_config() == DegraderParams()
    assert DegraderParams().gap_model == "lognormal"
    t = load_section("external_degrader_target")
    for key in ("ball_missing_pct", "gaps_per_10min", "players_mean", "frames_lt20_players_pct"):
        assert float(t[key]) > 0
    assert len(t["gap_s_p50_p90_p99"]) == 3 and len(t["hidden_mass_pct_lt5_5to20_20to60_gt60"]) == 4
    assert sum(t["hidden_mass_pct_lt5_5to20_20to60_gt60"]) == pytest.approx(100.0, abs=0.2)


def test_param_validation():
    with pytest.raises(ValueError):
        DegraderParams(gap_model="weibull")
    with pytest.raises(ValueError):
        DegraderParams(gap_min_s=30.0, gap_max_s=10.0)
    with pytest.raises(ValueError):
        DegraderParams(group_drop_players=(0, 3))


def test_lognormal_blocks_are_separated_and_have_the_configured_median(match):
    players, team, ball, gt = match
    p = DegraderParams(gap_model="lognormal", gap_median_s=11.5, gap_sigma=0.5, gap_min_s=5.0, guard_s=8.0)
    out = degrade_match(players, team, ball, gt, fps=FPS, seed=1, params=p, artefacts=False)
    starts, ends = _runs(out.hidden)
    assert len(starts) >= 20
    assert np.all(starts[1:] - ends[:-1] >= 8.0 * FPS - 1)                 # guard kept between blocks
    secs = np.array(out.report.block_seconds)
    assert (secs < 4.9).sum() <= 1
    assert 8.0 < np.median(secs) < 16.0


def test_classes_model_reproduces_issue_19_blocks(match):
    players, team, ball, gt = match
    p = DegraderParams(hide_frac=0.41, gap_model="classes", guard_s=0.0)
    out = degrade_match(players, team, ball, gt, fps=FPS, seed=2, params=p, artefacts=False)
    secs = np.array(out.report.block_seconds)
    assert secs.max() <= 120.0 + 1e-6
    assert out.report.hidden_alive_share == pytest.approx(0.41, abs=2e-3)


def test_group_dropouts_lose_several_players_together(match):
    players, team, ball, gt = match
    base = dict(hide_frac=0.0, noise_player_m=0.0, noise_ball_m=0.0, cut_rate_per_min=0.0,
                dup_rate_per_min=0.0, drop_rate_per_min=0.0)
    off = degrade_match(players, team, ball, gt, fps=FPS, seed=3,
                        params=DegraderParams(**base, group_drop_rate_per_min=0.0))
    on = degrade_match(players, team, ball, gt, fps=FPS, seed=3, params=DegraderParams(
        **base, group_drop_rate_per_min=2.0, group_drop_players=(4, 5), group_drop_duration_s=(5.0, 10.0)))
    n_off = np.isfinite(off.players_xy[..., 0]).sum(axis=1)
    n_on = np.isfinite(on.players_xy[..., 0]).sum(axis=1)
    assert n_off.min() == players.shape[1]
    assert n_on.mean() < n_off.mean() - 0.3
    assert (n_on <= players.shape[1] - 4).any()                    # at least four slots missing at once


def test_group_stream_off_does_not_touch_the_random_stream(match):
    players, team, ball, gt = match
    off = dict(group_drop_rate_per_min=0.0)
    a = degrade_match(players, team, ball, gt, fps=FPS, seed=4,
                      params=DegraderParams(**off, group_drop_players=(3, 6)))
    b = degrade_match(players, team, ball, gt, fps=FPS, seed=4,
                      params=DegraderParams(**off, group_drop_players=(2, 9)))
    assert np.array_equal(a.hidden, b.hidden)
    assert np.array_equal(a.players_xy, b.players_xy, equal_nan=True)


def test_degrader_is_deterministic(match):
    players, team, ball, gt = match
    p = dataclasses.replace(DegraderParams.from_config(), group_drop_rate_per_min=1.0)
    a = degrade_match(players, team, ball, gt, fps=FPS, seed=9, params=p)
    b = degrade_match(players, team, ball, gt, fps=FPS, seed=9, params=p)
    assert np.array_equal(a.players_xy, b.players_xy, equal_nan=True)
    assert np.array_equal(a.ball_xy, b.ball_xy, equal_nan=True)


def test_compare_input_stats_prints_target(match, capsys):
    cis = importlib.import_module("compare_input_stats")
    players, team, ball, gt = match
    out = degrade_match(players, team, ball, gt, fps=FPS, seed=0)
    blocks = [(1000, len(gt) // 2 - 1001), (len(gt) // 2 + 1000, len(gt) - 1)]
    stats = cis.pool([cis.input_stats(out.ball_xy, out.players_xy, blocks, FPS, graded_mask=gt != "X")])
    target = cis.target_from_config()
    assert target is not None
    cis.print_target(target, stats)
    text = capsys.readouterr().out
    assert "target" in text and "ball missing %" in text and "mass >60 s %" in text


def test_pff_heldout_script_reports_missing_data(tmp_path, capsys):
    mod = importlib.import_module("eval_pff_heldout")
    assert mod.main(["--bench-dir", str(tmp_path)]) == 1
    assert "no PFF" in capsys.readouterr().out


def test_legacy_section_is_the_issue_19_profile_and_selected_for_training():
    from mtah2027.models.specialist_data import TrainingDataParams

    legacy = DegraderParams.from_config(section="external_degrader_legacy")
    assert (legacy.hide_frac, legacy.gap_model, legacy.guard_s) == (0.41, "classes", 0.0)
    assert (legacy.drop_rate_per_min, legacy.drop_duration_s, legacy.group_drop_rate_per_min) == (
        0.5, (2.0, 20.0), 0.0)
    cal = DegraderParams.from_config()
    assert (legacy.noise_player_m, legacy.cut_rate_per_min) == (cal.noise_player_m, cal.cut_rate_per_min)
    assert cal.gap_model == "lognormal" and cal.hide_frac > 0.41
    # Issue #48: the adopted specialists are trained with the calibrated degrader (affinity 33); the Issue #19
    # profile produced models/specialists_mixed.json
    assert TrainingDataParams.from_config().degrader_section == "external_degrader"


def test_calibrated_degrader_hits_the_configured_target_on_a_long_match():
    cis = importlib.import_module("compare_input_stats")
    n = 600000
    players, team, ball, gt = _match(n=n, s=22)
    target = cis.target_from_config()
    out = degrade_match(players, team, ball, gt, fps=FPS, seed=0)
    blocks = [(1000, n // 2 - 1001), (n // 2 + 1000, n - 1)]
    stats = cis.pool([cis.input_stats(out.ball_xy, out.players_xy, blocks, FPS, graded_mask=gt != "X")])
    assert stats["gaps_per_10min"] == pytest.approx(target["gaps_per_10min"], rel=0.3)
    assert stats["gap_s_p50_p90_p99"][0] == pytest.approx(target["gap_s_p50_p90_p99"][0], rel=0.3)
    assert stats["hidden_mass_pct_lt5_5to20_20to60_gt60"][0] < 2.0       # no short gaps, like the ATD
    assert abs(stats["players_mean"] - target["players_mean"]) < 1.2


def test_calibrated_model_file_is_selectable_and_default_unchanged():
    from mtah2027.models import specialist_fusion as sf

    assert sf.load_specialists(ROOT / "models" / "specialists_calib.json").a is not None
    default = (ROOT / "models" / "specialists.json").read_bytes()
    # Issue #48: default = model trained at hidden<->dead affinity 33; the former default stays as mixed
    assert default == (ROOT / "models" / "specialists_gap.json").read_bytes()
    assert sf.load_specialists(ROOT / "models" / "specialists_mixed.json").b is not None
