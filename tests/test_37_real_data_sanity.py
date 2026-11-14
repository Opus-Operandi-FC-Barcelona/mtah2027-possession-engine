"""Issue #37: the aggregate-only ingestion report works on the synthetic match and never prints rows."""
from __future__ import annotations

import json

import numpy as np
import pytest
import real_data_sanity as rds

from mtah2027.config import MatchConfig

H, A, D, X = 0, 1, 2, 3


def _blocks(spec) -> MatchConfig:
    return MatchConfig(n_video_frames=spec.n_frames, half1=spec.half1, half2_dead=spec.half2_dead,
                       half2_play=spec.half2_play)


@pytest.fixture(scope="module")
def report(synthetic_dir, synthetic_spec):
    return rds.build_report(synthetic_dir, _blocks(synthetic_spec), sources=("tactical",))


def test_report_sections_and_shape(report, synthetic_spec):
    assert report["atd"]["frames_match_expected"] and "x" in report["atd"]["ball_channels"]
    tr = report["tracking"]
    assert tr["frames_parsed"] == synthetic_spec.n_frames
    assert 0.0 <= tr["ball_missing_pct"]["graded"] <= 100.0
    assert report["homography"]["tactical"]["valid_pct_graded"] > 0
    st = report["smart_tagging"]
    assert st["set_pieces"] > 0 and "set pieces" in st["by_code"] and st["instances_in_xml"] == st["events"]
    assert report["pipeline"]["route"] in ("heuristic", "specialists")
    assert "set_piece_anchors" in report["pipeline"]["stages"]


def test_d_diagnostic_is_consistent(report):
    d = report["d_diagnostic"]
    assert d["all_graded"]["frames_pct_of_graded"] == 100.0
    parts = d["ball_visible"]["frames_pct_of_graded"] + d["ball_hidden"]["frames_pct_of_graded"]
    assert parts == pytest.approx(100.0, abs=0.05)
    for cell in d.values():
        assert cell["D_pct"] + cell["H_pct"] + cell["A_pct"] <= 100.05


def test_report_is_json_and_has_no_frame_rows(report):
    text = rds.format_report(report)
    assert json.loads(text)["atd"]
    assert text.count("\n") < 600           # aggregates only: a few hundred lines at most


def test_d_table_unit():
    lab = np.array([H, D, D, A, D, H, H, D])
    graded = np.ones(8, bool)
    vis = np.array([1, 1, 0, 0, 0, 1, 1, 1], bool)
    out = rds.d_table(lab, graded, vis, np.full(8, 22), np.array([0, 1, 1, 0, 0, 0, 0, 0], bool))
    assert out["ball_hidden"]["D_pct"] == pytest.approx(66.67, abs=0.01)
    assert out["in_set_piece_window"]["D_pct"] == 100.0
    assert "players_lt20" not in out


def test_onpitch_counts_ignores_the_bench():
    xy = np.full((1, 4, 2), np.nan)
    xy[0] = [[50, 30], [50, -10], [20, 20], [np.nan, np.nan]]
    nh, na = rds.onpitch_counts(xy, np.array([0, 0, 1, 1]), (105.3, 68.0))
    assert (nh[0], na[0]) == (1, 1)
