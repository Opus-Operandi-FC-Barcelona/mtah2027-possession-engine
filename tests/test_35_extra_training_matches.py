"""Issue #35 -- SkillCorner / PFF matches as extra TRAINING rows of the specialists."""
from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from mtah2027.models import specialist_data as sd

PARAMS = dataclasses.replace(sd.TrainingDataParams(), seeds=(0, 1), stride=20, min_period_frames=100,
                             extra_sources=("skillcorner_", "pff_"))


@pytest.fixture(scope="module")
def bench(external_like_dir, tmp_path_factory):
    out = tmp_path_factory.mktemp("bench35")
    base = np.load(sorted(external_like_dir.glob("*.npz"))[0], allow_pickle=True)
    arrays = {k: base[k] for k in base.files}
    np.savez(out / "idsse_fake.npz", **arrays)
    n = len(arrays["gt"])
    observed = np.ones(n, bool)
    observed[::7] = False
    sc = dict(arrays, ball_xy=np.where(observed[:, None], arrays["ball_xy"], np.nan).astype(np.float32),
              observed=observed)
    np.savez(out / "skillcorner_1.npz", **sc)
    visible = (np.arange(n) // 400) % 2 == 0          # blocks, so the gaps are real occlusions
    np.savez(out / "pff_1.npz", **dict(arrays, ball_visible=visible))
    np.savez(out / "pff_2.npz", **dict(arrays, ball_visible=visible))
    return out


def test_file_selection_keeps_main_matches_first(bench):
    names = [f.stem for f in sd.list_external_matches(bench, PARAMS)]
    assert names == ["idsse_fake", "skillcorner_1", "pff_1", "pff_2"]
    plain = dataclasses.replace(PARAMS, extra_sources=())
    assert [f.stem for f in sd.list_external_matches(bench, plain)] == ["idsse_fake"]
    capped = dataclasses.replace(PARAMS, pff_max_matches=1)
    assert [f.stem for f in sd.list_external_matches(bench, capped)].count("pff_2") == 0
    assert sd.source_of("skillcorner_9") == "skillcorner" and sd.source_of("pff_9") == "pff"
    assert sd.source_of("idsse_J03") == "idsse" and sd.source_of("metrica_1") == "idsse"


def test_native_sources_are_not_degraded_and_respect_masks(bench):
    sc = sd.rows_for_match(bench / "skillcorner_1.npz", 0, stride=20, min_period_frames=100)[0]
    assert sc.source == "skillcorner" and len(sc.y_hid) == 0 and len(sc.y_vis) > 50   # ball missing <=> X
    pff = sd.rows_for_match(bench / "pff_1.npz", 0, stride=20, min_period_frames=100)[0]
    assert pff.source == "pff" and len(pff.y_hid) > 50 and len(pff.y_vis) > 50          # real occlusions


def test_training_set_weights_and_dead_rows(bench):
    ts = sd.build_training_set(bench, params=PARAMS)
    assert ts.summary()["sources"] == {"pff": ["pff_1", "pff_2"], "skillcorner": ["skillcorner_1"]}
    # native sources: one variant per swap, the main match: one per seed and swap
    assert sum(p.source == "skillcorner" for p in ts.parts) == 2
    assert sum(p.source == "idsse" for p in ts.parts) == 4
    x, y, g, w = ts.stack_weighted("vis")
    assert w is None and len(y) == len(x) == len(g)
    sub = ts.select(["idsse", "skillcorner"], {"skillcorner": 0.25}, drop_dead=["skillcorner"])
    x2, y2, g2, w2 = sub.stack_weighted("vis")
    sc_rows = g2 == "skillcorner_1"
    assert set(np.unique(w2[sc_rows])) == {0.25} and set(np.unique(w2[~sc_rows])) == {1.0}
    assert not (y2[sc_rows] == sd.D).any() and (y2[~sc_rows] == sd.D).any()
    assert "pff_1" not in set(g2)
    assert sub.summary()["source_weights"] == {"skillcorner": 0.25}


def test_default_configuration_is_the_adopted_mixed_set():
    p = sd.TrainingDataParams.from_config()
    assert p.extra_sources == ("pff_",) and p.pff_weight == 0.5 and p.pff_max_matches == 20
    assert p.skillcorner_weight == 1.0 and not p.skillcorner_drop_dead


def test_input_stats_are_aggregates():
    import compare_input_stats as cis

    n, fps = 5000, 25.0
    ball = np.ones((n, 2), np.float32)
    ball[1000:1250] = np.nan            # one 10 s gap
    ball[3000:3050] = np.nan            # one 2 s gap
    players = np.ones((n, 22, 2), np.float32)
    players[:100, :5] = np.nan
    st = cis.input_stats(ball, players, [(0, n - 1)], fps)
    assert st["frames"] == n and st["missing"] == 300 and sorted(st["gaps_s"]) == [2.0, 10.0]
    pooled = cis.pool([st, st])
    assert pooled["matches"] == 2 and pooled["ball_missing_pct"] == 6.0
    assert pooled["hidden_mass_pct_lt5_5to20_20to60_gt60"] == [16.7, 83.3, 0.0, 0.0]
    assert pooled["frames_lt20_players_pct"] == 2.0 and pooled["gaps_per_10min"] == 6.0
    assert "gaps_s" not in pooled
