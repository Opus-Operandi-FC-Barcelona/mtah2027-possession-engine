"""Issue #56: multi-scale context windows and goalkeeper-depth context columns (off by default)."""
from __future__ import annotations

import dataclasses

import numpy as np

from mtah2027.models import specialist_features as sfe


def test_default_config_is_unchanged():
    fp = sfe.FeatureParams.from_config()
    assert fp.context_windows == (50, 100) and not fp.gk_context and fp.gk_span == 150


def test_gk_context_is_block_local_nan_safe_and_deterministic():
    n = 600
    x = np.zeros((n, 2), np.float32)
    x[:, 0] = np.where(np.arange(n) < 300, 5.0, 15.0)       # home GK steps up 10 m at t = 300
    x[:, 1] = 90.0
    x[100:110, 0] = np.nan
    names = list(sfe.GK_SIGNALS)
    blocks = [(0, 299), (300, 599)]
    a, na = sfe.context_features(x, names, blocks, (150,), signals=sfe.GK_SIGNALS)
    b, _ = sfe.context_features(x, names, blocks, (150,), signals=sfe.GK_SIGNALS)
    np.testing.assert_array_equal(a, b)
    assert na == ["ctx:cp:gk_xy_home_x:cm150", "ctx:cp:gk_xy_home_x:dd150",
                  "ctx:cp:gk_xy_away_x:cm150", "ctx:cp:gk_xy_away_x:dd150"]
    assert np.isfinite(a).all()
    # no window crosses the block boundary: the step at 300 is invisible inside each block
    assert np.allclose(a[:300, 0], 5.0) and np.allclose(a[300:, 0], 15.0)
    # without blocks the step shows as a positive change before it and none for the constant team
    c, _ = sfe.context_features(x, names, None, (150,), signals=sfe.GK_SIGNALS)
    assert c[290, 1] > 4.0 and np.allclose(c[:, 3], 0.0)


def test_build_feature_matrix_gk_flag_adds_columns_only(synthetic_kin):
    from mtah2027.pipeline import compute_front_end, heuristic_emissions

    td, spec = synthetic_kin["td"], synthetic_kin["spec"]
    blocks = [spec.half1, spec.half2_play]
    fe = compute_front_end(td.players_xy, td.team_of_slot, td.ball_xy, play_blocks=blocks, collective=True)
    em = heuristic_emissions(fe, td.fps)
    base = sfe.build_feature_matrix(fe, em, play_blocks=blocks)
    gk = sfe.build_feature_matrix(fe, em, play_blocks=blocks,
                                  params=dataclasses.replace(sfe.FeatureParams(), gk_context=True))
    assert gk.names[:len(base.names)] == base.names and len(gk.names) == len(base.names) + 4
    np.testing.assert_array_equal(gk.x[:, :base.x.shape[1]], base.x)
    wide = sfe.build_feature_matrix(fe, em, play_blocks=blocks, params=dataclasses.replace(
        sfe.FeatureParams(), context_windows=(25, 75, 175, 250)))
    assert any(k.endswith("cm250") for k in wide.names)
