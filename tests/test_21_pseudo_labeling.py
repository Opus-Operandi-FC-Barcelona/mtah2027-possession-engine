"""Issue #21: block-masked self-supervised dataset built from the anchors of Specialist A."""
from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from mtah2027.config import ConfigError, params_from_mapping
from mtah2027.models import pseudo_labeling as pl

FPS = 25.0


def test_params_from_yaml_typos_and_validation():
    p = pl.PseudoLabelParams.from_config()
    assert (p.block_min_s, p.block_max_s) == (5.0, 45.0)
    with pytest.raises(ConfigError):
        params_from_mapping(pl.PseudoLabelParams, {"block_mins": 3}, "pseudo_labeling")
    with pytest.raises(ValueError):
        pl.PseudoLabelParams(block_min_s=50.0, block_max_s=10.0)
    with pytest.raises(ValueError):
        pl.PseudoLabelParams(cover_frac=0.0)


def _world(n=12000):
    visible = np.ones(n, bool)
    visible[3000:3500] = False                       # an existing real occlusion
    anchor = np.ones(n, bool)
    anchor[5000:5600] = False                        # a stretch without confident frames
    return visible, anchor, [(0, 5999), (6000, n - 1)]


def test_draw_blocks_respects_every_constraint():
    visible, anchor, periods = _world()
    p = pl.PseudoLabelParams(cover_frac=0.3, guard_s=5.0)
    blocks = pl.draw_blocks(anchor, visible, periods, FPS, p, np.random.default_rng(0))
    assert blocks and blocks == sorted(blocks)
    covered = 0
    for a, b in blocks:
        dur = (b - a + 1) / FPS
        assert 5.0 - 0.1 <= dur <= 45.0 + 0.1
        assert visible[a:b + 1].all()                                    # only visible frames are masked
        assert anchor[a:b + 1].mean() >= p.min_anchor_cover
        assert any(s <= a and b <= e for s, e in periods)                # never crosses a period border
        covered += b - a + 1
    for (a1, b1), (a2, b2) in zip(blocks, blocks[1:]):
        assert a2 - b1 > p.guard_s * FPS - 1                             # guard between blocks
    assert covered >= 0.3 * visible.sum() * 0.8
    again = pl.draw_blocks(anchor, visible, periods, FPS, p, np.random.default_rng(0))
    assert again == blocks                                               # deterministic
    nothing = np.zeros(len(visible), bool)
    assert pl.draw_blocks(anchor, nothing, periods, FPS, p, np.random.default_rng(0)) == []


def test_block_runs_of():
    m = np.array([0, 1, 1, 0, 1, 0, 0, 1, 1], bool)
    assert pl.block_runs_of(m) == [(1, 2), (4, 4), (7, 8)]


def test_pseudo_dataset_masks_the_ball_and_recomputes_memory(synthetic_kin):
    from mtah2027.models.specialist_features import build_feature_matrix
    from mtah2027.pipeline import compute_front_end, heuristic_emissions

    td, spec = synthetic_kin["td"], synthetic_kin["spec"]
    blocks = [spec.half1, spec.half2_play]
    raw = np.asarray(td.ball_xy, dtype=np.float32)
    labels = np.asarray(synthetic_kin["labels"])

    def rebuild(ball):
        return compute_front_end(td.players_xy, td.team_of_slot, ball, play_blocks=blocks, collective=True)

    def heur(fe):
        return heuristic_emissions(fe, td.fps)

    fe0 = rebuild(raw)
    names = build_feature_matrix(fe0, heur(fe0)).names
    gt_code = np.select([labels == "H", labels == "A", labels == "D"], [0, 1, 2], default=-1)
    anchor = (gt_code >= 0) & fe0.visible
    p = pl.PseudoLabelParams(n_variants=2, cover_frac=0.2, stride=5, min_rows=1, min_anchor_cover=0.5)
    ds = pl.build_pseudo_dataset(anchor, np.where(anchor, gt_code, -1), fe0.visible, raw, rebuild,
                                 play_blocks=blocks, heuristic=heur, fps=td.fps, params=p, names=names)
    assert len(ds) > 50 and ds.x.shape == (len(ds), len(names)) and ds.names == names
    assert ds.report["blocks"] > 0 and ds.report["rows"] == len(ds)
    assert set(np.unique(ds.y)) <= {0, 1, 2} and (ds.weight == p.weight).all()
    # every row lies in a masked block: its ball is gone (memory: time since the ball was seen > 0)
    since = ds.x[:, names.index("bm:since_seen_s")]
    visible_col = ds.x[:, names.index("visible")]
    assert (visible_col == 0).all() and (since > 0).all()
    # the same rows built with the real ball would see it
    clean = build_feature_matrix(fe0, heur(fe0))
    assert (clean.x[:, names.index("visible")] > 0).mean() > 0.5
    with pytest.raises(ValueError):
        pl.build_pseudo_dataset(anchor, np.where(anchor, gt_code, -1), fe0.visible, raw, rebuild,
                                play_blocks=blocks, heuristic=heur, fps=td.fps, params=p,
                                names=names[:-1])
    no_anchor = np.zeros(len(raw), bool)
    empty = pl.build_pseudo_dataset(no_anchor, np.full(len(raw), -1), fe0.visible, raw, rebuild,
                                    play_blocks=blocks, heuristic=heur, fps=td.fps,
                                    params=dataclasses.replace(p, max_attempts=200), names=names)
    assert len(empty) == 0 and empty.x.shape == (0, len(names))
