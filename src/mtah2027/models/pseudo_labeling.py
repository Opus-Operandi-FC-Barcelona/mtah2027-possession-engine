"""Self-supervised pseudo-labelling with block masking on the match itself (Issue #21).

The competition match has no labels, but Specialist A (Issue #20) is *ultra-confident* (``P > 0.98`` for at
least 2 s) on a good part of the frames whose ball is visible.  Those spans are algorithmic pseudo-labels
(``rules``: produced by an algorithm on the provided game, no hand labelling).  To teach the ball-absent
specialist (B) what this particular match looks like when the ball cannot be seen, we **mask the ball**:

1. Blocks of 5-45 s (log-uniform duration) are drawn inside the graded periods where the ball is visible and
   the anchors cover enough of the block (``min_anchor_cover``); they never overlap and keep a guard of
   ``guard_s`` seconds, so that the memory features still have a visible ball on both sides.
2. The ball of the raw tracking is erased in the blocks and the whole front end is **recomputed** on the
   masked copy (kinematics, ball kinematics, dead-ball signals, collective physics and -- from the start of
   each artificial mask -- the ball memory of Issue #18), so that every feature is exactly what it would be
   had the ball really been lost.
3. The rows of the masked frames are labelled with the anchor labels of Specialist A (frames of the block
   that are not anchors are skipped) and receive the sample ``weight`` of the configuration.

``n_variants`` independent maskings (seeded) are produced.  The result is a training set for B drawn from the
target match; on the external benchmark every held-out match plays the target: its pseudo-labels come only
from its own degraded tracking, never from its ground truth.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Sequence

import numpy as np

from mtah2027.config import FPS, load_section, params_from_mapping
from mtah2027.features._utils import mask_runs
from mtah2027.models.specialist_features import build_feature_matrix


@dataclass(frozen=True)
class PseudoLabelParams:
    block_min_s: float = 5.0
    block_max_s: float = 45.0
    cover_frac: float = 0.35          # target share of the visible graded frames to mask per variant
    min_anchor_cover: float = 0.5     # a block needs at least this share of anchor frames
    guard_s: float = 5.0              # minimum visible distance between two blocks
    n_variants: int = 2
    stride: int = 10                  # frames between two rows
    weight: float = 1.0               # sample weight of a pseudo-labelled row
    max_attempts: int = 20000
    seed: int = 0
    min_rows: int = 300               # fewer pseudo rows than this -> no adaptation
    include_dead: bool = True         # also keep the anchors labelled D (else only H / A possessions)

    def __post_init__(self):
        if not 0 < self.block_min_s <= self.block_max_s:
            raise ValueError("need 0 < block_min_s <= block_max_s")
        if not 0 < self.cover_frac <= 1 or not 0 <= self.min_anchor_cover <= 1:
            raise ValueError("cover_frac in (0, 1], min_anchor_cover in [0, 1]")

    @classmethod
    def from_config(cls, path=None) -> "PseudoLabelParams":
        return params_from_mapping(cls, load_section("pseudo_labeling", path), "pseudo_labeling")


@dataclass
class PseudoDataset:
    x: np.ndarray                     # [M, F] float32 features of the masked frames
    y: np.ndarray                     # [M] int64 anchor labels (0 H, 1 A, 2 D)
    weight: np.ndarray                # [M] float64
    names: list[str]
    blocks: list[list[tuple[int, int]]] = field(default_factory=list)   # per variant, inclusive frame pairs
    report: dict = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.y)


def draw_blocks(anchor: np.ndarray, visible: np.ndarray, periods: Sequence[tuple[int, int]], fps: float,
                params: PseudoLabelParams, rng: np.random.Generator) -> list[tuple[int, int]]:
    """Non-overlapping masked blocks ``(first, last)`` (inclusive) inside the visible anchored stretches."""
    n = len(visible)
    inside = np.zeros(n, bool)
    for a, b in periods:
        inside[a:b + 1] = True
    usable = inside & visible
    target = params.cover_frac * int(usable.sum())
    csum = np.concatenate([[0], np.cumsum(anchor.astype(np.int64))])
    vsum = np.concatenate([[0], np.cumsum(usable.astype(np.int64))])
    taken = np.zeros(n, bool)
    guard = int(round(params.guard_s * fps))
    lo, hi = np.log(params.block_min_s), np.log(params.block_max_s)
    blocks: list[tuple[int, int]] = []
    covered = 0
    cand = np.flatnonzero(usable)
    if not len(cand):
        return blocks
    for _ in range(params.max_attempts):
        if covered >= target:
            break
        dur = int(round(np.exp(rng.uniform(lo, hi)) * fps))
        start = int(cand[rng.integers(0, len(cand))])
        end = start + dur
        if end >= n or vsum[end + 1] - vsum[start] < dur + 1:        # every frame usable (visible, graded)
            continue
        if csum[end + 1] - csum[start] < params.min_anchor_cover * (dur + 1):
            continue
        a0, b0 = max(0, start - guard), min(n, end + guard + 1)
        if taken[a0:b0].any():
            continue
        taken[start:end + 1] = True
        blocks.append((start, end))
        covered += dur + 1
    return sorted(blocks)


def build_pseudo_dataset(
        anchor_mask: np.ndarray, anchor_label: np.ndarray, visible: np.ndarray, raw_ball_xy: np.ndarray,
        rebuild: Callable[[np.ndarray], object], *, play_blocks: Sequence[tuple[int, int]],
        heuristic: Callable[[object], object], fps: float = FPS, params: PseudoLabelParams | None = None,
        names: Sequence[str] | None = None) -> PseudoDataset:
    """Block-masked, pseudo-labelled rows of the target match.

    ``anchor_mask`` / ``anchor_label`` come from Specialist A (``VisibleResult``); ``visible`` is the ball
    visibility of the oriented front end; ``raw_ball_xy`` the raw ball (it is erased in the blocks);
    ``rebuild(ball) -> FrontEnd`` recomputes the whole front end (with the collective features) and
    ``heuristic(fe) -> EmissionResult`` its wave-1 emissions (a stacked feature).
    """
    p = params or PseudoLabelParams.from_config()
    n = len(visible)
    periods = [(int(a), int(b)) for a, b in play_blocks]
    xs, ys, ws, all_blocks = [], [], [], []
    feat_names: list[str] = list(names) if names is not None else []
    for variant in range(p.n_variants):
        rng = np.random.default_rng(p.seed + 1000 * variant)
        blocks = draw_blocks(np.asarray(anchor_mask, bool), np.asarray(visible, bool), periods, fps, p, rng)
        all_blocks.append(blocks)
        if not blocks:
            continue
        masked = np.array(raw_ball_xy, dtype=np.float32, copy=True)
        in_block = np.zeros(n, bool)
        for a, b in blocks:
            masked[a:b + 1] = np.nan
            in_block[a:b + 1] = True
        fe = rebuild(masked)
        fm = build_feature_matrix(fe, heuristic(fe))
        if not feat_names:
            feat_names = list(fm.names)
        elif list(fm.names) != feat_names:
            raise ValueError("feature layout of the masked match differs from the specialists'")
        on_grid = (np.arange(n) % p.stride) == variant
        rows = np.flatnonzero(in_block & np.asarray(anchor_mask, bool) & on_grid)
        rows = rows[np.asarray(anchor_label)[rows] >= 0]
        if not p.include_dead:
            rows = rows[np.asarray(anchor_label)[rows] != 2]
        xs.append(fm.x[rows])
        ys.append(np.asarray(anchor_label)[rows].astype(np.int64))
        ws.append(np.full(len(rows), p.weight))
    x = np.concatenate(xs) if xs else np.zeros((0, len(feat_names)), np.float32)
    y = np.concatenate(ys) if ys else np.zeros(0, np.int64)
    w = np.concatenate(ws) if ws else np.zeros(0)
    n_blocks = sum(len(b) for b in all_blocks)
    masked_frames = sum(b - a + 1 for blocks in all_blocks for a, b in blocks)
    report = {"variants": p.n_variants, "blocks": n_blocks, "masked_frames": int(masked_frames),
              "rows": int(len(y)), "class_counts": [int((y == k).sum()) for k in range(3)],
              "anchor_share_of_visible": float(np.asarray(anchor_mask)[np.asarray(visible)].mean())
              if np.asarray(visible).any() else 0.0,
              "mean_block_s": float(masked_frames / max(n_blocks, 1) / fps)}
    return PseudoDataset(x=x, y=y, weight=w, names=feat_names, blocks=all_blocks, report=report)


def block_runs_of(mask: np.ndarray) -> list[tuple[int, int]]:
    """Inclusive ``(first, last)`` pairs of the True runs of ``mask`` (helper for tests and reports)."""
    starts, ends = mask_runs(mask)
    return [(int(a), int(b) - 1) for a, b in zip(starts, ends)]
