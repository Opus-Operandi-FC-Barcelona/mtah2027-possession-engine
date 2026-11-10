"""Broadcast-to-tactical synchronisation by continuous blocks and ball injection (Issue #10).

The TV tracking is **not aligned** to the tactical one (``games.py``): its frame axis is different (fewer
frames, cuts, replays) and every continuous shot has its own time offset.  The broadcast camera follows the ball,
so it often sees the ball where the tactical tracking lost it; once a shot is placed on the tactical clock, its
ball positions can fill the tactical holes.

1. :func:`split_blocks` -- the broadcast timeline is cut into continuous blocks: frames whose homography is
   ``valid`` with ``reprojection_error < max_reproj_error`` (and that carry data), at least ``min_block_frames``
   long; long shots are split every ``max_block_s`` so a replay or a drift inside one shot stays local.
2. :func:`fingerprint` -- a 1-D crop-invariant signature per frame from the visible players only: their mean
   speed and the median distance of a player to the nearest visible opponent.  Both are smoothed over
   ``fingerprint_smooth_s``.  The camera crops different players, but the *tempo* of the game (everybody stops at
   a dead ball, runs in a transition, spreads or compresses) is the same on both clocks.
3. :func:`correlate_block` -- z-normalised cross-correlation of the block's signature against the whole tactical
   signature (one FFT per block and feature), bounded by ``max_offset_s``; the best lag gives the offset
   ``Delta`` (tactical frame = broadcast frame + ``Delta``), its correlation and the margin over the best peak
   outside ``margin_excl_s``.
4. :func:`refine_offset` -- the lag is refined around the coarse peak with the **positions**: for every candidate
   ``Delta`` the median distance from each visible broadcast player to the nearest tactical player of the same
   team (label-free, the slots are not matched) is computed on sub-sampled frames; its minimum is the final
   offset and its value a geometric acceptance test.
5. A block is accepted when ``corr >= min_corr``, ``margin >= min_margin``, the position residual is below
   ``max_refine_dist_m`` and -- where both balls are seen -- the balls agree within ``max_ball_disagree_m``.
6. **Local pass** -- the strict global criterion only anchors the longer, distinctive shots.  Each remaining block
   is then searched in a narrow window (``local_window_s``) around the offset of the nearest accepted block
   (broadcast and tactical clocks drift only at replays / cut-aways), with a relaxed fingerprint threshold
   (``min_corr_local``) and the same position test; it repeats ``local_rounds`` times so that newly accepted
   blocks anchor their neighbours.
7. :func:`inject_broadcast_ball` -- the broadcast ball of the accepted blocks is written **only** where the
   tactical ball is NaN (never over the tracking); ``ball_source`` flags them with :data:`SRC_BROADCAST` (4),
   continuing the codes of ``vision.recovery`` (0 missing, 1 tracking, 2/3 vision).

Everything is vectorised NumPy; the signatures of a full match (199,319 frames) are built in a few seconds.
Only aggregates are reported, never per-frame data.  The stage is OFF by default (``broadcast_sync.enabled``)
until it has been validated locally on the real broadcast files.
"""
# ruff: noqa: E501  (long docstring / comment lines)
from __future__ import annotations

import os
import time
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import numpy as np

from mtah2027.config import FPS, PITCH, load_section, params_from_mapping
from mtah2027.features._utils import mask_runs
from mtah2027.vision.recovery import SRC_MISSING, SRC_TRACKING

SRC_BROADCAST = 4
_BIG = 1.0e4


@dataclass(frozen=True)
class BroadcastSyncParams:
    enabled: bool = False
    min_block_frames: int = 75        # a continuous block is at least this long (3 s)
    max_block_s: float = 90.0         # longer shots are split into pieces of at most this length (0 = no split)
    max_reproj_error: float = 5.0     # broadcast homography frames with a larger error are not usable
    pitch_margin_m: float = 1.0       # players farther outside the pitch are ignored
    min_players: int = 4              # fingerprint needs at least this many visible players
    speed_half_span: int = 2          # frames each side of the central difference speed
    fingerprint_smooth_s: float = 1.0
    min_corr: float = 0.65            # accepted fingerprint correlation
    min_margin: float = 0.0           # ... and its lead over the best peak outside margin_excl_s (0 = off)
    margin_excl_s: float = 2.0
    max_offset_s: float = 2400.0      # search window |tactical - broadcast| in seconds (0 = unbounded)
    std_floor_rel: float = 0.1        # tactical windows flatter than this x global std are not correlated
    refine_s: float = 3.0             # position refinement search half-width (0 = off)
    refine_step: int = 5              # frame sub-sampling of the refinement
    refine_max_samples: int = 80
    max_refine_dist_m: float = 2.0    # accepted median nearest-same-team distance after refinement
    max_ball_disagree_m: float = 3.0  # accepted median ball distance where both balls are seen
    min_ball_overlap: int = 25        # frames needed for the ball test
    ball_margin_m: float = 1.0        # injected broadcast ball must lie inside the pitch + margin
    local_rounds: int = 3             # propagation rounds from the accepted blocks to their neighbours (0 = off)
    local_reach_s: float = 180.0      # a block is only matched to anchors at most this far away in time
    local_window_s: float = 60.0      # ... searching offsets within +- this of the nearest anchor's offset
    min_corr_local: float = 0.5       # relaxed fingerprint correlation inside that narrow window

    @classmethod
    def from_config(cls, path=None) -> "BroadcastSyncParams":
        return params_from_mapping(cls, load_section("broadcast_sync", path), "broadcast_sync")


@dataclass
class BlockSync:
    """One continuous broadcast block and its synchronisation outcome."""

    start: int                 # first broadcast frame
    stop: int                  # exclusive end
    offset: int = 0            # tactical frame = broadcast frame + offset (valid when ``accepted``)
    coarse_offset: int = 0     # fingerprint peak before the position refinement
    corr: float = float("nan")
    margin: float = float("nan")
    cost: float = float("nan")         # median nearest-same-team distance (m) at the refined offset
    ball_disagree: float = float("nan")
    accepted: bool = False
    reason: str = ""

    @property
    def length(self) -> int:
        return self.stop - self.start


@dataclass
class BroadcastSyncResult:
    ball_xy: np.ndarray            # [N, 2] tactical ball with the broadcast points injected
    ball_source: np.ndarray        # [N] uint8: 0 missing, 1 tracking, 4 broadcast
    blocks: list[BlockSync]
    stats: dict = field(default_factory=dict)


# --------------------------------------------------------------------------------------------
# Fingerprint
# --------------------------------------------------------------------------------------------
def mask_off_pitch(players_xy: np.ndarray, pitch: tuple[float, float], margin: float) -> np.ndarray:
    """Copy of ``players_xy`` [N, S, 2] with positions beyond ``margin`` metres outside the pitch set to NaN."""
    p = np.array(players_xy, dtype=np.float32, copy=True)
    out = ((p[..., 0] < -margin) | (p[..., 0] > pitch[0] + margin)
           | (p[..., 1] < -margin) | (p[..., 1] > pitch[1] + margin))
    p[out] = np.nan
    return p


def _moving_nanmean(a: np.ndarray, width: int) -> np.ndarray:
    """Centered moving average over ``width`` frames ignoring NaN (NaN where the window is empty)."""
    n = len(a)
    ok = np.isfinite(a)
    v = np.where(ok, a, 0.0)
    cs = np.concatenate([[0.0], np.cumsum(v)])
    cc = np.concatenate([[0], np.cumsum(ok)])
    half = width // 2
    lo = np.clip(np.arange(n) - half, 0, n)
    hi = np.clip(np.arange(n) + half + 1, 0, n)
    cnt = cc[hi] - cc[lo]
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(cnt > 0, (cs[hi] - cs[lo]) / np.maximum(cnt, 1), np.nan)


def fingerprint(players_xy: np.ndarray, team_of_slot: np.ndarray, fps: float,
                params: BroadcastSyncParams | None = None, pitch: tuple[float, float] = PITCH,
                chunk: int = 4000) -> np.ndarray:
    """Crop-invariant signature ``[N, 2]``: mean speed (m/s) of the visible players and median distance (m) of a
    visible player to the nearest visible opponent, smoothed; NaN where fewer than ``min_players`` are visible."""
    p = params or BroadcastSyncParams()
    xy = mask_off_pitch(players_xy, pitch, p.pitch_margin_m)
    n = xy.shape[0]
    team = np.asarray(team_of_slot)
    h = int(p.speed_half_span)
    speed = np.full(xy.shape[:2], np.nan, dtype=np.float32)
    if n > 2 * h:
        d = xy[2 * h:] - xy[:n - 2 * h]
        speed[h:n - h] = np.sqrt((d ** 2).sum(-1)) * (fps / (2 * h))
    mean_speed = np.full(n, np.nan)
    nn_med = np.full(n, np.nan)
    home, away = np.flatnonzero(team == 0), np.flatnonzero(team == 1)
    count = np.isfinite(xy[..., 0]).sum(axis=1)
    with np.errstate(all="ignore"):
        spd_sum = np.nansum(speed, axis=1)
        spd_cnt = np.isfinite(speed).sum(axis=1)
        mean_speed = np.where(spd_cnt > 0, spd_sum / np.maximum(spd_cnt, 1), np.nan)
    for a in range(0, n, chunk):
        b = min(n, a + chunk)
        ph, pa = xy[a:b][:, home], xy[a:b][:, away]
        diff = ph[:, :, None, :] - pa[:, None, :, :]
        d2 = (diff ** 2).sum(-1)                                 # [c, Sh, Sa], NaN if either is missing
        d2 = np.where(np.isfinite(d2), d2, np.inf)
        dh = d2.min(axis=2)                                      # nearest visible opponent of each home player
        da = d2.min(axis=1)
        allv = np.concatenate([dh, da], axis=1)
        allv = np.where(np.isfinite(allv), np.sqrt(allv), np.nan)
        with np.errstate(all="ignore"), warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            nn_med[a:b] = np.nanmedian(allv, axis=1)
    ok = count >= p.min_players
    mean_speed[~ok] = np.nan
    nn_med[~ok] = np.nan
    width = max(1, int(round(p.fingerprint_smooth_s * fps)))
    return np.stack([_moving_nanmean(mean_speed, width), _moving_nanmean(nn_med, width)], axis=1)


def _fill_nan(a: np.ndarray) -> np.ndarray:
    """Linear interpolation over NaN (nearest value at the edges); an all-NaN column becomes zeros."""
    out = np.array(a, dtype=np.float64, copy=True)
    for j in range(out.shape[1]):
        col = out[:, j]
        ok = np.isfinite(col)
        if not ok.any():
            col[:] = 0.0
        elif not ok.all():
            idx = np.arange(len(col))
            col[~ok] = np.interp(idx[~ok], idx[ok], col[ok])
    return out


# --------------------------------------------------------------------------------------------
# Blocks
# --------------------------------------------------------------------------------------------
def split_blocks(usable: np.ndarray, params: BroadcastSyncParams | None = None,
                 fps: float = FPS) -> list[tuple[int, int]]:
    """Continuous ``(start, stop)`` ranges of the True runs of ``usable`` that are long enough (pieces of at most
    ``max_block_s``; a trailing remainder shorter than ``min_block_frames`` is merged into the previous piece)."""
    p = params or BroadcastSyncParams()
    starts, ends = mask_runs(np.asarray(usable, dtype=bool))
    cap = int(round(p.max_block_s * fps)) if p.max_block_s > 0 else 0
    out: list[tuple[int, int]] = []
    for a, b in zip(starts.tolist(), ends.tolist()):
        if b - a < p.min_block_frames:
            continue
        if cap <= 0 or b - a <= cap:
            out.append((a, b))
            continue
        k = max(1, int(round((b - a) / cap)))
        cuts = np.linspace(a, b, k + 1).round().astype(int)
        out.extend((int(x), int(y)) for x, y in zip(cuts[:-1], cuts[1:]))
    return out


# --------------------------------------------------------------------------------------------
# Coarse offset: z-normalised cross-correlation of the signatures
# --------------------------------------------------------------------------------------------
class _TacticalIndex:
    """Pre-computed FFTs and sliding statistics of the tactical signature (shared by all blocks)."""

    def __init__(self, fp: np.ndarray, max_block: int, floor_rel: float):
        self.n = len(fp)
        self.fp = fp
        self.m = 1 << int(np.ceil(np.log2(self.n + max_block + 1)))
        self.cols = []
        for j in range(fp.shape[1]):
            t = fp[:, j] - fp[:, j].mean()
            self.cols.append({"t": t, "fft": np.fft.rfft(t, self.m), "cs": np.concatenate([[0.0], np.cumsum(t)]),
                              "cs2": np.concatenate([[0.0], np.cumsum(t * t)]),
                              "floor": floor_rel * max(float(t.std()), 1e-9)})

    def window_std(self, j: int, length: int) -> np.ndarray:
        c = self.cols[j]
        k = self.n - length + 1
        mean = (c["cs"][length:length + k] - c["cs"][:k]) / length
        var = (c["cs2"][length:length + k] - c["cs2"][:k]) / length - mean ** 2
        return np.sqrt(np.maximum(var, 0.0))

    def correlate(self, block_fp: np.ndarray) -> np.ndarray | None:
        """Mean over the usable features of the Pearson correlation for every lag ``p`` in ``[0, N - L]``."""
        length = len(block_fp)
        if length > self.n:
            return None
        acc, used = None, 0
        for j, c in enumerate(self.cols):
            b = block_fp[:, j]
            sd = b.std()
            if sd < 1e-6:
                continue
            bz = (b - b.mean()) / sd
            raw = np.fft.irfft(c["fft"] * np.conj(np.fft.rfft(bz, self.m)), self.m)[:self.n - length + 1]
            sigma = np.maximum(self.window_std(j, length), c["floor"])
            corr = raw / (length * sigma)
            acc = corr if acc is None else acc + corr
            used += 1
        return None if acc is None else acc / used


def correlate_block(index: _TacticalIndex, block_fp: np.ndarray, start: int, params: BroadcastSyncParams,
                    fps: float, around: int | None = None,
                    half_width: int | None = None) -> tuple[int, float, float] | None:
    """``(offset, corr, margin)`` of the best tactical lag for a block starting at broadcast frame ``start``.

    The lags are bounded by ``max_offset_s``; with ``around`` / ``half_width`` only offsets within that distance of
    ``around`` are searched (the local pass, anchored on an already synchronised neighbour)."""
    c = index.correlate(block_fp)
    if c is None:
        return None
    p_idx = np.arange(len(c))
    if params.max_offset_s > 0:
        lim = int(round(params.max_offset_s * fps))
        allowed = np.abs(p_idx - start) <= lim
        if not allowed.any():
            return None
        c = np.where(allowed, c, -np.inf)
    if around is not None and half_width is not None:
        c = np.where(np.abs(p_idx - start - around) <= half_width, c, -np.inf)
        if not np.isfinite(c).any():
            return None
    best = int(np.argmax(c))
    if not np.isfinite(c[best]):
        return None
    excl = max(1, int(round(params.margin_excl_s * fps)))
    lo, hi = max(0, best - excl), min(len(c), best + excl + 1)
    rest = np.concatenate([c[:lo], c[hi:]])
    second = float(rest.max()) if len(rest) and np.isfinite(rest.max()) else -1.0
    return best - start, float(c[best]), float(c[best] - second)


# --------------------------------------------------------------------------------------------
# Position refinement
# --------------------------------------------------------------------------------------------
def refine_offset(b_players: np.ndarray, b_team: np.ndarray, t_players: np.ndarray, t_team: np.ndarray,
                  start: int, stop: int, offset0: int, params: BroadcastSyncParams,
                  fps: float) -> tuple[int, float]:
    """``(offset, cost)``: the offset in ``offset0 +- refine_s`` minimising the median distance from the visible
    broadcast players to the nearest tactical player of the same team (sub-sampled frames, label-free)."""
    n_t = t_players.shape[0]
    frames = np.arange(start, stop, max(1, params.refine_step))
    if len(frames) > params.refine_max_samples:
        frames = frames[np.linspace(0, len(frames) - 1, params.refine_max_samples).round().astype(int)]
    radius = int(round(params.refine_s * fps))
    teams = [(np.flatnonzero(b_team == k), np.flatnonzero(t_team == k)) for k in (0, 1)]
    bsel = [b_players[frames][:, bs] for bs, _ in teams]
    bok = [np.isfinite(b[..., 0]) for b in bsel]
    if sum(int(o.sum()) for o in bok) == 0:
        return offset0, float("inf")
    t_clean = np.where(np.isfinite(t_players), t_players, _BIG).astype(np.float32)
    memo: dict[int, float] = {}

    def cost(off: int) -> float:
        if off in memo:
            return memo[off]
        tf = frames + off
        inside = (tf >= 0) & (tf < n_t)
        value = np.inf
        if inside.sum() >= max(3, len(frames) // 2):
            vals = []
            for (bs, ts), b, ok in zip(teams, bsel, bok):
                if len(ts) == 0 or len(bs) == 0:
                    continue
                tt = t_clean[tf[inside]][:, ts]                                   # [F, St, 2]
                d2 = ((b[inside][:, :, None, :] - tt[:, None, :, :]) ** 2).sum(-1)    # [F, Sb, St]
                vals.append(d2.min(axis=2)[ok[inside]])
            v = np.concatenate(vals) if vals else np.empty(0)
            v = v[np.isfinite(v)]
            if len(v):
                value = float(np.sqrt(np.median(v)))
        memo[off] = value
        return value

    stride = max(1, radius // 12)                       # coarse scan, then a unit-step scan around its minimum
    coarse = np.arange(offset0 - radius, offset0 + radius + 1, stride)
    best = int(coarse[int(np.argmin([cost(int(o)) for o in coarse]))])
    fine = np.arange(best - stride, best + stride + 1)
    scores = [cost(int(o)) for o in fine]
    k = int(np.argmin(scores))
    return int(fine[k]), float(scores[k])


# --------------------------------------------------------------------------------------------
# Ball injection
# --------------------------------------------------------------------------------------------
def ball_disagreement(b_ball: np.ndarray, t_ball: np.ndarray, start: int, stop: int, offset: int,
                      min_overlap: int) -> float:
    """Median distance between the two balls where both are visible (NaN when fewer than ``min_overlap``)."""
    j = np.arange(start, stop)
    t = j + offset
    ok = (t >= 0) & (t < len(t_ball))
    j, t = j[ok], t[ok]
    both = np.isfinite(b_ball[j]).all(axis=1) & np.isfinite(t_ball[t]).all(axis=1)
    if both.sum() < min_overlap:
        return float("nan")
    return float(np.median(np.linalg.norm(b_ball[j[both]] - t_ball[t[both]], axis=1)))


def inject_broadcast_ball(ball_xy: np.ndarray, b_ball: np.ndarray, blocks: Sequence[BlockSync],
                          pitch: tuple[float, float] = PITCH,
                          margin: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    """Write the broadcast ball of the accepted blocks into the NaN frames of ``ball_xy``.

    Returns ``(ball_xy, ball_source)`` (``0`` missing, ``1`` tracking, :data:`SRC_BROADCAST`).  Tracking is never
    overwritten; where two blocks claim the same frame the one with the higher correlation wins."""
    out = np.array(ball_xy, dtype=np.float32, copy=True)
    seen = np.isfinite(out).all(axis=1)
    source = np.where(seen, SRC_TRACKING, SRC_MISSING).astype(np.uint8)
    for blk in sorted((b for b in blocks if b.accepted), key=lambda b: -b.corr):
        j = np.arange(blk.start, blk.stop)
        t = j + blk.offset
        keep = (t >= 0) & (t < len(out))
        j, t = j[keep], t[keep]
        bb = b_ball[j]
        inside = (np.isfinite(bb).all(axis=1) & (bb[:, 0] >= -margin) & (bb[:, 0] <= pitch[0] + margin)
                  & (bb[:, 1] >= -margin) & (bb[:, 1] <= pitch[1] + margin))
        sel = inside & (source[t] == SRC_MISSING)
        out[t[sel]] = bb[sel]
        source[t[sel]] = SRC_BROADCAST
    return out, source


# --------------------------------------------------------------------------------------------
# Orchestration
# --------------------------------------------------------------------------------------------
def sync_stats(blocks: Sequence[BlockSync], ball_before: np.ndarray, source: np.ndarray,
               graded: Sequence[tuple[int, int]], seconds: float, n_broadcast: int, usable: np.ndarray) -> dict:
    """Aggregate counters (no per-frame data)."""
    n = len(ball_before)
    g = np.zeros(n, bool)
    for a, b in graded:
        g[max(int(a), 0):min(int(b), n - 1) + 1] = True
    hidden = g & ~np.isfinite(ball_before).all(axis=1)
    filled = hidden & (source == SRC_BROADCAST)
    acc = [b for b in blocks if b.accepted]
    reasons: dict[str, int] = {}
    for b in blocks:
        if not b.accepted:
            reasons[b.reason] = reasons.get(b.reason, 0) + 1
    return {
        "status": "ok", "broadcast_frames": int(n_broadcast), "usable_frames": int(usable.sum()),
        "blocks": len(blocks), "blocks_accepted": len(acc),
        "block_frames": int(sum(b.length for b in blocks)), "accepted_frames": int(sum(b.length for b in acc)),
        "rejected_by": reasons,
        "corr_median": float(np.median([b.corr for b in acc])) if acc else float("nan"),
        "cost_median_m": float(np.median([b.cost for b in acc])) if acc else float("nan"),
        "offset_median_frames": float(np.median([b.offset for b in acc])) if acc else float("nan"),
        "hidden_graded_frames": int(hidden.sum()), "frames_filled": int(filled.sum()),
        "fill_rate": float(filled.sum() / max(hidden.sum(), 1)),
        "seconds": round(float(seconds), 2),
    }


def sync_broadcast(t_players: np.ndarray, t_team: np.ndarray, t_ball: np.ndarray,
                   b_players: np.ndarray, b_team: np.ndarray, b_ball: np.ndarray, b_usable: np.ndarray, *,
                   fps: float = FPS, pitch: tuple[float, float] = PITCH,
                   graded: Sequence[tuple[int, int]] = (), params: BroadcastSyncParams | None = None
                   ) -> BroadcastSyncResult:
    """Place every continuous broadcast block on the tactical clock and inject its ball (see the module doc).

    ``t_*`` are the tactical arrays (players ``[N, S, 2]`` metres, team 0/1 per slot, ball ``[N, 2]``); ``b_*`` the
    broadcast ones on their own clock; ``b_usable`` ``[Nb]`` marks the frames with a valid homography."""
    p = params or BroadcastSyncParams()
    t0 = time.perf_counter()
    nb = b_players.shape[0]
    usable = np.zeros(nb, bool)
    m = min(nb, len(b_usable))
    usable[:m] = np.asarray(b_usable, dtype=bool)[:m]
    usable &= np.isfinite(b_players[..., 0]).any(axis=1) | np.isfinite(b_ball).all(axis=1)
    ranges = split_blocks(usable, p, fps)
    t_mask = mask_off_pitch(t_players, pitch, p.pitch_margin_m)
    b_mask = mask_off_pitch(b_players, pitch, p.pitch_margin_m)
    t_team, b_team = np.asarray(t_team), np.asarray(b_team)
    t_ball = np.asarray(t_ball, dtype=np.float32)
    b_ball = np.asarray(b_ball, dtype=np.float32)
    blocks = [BlockSync(a, b) for a, b in ranges]
    if blocks:
        t_fp = _fill_nan(fingerprint(t_mask, t_team, fps, p, pitch))
        b_fp = fingerprint(b_mask, b_team, fps, p, pitch)
        index = _TacticalIndex(t_fp, max(blk.length for blk in blocks), p.std_floor_rel)

        def verify(blk: BlockSync, max_cost: float) -> bool:
            """Position refinement + residual + ball tests of ``blk`` (offset = its coarse offset)."""
            if p.refine_s > 0:
                blk.offset, blk.cost = refine_offset(b_mask, b_team, t_mask, t_team, blk.start, blk.stop,
                                                     blk.coarse_offset, p, fps)
                if not blk.cost <= max_cost:
                    blk.reason = "position"
                    return False
            blk.ball_disagree = ball_disagreement(b_ball, t_ball, blk.start, blk.stop, blk.offset,
                                                  p.min_ball_overlap)
            if np.isfinite(blk.ball_disagree) and blk.ball_disagree > p.max_ball_disagree_m:
                blk.reason = "ball"
                return False
            blk.accepted, blk.reason = True, ""
            return True

        # pass 1: global search, strict fingerprint thresholds -> anchors
        for blk in blocks:
            seg = b_fp[blk.start:blk.stop]
            if np.isfinite(seg).mean() < 0.5:
                blk.reason = "fingerprint"
                continue
            res = correlate_block(index, _fill_nan(seg), blk.start, p, fps)
            if res is None:
                blk.reason = "no lag"
                continue
            blk.coarse_offset, blk.corr, blk.margin = res
            blk.offset = blk.coarse_offset
            if blk.corr < p.min_corr:
                blk.reason = "corr"
            elif blk.margin < p.min_margin:
                blk.reason = "margin"
            else:
                verify(blk, p.max_refine_dist_m)
        # pass 2: shots next to synchronised ones -> narrow window around the neighbour's offset
        tried: dict[int, set] = {}
        reach = int(round(p.local_reach_s * fps))
        window = int(round(p.local_window_s * fps))
        for _ in range(max(0, p.local_rounds)):
            anchors = [b for b in blocks if b.accepted]
            if not anchors:
                break
            a_mid = np.array([(b.start + b.stop) / 2 for b in anchors])
            a_off = np.array([b.offset for b in anchors])
            progress = False
            for blk in blocks:
                if blk.accepted or blk.reason == "fingerprint":
                    continue
                near = int(np.argmin(np.abs(a_mid - (blk.start + blk.stop) / 2)))
                if abs(a_mid[near] - (blk.start + blk.stop) / 2) > reach:
                    continue
                res = correlate_block(index, _fill_nan(b_fp[blk.start:blk.stop]), blk.start, p, fps,
                                      around=int(a_off[near]), half_width=window)
                coarse = int(a_off[near])
                if res is not None and res[1] >= p.min_corr_local:
                    coarse, blk.corr, blk.margin = res
                if coarse in tried.setdefault(id(blk), set()):
                    continue                      # the very same candidate already failed in an earlier round
                tried[id(blk)].add(coarse)
                blk.coarse_offset = coarse
                progress |= verify(blk, p.max_refine_dist_m)
            if not progress:
                break
    ball, source = inject_broadcast_ball(t_ball, b_ball, blocks, pitch, p.ball_margin_m)
    stats = sync_stats(blocks, t_ball, source, graded, time.perf_counter() - t0, nb, usable)
    return BroadcastSyncResult(ball, source, blocks, stats)


def broadcast_files_present(data_dir: str | os.PathLike | None = None) -> bool:
    """True when the broadcast ATD (xml + txt) and homography files exist in the data bundle."""
    from mtah2027.config import load_games_module

    games = load_games_module()
    if games is None:
        return False
    try:
        paths = games.asset_paths(data_dir=str(data_dir) if data_dir is not None else None, source="broadcast")
    except KeyError:
        return False
    return all(Path(paths[k]).is_file() for k in ("atd_meta", "atd_raw", "homography") if k in paths)


def sync_broadcast_files(td, data_dir: str | os.PathLike | None, graded: Sequence[tuple[int, int]], *,
                         params: BroadcastSyncParams | None = None,
                         cache_dir: str | os.PathLike | None = None) -> BroadcastSyncResult:
    """Load the broadcast ATD + homographies of the bundle and synchronise them to the tactical ``td``."""
    from mtah2027.io.atd_loader import load_atd
    from mtah2027.io.homography_loader import load_homography

    p = params or BroadcastSyncParams.from_config()
    btd = load_atd(data_dir, source="broadcast", cache_dir=cache_dir, use_cache=cache_dir is not None)
    bhom = load_homography(data_dir, source="broadcast")
    nb = btd.n_frames
    usable = np.zeros(nb, bool)
    m = min(nb, bhom.n_frames)
    usable[:m] = (bhom.valid & (bhom.reprojection_error < p.max_reproj_error))[:m]
    return sync_broadcast(td.players_xy, td.team_of_slot, td.ball_xy, btd.players_xy, btd.team_of_slot,
                          btd.ball_xy, usable, fps=float(td.fps), pitch=tuple(td.pitch), graded=graded, params=p)
