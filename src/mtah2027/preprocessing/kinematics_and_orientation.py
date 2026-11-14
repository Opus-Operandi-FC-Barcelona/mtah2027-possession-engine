"""Savitzky-Golay kinematics and attack-direction normalisation (Issue #9).

1. **Gap interpolation.**  NaN gaps of at most ``max_gap`` (5) frames -- 0.2 s at 25 fps -- that have
   valid samples on both sides are linearly interpolated; longer gaps and the NaN margins stay NaN.
2. **Velocity / acceleration.**  On every *contiguous run* of valid samples of an entity (a player
   slot or the ball) the Savitzky-Golay filter (``window=7``, ``polyorder=2``, ``deriv=1`` and ``2``,
   ``delta=1/fps``) gives ``v`` in m/s and ``a`` in m/s^2.  Runs shorter than the window use finite
   differences (central differences inside, one-sided at the ends); a run of one sample has no
   derivative (NaN) and one of two has velocity but no acceleration.  The filter never mixes samples
   across a NaN gap.
3. **Orientation.**  The side each team defends in each half is read from the mean ``x`` of its
   players during the first minutes of play of the half (all players are in their own half at the
   kick-off); if that window has no data, from the whole-half centroid.  Every half in which the home
   team (code 0) attacks ``-x`` is mirrored -- ``(x, y) -> (L - x, W - y)``, ``(v, a) -> (-v, -a)`` --
   so that **home always attacks +x**.  Kinematics are computed on the raw coordinates *before*
   mirroring, so a run that spans the half-time flip never produces a spurious jump.

Run :func:`mtah2027.preprocessing.clean_tracks.clean_tracks` first so that staff and duplicates do not
bias the direction estimate.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.signal import savgol_filter

from mtah2027.config import FPS, HALF1, HALF2_PLAY, PITCH
from mtah2027.preprocessing.clean_tracks import run_lengths

MAX_GAP = 5
SG_WINDOW = 7
SG_POLY = 2
FIRST_MINUTES = 2.0
MIN_DIRECTION_SAMPLES = 25


# --------------------------------------------------------------------------------------------
# Gap interpolation
# --------------------------------------------------------------------------------------------
def interpolate_gaps(values: np.ndarray, max_gap: int = MAX_GAP) -> tuple[np.ndarray, np.ndarray]:
    """Linearly fill interior NaN gaps of length <= ``max_gap`` along axis 0.

    ``values`` is ``[N, ...]``; returns ``(filled float32 copy, filled mask)``.
    """
    arr = np.asarray(values)
    shape = arr.shape
    flat = np.array(arr.reshape(shape[0], -1), dtype=np.float32, copy=True)
    nan = np.isnan(flat)
    filled = np.zeros_like(nan)
    if max_gap > 0 and nan.any():
        gaps = run_lengths(nan)
        for k in np.flatnonzero(nan.any(axis=0) & (~nan).sum(axis=0).astype(bool)):
            valid_idx = np.flatnonzero(~nan[:, k])
            if len(valid_idx) < 2:
                continue
            fill = nan[:, k] & (gaps[:, k] <= max_gap)
            fill[:valid_idx[0]] = False
            fill[valid_idx[-1] + 1:] = False
            pos = np.flatnonzero(fill)
            if len(pos):
                flat[pos, k] = np.interp(pos, valid_idx, flat[valid_idx, k])
                filled[pos, k] = True
    return flat.reshape(shape), filled.reshape(shape)


# --------------------------------------------------------------------------------------------
# Savitzky-Golay kinematics
# --------------------------------------------------------------------------------------------
def _runs(valid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    padded = np.concatenate(([0], valid.astype(np.int8), [0]))
    edges = np.diff(padded)
    return np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)


def _finite_difference(seg: np.ndarray, dt: float) -> tuple[np.ndarray, np.ndarray]:
    """Velocity and acceleration of a short run (``len(seg) < window``)."""
    n = len(seg)
    vel = np.full(seg.shape, np.nan)
    acc = np.full(seg.shape, np.nan)
    if n >= 2:
        vel = np.gradient(seg, dt, axis=0)
    if n >= 3:
        acc = np.gradient(vel, dt, axis=0)
    return vel, acc


def savgol_kinematics(xy: np.ndarray, fps: float = FPS, window: int = SG_WINDOW, poly: int = SG_POLY
                      ) -> tuple[np.ndarray, np.ndarray]:
    """Velocity and acceleration of ``xy[N, E, 2]`` (metres) per contiguous valid run.

    Entities with a NaN in either coordinate are invalid for that frame.  Returns float32 arrays of
    the same shape as ``xy`` (m/s and m/s^2), NaN wherever the position is NaN.
    """
    if window % 2 == 0 or window <= poly:
        raise ValueError("window must be odd and larger than poly")
    n, e, _ = xy.shape
    dt = 1.0 / fps
    vel = np.full(xy.shape, np.nan, dtype=np.float32)
    acc = np.full(xy.shape, np.nan, dtype=np.float32)
    valid = np.isfinite(xy).all(axis=2)
    for k in np.flatnonzero(valid.any(axis=0)):
        starts, ends = _runs(valid[:, k])
        col = xy[:, k, :].astype(np.float64)
        for a, b in zip(starts, ends):
            seg = col[a:b]
            if b - a >= window:
                v = savgol_filter(seg, window, poly, deriv=1, delta=dt, axis=0, mode="interp")
                ac = savgol_filter(seg, window, poly, deriv=2, delta=dt, axis=0, mode="interp")
            else:
                v, ac = _finite_difference(seg, dt)
            vel[a:b, k, :] = v
            acc[a:b, k, :] = ac
    return vel, acc


# --------------------------------------------------------------------------------------------
# Orientation
# --------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class AttackDirection:
    """Home attack direction per half: +1 (towards +x), -1, or 0 when it cannot be determined."""

    home_dir: tuple[int, int]
    gap_m: tuple[float, float]        # mean_x(away) - mean_x(home) on the estimation window
    method: tuple[str, str]           # 'kickoff-window', 'whole-half' or 'unknown'


def _half_of_frame(n: int, halves: tuple[tuple[int, int], tuple[int, int]]) -> np.ndarray:
    """Half index (0/1) of every frame: split at the middle of the half-time break."""
    split = (halves[0][1] + halves[1][0]) // 2
    return (np.arange(n) > split).astype(np.int8)


def detect_attack_direction(players_xy: np.ndarray, team_of_slot: np.ndarray, *,
                            halves: tuple[tuple[int, int], tuple[int, int]] = (HALF1, HALF2_PLAY),
                            fps: float = FPS, first_minutes: float = FIRST_MINUTES,
                            pitch: tuple[float, float] = PITCH,
                            min_samples: int = MIN_DIRECTION_SAMPLES) -> AttackDirection:
    """Side defended by the home team in each half (see the module docstring).

    The home team (slot team 0) attacks ``+x`` when its mean ``x`` is smaller than the away team's.
    """
    n = players_xy.shape[0]
    team = np.asarray(team_of_slot)
    x, y = players_xy[..., 0], players_xy[..., 1]
    with np.errstate(invalid="ignore"):
        on_pitch = (x >= 0) & (x <= pitch[0]) & (y >= 0) & (y <= pitch[1])
    xm = np.where(on_pitch, x, np.nan)
    dirs, gaps, methods = [], [], []
    for first, last in halves:
        first, last = max(0, int(first)), min(n - 1, int(last))
        chosen = None
        windows = ((first, min(last, first + int(first_minutes * 60 * fps))), (first, last))
        for (a, b), name in zip(windows, ("kickoff-window", "whole-half")):
            if b < a:
                continue
            block = xm[a:b + 1]
            home, away = block[:, team == 0], block[:, team == 1]
            if np.isfinite(home).sum() < min_samples or np.isfinite(away).sum() < min_samples:
                continue
            gap = float(np.nanmean(away) - np.nanmean(home))
            if gap != 0.0:
                chosen = (1 if gap > 0 else -1, gap, name)
                break
        chosen = chosen or (0, 0.0, "unknown")
        dirs.append(chosen[0])
        gaps.append(chosen[1])
        methods.append(chosen[2])
    return AttackDirection(tuple(dirs), tuple(gaps), tuple(methods))


@dataclass
class OrientedKinematics:
    """Interpolated, filtered and orientation-normalised tracking (home attacks +x)."""

    players_xy: np.ndarray        # [N, S, 2] metres, gaps <= max_gap interpolated, mirrored
    ball_xy: np.ndarray           # [N, 2]
    players_vel: np.ndarray       # [N, S, 2] m/s
    players_acc: np.ndarray       # [N, S, 2] m/s^2
    ball_vel: np.ndarray          # [N, 2]
    ball_acc: np.ndarray          # [N, 2]
    players_speed: np.ndarray     # [N, S] |v| m/s
    ball_speed: np.ndarray        # [N]
    players_interpolated: np.ndarray   # [N, S] bool
    ball_interpolated: np.ndarray      # [N] bool
    flip_halves: tuple[bool, bool]     # True where that half was mirrored
    flipped: np.ndarray                # [N] bool, per-frame flip flag
    direction: AttackDirection         # home attack direction in the *raw* coordinates


def mirror(arr: np.ndarray, flipped: np.ndarray, pitch: tuple[float, float], *, vector: bool) -> np.ndarray:
    """Mirror positions (``vector=False``) or velocity/acceleration (``vector=True``) where ``flipped``.

    ``arr`` is ``[N, ..., 2]`` and ``flipped`` ``[N]``; returns a new array.
    """
    out = np.array(arr, copy=True)
    idx = np.flatnonzero(flipped)
    if len(idx):
        if vector:
            out[idx] = -out[idx]
        else:
            out[idx, ..., 0] = pitch[0] - out[idx, ..., 0]
            out[idx, ..., 1] = pitch[1] - out[idx, ..., 1]
    return out


def fill_speed_from_native(speed: np.ndarray, valid: np.ndarray, native: np.ndarray,
                           edge_frames: int = 3) -> tuple[np.ndarray, np.ndarray]:
    """Use the provider's own speed (``native``, m/s) where the Savitzky-Golay one is unreliable (Issue #42).

    ``speed`` and ``native`` are ``[N, S]``; ``valid`` marks the frames with a position.  The native value
    replaces (a) a missing SG speed at a valid position and (b) the SG speed within ``edge_frames`` of the
    start / end of every continuous run of positions (where a fragmented track has the least support).
    Returns ``(speed, replaced_mask)``; frames without a finite native value keep the SG speed."""
    nat = np.asarray(native, dtype=np.float32)
    ok = np.asarray(valid, dtype=bool) & np.isfinite(nat) & (nat >= 0)
    inner = np.asarray(valid, dtype=bool).copy()
    for j in range(1, int(edge_frames) + 1):
        inner[j:] &= valid[:-j]
        inner[:-j] &= valid[j:]
        inner[:j] = False
        inner[-j:] = False
    use = ok & (~inner | ~np.isfinite(speed))
    return np.where(use, nat, speed).astype(np.float32), use


def compute_kinematics_and_orientation(
        players_xy: np.ndarray, ball_xy: np.ndarray, team_of_slot: np.ndarray, *, fps: float = FPS,
        pitch: tuple[float, float] = PITCH,
        halves: tuple[tuple[int, int], tuple[int, int]] = (HALF1, HALF2_PLAY), max_gap: int = MAX_GAP,
        window: int = SG_WINDOW, poly: int = SG_POLY, first_minutes: float = FIRST_MINUTES,
        min_samples: int = MIN_DIRECTION_SAMPLES,
        players_speed_native: np.ndarray | None = None, native_edge_frames: int = 3
) -> OrientedKinematics:
    """Interpolate, differentiate and mirror so that the home team attacks +x in both halves.

    ``players_xy`` is ``[N, S, 2]`` metres (cleaned), ``ball_xy`` ``[N, 2]`` metres.  ``halves`` are
    the (first, last) graded frames of play of each half (config default).
    """
    n, s, _ = players_xy.shape
    if ball_xy.shape != (n, 2):
        raise ValueError(f"ball_xy has shape {ball_xy.shape}, expected ({n}, 2)")
    ents = np.concatenate([players_xy, ball_xy[:, None, :]], axis=1).astype(np.float32)
    ents, filled = interpolate_gaps(ents, max_gap)
    vel, acc = savgol_kinematics(ents, fps, window, poly)

    direction = detect_attack_direction(ents[:, :s], team_of_slot, halves=halves, fps=fps,
                                        first_minutes=first_minutes, pitch=pitch,
                                        min_samples=min_samples)
    flip_halves = tuple(d == -1 for d in direction.home_dir)
    half_idx = _half_of_frame(n, halves)
    flipped = np.asarray(flip_halves)[half_idx]

    pos = mirror(ents, flipped, pitch, vector=False)
    vel = mirror(vel, flipped, pitch, vector=True)
    acc = mirror(acc, flipped, pitch, vector=True)
    speed = np.linalg.norm(vel, axis=-1).astype(np.float32)
    if players_speed_native is not None:       # Issue #42: provider speed channel as fallback at track edges
        speed[:, :s], _ = fill_speed_from_native(speed[:, :s], np.isfinite(ents[:, :s]).all(axis=2),
                                                 players_speed_native, native_edge_frames)
    return OrientedKinematics(
        players_xy=pos[:, :s], ball_xy=pos[:, s], players_vel=vel[:, :s], players_acc=acc[:, :s],
        ball_vel=vel[:, s], ball_acc=acc[:, s], players_speed=speed[:, :s], ball_speed=speed[:, s],
        players_interpolated=filled[:, :s, 0], ball_interpolated=filled[:, s, 0],
        flip_halves=(bool(flip_halves[0]), bool(flip_halves[1])), flipped=flipped, direction=direction)
