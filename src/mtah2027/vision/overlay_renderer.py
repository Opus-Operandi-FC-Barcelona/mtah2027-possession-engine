"""Audit-video overlay: pitch lines, 2-D minimap and HUD over the tactical video (Issue #36).

A debugging aid for the PIPELINE (alignment of the homography, format, obvious bugs), run locally.  The
rendered video of the real match is never to be used to label frames by hand or to tune parameters, and it
is never uploaded (``*.mp4`` and ``outputs/`` are git-ignored).

Per frame it draws

* the pitch lines (outline, halfway line, centre circle, boxes, penalty spots and arcs) projected with the
  homography ``H_t`` of that frame;
* the 22 players (team colours) and the ball with a trail of the last ``trail`` frames;
* a halo on the **possessor**: the player of the predicted team nearest to the ball (only for ``H``/``A``);
* a **minimap** (top view of the pitch) with players, ball, possessor and an arrow per team for the
  *collective coherence* -- the mean unit velocity vector of its outfield-and-keeper slots, whose length
  encodes ``R in [0, 1]`` (1 = the whole team moves the same way);
* a **HUD**: frame / clock, the predicted ``H``/``A``/``D``/``X``, ``P(H)``, ``P(A)``, ``P(D)`` bars and
  whether the ball is visible or hidden in the tracking.

Everything but the drawing (``cv2``) is NumPy and unit-tested without a video.  When the tactical mp4 is not
given or not readable the background is a plain green pitch, so the overlay can be checked on synthetic data.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import numpy as np

from mtah2027.config import FPS, PITCH
from mtah2027.io.homography_loader import project_pitch_to_pixels

LETTERS = "HADX"
# BGR colours
COL_HOME = (190, 90, 30)       # blue-garnet (FC Barcelona)
COL_AWAY = (50, 50, 215)       # red (Athletic Club)
COL_BALL = (255, 255, 255)
COL_HALO = (0, 220, 255)
COL_LINES = (235, 235, 235)
COL_GRASS = (46, 110, 52)
COL_LABEL = {0: COL_HOME, 1: COL_AWAY, 2: (60, 190, 230), 3: (140, 140, 140)}


class OverlayUnavailableError(RuntimeError):
    """OpenCV is not installed."""


def require_cv2():
    try:
        import cv2
    except ImportError as exc:
        raise OverlayUnavailableError("OpenCV is required to render the audit video: "
                                      "pip install -e .[vision]") from exc
    return cv2


# --------------------------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------------------------
@dataclass
class AuditData:
    """Everything the renderer needs, indexed by video frame (``N`` rows)."""

    players_xy: np.ndarray                  # [N, S, 2] metres (NaN where absent)
    team_of_slot: np.ndarray                # [S] 0 home / 1 away
    ball_xy: np.ndarray                     # [N, 2] metres (NaN where hidden)
    labels: np.ndarray                      # [N] codes H=0 A=1 D=2 X=3
    H: np.ndarray | None = None             # [N, 3, 3] normalised pitch -> pixels (None: no projection)
    probs: np.ndarray | None = None         # [N, 3] P(H), P(A), P(D) (None: bars omitted)
    ball_visible: np.ndarray | None = None  # [N] bool (default: ball_xy finite)
    pitch: tuple[float, float] = PITCH
    fps: float = float(FPS)
    frame_size: tuple[int, int] = (1920, 1080)
    possessor: np.ndarray = field(init=False)
    coherence: np.ndarray = field(init=False)       # [N, 2] R of home / away
    mean_dir: np.ndarray = field(init=False)        # [N, 2, 2] mean unit velocity of home / away
    centroid: np.ndarray = field(init=False)        # [N, 2, 2] team centroids

    def __post_init__(self):
        self.players_xy = np.asarray(self.players_xy, dtype=np.float32)
        self.ball_xy = np.asarray(self.ball_xy, dtype=np.float32)
        self.labels = np.asarray(self.labels).astype(np.int64)
        self.team_of_slot = np.asarray(self.team_of_slot).astype(np.int64)
        if self.ball_visible is None:
            self.ball_visible = np.isfinite(self.ball_xy).all(axis=1)
        self.possessor = possessor_slots(self.players_xy, self.team_of_slot, self.ball_xy, self.labels)
        self.coherence, self.mean_dir, self.centroid = team_coherence(
            self.players_xy, self.team_of_slot, self.fps)

    @property
    def n_frames(self) -> int:
        return int(len(self.labels))


def probabilities_from_log(log_probs: np.ndarray) -> np.ndarray:
    """Row-wise softmax of ``[N, 3]`` log-emissions: ``P(H), P(A), P(D)`` (sums to 1, NaN-safe)."""
    lp = np.asarray(log_probs, dtype=np.float64)
    lp = np.where(np.isfinite(lp), lp, -50.0)
    lp = lp - lp.max(axis=1, keepdims=True)
    e = np.exp(lp)
    return (e / e.sum(axis=1, keepdims=True)).astype(np.float32)


def possessor_slots(players_xy: np.ndarray, team_of_slot: np.ndarray, ball_xy: np.ndarray,
                    labels: np.ndarray, max_dist_m: float = 8.0) -> np.ndarray:
    """Slot of the player of the predicted team nearest to the ball per frame (-1: none).

    Only frames labelled ``H`` / ``A`` with a visible ball and a player within ``max_dist_m`` have one.
    """
    n = len(labels)
    out = np.full(n, -1, dtype=np.int64)
    rel = players_xy.astype(np.float64) - ball_xy.astype(np.float64)[:, None, :]
    dist = np.hypot(rel[..., 0], rel[..., 1])                                  # [N, S]
    for team in (0, 1):
        dist_t = np.where((team_of_slot == team)[None, :] & np.isfinite(dist), dist, np.inf)
        best = np.argmin(dist_t, axis=1)
        ok = (labels == team) & (dist_t[np.arange(n), best] <= max_dist_m)
        out[ok] = best[ok]
    return out


def team_coherence(players_xy: np.ndarray, team_of_slot: np.ndarray, fps: float, half: int = 2
                   ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per team: ``R`` (norm of the mean unit velocity), that mean vector and the centroid.

    Velocities are central differences over ``+- half`` frames; players that do not move (speed below
    0.3 m/s) contribute a zero vector, so a standing team has ``R`` near 0.
    """
    n = len(players_xy)
    p = players_xy.astype(np.float64)
    lo, hi = np.clip(np.arange(n) - half, 0, n - 1), np.clip(np.arange(n) + half, 0, n - 1)
    dt = np.maximum(hi - lo, 1)[:, None, None] / fps
    vel = (p[hi] - p[lo]) / dt
    speed = np.hypot(vel[..., 0], vel[..., 1])
    unit = np.where((speed > 0.3)[..., None], vel / np.maximum(speed, 1e-9)[..., None], 0.0)
    unit = np.nan_to_num(unit)
    r = np.zeros((n, 2))
    mean_dir = np.zeros((n, 2, 2))
    cen = np.full((n, 2, 2), np.nan)
    for team in (0, 1):
        sel = team_of_slot == team
        present = np.isfinite(p[:, sel, 0]) & np.isfinite(p[:, sel, 1])
        cnt = np.maximum(present.sum(axis=1), 1)[:, None]
        mean_dir[:, team] = unit[:, sel].sum(axis=1) / cnt
        r[:, team] = np.hypot(mean_dir[:, team, 0], mean_dir[:, team, 1])
        pos = np.where(present[..., None], p[:, sel], 0.0)
        c = pos.sum(axis=1) / cnt
        cen[:, team] = np.where(present.any(axis=1)[:, None], c, np.nan)
    return r.astype(np.float32), mean_dir.astype(np.float32), cen.astype(np.float32)


# --------------------------------------------------------------------------------------------
# Geometry
# --------------------------------------------------------------------------------------------
def pitch_polylines(pitch: tuple[float, float] = PITCH, arc_points: int = 48) -> list[np.ndarray]:
    """Standard pitch markings as polylines ``[K, 2]`` in metres."""
    L, W = pitch
    cy = W / 2.0
    lines: list[np.ndarray] = [np.array([[0, 0], [L, 0], [L, W], [0, W], [0, 0]], dtype=np.float64),
                               np.array([[L / 2, 0], [L / 2, W]], dtype=np.float64)]
    t = np.linspace(0.0, 2.0 * np.pi, 2 * arc_points)
    lines.append(np.stack([L / 2 + 9.15 * np.cos(t), cy + 9.15 * np.sin(t)], axis=1))
    for side in (0, 1):
        sgn = 1.0 if side == 0 else -1.0
        x0 = 0.0 if side == 0 else L
        for depth, half_w in ((16.5, 20.16), (5.5, 9.16)):
            xs = x0 + sgn * depth
            lines.append(np.array([[x0, cy - half_w], [xs, cy - half_w], [xs, cy + half_w],
                                   [x0, cy + half_w]]))
        spot = x0 + sgn * 11.0
        lines.append(np.array([[spot - 0.15, cy], [spot + 0.15, cy]]))
        ang = np.arccos(5.5 / 9.15)
        a = np.linspace(-ang, ang, arc_points)
        lines.append(np.stack([spot + sgn * 9.15 * np.cos(a), cy + 9.15 * np.sin(a)], axis=1))
    return lines


def project_points(H_t: np.ndarray | None, xy: np.ndarray, pitch: tuple[float, float] = PITCH) -> np.ndarray:
    """Pitch metres ``[K, 2]`` -> pixels ``[K, 2]`` (NaN when there is no valid homography)."""
    xy = np.asarray(xy, dtype=np.float64).reshape(-1, 2)
    if H_t is None or not np.isfinite(H_t).all():
        return np.full(xy.shape, np.nan)
    u, v = project_pitch_to_pixels(np.asarray(H_t, dtype=np.float64), xy[:, 0], xy[:, 1], pitch)
    return np.stack([u, v], axis=1)


def audit_windows(frames: Sequence[int], n_frames: int, pad: int) -> list[tuple[int, int]]:
    """Merged inclusive ``[start, end]`` windows of ``+- pad`` frames around each frame."""
    spans = sorted((max(int(f) - pad, 0), min(int(f) + pad, n_frames - 1)) for f in frames)
    merged: list[list[int]] = []
    for a, b in spans:
        if merged and a <= merged[-1][1] + 1:
            merged[-1][1] = max(merged[-1][1], b)
        else:
            merged.append([a, b])
    return [(a, b) for a, b in merged]


# --------------------------------------------------------------------------------------------
# Drawing
# --------------------------------------------------------------------------------------------
def _pt(p) -> tuple[int, int]:
    return int(round(float(p[0]) * 16)), int(round(float(p[1]) * 16))     # shift=4 fixed point


def _ok(p) -> bool:
    return bool(np.isfinite(p).all() and abs(p[0]) < 20000 and abs(p[1]) < 20000)


def blank_background(size: tuple[int, int]) -> np.ndarray:
    """Plain green pitch with mowing stripes (used when there is no video)."""
    w, h = size
    img = np.empty((h, w, 3), np.uint8)
    img[:] = COL_GRASS
    stripe = (np.arange(w) // max(w // 14, 1)) % 2 == 1
    img[:, stripe] = (np.array(COL_GRASS) + 8).astype(np.uint8)
    return img


def draw_pitch_lines(img: np.ndarray, H_t: np.ndarray | None, pitch: tuple[float, float] = PITCH) -> int:
    """Project and draw the pitch markings; returns the number of polylines drawn."""
    cv2 = require_cv2()
    drawn = 0
    for line in pitch_polylines(pitch):
        uv = project_points(H_t, line, pitch)
        ok = np.isfinite(uv).all(axis=1)
        if ok.sum() < 2:
            continue
        pts = np.array([_pt(p) for p in uv[ok] if _ok(p)], dtype=np.int32)
        if len(pts) >= 2:
            cv2.polylines(img, [pts.reshape(-1, 1, 2)], False, COL_LINES, 2, cv2.LINE_AA, 4)
            drawn += 1
    return drawn


def _minimap_xy(xy, box: tuple[int, int, int, int], pitch: tuple[float, float]):
    x0, y0, bw, bh = box
    return (x0 + float(xy[0]) / pitch[0] * bw, y0 + float(xy[1]) / pitch[1] * bh)


def draw_minimap(img: np.ndarray, data: AuditData, f: int, box: tuple[int, int, int, int]) -> None:
    """Top view of the pitch in ``box = (x, y, w, h)`` with players, ball, possessor and coherence arrows."""
    cv2 = require_cv2()
    x0, y0, bw, bh = box
    pitch = data.pitch
    overlay = img.copy()
    cv2.rectangle(overlay, (x0, y0), (x0 + bw, y0 + bh), (35, 80, 40), -1)
    cv2.addWeighted(overlay, 0.78, img, 0.22, 0, img)
    for line in pitch_polylines(pitch):
        pts = np.array([_pt(_minimap_xy(p, box, pitch)) for p in line], dtype=np.int32)
        cv2.polylines(img, [pts.reshape(-1, 1, 2)], False, (200, 200, 200), 1, cv2.LINE_AA, 4)
    r = max(3, bw // 90)
    for s in range(data.players_xy.shape[1]):
        p = data.players_xy[f, s]
        if not np.isfinite(p).all():
            continue
        c = _minimap_xy(p, box, pitch)
        col = COL_HOME if data.team_of_slot[s] == 0 else COL_AWAY
        cv2.circle(img, (int(round(c[0])), int(round(c[1]))), r, col, -1, cv2.LINE_AA)
    poss = int(data.possessor[f])
    if poss >= 0 and np.isfinite(data.players_xy[f, poss]).all():
        c = _minimap_xy(data.players_xy[f, poss], box, pitch)
        cv2.circle(img, (int(round(c[0])), int(round(c[1]))), r + 4, COL_HALO, 2, cv2.LINE_AA)
    for team in (0, 1):
        cen, vec = data.centroid[f, team], data.mean_dir[f, team]
        if not np.isfinite(cen).all():
            continue
        mag = float(np.hypot(*vec))
        a = _minimap_xy(cen, box, pitch)
        # arrow length: R * 18 m on the pitch; thicker for the team in possession
        tip = _minimap_xy(cen + vec / max(mag, 1e-9) * (18.0 * mag), box, pitch)
        col = COL_HOME if team == 0 else COL_AWAY
        thick = 3 if int(data.labels[f]) == team else 1
        cv2.arrowedLine(img, (int(round(a[0])), int(round(a[1]))), (int(round(tip[0])), int(round(tip[1]))),
                        col, thick, cv2.LINE_AA, tipLength=0.35)
    if np.isfinite(data.ball_xy[f]).all():
        c = _minimap_xy(data.ball_xy[f], box, pitch)
        cv2.circle(img, (int(round(c[0])), int(round(c[1]))), r - 1, COL_BALL, -1, cv2.LINE_AA)
        cv2.circle(img, (int(round(c[0])), int(round(c[1]))), r - 1, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.rectangle(img, (x0, y0), (x0 + bw, y0 + bh), (230, 230, 230), 1)


def draw_hud(img: np.ndarray, data: AuditData, f: int, scale_ref: int = 1080) -> None:
    """Top-left panel: frame, clock, predicted label, P(H/A/D) bars and ball visibility."""
    cv2 = require_cv2()
    k = img.shape[0] / float(scale_ref)
    fs = 0.75 * k
    th = max(1, int(round(2 * k)))
    x0, y0, pw, ph = int(16 * k), int(16 * k), int(430 * k), int(200 * k)
    overlay = img.copy()
    cv2.rectangle(overlay, (x0, y0), (x0 + pw, y0 + ph), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.65, img, 0.35, 0, img)
    t = f / data.fps
    lab = int(data.labels[f])
    line = int(34 * k)
    cv2.putText(img, f"frame {f}   {int(t // 60):02d}:{t % 60:05.2f}", (x0 + int(10 * k), y0 + line),
                cv2.FONT_HERSHEY_SIMPLEX, fs, (240, 240, 240), th, cv2.LINE_AA)
    cv2.putText(img, f"pred: {LETTERS[lab]}", (x0 + int(10 * k), y0 + 2 * line),
                cv2.FONT_HERSHEY_SIMPLEX, 1.15 * k, COL_LABEL.get(lab, (200, 200, 200)), th + 1, cv2.LINE_AA)
    visible = bool(data.ball_visible[f])
    cv2.putText(img, "ball visible" if visible else "ball HIDDEN", (x0 + int(190 * k), y0 + 2 * line),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7 * k, (90, 220, 90) if visible else (60, 80, 240), th,
                cv2.LINE_AA)
    names = ("H", "A", "D")
    for i, name in enumerate(names):
        y = y0 + int((100 + 30 * i) * k)
        pr = float(data.probs[f, i]) if data.probs is not None else float("nan")
        cv2.putText(img, f"P({name})", (x0 + int(10 * k), y), cv2.FONT_HERSHEY_SIMPLEX, 0.6 * k,
                    (230, 230, 230), max(1, th - 1), cv2.LINE_AA)
        bx, bw_ = x0 + int(80 * k), int(250 * k)
        cv2.rectangle(img, (bx, y - int(14 * k)), (bx + bw_, y + int(2 * k)), (90, 90, 90), 1)
        if np.isfinite(pr):
            cv2.rectangle(img, (bx, y - int(14 * k)), (bx + int(bw_ * pr), y + int(2 * k)),
                          COL_LABEL[i], -1)
            cv2.putText(img, f"{pr:.2f}", (bx + bw_ + int(10 * k), y), cv2.FONT_HERSHEY_SIMPLEX, 0.6 * k,
                        (230, 230, 230), max(1, th - 1), cv2.LINE_AA)
        else:
            cv2.putText(img, "n/a", (bx + bw_ + int(10 * k), y), cv2.FONT_HERSHEY_SIMPLEX, 0.6 * k,
                        (160, 160, 160), max(1, th - 1), cv2.LINE_AA)


def render_frame(data: AuditData, f: int, background: np.ndarray | None = None, *, trail: int = 25
                 ) -> np.ndarray:
    """One annotated BGR frame (``frame_size``) for video frame ``f``."""
    cv2 = require_cv2()
    w, h = data.frame_size
    img = blank_background(data.frame_size) if background is None else background.copy()
    if img.shape[1] != w or img.shape[0] != h:
        img = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)
    H_t = data.H[f] if data.H is not None else None
    draw_pitch_lines(img, H_t, data.pitch)
    # players
    pl_uv = project_points(H_t, data.players_xy[f], data.pitch)
    for s, uv in enumerate(pl_uv):
        if _ok(uv):
            col = COL_HOME if data.team_of_slot[s] == 0 else COL_AWAY
            cv2.circle(img, (int(round(uv[0])), int(round(uv[1]))), 9, col, -1, cv2.LINE_AA)
            cv2.circle(img, (int(round(uv[0])), int(round(uv[1]))), 9, (255, 255, 255), 1, cv2.LINE_AA)
    # halo on the possessor
    poss = int(data.possessor[f])
    if poss >= 0 and _ok(pl_uv[poss]):
        centre = (int(round(pl_uv[poss][0])), int(round(pl_uv[poss][1])))
        cv2.circle(img, centre, 20, COL_HALO, 3, cv2.LINE_AA)
    # ball trail (positions of the previous frames seen through the CURRENT homography)
    lo = max(f - int(trail), 0)
    tr = project_points(H_t, data.ball_xy[lo:f + 1], data.pitch)
    pts = [(i, p) for i, p in enumerate(tr) if _ok(p)]
    for (i0, p0), (i1, p1) in zip(pts[:-1], pts[1:]):
        if i1 - i0 > 3:
            continue
        a = (i1 + 1) / float(len(tr))
        cv2.line(img, (int(round(p0[0])), int(round(p0[1]))), (int(round(p1[0])), int(round(p1[1]))),
                 (int(40 + 215 * a), int(40 + 215 * a), 255), max(1, int(round(1 + 3 * a))), cv2.LINE_AA)
    if _ok(tr[-1]):
        c = (int(round(tr[-1][0])), int(round(tr[-1][1])))
        cv2.circle(img, c, 7, COL_BALL, -1, cv2.LINE_AA)
        cv2.circle(img, c, 7, (0, 0, 0), 1, cv2.LINE_AA)
    mw = int(w * 0.24)
    mh = int(round(mw * data.pitch[1] / data.pitch[0]))
    draw_minimap(img, data, f, (w - mw - 16, h - mh - 16, mw, mh))
    draw_hud(img, data, f, scale_ref=h)
    return img


# --------------------------------------------------------------------------------------------
# Video
# --------------------------------------------------------------------------------------------
class _VideoSource:
    """Sequential frame reader of the tactical mp4 (``None`` source -> no background)."""

    def __init__(self, path):
        self.cap = None
        self.pos = -1
        if path is not None and Path(path).is_file():
            cv2 = require_cv2()
            cap = cv2.VideoCapture(str(path))
            if cap.isOpened():
                self.cap = cap

    def get(self, f: int, size: tuple[int, int]):
        if self.cap is None:
            return None
        cv2 = require_cv2()
        if f != self.pos + 1:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, f)
        ok, img = self.cap.read()
        self.pos = f
        if not ok or img is None:
            return None
        return img if (img.shape[1], img.shape[0]) == tuple(size) else cv2.resize(img, tuple(size))

    def close(self):
        if self.cap is not None:
            self.cap.release()


def render_video(data: AuditData, windows: Sequence[tuple[int, int]], out_path: str | Path, *,
                 video_path: str | Path | None = None, scale: float = 1.0, trail: int = 25,
                 max_frames: int = 0) -> dict:
    """Write the annotated frames of the inclusive ``windows`` to an mp4 (mp4v); returns aggregates only."""
    cv2 = require_cv2()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    w, h = data.frame_size
    ow, oh = max(2, int(round(w * scale)) // 2 * 2), max(2, int(round(h * scale)) // 2 * 2)
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), float(data.fps), (ow, oh))
    if not writer.isOpened():
        raise OverlayUnavailableError(f"cannot open '{out_path.name}' for writing (mp4v codec)")
    src = _VideoSource(video_path)
    n = 0
    used_video = src.cap is not None
    try:
        for a, b in windows:
            for f in range(int(a), int(b) + 1):
                if max_frames and n >= max_frames:
                    break
                img = render_frame(data, f, src.get(f, data.frame_size), trail=trail)
                if (ow, oh) != (w, h):
                    img = cv2.resize(img, (ow, oh), interpolation=cv2.INTER_AREA)
                writer.write(img)
                n += 1
    finally:
        writer.release()
        src.close()
    return {"frames_written": n, "clips": len(windows), "width": ow, "height": oh,
            "background": "video" if used_video else "synthetic pitch", "path": str(out_path)}
