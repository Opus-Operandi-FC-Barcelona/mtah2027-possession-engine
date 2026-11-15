#!/usr/bin/env python
"""Issue #70 probe: value of the broadcast ATD beyond the injected ball (aggregates only, real match).

After the existing Issue #39 block synchronisation (accepted blocks only) it measures, over the graded frames:

* coverage of the tactical timeline by synchronised blocks, and the TV ball visibility conditional on the tactical
  ball being visible / hidden (is the TV tracker also blind where the tactical one is?);
* tracked players the TV view sees and the tactical tracking does not (same team, farther than 2 m from every
  tactical player), for frames with fewer than 20 / 16 tactical players;
* the pitch point under the TV image centre (camera look-at) as a coarse ball proxy: availability and distance to the
  tactical ball where both exist, and TV field-of-view width as a replay / close-up cue.

Nothing per frame is printed and no label is read."""
# ruff: noqa: E501
from __future__ import annotations

import argparse
import dataclasses
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
for extra in (ROOT / "src", ROOT, ROOT / "scripts"):
    sys.path.insert(0, str(extra))

from mtah2027.config import load_config  # noqa: E402
from mtah2027.io.atd_loader import load_atd  # noqa: E402
from mtah2027.io.homography_loader import load_homography, project_pixels_to_pitch  # noqa: E402
from mtah2027.preprocessing import broadcast_sync as bs  # noqa: E402


def pct(x):
    x = np.asarray(x)
    return f"{100.0 * float(np.mean(x)):.2f}%" if x.size else "n/a"


def q(v, qs=(10, 50, 90)):
    v = np.asarray(v, float)
    v = v[np.isfinite(v)]
    return "n/a" if not v.size else "/".join(f"{np.percentile(v, p):.1f}" for p in qs)


def synced_cover(data_dir, cache_dir):
    """Tactical/broadcast data, the accepted-block sync result and the tactical -> broadcast frame map."""
    cfg = load_config()
    play = [cfg.match.half1, cfg.match.half2_play]
    td = load_atd(data_dir, cache_dir=cache_dir)
    btd = load_atd(data_dir, source="broadcast", cache_dir=cache_dir)
    bh = load_homography(data_dir, source="broadcast")
    n = td.n_frames
    p = dataclasses.replace(bs.BroadcastSyncParams.from_config(), enabled=True)
    usable = np.zeros(btd.n_frames, bool)
    m = min(btd.n_frames, bh.n_frames)
    usable[:m] = (bh.valid & (bh.reprojection_error < p.max_reproj_error))[:m]
    res = bs.sync_broadcast(td.players_xy, td.team_of_slot, td.ball_xy, btd.players_xy, btd.team_of_slot,
                            btd.ball_xy, usable, fps=float(td.fps), pitch=tuple(td.pitch), graded=play, params=p)
    graded = np.zeros(n, bool)
    for a, b in play:
        graded[a:b + 1] = True
    bf = np.full(n, -1, dtype=np.int64)            # tactical frame -> broadcast frame (accepted blocks)
    for b in res.blocks:
        if b.accepted:
            lo, hi = max(b.start + b.offset, 0), min(b.stop + b.offset, n)
            if hi > lo:
                bf[lo:hi] = np.arange(lo, hi) - b.offset
    return td, btd, bh, res, graded, bf, p


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--cache-dir", default=str(ROOT / "cache"))
    args = ap.parse_args(argv)
    td, btd, bh, res, graded, bf, p = synced_cover(args.data_dir, args.cache_dir)
    n = td.n_frames
    cov = (bf >= 0) & graded
    vt = td.ball_visible
    idx = np.where(cov)[0]
    b_idx = bf[idx]
    vb = np.zeros(n, bool)
    vb[idx] = btd.ball_visible[b_idx]
    print(f"graded {int(graded.sum())}; covered by accepted blocks {pct(cov[graded])}; tactical ball hidden "
          f"{pct(~vt[graded])} (in covered frames {pct(~vt[cov])})")
    print(f"TV ball visible | tactical visible: {pct(vb[cov & vt])}   | tactical hidden: {pct(vb[cov & ~vt])}")
    both = cov & vt & vb
    d = np.linalg.norm(td.ball_xy[both] - btd.ball_xy[bf[both]], axis=1)
    print(f"both visible n={int(both.sum())}: distance m p10/50/90 {q(d)}; within 1 m {pct(d < 1)}; > 5 m {pct(d > 5)}")
    print(f"hidden-tactical graded frames: covered {pct(cov[graded & ~vt])}; TV ball visible among covered "
          f"{pct(vb[cov & ~vt])}; TV ball visible among all hidden {pct((cov & vb)[graded & ~vt])}")

    L, W = td.pitch
    mg = p.pitch_margin_m

    def on_pitch(xy):
        return (np.isfinite(xy[..., 0]) & (xy[..., 0] > -mg) & (xy[..., 0] < L + mg) &
                (xy[..., 1] > -mg) & (xy[..., 1] < W + mg))

    tp, tt = td.players_xy, np.asarray(td.team_of_slot)
    bp, btm = btd.players_xy, np.asarray(btd.team_of_slot)
    n_t_all = on_pitch(tp).sum(axis=1)
    extras = np.zeros(n, np.int16)
    n_b = np.zeros(n, np.int16)
    for c0 in range(0, len(idx), 3000):
        ii = idx[c0:c0 + 3000]
        T, B = tp[ii], bp[bf[ii]]
        okT, okB = on_pitch(T), on_pitch(B)
        n_b[ii] = okB.sum(axis=1)
        ex = np.zeros(len(ii), np.int16)
        for team in (0, 1):
            Tt = np.where(okT[:, tt == team, None], T[:, tt == team], np.inf)
            Bt = np.where(okB[:, btm == team, None], B[:, btm == team], np.nan)
            with np.errstate(invalid="ignore"):
                dd = np.linalg.norm(Bt[:, :, None, :] - Tt[:, None, :, :], axis=-1)
            dd = np.where(np.isfinite(dd), dd, np.inf)
            dmin = dd.min(axis=2) if dd.shape[2] else np.full(dd.shape[:2], np.inf)
            ex += ((dmin > 2.0) & okB[:, btm == team]).sum(axis=1).astype(np.int16)
        extras[ii] = ex
    print(f"covered frames: tactical players on pitch mean {n_t_all[cov].mean():.2f}; TV players mean "
          f"{n_b[cov].mean():.2f}; extra TV players (same team, >2 m from every tactical one) mean {extras[cov].mean():.2f}")
    for thr in (20, 16):
        lo = cov & (n_t_all < thr)
        gl = graded & (n_t_all < thr)
        if lo.any():
            gain = n_t_all[lo] + np.minimum(extras[lo], 22 - n_t_all[lo])
            print(f"tactical < {thr} players: {pct(gl[graded])} of graded; covered by a block {pct(cov[gl])}; "
                  f"covered ones with >= 1 extra TV player {pct(extras[lo] >= 1)}; mean extras {extras[lo].mean():.2f}; "
                  f"reach >= {thr} players with the extras {pct(gain >= thr)}")

    valid_b = (b_idx < bh.n_frames)
    valid_b[valid_b] = bh.valid[b_idx[valid_b]]
    ci = idx[valid_b]
    bi = bf[ci]
    Hi = bh.H_inv[bi]
    cx, cy = project_pixels_to_pitch(Hi, np.full(len(bi), 960.0), np.full(len(bi), 540.0), tuple(td.pitch))
    xl, _ = project_pixels_to_pitch(Hi, np.full(len(bi), 0.0), np.full(len(bi), 540.0), tuple(td.pitch))
    xr, _ = project_pixels_to_pitch(Hi, np.full(len(bi), 1920.0), np.full(len(bi), 540.0), tuple(td.pitch))
    fov = np.abs(xr - xl)
    ok = np.isfinite(cx) & np.isfinite(cy) & (cx > -20) & (cx < L + 20) & (cy > -20) & (cy < W + 20)
    print(f"covered frames with a valid broadcast homography: {pct(valid_b)}; usable look-at point {pct(ok)}; "
          f"FOV width m p10/50/90 {q(fov[ok])}")
    cxy = np.stack([cx, cy], 1)
    mmv = vt[ci] & ok
    dl = np.linalg.norm(cxy[mmv] - td.ball_xy[ci[mmv]], axis=1)
    print(f"look-at vs tactical ball (visible, n={int(mmv.sum())}): distance m p10/50/90 {q(dl)}; within 10 m "
          f"{pct(dl < 10)}; within 20 m {pct(dl < 20)}")
    dx = np.abs(cxy[mmv, 0] - td.ball_xy[ci[mmv], 0])
    dy = np.abs(cxy[mmv, 1] - td.ball_xy[ci[mmv], 1])
    print(f"look-at |dx| m p10/50/90 {q(dx)}; |dy| {q(dy)}")
    has = np.zeros(n, bool)
    has[ci[ok]] = True
    print(f"hidden graded frames with a look-at proxy: {pct(has[graded & ~vt])}")
    for w in (20.0, 30.0):
        print(f"FOV < {w:.0f} m among covered usable frames: {pct(fov[ok] < w)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
