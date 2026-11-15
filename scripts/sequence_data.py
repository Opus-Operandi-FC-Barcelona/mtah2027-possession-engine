#!/usr/bin/env python
"""Issue #72: dense per-frame training sequences of the sequence model (external matches only; never BAR-ATH).

For every external match and degradation world, the front end + feature matrix of the specialists is computed on
the degraded tracking (home / away swapped copies included) and stored frame by frame (float16, NaN kept) with
the labels, the ball visibility and the play blocks, so that ``scripts/train_sequence.py`` can sample windows.

Worlds: IDSSE degraded with the calibrated degrader at ``dead_affinity`` 33 (two training seeds) and 1 (one seed);
PFF with its real occlusions (no synthetic degradation).  The training seeds (101..) never coincide with the
evaluation seeds of ``scripts/sequence_eval.py`` (0..8).

    python scripts/sequence_data.py build --out-dir D --jobs 6
"""
# ruff: noqa: E501
from __future__ import annotations

import argparse
import dataclasses
import json
import os
import sys
import time
from multiprocessing import get_context
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMBA_NUM_THREADS", "1")

import numpy as np  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
for _extra in (REPO_ROOT / "src", REPO_ROOT):
    if str(_extra) not in sys.path:
        sys.path.insert(0, str(_extra))

BENCH = REPO_ROOT.parent / "mtah2027-external-benchmarks" / "cache"
IDSSE_WORLDS = ((33.0, 101), (33.0, 102), (1.0, 103))      # (dead_affinity, base training seed)
PFF_TRAIN = slice(20, 56)                                    # PFF matches never used by the A / unseen cells


def _prepare(path, aff, seed):
    """Tracking of one world (degraded IDSSE or native PFF) exactly as ``_job`` builds it."""
    from mtah2027.models.external_degrader import DegraderParams, degrade_match
    from mtah2027.validation import eval_bridge as eb

    z = np.load(path, allow_pickle=True)
    players, team = z["players_xy"].astype(np.float32), z["team_of_slot"].astype(np.int8)
    ball = z["ball_xy"].astype(np.float32)
    fps, pitch = float(z["fps"]), tuple(float(v) for v in z["pitch"])
    periods = [(int(a), int(b)) for a, b in z["period_bounds"] if int(b) - int(a) >= 250]
    gt = eb.to_codes(z["gt"])
    if aff is not None:
        base = DegraderParams.from_config(None, "external_degrader")
        dm = degrade_match(players, team, ball, gt, fps=fps, seed=seed,
                           params=dataclasses.replace(base, dead_affinity=float(aff)))
        players, ball = dm.players_xy, dm.ball_xy
    else:
        if "ball_visible" in z.files:
            ball = np.where(z["ball_visible"][:, None], ball, np.float32(np.nan)).astype(np.float32)
    in_block = np.zeros(len(gt), bool)
    for a, b in periods:
        in_block[a:b + 1] = True
    return players, team, ball, fps, pitch, periods, gt, in_block


def _job_extra(args):
    """Issue #74: only the carrier / swarm columns (#64) of the worlds of ``_job`` (same file names, in ``out_dir``);
    the stored ball visibility is checked against the one of the stored sequence (same degradation)."""
    path, tag, aff, seed, out_dir, ref_dir = args
    from mtah2027.models.specialist_features import carrier_swarm_columns
    from mtah2027.pipeline import compute_front_end

    t0 = time.perf_counter()
    players, team, ball, fps, pitch, periods, _gt, _blk = _prepare(path, aff, seed)
    for swap in (False, True):
        tm = (1 - team).astype(np.int8) if swap else team
        name = f"{Path(path).stem}__{tag}__s{int(swap)}"
        fe = compute_front_end(players, tm, ball, play_blocks=periods, fps=fps, pitch=pitch, collective=True)
        ref = np.load(Path(ref_dir) / f"{name}.npz")
        if not np.array_equal(np.asarray(fe.visible, bool), ref["vis"]):
            raise RuntimeError(f"{name}: degradation does not reproduce the stored sequence")
        sx, names = carrier_swarm_columns(fe)
        np.savez(Path(out_dir) / f"{name}.npz", x=sx.astype(np.float16))
        (Path(out_dir) / "names.json").write_text(json.dumps(names), encoding="utf-8")
    return f"{Path(path).stem} {tag}: {time.perf_counter() - t0:.0f} s"


def _job(args):
    path, tag, aff, seed, out_dir = args
    from mtah2027.models.sequence_model import select_columns
    from mtah2027.models.specialist_data import swap_codes
    from mtah2027.models.specialist_features import build_feature_matrix
    from mtah2027.pipeline import compute_front_end, heuristic_emissions

    t0 = time.perf_counter()
    players, team, ball, fps, pitch, periods, gt, in_block = _prepare(path, aff, seed)
    for swap in (False, True):
        tm = (1 - team).astype(np.int8) if swap else team
        labels = swap_codes(gt) if swap else gt
        fe = compute_front_end(players, tm, ball, play_blocks=periods, fps=fps, pitch=pitch, collective=True)
        em = heuristic_emissions(fe, fps)
        fm = build_feature_matrix(fe, em, play_blocks=periods)
        names = select_columns(fm.names)
        idx = [fm.names.index(k) for k in names]
        y = np.where(in_block, labels, 3).astype(np.int8)
        name = f"{Path(path).stem}__{tag}__s{int(swap)}"
        np.savez(Path(out_dir) / f"{name}.npz", x=fm.x[:, idx].astype(np.float16), y=y,
                 vis=np.asarray(fe.visible, bool), blocks=np.asarray(periods, np.int64))
    (Path(out_dir) / "names.json").write_text(json.dumps(names), encoding="utf-8")
    return f"{Path(path).stem} {tag}: {time.perf_counter() - t0:.0f} s"


def cmd_build(a) -> int:
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    extra = bool(getattr(a, "ref_dir", None))
    jobs = []
    for i, f in enumerate(sorted(BENCH.glob("idsse_*.npz"))):
        for aff, seed in IDSSE_WORLDS:
            tag = f"a{int(aff)}_{seed}"
            if not (out / f"{f.stem}__{tag}__s1.npz").is_file():
                jobs.append((str(f), tag, aff, seed + i, str(out)) + ((a.ref_dir,) if extra else ()))
    for f in sorted(BENCH.glob("pff_*.npz"))[PFF_TRAIN]:
        if not (out / f"{f.stem}__native__s1.npz").is_file():
            jobs.append((str(f), "native", None, 0, str(out)) + ((a.ref_dir,) if extra else ()))
    print(f"{len(jobs)} jobs", flush=True)
    with get_context("spawn").Pool(a.jobs) as pool:
        for msg in pool.imap_unordered(_job_extra if extra else _job, jobs):
            print(msg, flush=True)
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--out-dir", required=True)
    b.add_argument("--jobs", type=int, default=6)
    b.add_argument("--ref-dir", default=None, help="Issue #74: only the carrier / swarm columns, checked against "
                   "the sequences stored in this directory")
    args = ap.parse_args(argv)
    return cmd_build(args)


if __name__ == "__main__":
    raise SystemExit(main())
