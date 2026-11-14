#!/usr/bin/env python
"""Render an audit video with the pipeline's view of the match over the tactical video (Issue #36).

Pitch lines projected with the per-frame homography, a 2-D minimap (22 players, ball, possessor, collective
coherence arrows), a 25-frame ball trail, a halo on the possessor and a HUD with the predicted H/A/D,
P(H)/P(A)/P(D) and whether the ball is visible or hidden.

    python scripts/render_audit_video.py --synthetic                              # synthetic match, no data
    python scripts/render_audit_video.py --data-dir data --start-frame 11185 --end-frame 11500
    python scripts/render_audit_video.py --data-dir data --from-audit-csv outputs/audit_transitions.csv

``--from-audit-csv`` renders a window around every transition exported by KPI 5 of the dashboard.  The
background is the tactical mp4 when it exists under ``<data-dir>/tactical/`` (``--video`` overrides it) and a
plain pitch otherwise.  The output goes to ``outputs/audit_video.mp4`` (git-ignored, like every ``*.mp4``).

RULES: the audit video of BAR-ATH is for LOCAL pipeline debugging only (alignment, format, bugs).  Never use
it to label frames by hand or to tune parameters, and never upload it anywhere.  With real data this script
prints only aggregates and paths, never per-frame rows.
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
for extra in (REPO_ROOT / "src", REPO_ROOT, REPO_ROOT / "scripts"):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

from mtah2027.config import MatchConfig  # noqa: E402
from mtah2027.vision import overlay_renderer as ovl  # noqa: E402

LETTER_TO_CODE = {"H": 0, "A": 1, "D": 2, "X": 3}


def read_audit_csv(path) -> list[int]:
    """Frames of the transitions exported by KPI 5 (column ``frame``)."""
    with open(path, newline="", encoding="utf-8") as handle:
        return [int(row["frame"]) for row in csv.DictReader(handle)]


def read_labels_csv(path, n: int) -> np.ndarray:
    """A submission ``frame,label`` CSV as codes (H=0 A=1 D=2 X=3), padded / cut to ``n`` frames."""
    codes = np.full(n, 3, dtype=np.int8)
    with open(path, newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            f = int(row["frame"])
            if 0 <= f < n:
                codes[f] = LETTER_TO_CODE[row["label"].strip().upper()]
    return codes


def build_audit_data(data_dir, match: MatchConfig, *, config=None, labels_csv=None, log=None):
    """Run the pipeline on ``data_dir`` and gather the arrays of the overlay: ``(AuditData, video_path)``."""
    import games
    from mtah2027.io.atd_loader import load_atd
    from mtah2027.io.homography_loader import load_homography
    from mtah2027.pipeline import _fit_length, run_pipeline

    n = match.n_video_frames
    res = run_pipeline(data_dir, config, match, log=log or (lambda m: None), keep_features=True)
    td = load_atd(data_dir, n_frames=n)
    try:
        hom = load_homography(data_dir)
        H = _fit_length(hom.H.reshape(len(hom.H), 9), n).reshape(n, 3, 3)
    except (OSError, ValueError, KeyError, FileNotFoundError):
        H = None
    log_probs = res.features.get("log_probs")
    probs = ovl.probabilities_from_log(log_probs) if log_probs is not None else None
    labels = read_labels_csv(labels_csv, n) if labels_csv else res.labels
    visible = res.features.get("visible")
    data = ovl.AuditData(
        players_xy=_fit_length(td.players_xy, n), team_of_slot=td.team_of_slot,
        ball_xy=_fit_length(td.ball_xy, n), labels=labels, H=H, probs=probs,
        ball_visible=np.asarray(visible, dtype=bool) if visible is not None else None,
        pitch=tuple(td.pitch), fps=float(td.fps))
    video = Path(games.asset_paths(data_dir=str(data_dir) if data_dir else None)["video"])
    return data, video


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    src = ap.add_mutually_exclusive_group()
    src.add_argument("--synthetic", action="store_true", help="render a generated synthetic match (no data)")
    src.add_argument("--data-dir", default=None, help="folder with tactical/ (real or synthetic bundle)")
    ap.add_argument("--synthetic-frames", type=int, default=12000)
    ap.add_argument("--start-frame", type=int, default=None)
    ap.add_argument("--end-frame", type=int, default=None, help="inclusive")
    ap.add_argument("--from-audit-csv", default=None, help="outputs/audit_transitions.csv of KPI 5")
    ap.add_argument("--pad-seconds", type=float, default=3.0, help="window around each audited transition")
    ap.add_argument("--labels-csv", default=None, help="draw the labels of this submission CSV instead")
    ap.add_argument("--video", default=None, help="tactical mp4 (default: the one of the data bundle)")
    ap.add_argument("--config", default=None, help="YAML configuration (default configs/default.yaml)")
    ap.add_argument("--out", default=None, help="output mp4 (default outputs/audit_video.mp4)")
    ap.add_argument("--scale", type=float, default=0.5, help="output size vs the video (default 0.5)")
    ap.add_argument("--trail", type=int, default=25, help="ball trail length in frames")
    ap.add_argument("--max-frames", type=int, default=0, help="stop after this many frames (0 = no cap)")
    args = ap.parse_args(argv)

    out = Path(args.out) if args.out else REPO_ROOT / "outputs" / "audit_video.mp4"
    tmp = None
    try:
        if args.synthetic:
            import tempfile

            import generate_synthetic_fixtures as gsf

            tmp = tempfile.TemporaryDirectory(prefix="mtah_audit_")
            info = gsf.generate_synthetic_match(tmp.name, seed=0, n_frames=args.synthetic_frames)
            spec = info["spec"]
            match = MatchConfig(n_video_frames=spec.n_frames, half1=spec.half1, half2_dead=spec.half2_dead,
                                half2_play=spec.half2_play)
            data_dir = tmp.name
            if args.start_frame is None and not args.from_audit_csv:
                args.start_frame = spec.half1[0] + 100
                args.end_frame = args.start_frame + 100
        else:
            match = MatchConfig()
            data_dir = args.data_dir
            if args.start_frame is None and not args.from_audit_csv:
                ap.error("give --start-frame/--end-frame or --from-audit-csv (or use --synthetic)")
        data, video = build_audit_data(data_dir, match, config=args.config, labels_csv=args.labels_csv)
        video = Path(args.video) if args.video else video
        n = data.n_frames
        if args.from_audit_csv:
            frames = read_audit_csv(args.from_audit_csv)
            windows = ovl.audit_windows(frames, n, int(round(args.pad_seconds * data.fps)))
        else:
            end = args.end_frame if args.end_frame is not None else args.start_frame + int(5 * data.fps)
            windows = [(max(int(args.start_frame), 0), min(int(end), n - 1))]
        stats = ovl.render_video(data, windows, out, video_path=video, scale=args.scale, trail=args.trail,
                                 max_frames=args.max_frames)
    except (ovl.OverlayUnavailableError, ImportError, FileNotFoundError) as exc:
        print(f"audit video unavailable: {exc}")
        return 2
    finally:
        if tmp is not None:
            tmp.cleanup()
    print(f"audit video: {stats['frames_written']} frames in {stats['clips']} clip(s), "
          f"{stats['width']}x{stats['height']}, background={stats['background']}")
    print(f"written to {stats['path']} (git-ignored; local debugging only, never label or tune on it)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
