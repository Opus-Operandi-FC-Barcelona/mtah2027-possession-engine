"""Quality-control dashboard with seven KPIs (Issue #27).  Aggregate numbers only -- never per-frame rows.

The report (a dict, printed as a table and written as JSON) is safe to paste: it holds counts, shares, scores,
hashes and timings.  The only per-frame artefact is the optional audit CSV (KPI 5), which is written to the
git-ignored ``outputs/`` folder and reported by its *count* only.

1. Macroscopic sanity ALARMS (possessions, mean duration, %D, %H of alive): OK / WARN.
2. Artificial blindness (no GT): hide the ball in extra blocks on the first graded minutes, rerun, compare
   with the unmasked run (spellF1 >= ``blindness_target``).
3. Degraded external evaluation (other matches, Issue #19 degrader): spellF1 >= 0.65, latency < 8 frames.
4. Pipeline smoke test on the public 5 minutes (real data only; NEVER for model selection): INFO.
5. Export of ``audit_n`` transitions for the visual audit: count only.
6. Official ``validate_submission`` + SHA256 of the submission file: OK / FAIL.
7. Route comparison without ground truth: specialists vs heuristic route, agreement diagnostics and the
   external-evidence decision rule.

The KPI ranges of KPI 1 are generic football priors used as *alarms*; no KPI is ever used to choose
hyper-parameters (``EVALUATION.md``).  The route decision of KPI 7 uses the external benchmark only
(leave-one-match-out specialists vs heuristic route on degraded matches); the agreement between the two routes
on the target match is a diagnostic, and the public ground truth is never consulted for it.
"""

from __future__ import annotations

import csv
import hashlib
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np

from mtah2027.config import FPS, REPO_ROOT, KpiAlarms, load_section, params_from_mapping
from mtah2027.models.external_degrader import degrade_match
from mtah2027.postprocessing.submission_contract import graded_runs
from mtah2027.validation import eval_bridge as eb

H, A, D, X = 0, 1, 2, 3
LETTERS = "HADX"
OK, WARN, FAIL, INFO, SKIPPED = "OK", "WARN", "FAIL", "INFO", "SKIPPED"
SMOKE_NOTE = "pipeline check only - not for model selection"


@dataclass(frozen=True)
class KpiParams:
    blindness_minutes: float = 20.0
    blindness_hide_frac: float = 0.41
    blindness_seed: int = 0
    blindness_target: float = 0.75
    external_spell_target: float = 0.65
    latency_target_frames: float = 8.0
    latency_match_window_s: float = 5.0
    external_seeds: tuple = (0,)
    bootstrap_n: int = 200
    audit_n: int = 20
    route_margin_spellf1: float = 0.04

    @classmethod
    def from_config(cls, path=None) -> "KpiParams":
        return params_from_mapping(cls, load_section("kpi_dashboard", path), "kpi_dashboard")


# --------------------------------------------------------------------------------------------
# Label statistics and KPI 1
# --------------------------------------------------------------------------------------------
def label_statistics(codes: np.ndarray, runs: Sequence[tuple[int, int]], fps: float = FPS) -> dict:
    """Aggregate statistics of ``codes`` over the graded ``runs`` (inclusive blocks)."""
    codes = np.asarray(codes)
    n_graded = alive = n_h = n_d = n_poss = 0
    for a, b in runs:
        seg = codes[a : b + 1]
        if not len(seg):
            continue
        n_graded += len(seg)
        n_h += int((seg == H).sum())
        n_d += int((seg == D).sum())
        alive += int(((seg == H) | (seg == A)).sum())
        starts = np.concatenate([[True], seg[1:] != seg[:-1]])
        n_poss += int(((seg == H) | (seg == A))[starts].sum())
    return {
        "graded_frames": n_graded,
        "n_possessions": n_poss,
        "mean_possession_s": alive / max(n_poss, 1) / fps,
        "dead_pct_of_graded": 100.0 * n_d / max(n_graded, 1),
        "home_pct_of_alive": 100.0 * n_h / max(alive, 1),
    }


def kpi_alarms(stats: dict, alarms: KpiAlarms) -> dict:
    """KPI 1: macroscopic sanity alarms (WARN outside the generic ranges; never a tuning target)."""
    spec = (
        ("possessions", stats["n_possessions"], alarms.possessions),
        ("mean_duration_s", stats["mean_possession_s"], alarms.mean_duration_s),
        ("dead_pct_of_graded", stats["dead_pct_of_graded"], alarms.dead_pct_of_graded),
        ("home_pct_of_alive", stats["home_pct_of_alive"], alarms.home_pct_of_alive),
    )
    items = []
    for name, value, (low, high) in spec:
        items.append(
            {
                "name": name,
                "value": float(value),
                "low": float(low),
                "high": float(high),
                "status": OK if low <= value <= high else WARN,
            }
        )
    status = OK if all(i["status"] == OK for i in items) else WARN
    return {
        "status": status,
        "alarms": items,
        "graded_frames": stats["graded_frames"],
        "note": "alarms from generic football priors; never used for tuning",
    }


# --------------------------------------------------------------------------------------------
# KPI 2: artificial blindness
# --------------------------------------------------------------------------------------------
def blindness_region(half1: tuple[int, int], minutes: float, fps: float) -> tuple[int, int]:
    """First ``minutes`` of the first graded block (clipped to the block)."""
    a = int(half1[0])
    return a, min(int(half1[1]), a + int(round(minutes * 60.0 * fps)) - 1)


def kpi_blindness(
    players_xy: np.ndarray,
    team_of_slot: np.ndarray,
    ball_xy: np.ndarray,
    half1: tuple[int, int],
    *,
    smart_tagging: Any = None,
    config_path=None,
    specialists: Any = None,
    fps: float = FPS,
    pitch: tuple[float, float] = (105.3, 68.0),
    params: KpiParams | None = None,
) -> dict:
    """KPI 2.  Self-consistency without ground truth on the first graded minutes of the target match.

    The unmasked run's labels act as the reference.  The ball is then additionally hidden in synthetic blocks
    (the realistic crowding-weighted occlusion model of Issue #19, no tracking noise; alive frames are those
    of the reference run) and the pipeline is rerun; the score is the official spellF1 of masked vs reference.
    """
    from mtah2027.pipeline import run_core

    p = params or KpiParams.from_config(config_path)
    a, b = blindness_region(half1, p.blindness_minutes, fps)

    def run(ball: np.ndarray) -> np.ndarray:
        return run_core(
            players_xy,
            team_of_slot,
            ball,
            play_blocks=[(a, b)],
            fps=fps,
            pitch=pitch,
            smart_tagging=smart_tagging,
            config_path=config_path,
            log=lambda m: None,
            specialists=specialists,
        ).codes

    reference = run(ball_xy)
    degraded = degrade_match(
        players_xy,
        team_of_slot,
        ball_xy,
        reference,
        fps=fps,
        seed=p.blindness_seed,
        hide_frac=p.blindness_hide_frac,
        artefacts=False,
    )
    masked = run(degraded.ball_xy)
    score = eb.evaluate_arrays(reference, masked)
    alive = np.zeros(len(reference), bool)
    alive[a : b + 1] = (reference[a : b + 1] == H) | (reference[a : b + 1] == A)
    seen = np.isfinite(ball_xy).all(axis=1)
    newly = seen & ~np.isfinite(degraded.ball_xy).all(axis=1) & alive
    spell = float(score["spell_f1_mean"])
    return {
        "status": OK if spell >= p.blindness_target else WARN,
        "spell_f1": spell,
        "macro_f1": float(score["macro_f1"]),
        "target": p.blindness_target,
        "minutes": (b - a + 1) / fps / 60.0,
        "ball_visible_share_alive": float(seen[alive].mean()) if alive.any() else 0.0,
        "newly_hidden_share_of_visible_alive": float(newly.sum() / max((seen & alive).sum(), 1)),
        "possessions_reference": int(score["n_gt_spells"]),
        "possessions_masked": int(score["n_pred_spells"]),
        "note": "self-consistency, no ground truth",
    }


# --------------------------------------------------------------------------------------------
# KPI 3: degraded external matches
# --------------------------------------------------------------------------------------------
def _transitions(codes: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    c = np.asarray(codes).astype(np.int64)
    idx = np.flatnonzero((c[1:] != c[:-1]) & (c[1:] != X) & (c[:-1] != X)) + 1
    return idx, c[idx]


def transition_latency(
    gt: np.ndarray, pred: np.ndarray, fps: float, window_s: float
) -> tuple[np.ndarray, int]:
    """Absolute latency (frames) of every true transition matched to the nearest predicted transition into the
    same label within ``window_s``; returns ``(latencies, n_true_transitions)`` (unmatched: counted only)."""
    gi, gl = _transitions(gt)
    pi, pl = _transitions(pred)
    window = int(round(window_s * fps))
    out: list[np.ndarray] = []
    for label in (H, A, D):
        g, p = gi[gl == label], pi[pl == label]
        if not len(g) or not len(p):
            continue
        pos = np.searchsorted(p, g)
        left, right = p[np.clip(pos - 1, 0, len(p) - 1)], p[np.clip(pos, 0, len(p) - 1)]
        d = np.minimum(np.abs(g - left), np.abs(g - right))
        out.append(d[d <= window])
    lat = np.concatenate(out) if out else np.zeros(0, dtype=np.int64)
    return lat, int(len(gi))


def _eval_external_module():
    scripts = str(REPO_ROOT / "scripts")
    if scripts not in sys.path:
        sys.path.insert(0, scripts)
    import eval_external

    return eval_external


def _route_summary(rows: list[dict], key: str, fps_of: Callable[[dict], float], window_s: float) -> dict:
    lats, n_true = [], 0
    for r in rows:
        pred = r["_codes"][key]
        lat, n = transition_latency(r["_codes"]["gt"], pred, fps_of(r), window_s)
        lats.append(lat)
        n_true += n
    lat = np.concatenate(lats) if lats else np.zeros(0)
    return {
        "spell_f1": float(np.mean([r[key]["spell_f1"] for r in rows])),
        "macro_f1": float(np.mean([r[key]["macro_f1"] for r in rows])),
        "possessions_pred_mean": float(np.mean([r[key]["n_pred_spells"] for r in rows])),
        "possessions_true_mean": float(np.mean([r[key]["n_gt_spells"] for r in rows])),
        "latency_median_frames": float(np.median(lat)) if len(lat) else float("nan"),
        "latency_matched_share": float(len(lat) / max(n_true, 1)),
        "n_matches": len(rows),
    }


def kpi_external(
    bench_dir: str | Path,
    *,
    config_path=None,
    params: KpiParams | None = None,
    lomo: bool = False,
    log: Callable[[str], None] | None = None,
) -> dict:
    """KPI 3: realistic degradation of the external matches; the heuristic route (and, with ``lomo``, the
    specialists trained leave-one-match-out) scored with the official metric, plus the transition latency."""
    p = params or KpiParams.from_config(config_path)
    say = log or (lambda m: None)
    ee = _eval_external_module()
    files = sorted(Path(bench_dir).glob("*.npz"))
    if not files:
        return {"status": SKIPPED, "reason": f"no benchmark files in {bench_dir}"}
    cfg = str(config_path) if config_path is not None else None
    models: dict = {}
    training = None
    if lomo:
        from mtah2027.models import specialist_fusion as sf
        from mtah2027.models.specialist_data import build_training_set

        training = build_training_set(bench_dir, config_path=cfg, log=say)

        def specialists_for(stem: str):
            if stem not in models:
                models[stem] = sf.train_specialists(
                    bench_dir, config_path=cfg, training=training, exclude_match=stem
                )
            return models[stem]

    rows = []
    for seed in p.external_seeds:
        for i, f in enumerate(files):
            row = ee.evaluate_match(
                f,
                0.0,
                int(seed) + i,
                cfg,
                False,
                True,
                specialists_for(f.stem) if lomo else None,
                return_codes=True,
            )
            rows.append(row)
            say(f"  external {f.stem} seed {seed} done")
    fps_of = lambda r: float(r["_codes"]["fps"])  # noqa: E731
    routes = {"heuristic": _route_summary(rows, "engine", fps_of, p.latency_match_window_s)}
    if lomo:
        routes["specialists_lomo"] = _route_summary(rows, "specialists", fps_of, p.latency_match_window_s)
    primary = "specialists_lomo" if lomo else "heuristic"
    r0 = routes[primary]
    ok = r0["spell_f1"] >= p.external_spell_target and r0["latency_median_frames"] < p.latency_target_frames
    return {
        "status": OK if ok else WARN,
        "primary_route": primary,
        "routes": routes,
        "targets": {"spell_f1": p.external_spell_target, "latency_median_frames": p.latency_target_frames},
        "n_matches": len(files),
        "seeds": [int(s) for s in p.external_seeds],
        "note": "other matches, realistic degrader; no BAR-ATH data",
    }


# --------------------------------------------------------------------------------------------
# KPI 4, 5, 6
# --------------------------------------------------------------------------------------------
def kpi_public_smoke(labels: np.ndarray, *, bootstrap_n: int = 200) -> dict:
    """KPI 4: official metrics against the public 5-minute ground truth (pipeline check only)."""
    r = eb.score_against_public_gt(labels, bootstrap_n=bootstrap_n)
    return {
        "status": INFO,
        "note": SMOKE_NOTE,
        "spell_f1": float(r["spell_f1_mean"]),
        "spell_f1_low": float(r["spell_bootstrap_low"]),
        "spell_f1_high": float(r["spell_bootstrap_high"]),
        "macro_f1": float(r["macro_f1"]),
        "bootstrap_n": int(bootstrap_n),
        "n_graded": int(r["n_graded"]),
        "abstain_on_graded": int(r["n_abstain_on_graded"]),
    }


def select_audit_transitions(codes: np.ndarray, runs: Sequence[tuple[int, int]], n: int) -> np.ndarray:
    """``n`` transition frames (label changes inside the graded runs) evenly spread over the match."""
    frames = []
    for a, b in runs:
        seg = np.asarray(codes[a : b + 1])
        frames.append(np.flatnonzero(seg[1:] != seg[:-1]) + 1 + a)
    allf = np.concatenate(frames) if frames else np.zeros(0, dtype=np.int64)
    if len(allf) <= n:
        return allf.astype(np.int64)
    pick = np.round(np.linspace(0, len(allf) - 1, n)).astype(np.int64)
    return allf[np.unique(pick)].astype(np.int64)


def kpi_audit(
    codes: np.ndarray, runs: Sequence[tuple[int, int]], path: str | Path, n: int, fps: float = FPS
) -> dict:
    """KPI 5: write ``frame,time_s,clock,from,to`` of ``n`` transitions to ``path``; count only."""
    frames = select_audit_transitions(codes, runs, n)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(["frame", "time_s", "clock", "from", "to"])
        for f in frames:
            t = float(f) / fps
            writer.writerow(
                [
                    int(f),
                    f"{t:.2f}",
                    f"{int(t // 60):02d}:{t % 60:05.2f}",
                    LETTERS[int(codes[f - 1])],
                    LETTERS[int(codes[f])],
                ]
            )
    return {
        "status": INFO,
        "n_exported": int(len(frames)),
        "requested": int(n),
        "note": "derived data, git-ignored (outputs/); never commit or paste",
    }


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def kpi_format(path: str | Path, n_expected: int) -> dict:
    """KPI 6: official format check and SHA256 of the submission file."""
    result = eb.validate_submission_file(path, n_expected)
    out = {
        "status": OK if result else FAIL,
        "validate_submission": "OK" if result else f"FAIL: {result.reason}",
        "rows": int(n_expected),
    }
    if Path(path).is_file():
        out["sha256"] = file_sha256(path)
    return out


# --------------------------------------------------------------------------------------------
# KPI 7: route comparison without ground truth
# --------------------------------------------------------------------------------------------
def decide_route(external: dict | None, agreement: dict | None, margin: float) -> dict:
    """Which route to submit, from the external-benchmark evidence only (never the public GT).

    ``specialists`` when the leave-one-match-out specialists beat the heuristic route on the degraded external
    matches by at least ``margin`` spellF1 (the metric's noise level); ``heuristic`` when they do worse than
    ``-margin`` or the difference is within the noise (the training-free route is the safer default); and
    ``undetermined`` when the external evidence for both routes is missing.  The agreement of the two routes
    on the target match is reported as a domain-shift diagnostic only.
    """
    routes = (external or {}).get("routes", {})
    note = None
    if agreement is not None and agreement.get("spell_f1") is not None and agreement["spell_f1"] < 0.5:
        note = "routes disagree strongly on the target match: possible domain shift; use external evidence"
    if "heuristic" not in routes or "specialists_lomo" not in routes:
        return {
            "recommended": "undetermined",
            "basis": "external leave-one-match-out evidence missing "
            "(run with the external benchmark and --external-lomo)",
            "agreement_note": note,
        }
    gain = routes["specialists_lomo"]["spell_f1"] - routes["heuristic"]["spell_f1"]
    if gain >= margin:
        choice, why = "specialists", f"LOMO gain {gain:+.4f} >= margin {margin:.2f}"
    elif gain <= -margin:
        choice, why = "heuristic", f"LOMO gain {gain:+.4f} <= -{margin:.2f}"
    else:
        choice, why = "heuristic", f"LOMO gain {gain:+.4f} within the noise (+-{margin:.2f}); simpler route"
    return {
        "recommended": choice,
        "basis": why,
        "external_gain_spell_f1": float(gain),
        "agreement_note": note,
    }


def kpi_routes(
    labels_specialists: np.ndarray | None,
    labels_heuristic: np.ndarray,
    runs: Sequence[tuple[int, int]],
    fps: float,
    external: dict | None,
    margin: float,
) -> dict:
    """KPI 7: agreement of the learned and the heuristic route on the target match plus the decision rule."""
    stats_h = label_statistics(labels_heuristic, runs, fps)
    out: dict = {"status": INFO, "heuristic": _stat_row(stats_h)}
    if labels_specialists is None:
        out["specialists"] = None
        out["agreement"] = None
        out["decision"] = decide_route(external, None, margin)
        out["note"] = "specialists unavailable: heuristic route only"
        return out
    stats_s = label_statistics(labels_specialists, runs, fps)
    r = eb.evaluate_arrays(labels_heuristic, labels_specialists)
    graded = np.zeros(len(labels_heuristic), bool)
    for a, b in runs:
        graded[a : b + 1] = True
    agree = {
        "spell_f1": float(r["spell_f1_mean"]),
        "macro_f1": float(r["macro_f1"]),
        "frame_agreement": float(
            (np.asarray(labels_heuristic)[graded] == np.asarray(labels_specialists)[graded]).mean()
        ),
    }
    out.update(
        {
            "specialists": _stat_row(stats_s),
            "agreement": agree,
            "decision": decide_route(external, agree, margin),
            "note": "agreement is a diagnostic; the route is chosen from external evidence only",
        }
    )
    return out


def _stat_row(s: dict) -> dict:
    return {
        "possessions": int(s["n_possessions"]),
        "mean_possession_s": float(s["mean_possession_s"]),
        "dead_pct_of_graded": float(s["dead_pct_of_graded"]),
        "home_pct_of_alive": float(s["home_pct_of_alive"]),
    }


# --------------------------------------------------------------------------------------------
# Orchestration and presentation
# --------------------------------------------------------------------------------------------
def _timed(report: dict, name: str, fn: Callable[[], dict]) -> dict:
    t0 = time.perf_counter()
    try:
        result = fn()
    except Exception as exc:  # noqa: BLE001 - a failing KPI must not hide the others
        result = {"status": FAIL, "error": f"{type(exc).__name__}: {exc}"}
    result["seconds"] = round(time.perf_counter() - t0, 1)
    report["kpis"][name] = result
    return result


def run_dashboard(
    *,
    data_dir,
    blocks,
    specialists=None,
    submission_path: str | Path | None = None,
    config_path=None,
    bench_dir=None,
    skip_external: bool = False,
    external_lomo: bool = False,
    real_data: bool = False,
    out_json: str | Path | None = None,
    audit_path: str | Path | None = None,
    cache_dir=None,
    truth: np.ndarray | None = None,
    log: Callable[[str], None] | None = None,
) -> dict:
    """Run the seven KPIs on the match in ``data_dir`` and return (and optionally write) the aggregate report.

    ``blocks`` is the :class:`~mtah2027.config.MatchConfig`; ``specialists`` the learned set (or ``None``);
    ``submission_path`` an existing submission CSV (otherwise the primary route is run and written to
    ``outputs/kpi_candidate.csv``); ``real_data`` enables KPI 4; ``truth`` (synthetic runs only) adds the true
    score of both routes under the key ``synthetic_truth``.
    """
    import json

    from mtah2027.config import load_config, set_global_seeds
    from mtah2027.io.atd_loader import load_atd
    from mtah2027.io.smart_tagging_loader import load_smart_tagging
    from mtah2027.pipeline import _fit_length, resolve_config, run_pipeline
    from mtah2027.postprocessing.submission_contract import write_submission_csv

    say = log or (lambda m: None)
    cfg_path, _ = resolve_config(config_path)
    config = load_config(cfg_path)
    p = KpiParams.from_config(cfg_path)
    set_global_seeds(config.seed)
    n = blocks.n_video_frames
    runs = graded_runs(blocks)
    fps = float(blocks.fps)
    outputs = REPO_ROOT / "outputs"
    report: dict = {
        "match": {"frames": n, "graded_frames": int(sum(b - a + 1 for a, b in runs))},
        "kpis": {},
        "note": "aggregate numbers only",
    }
    t_all = time.perf_counter()

    def route(sp) -> np.ndarray:
        return run_pipeline(
            data_dir, cfg_path, blocks, cache_dir=cache_dir, log=lambda m: None, specialists=sp
        ).labels

    t0 = time.perf_counter()
    labels_heur = route(None)
    labels_spec = route(specialists) if specialists is not None else None
    report["route_seconds"] = round(time.perf_counter() - t0, 1)
    primary = labels_spec if labels_spec is not None else labels_heur
    report["primary_route"] = "specialists" if labels_spec is not None else "heuristic"
    if submission_path is not None:
        sub_path = Path(submission_path)
        labels = (
            eb.read_submission_codes(sub_path, n) if eb.validate_submission_file(sub_path, n) else primary
        )
        report["primary_route"] = "submission file"
    else:
        sub_path = outputs / "kpi_candidate.csv"
        write_submission_csv(sub_path, primary, n_expected=n)
        labels = primary

    _timed(report, "1_alarms", lambda: kpi_alarms(label_statistics(labels, runs, fps), config.kpi_alarms))

    def blindness() -> dict:
        td = load_atd(data_dir, cache_dir=cache_dir, use_cache=cache_dir is not None, n_frames=n)
        smart = load_smart_tagging(data_dir, n_frames=n)
        return kpi_blindness(
            _fit_length(td.players_xy, n),
            td.team_of_slot,
            _fit_length(td.ball_xy, n),
            blocks.half1,
            smart_tagging=smart,
            config_path=cfg_path,
            specialists=specialists,
            fps=float(td.fps),
            pitch=tuple(td.pitch),
            params=p,
        )

    _timed(report, "2_blindness", blindness)

    bench = Path(bench_dir) if bench_dir else None
    if skip_external or bench is None or not bench.is_dir():
        reason = "skipped by option" if skip_external else "external benchmark directory not found"
        report["kpis"]["3_external"] = {"status": SKIPPED, "reason": reason}
    else:
        _timed(
            report,
            "3_external",
            lambda: kpi_external(bench, config_path=cfg_path, params=p, lomo=external_lomo, log=say),
        )
    external = report["kpis"]["3_external"] if report["kpis"]["3_external"].get("routes") else None

    from mtah2027.config import N_VIDEO_FRAMES

    if real_data and len(labels) == N_VIDEO_FRAMES and eb.PUBLIC_GT_PATH.is_file():
        _timed(report, "4_public_smoke", lambda: kpi_public_smoke(labels, bootstrap_n=p.bootstrap_n))
    else:
        report["kpis"]["4_public_smoke"] = {
            "status": SKIPPED,
            "reason": "needs the real match (not a synthetic run)",
            "note": SMOKE_NOTE,
        }

    audit = Path(audit_path) if audit_path else outputs / "audit_transitions.csv"
    _timed(report, "5_audit", lambda: kpi_audit(labels, runs, audit, p.audit_n, fps))
    _timed(report, "6_format", lambda: kpi_format(sub_path, n))
    _timed(
        report,
        "7_routes",
        lambda: kpi_routes(labels_spec, labels_heur, runs, fps, external, p.route_margin_spellf1),
    )
    if truth is not None:
        sy = {"heuristic": eb.evaluate_arrays(truth, labels_heur)}
        if labels_spec is not None:
            sy["specialists"] = eb.evaluate_arrays(truth, labels_spec)
        report["synthetic_truth"] = {
            k: {
                "spell_f1": float(v["spell_f1_mean"]),
                "macro_f1": float(v["macro_f1"]),
                "possessions_pred": int(v["n_pred_spells"]),
                "possessions_true": int(v["n_gt_spells"]),
            }
            for k, v in sy.items()
        }
    report["seconds_total"] = round(time.perf_counter() - t_all, 1)
    if out_json is not None:
        out_json = Path(out_json)
        out_json.parent.mkdir(parents=True, exist_ok=True)
        out_json.write_text(json.dumps(report, indent=2, default=float), encoding="utf-8")
    return report


def _fmt(x: Any, nd: int = 3) -> str:
    return (
        "n/a"
        if x is None or (isinstance(x, float) and x != x)
        else (f"{x:.{nd}f}" if isinstance(x, float) else str(x))
    )


def format_report(report: dict) -> str:
    """Human-readable table of the report (aggregate numbers only)."""
    k = report["kpis"]
    lines = [
        f"KPI dashboard   frames={report['match']['frames']}  graded={report['match']['graded_frames']}  "
        f"route={report.get('primary_route')}",
        "-" * 100,
    ]

    def row(name: str, status: str, text: str) -> None:
        lines.append(f"{name:<22}{status:<9}{text}")

    a = k["1_alarms"]
    if "alarms" in a:
        row(
            "1 sanity alarms",
            a["status"],
            "  ".join(
                f"{i['name']}={_fmt(i['value'], 1)}[{i['low']:g}..{i['high']:g}]"
            f"{'' if i['status'] == OK else '!'}"
                for i in a["alarms"]
            ),
        )
    else:
        row("1 sanity alarms", a["status"], a.get("error", ""))
    b = k["2_blindness"]
    row(
        "2 blindness",
        b["status"],
        b.get("error")
        or (
            f"spellF1 {_fmt(b['spell_f1'])} (target >= {b['target']:g}) over {_fmt(b['minutes'], 1)} min; "
            f"{100 * b['newly_hidden_share_of_visible_alive']:.0f}% of the visible alive ball newly hidden; "
            f"possessions {b['possessions_reference']} -> {b['possessions_masked']}"
        ),
    )
    e = k["3_external"]
    if e.get("routes"):
        txt = "; ".join(
            f"{name}: spellF1 {_fmt(v['spell_f1'])} latency {_fmt(v['latency_median_frames'], 1)} fr "
            f"(matched {100 * v['latency_matched_share']:.0f}%)"
            for name, v in e["routes"].items()
        )
        row(
            "3 external degraded",
            e["status"],
            f"{txt} (targets >= {e['targets']['spell_f1']:g}, "
            f"< {e['targets']['latency_median_frames']:g} fr; "
            f"{e['n_matches']} matches)",
        )
    else:
        row("3 external degraded", e["status"], e.get("reason") or e.get("error", ""))
    s = k["4_public_smoke"]
    if "spell_f1" in s:
        row(
            "4 public 5 min",
            s["status"],
            f"spellF1 {_fmt(s['spell_f1'])} [{_fmt(s['spell_f1_low'])}, "
            f"{_fmt(s['spell_f1_high'])}]  macroF1 {_fmt(s['macro_f1'])}  -- {SMOKE_NOTE}",
        )
    else:
        row("4 public 5 min", s["status"], f"{s.get('reason') or s.get('error', '')} -- {SMOKE_NOTE}")
    u = k["5_audit"]
    row(
        "5 audit export",
        u["status"],
        u.get("error") or f"{u['n_exported']} transitions written (git-ignored)",
    )
    f = k["6_format"]
    row(
        "6 format + SHA256",
        f["status"],
        f.get("error") or f"validate_submission {f['validate_submission']}  sha256 {f.get('sha256', 'n/a')}",
    )
    r = k["7_routes"]
    if "heuristic" in r:
        h = r["heuristic"]
        txt = (
            f"heuristic: {h['possessions']} poss, {h['mean_possession_s']:.1f} s, "
            f"D {h['dead_pct_of_graded']:.0f}%"
        )
        if r.get("specialists"):
            sp = r["specialists"]
            ag = r["agreement"]
            txt += (
                f" | specialists: {sp['possessions']} poss, {sp['mean_possession_s']:.1f} s, "
                f"D {sp['dead_pct_of_graded']:.0f}% | agreement spellF1 {_fmt(ag['spell_f1'])} "
                f"macroF1 {_fmt(ag['macro_f1'])} frames {100 * ag['frame_agreement']:.1f}%"
            )
        row("7 route comparison", r["status"], txt)
        d = r["decision"]
        row(
            "  decision",
            "",
            f"{d['recommended']} ({d['basis']})"
            + (f"; {d['agreement_note']}" if d.get("agreement_note") else ""),
        )
    else:
        row("7 route comparison", r["status"], r.get("error", ""))
    if "synthetic_truth" in report:
        lines.append(
            "synthetic ground truth (own generator): "
            + "; ".join(
                f"{name} spellF1 {_fmt(v['spell_f1'])} macroF1 {_fmt(v['macro_f1'])} "
                f"poss {v['possessions_pred']}/"
                f"{v['possessions_true']}"
                for name, v in report["synthetic_truth"].items()
            )
        )
    lines.append(
        f"runtime: {report.get('seconds_total', 0):.0f} s   "
        + "  ".join(f"{name.split('_')[0]}:{v.get('seconds', 0):.0f}s" for name, v in k.items())
    )
    return "\n".join(lines)
