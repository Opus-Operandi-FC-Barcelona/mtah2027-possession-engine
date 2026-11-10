"""End-to-end wave-1 pipeline (Issue #28): tracking files -> ``labels[N]`` honouring the submission contract.

Stages (each optional ones are wrapped: a missing input or a failing module is logged in the diagnostics
and skipped -- *progressive fallback* -- and the route continues with what remains):

====  ====================================  =====================================================
 1    loaders                               ATD (#5), Smart Tagging (#6)               [required: ATD]
 2    ``clean_tracks``  (#8)                off-pitch staff, duplicate slots           [optional]
 3    ``kinematics``  (#9)                  interpolation, Savitzky-Golay, orientation [optional]
 4    ``dead_ball_signals``  (#13)          ``dead_score``, geometry, status channel   [optional]
 5    ``ball_kinematics``  (#16)            distances, control, touches, evidence      [optional]
 6    ``set_piece_anchors``  (#14)          restart frames ``t*``, forced dead intervals [optional]
 7    ``emissions``                         wave-1 heuristic ``log_probs[N, 3]``       [required]
 8    ``transitions``  (#23)                ``log_trans[N, 3, 3]``                     [optional]
 9    ``viterbi``  (#24)                    each graded block decoded independently     [required]
 10   ``spell_optimizer``  (#25)            micro-possession / D clean-up, D bridge, forced D [optional]
 11   ``contract``  (#26)                   fixed X/D blocks, no internal X            [required]
====  ====================================  =====================================================

If a required stage of the heuristic route fails, the whole route falls back to the nearest-player baseline so
that a valid file is always produced.  ``run_core`` is the array-level part (no file loaders, no Smart
Tagging unless supplied) shared with the external-benchmark evaluation; ``run_pipeline`` adds the loaders and
the contract.  The wave-2 specialists (#20-#22, ``specialists=``) replace stage 7 (the heuristic
emissions stay as the fallback) and add the collective-physics / ball-memory features to the front end.
"""
from __future__ import annotations

import dataclasses
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np

from mtah2027.config import (
    DEFAULT_CONFIG_PATH,
    FPS,
    PITCH,
    Config,
    MatchConfig,
    load_section,
    params_from_mapping,
)
from mtah2027.decoding.dynamic_viterbi import decode_match_halves, default_log_init
from mtah2027.decoding.transition_matrix import TransitionParams, build_log_transitions
from mtah2027.features._utils import speed_from_xy
from mtah2027.features.ball_kinematics import BallKinParams, compute_ball_kinematics
from mtah2027.features.ball_memory import BallMemoryParams, compute_ball_memory
from mtah2027.features.collective_physics import CollectivePhysicsParams, compute_collective_physics
from mtah2027.features.dead_ball_signals import DeadBallParams, compute_dead_ball_signals
from mtah2027.features.setpiece_anchor import AnchorParams, anchor_set_pieces
from mtah2027.features.throwin_resolver import (
    ThrowInParams,
    apply_resolutions,
    resolve_throw_ins,
    restart_emission_prior,
)
from mtah2027.models.baseline_labels import baseline_labels
from mtah2027.models.heuristic_emissions import (
    EmissionParams,
    compute_emissions,
    resolve_owner_team,
    touch_signal,
)
from mtah2027.postprocessing.spell_optimizer import SpellParams, optimize_spells
from mtah2027.postprocessing.submission_contract import enforce_submission_contract
from mtah2027.preprocessing.clean_tracks import clean_tracks
from mtah2027.preprocessing.kinematics_and_orientation import compute_kinematics_and_orientation
from mtah2027.vision.aerial_filter import AerialParams, compute_is_aerial, find_z_channel
from mtah2027.vision.attention_crops import VisionParams

H, A, D, X = 0, 1, 2, 3


@dataclass(frozen=True)
class PipelineParams:
    use_clean_tracks: bool = True
    use_set_piece_anchors: bool = True
    use_forced_dead: bool = True
    use_phase_prior: bool = True
    use_owner_channel: bool = True
    use_spell_optimizer: bool = True
    use_aerial_filter: bool = False   # Issue #12: is_aerial mask suppresses the touches of airborne frames
    use_homography: bool = False      # feed the tactical homographies to the aerial filter (jump cue)

    @classmethod
    def from_config(cls, path=None) -> "PipelineParams":
        return params_from_mapping(cls, load_section("pipeline", path), "pipeline")


@dataclass
class CoreResult:
    """Output of :func:`run_core`: label codes (``X`` outside the decoded blocks) and diagnostics."""

    codes: np.ndarray
    diagnostics: dict
    features: dict = field(default_factory=dict)


@dataclass
class PipelineResult:
    """Output of :func:`run_pipeline`: contract-compliant label codes (int8 H=0, A=1, D=2, X=3)."""

    labels: np.ndarray
    diagnostics: dict


def _log_default(message: str) -> None:
    print(f"[pipeline] {message}")


class _Stages:
    """Runs optional stages, recording status and wall time (no per-frame data is ever recorded)."""

    def __init__(self, log: Callable[[str], None] | None):
        self.log = log or _log_default
        self.info: dict[str, dict] = {}

    def run(self, name: str, fn: Callable[[], Any], *, default: Any = None, required: bool = False) -> Any:
        t0 = time.perf_counter()
        try:
            result = fn()
        except Exception as exc:  # noqa: BLE001 - deliberate: any failing optional stage is skipped
            self.info[name] = {"status": f"failed: {type(exc).__name__}: {exc}",
                               "seconds": round(time.perf_counter() - t0, 3)}
            self.log(f"stage '{name}' failed ({type(exc).__name__}: {exc})"
                     + ("" if required else "; continuing without it"))
            if required:
                raise
            return default
        self.info[name] = {"status": "ok", "seconds": round(time.perf_counter() - t0, 3)}
        return result

    def skip(self, name: str, reason: str) -> None:
        self.info[name] = {"status": f"skipped: {reason}", "seconds": 0.0}
        self.log(f"stage '{name}' skipped ({reason})")


def _count_possessions(codes: np.ndarray, blocks: Sequence[tuple[int, int]]) -> int:
    total = 0
    for a, b in blocks:
        seg = codes[a:b + 1]
        if not len(seg):
            continue
        starts = np.concatenate([[True], seg[1:] != seg[:-1]])
        total += int(((seg == H) | (seg == A))[starts].sum())
    return total


def summarize_labels(codes: np.ndarray, play_blocks: Sequence[tuple[int, int]], fps: float = FPS) -> dict:
    """Aggregate label statistics (never per-frame data)."""
    graded = np.zeros(len(codes), bool)
    for a, b in play_blocks:
        graded[a:b + 1] = True
    sel = codes[graded]
    n = max(int(graded.sum()), 1)
    n_poss = _count_possessions(codes, play_blocks)
    alive = int(((sel == H) | (sel == A)).sum())
    return {"graded_frames": int(graded.sum()), "share_H": float((sel == H).sum() / n),
            "share_A": float((sel == A).sum() / n), "share_D": float((sel == D).sum() / n),
            "n_possessions": n_poss, "mean_possession_s": float(alive / max(n_poss, 1) / fps)}


@dataclass
class FrontEnd:
    """Per-frame feature bundle shared by the heuristic route and the learned specialists (stages 2-6).

    ``pxy`` / ``bxy`` are cleaned and oriented (home attacks +x); optional pieces are ``None`` when their
    stage failed or its inputs were missing.
    """

    pxy: np.ndarray
    bxy: np.ndarray
    team: np.ndarray
    pvel: np.ndarray | None
    bvel: np.ndarray | None
    bacc: np.ndarray | None
    pspeed: np.ndarray
    bspeed: np.ndarray
    visible: np.ndarray
    sig: Any
    bk: Any
    anchors: Any
    dead_score: np.ndarray
    owner: np.ndarray | None
    owner_key: str | None
    phase: np.ndarray | None
    touches: np.ndarray | None
    phys: Any = None
    mem: Any = None
    throwins: Any = None          # list of ThrowInResolution (Issue #15), None when not run
    is_aerial: np.ndarray | None = None   # Issue #12 mask (None when the filter is off or failed)


def compute_front_end(
        players_xy: np.ndarray, team_of_slot: np.ndarray, ball_xy: np.ndarray, *,
        play_blocks: Sequence[tuple[int, int]], fps: float = FPS, pitch: tuple[float, float] = PITCH,
        ball_channels: dict[str, np.ndarray] | None = None,
        ball_channel_codes: dict[str, dict[str, int]] | None = None,
        team_ids: tuple = (), team_names: tuple = (), smart_tagging: Any = None,
        config_path: str | Path | None = None, stages: "_Stages | None" = None,
        collective: bool = False, homography: np.ndarray | None = None) -> FrontEnd:
    """Stages 2-6 (cleaning, kinematics + orientation, dead-ball signals, ball kinematics, set-piece anchors)
    plus the owner / phase / touch priors; with ``collective`` also the collective physics (#17) and the
    ball memory (#18) that feed the learned specialists."""
    cfg = config_path
    pp = PipelineParams.from_config(cfg)
    st = stages or _Stages(None)
    n = len(ball_xy)
    team = np.asarray(team_of_slot)
    play_blocks = [(int(a), int(b)) for a, b in play_blocks]

    # 2. cleaning ---------------------------------------------------------------------------
    pxy = np.asarray(players_xy, dtype=np.float32)
    if pp.use_clean_tracks:
        pxy = st.run("clean_tracks", lambda: clean_tracks(pxy, team, pitch=pitch).players_xy, default=pxy)
    else:
        st.skip("clean_tracks", "disabled by configuration")

    # 3. kinematics and orientation -------------------------------------------------------------
    halves = (play_blocks[0], play_blocks[1] if len(play_blocks) > 1 else play_blocks[0])
    kin = st.run("kinematics", lambda: compute_kinematics_and_orientation(
        pxy, np.asarray(ball_xy, dtype=np.float32), team, fps=fps, pitch=pitch, halves=halves))
    if kin is not None:
        pxy, bxy = kin.players_xy, kin.ball_xy
        pvel, bvel, bacc = kin.players_vel, kin.ball_vel, kin.ball_acc
        pspeed, bspeed = kin.players_speed, kin.ball_speed
    else:
        bxy = np.asarray(ball_xy, dtype=np.float32)
        pvel = bvel = bacc = None
        pspeed, bspeed = speed_from_xy(pxy, fps, 1), speed_from_xy(bxy, fps, 2)
    visible = np.isfinite(bxy).all(axis=1)

    # 4. dead-ball signals ----------------------------------------------------------------------
    sig = st.run("dead_ball_signals", lambda: compute_dead_ball_signals(
        pxy, team, bxy, players_speed=pspeed, ball_speed=bspeed, ball_channels=ball_channels,
        ball_channel_codes=ball_channel_codes, fps=fps, pitch=pitch, params=DeadBallParams.from_config(cfg)))
    dead_score = sig.dead_score if sig is not None else np.zeros(n, dtype=np.float32)

    # 4b. aerial ball (Issue #12) -----------------------------------------------------------------
    is_aerial = None
    if pp.use_aerial_filter:
        ap = AerialParams.from_config(cfg)
        is_aerial = st.run("aerial_filter", lambda: compute_is_aerial(
            bxy, fps=fps, ball_z=find_z_channel(ball_channels, ap.z_channels, n),
            H=homography if pp.use_homography else None, pitch=pitch, params=ap))
    else:
        st.skip("aerial_filter", "disabled by configuration")

    # 5. ball kinematics -------------------------------------------------------------------------
    bk = st.run("ball_kinematics", lambda: compute_ball_kinematics(
        pxy, team, bxy, players_vel=pvel, ball_vel=bvel, ball_acc=bacc, is_aerial=is_aerial, fps=fps,
        params=BallKinParams.from_config(cfg)))

    # 6. set-piece anchors -----------------------------------------------------------------------
    anchors = None
    if smart_tagging is not None and pp.use_set_piece_anchors:
        anchors = st.run("set_piece_anchors", lambda: anchor_set_pieces(
            bxy, pxy, team, smart_tagging.set_pieces, smart_tagging.goals, ball_speed=bspeed,
            players_speed=pspeed, dead_score=dead_score, fps=fps, pitch=pitch,
            params=AnchorParams.from_config(cfg)))
    else:
        reason = "no Smart Tagging" if smart_tagging is None else "disabled by configuration"
        st.skip("set_piece_anchors", reason)

    # 6b. throw-in restart team (Issue #15): Team = N/A throw-ins get a team from the taker / first controller
    throwins = None
    tp = ThrowInParams.from_config(cfg)
    if anchors is not None and tp.enabled:
        def _resolve_throw_ins():
            res = resolve_throw_ins(anchors, pxy, team, bxy, pitch=pitch, fps=fps, params=tp)
            return res, apply_resolutions(anchors, res)

        out = st.run("throwin_resolver", _resolve_throw_ins)
        if out is not None:
            throwins, anchors = out
    else:
        st.skip("throwin_resolver", "no anchors" if anchors is None else "disabled by configuration")

    # priors for the emissions ----------------------------------------------------------------------
    owner, owner_key = None, None
    if pp.use_owner_channel and ball_channels:
        ep = EmissionParams.from_config(cfg)
        owner, owner_key = resolve_owner_team(ball_channels, ball_channel_codes, ep.owner_channels, team_ids,
                                              team_names, n)
    phase = None
    if smart_tagging is not None and pp.use_phase_prior:
        phase = np.asarray(smart_tagging.phase_prior_home, dtype=np.float64)
        if phase.shape != (n,):
            phase = None
    touches = None
    if bk is not None:
        touches = touch_signal(bk.touch_mask, bk.delta_d, EmissionParams.from_config(cfg).touch_margin_m)

    phys = mem = None
    if collective:
        phys = st.run("collective_physics", lambda: compute_collective_physics(
            pxy, team, players_vel=pvel, fps=fps, params=CollectivePhysicsParams.from_config(cfg)))
        mem = st.run("ball_memory", lambda: compute_ball_memory(
            pxy, team, bxy, players_vel=pvel, ball_vel=bvel,
            touch_mask=bk.touch_mask if bk is not None else None,
            delta_d=bk.delta_d if bk is not None else None,
            possession_evidence=bk.possession_evidence if bk is not None else None, fps=fps,
            params=BallMemoryParams.from_config(cfg)))
    return FrontEnd(pxy=pxy, bxy=bxy, team=team, pvel=pvel, bvel=bvel, bacc=bacc, pspeed=pspeed,
                    bspeed=bspeed, visible=visible, sig=sig, bk=bk, anchors=anchors, dead_score=dead_score,
                    owner=owner, owner_key=owner_key, phase=phase, touches=touches, phys=phys, mem=mem,
                    throwins=throwins, is_aerial=is_aerial)


def heuristic_emissions(fe: FrontEnd, fps: float, config_path=None, stages: "_Stages | None" = None):
    """Wave-1 emissions of a :class:`FrontEnd` (an ``EmissionResult``)."""
    st = stages or _Stages(None)
    n = len(fe.visible)
    bk, sig = fe.bk, fe.sig
    return st.run("emissions", lambda: compute_emissions(
        fe.dead_score, bk.possession_evidence if bk is not None else np.zeros(n),
        bk.evidence_valid if bk is not None else np.zeros(n, bool), fe.visible,
        status_dead=sig.status_dead if (sig is not None and sig.has_status) else None,
        phase_prior_home=fe.phase, owner_team=fe.owner, touch_signed=fe.touches, fps=fps,
        params=EmissionParams.from_config(config_path)), required=True)


def run_core(
        players_xy: np.ndarray, team_of_slot: np.ndarray, ball_xy: np.ndarray, *,
        play_blocks: Sequence[tuple[int, int]], dead_blocks: Sequence[tuple[int, int]] = (),
        fps: float = FPS, pitch: tuple[float, float] = PITCH,
        ball_channels: dict[str, np.ndarray] | None = None,
        ball_channel_codes: dict[str, dict[str, int]] | None = None,
        team_ids: tuple = (), team_names: tuple = (), smart_tagging: Any = None,
        config_path: str | Path | None = None, log: Callable[[str], None] | None = None,
        keep_features: bool = False, specialists: Any = None,
        homography: np.ndarray | None = None) -> CoreResult:
    """Features -> emissions -> transitions -> Viterbi per block -> spell optimiser (array level).

    ``players_xy[N, S, 2]``, ``ball_xy[N, 2]`` in metres (NaN when missing), ``team_of_slot[S]`` 0 home /
    1 away.  ``play_blocks`` are the inclusive graded blocks decoded independently (the first two also give
    the halves used to orient the pitch); ``dead_blocks`` are forced to ``D``.  ``smart_tagging`` (optional)
    is a :class:`~mtah2027.io.smart_tagging_loader.SmartTagging`.  Frames outside the blocks are ``X``.

    ``specialists`` (a :class:`~mtah2027.models.specialist_fusion.SpecialistSet`, optional) replaces the
    heuristic emissions by the learned ones (Issues #20-#22); if it is ``None`` or fails at any point the
    wave-1 heuristic emissions are used (progressive fallback).
    """
    cfg = config_path
    pp = PipelineParams.from_config(cfg)
    st = _Stages(log)
    n = len(ball_xy)
    play_blocks = [(int(a), int(b)) for a, b in play_blocks]
    feats: dict[str, np.ndarray] = {}

    def front_end(ball: np.ndarray, quiet: bool = False) -> FrontEnd:
        return compute_front_end(
            players_xy, team_of_slot, ball, play_blocks=play_blocks, fps=fps, pitch=pitch,
            ball_channels=ball_channels, ball_channel_codes=ball_channel_codes, team_ids=team_ids,
            team_names=team_names, smart_tagging=smart_tagging, config_path=cfg,
            stages=_Stages(lambda m: None) if quiet else st, collective=specialists is not None,
            homography=homography)

    fe = front_end(ball_xy)
    sig, bk, anchors, visible, dead_score = fe.sig, fe.bk, fe.anchors, fe.visible, fe.dead_score
    owner_key = fe.owner_key

    # 7. emissions ---------------------------------------------------------------------------------
    em = heuristic_emissions(fe, fps, cfg, st)
    learned = None
    if specialists is not None:
        from mtah2027.models.specialist_fusion import fuse_emissions

        learned = st.run("specialists", lambda: fuse_emissions(
            specialists, fe, em, play_blocks=play_blocks, fps=fps,
            rebuild=lambda ball: front_end(ball, quiet=True), raw_ball=np.asarray(ball_xy, np.float32),
            config_path=cfg))
        if learned is not None:
            em = learned.emissions
    emission_source = "specialists" if learned is not None else "heuristic"
    if fe.throwins:       # Issue #15: soft prior towards the recovered restart team just after t*
        em = dataclasses.replace(em, log_probs=restart_emission_prior(
            em.log_probs, fe.throwins, fps=fps, params=ThrowInParams.from_config(cfg)))

    # 8. transitions -------------------------------------------------------------------------------
    restart_frames = restart_teams = None
    if anchors is not None:
        resolved = [a for a in anchors.anchors if a.resolved]
        restart_frames = np.array([a.t_star for a in resolved], dtype=np.int64)
        restart_teams = np.array([a.team for a in resolved], dtype=np.int64)

    def _transitions():
        return build_log_transitions(
            n, touch_score=bk.touch_score if bk is not None else None,
            d_home=bk.d_home[:, 0] if bk is not None else None,
            d_away=bk.d_away[:, 0] if bk is not None else None,
            ball_visible=visible, dead_score=dead_score, ball_out=sig.ball_out if sig is not None else None,
            restart_frames=restart_frames, restart_teams=restart_teams, fps=fps,
            params=TransitionParams.from_config(cfg))

    log_trans = st.run("transitions", _transitions)
    if log_trans is None:     # closed matrix: stay costs 0, every change is expensive
        log_trans = build_log_transitions(n, params=TransitionParams.from_config(cfg))

    # 9. Viterbi ------------------------------------------------------------------------------------
    codes = st.run("viterbi", lambda: decode_match_halves(
        em.log_probs, log_trans, play_blocks=play_blocks, dead_blocks=dead_blocks,
        log_init=default_log_init(cfg)), required=True)
    raw_summary = summarize_labels(codes, play_blocks, fps)

    # 10. spell optimiser ---------------------------------------------------------------------------
    opt_stats: dict = {}
    if pp.use_spell_optimizer:
        support = np.zeros(n, bool)
        if sig is not None:
            support = (sig.geom_touchline > 0) | (sig.geom_corner > 0) | (sig.geom_goal_area > 0) | \
                      (sig.geom_kickoff > 0) | (sig.ball_out > 0)
        if smart_tagging is not None and getattr(smart_tagging, "in_set_piece", None) is not None:
            support = support | np.asarray(smart_tagging.in_set_piece, dtype=bool)
        forced = anchors.forced_dead_mask if (anchors is not None and pp.use_forced_dead) else None

        def _optimize():
            return optimize_spells(
                codes, play_blocks=play_blocks, ball_visible=visible,
                touch_score=bk.touch_score if bk is not None else None, dead_score=dead_score,
                dead_support=support, forced_dead_mask=forced, fps=fps, params=SpellParams.from_config(cfg))

        res = st.run("spell_optimizer", _optimize)
        if res is not None:
            codes, opt_stats = res
    else:
        st.skip("spell_optimizer", "disabled by configuration")

    diagnostics = {
        "stages": st.info, "n_frames": int(n), "n_slots": int(players_xy.shape[1]),
        "ball_visible_share": float(visible.mean()) if n else 0.0,
        "aerial_share": float(fe.is_aerial.mean()) if fe.is_aerial is not None and n else None,
        "has_status_channel": bool(sig is not None and sig.has_status),
        "owner_channel": owner_key, "n_anchors": len(anchors.anchors) if anchors is not None else 0,
        "n_anchors_resolved": anchors.n_resolved if anchors is not None else 0,
        "throw_ins": {"resolved": sum(r.resolved for r in fe.throwins), "total": len(fe.throwins)}
        if fe.throwins is not None else None,
        "emission_source": emission_source,
        "specialists": learned.report if learned is not None else {},
        "viterbi_raw": raw_summary, "labels": summarize_labels(codes, play_blocks, fps),
        "optimizer": opt_stats,
    }
    if keep_features:
        feats.update({"dead_score": dead_score, "visible": visible, "log_probs": em.log_probs,
                      "home_logit": em.home_logit})
        if bk is not None:
            feats.update({"possession_evidence": bk.possession_evidence, "touch_score": bk.touch_score})
    return CoreResult(codes=codes, diagnostics=diagnostics, features=feats)


def _fit_length(arr: np.ndarray, n: int) -> np.ndarray:
    """Pad with NaN rows / truncate along axis 0 to exactly ``n`` frames."""
    if len(arr) == n:
        return arr
    if len(arr) > n:
        return arr[:n]
    pad = np.full((n - len(arr),) + arr.shape[1:], np.nan, dtype=arr.dtype)
    return np.concatenate([arr, pad], axis=0)


def resolve_config(config: str | Path | Config | None) -> tuple[Path, Config | None]:
    """``(yaml_path, typed_config_or_None)`` from a path, a :class:`Config` or ``None`` (default YAML)."""
    if isinstance(config, Config):
        return DEFAULT_CONFIG_PATH, config
    return (Path(config) if config is not None else DEFAULT_CONFIG_PATH), None


def run_pipeline(data_dir: str | Path | None = None, config: str | Path | Config | None = None,
                 blocks: MatchConfig | None = None, *, cache_dir: str | Path | None = None,
                 log: Callable[[str], None] | None = None, specialists: Any = None) -> PipelineResult:
    """Tracking bundle in ``data_dir`` -> contract-compliant labels (see the module docstring).

    ``config`` is a YAML path or a typed :class:`Config` (default ``configs/default.yaml``); ``blocks`` the
    match layout (default: the official BAR-ATH one, or ``config.match``).  ``cache_dir`` enables the
    ``.npz`` cache of the parsed ATD (``None`` = no cache).  ``specialists`` (optional
    :class:`~mtah2027.models.specialist_fusion.SpecialistSet`) switches the emissions from the heuristic
    to the learned ones (Issues #20-#22); the heuristic emissions remain the fallback.
    """
    from mtah2027.io.atd_loader import load_atd
    from mtah2027.io.smart_tagging_loader import load_smart_tagging

    cfg_path, typed = resolve_config(config)
    match = blocks or (typed.match if typed is not None else MatchConfig())
    n = match.n_video_frames
    st = _Stages(log)
    td = st.run("load_atd", lambda: load_atd(data_dir, cache_dir=cache_dir, use_cache=cache_dir is not None,
                                             n_frames=n), required=True)
    smart = st.run("load_smart_tagging", lambda: load_smart_tagging(data_dir, n_frames=n))
    pp = PipelineParams.from_config(cfg_path)
    vp = VisionParams.from_config(cfg_path)
    td, broadcast = _apply_broadcast_sync(td, data_dir, match, cfg_path, st, cache_dir)
    homography, hom = None, None
    if (pp.use_aerial_filter and pp.use_homography) or vp.enabled:
        from mtah2027.io.homography_loader import load_homography

        hom = st.run("load_homography", lambda: load_homography(data_dir))
        homography = _fit_length(hom.H, n) if (hom is not None and pp.use_aerial_filter
                                               and pp.use_homography) else None
    vision = None
    if vp.enabled and hom is not None:
        vision = st.run("vision_recovery", lambda: _merge_cached_vision(td, hom, data_dir, match, vp, n))
        if vision is not None:
            st.log(f"vision: {vision.stats['status']}, {vision.stats['frames_filled']} hidden frames filled "
                   f"({100 * vision.stats['fill_rate']:.1f}%)")
    elif not vp.enabled:
        st.skip("vision_recovery", "disabled by configuration")
    return _decode_loaded(td, smart, match, cfg_path, st, specialists, homography, vision, broadcast)


def _apply_broadcast_sync(td, data_dir, match: MatchConfig, cfg_path, st: _Stages, cache_dir):
    """Issue #10: broadcast ball injected into the tactical holes; ``(td, stats)``."""
    from mtah2027.preprocessing import broadcast_sync as bsync

    bp = bsync.BroadcastSyncParams.from_config(cfg_path)
    if not bp.enabled:
        st.skip("broadcast_sync", "disabled by configuration")
        return td, None
    if not bsync.broadcast_files_present(data_dir):
        st.skip("broadcast_sync", "broadcast files not found")
        return td, None
    res = st.run("broadcast_sync", lambda: bsync.sync_broadcast_files(
        td, data_dir, [match.half1, match.half2_play], params=bp, cache_dir=cache_dir))
    if res is None:
        return td, None
    s = res.stats
    st.log(f"broadcast: {s['blocks_accepted']}/{s['blocks']} blocks synchronised, "
           f"{s['frames_filled']} hidden frames filled ({100 * s['fill_rate']:.1f}%)")
    ball = res.ball_xy
    channels = {**td.ball_channels, "x": np.ascontiguousarray(ball[:, 0]),
                "y": np.ascontiguousarray(ball[:, 1])}
    return dataclasses.replace(td, ball_xy=ball, ball_channels=channels), dict(s)


def _merge_cached_vision(td, hom, data_dir, match: MatchConfig, vp: VisionParams, n: int):
    """Cached video detections (``scripts/run_vision_recovery.py``) merged into the tracked ball.

    Only the cache is read (``detector=None``): the video is never decoded here.  Without a cache the ball is
    returned unchanged and ``stats['status']`` says why."""
    import games
    from mtah2027.vision.recovery import recover_ball

    video = games.asset_paths(data_dir=str(data_dir) if data_dir is not None else None)["video"]
    return recover_ball(_fit_length(td.ball_xy, n), _fit_length(td.players_xy, n), hom,
                        [match.half1, match.half2_play], video, detector=None, params=vp, fps=float(td.fps),
                        pitch=tuple(td.pitch))


def match_from_periods(periods: Sequence[tuple[int, int]], n_frames: int, fps: float = FPS,
                       pitch: tuple[float, float] = PITCH) -> MatchConfig:
    """Match layout of a file with two declared periods and no dead pre-block (e.g. open Metrica data)."""
    if len(periods) < 2:
        raise ValueError(f"need the two periods (first/second half), got {list(periods)}")
    (a1, b1), (a2, b2) = ((int(a), int(b)) for a, b in periods[:2])
    return MatchConfig(game="external", n_video_frames=int(n_frames), fps=int(round(fps)), pitch=pitch,
                       half1=(a1, b1), half2_dead=(a2, a2 - 1), half2_play=(a2, b2))


def run_pipeline_files(meta_path: str | Path, raw_path: str | Path, *,
                       smart_tagging_path: str | Path | None = None,
                       config: str | Path | Config | None = None,
                       periods: Sequence[tuple[int, int]] | None = None,
                       log: Callable[[str], None] | None = None, specialists: Any = None) -> PipelineResult:
    """Like :func:`run_pipeline` for an explicit EPTS ``(xml, txt)`` pair, e.g. a Metrica open sample.

    The periods default to the ``first_half_start`` ... provider parameters of the XML; ``smart_tagging_path``
    is optional (without it the Smart-Tagging-dependent stages are skipped and logged).  The pitch is the
    ``<FieldSize>`` of the XML.
    """
    from mtah2027.io.atd_loader import load_atd_files
    from mtah2027.io.smart_tagging_loader import load_smart_tagging

    cfg_path, _ = resolve_config(config)
    st = _Stages(log)
    td = st.run("load_atd", lambda: load_atd_files(meta_path, raw_path), required=True)
    match = match_from_periods(periods or td.periods, td.n_frames, td.fps, tuple(td.pitch))
    if smart_tagging_path is not None:
        smart = st.run("load_smart_tagging", lambda: load_smart_tagging(
            n_frames=td.n_frames, fps=int(round(td.fps)), path=smart_tagging_path))
    else:
        smart = None
        st.skip("load_smart_tagging", "no Smart Tagging file given")
    return _decode_loaded(td, smart, match, cfg_path, st, specialists)


def _decode_loaded(td, smart, match: MatchConfig, cfg_path, st: _Stages, specialists: Any,
                   homography: np.ndarray | None = None, vision: Any = None,
                   broadcast: dict | None = None) -> PipelineResult:
    """Loaded tracking (+ optional Smart Tagging) -> contract-compliant labels."""
    n = match.n_video_frames
    t_total = time.perf_counter()
    play_blocks = [match.half1, match.half2_play]
    dead_blocks = [match.half2_dead] if match.half2_dead[1] >= match.half2_dead[0] else []
    players, ball = _fit_length(td.players_xy, n), _fit_length(td.ball_xy, n)
    if vision is not None and vision.stats.get("status") == "ok":
        ball = vision.ball_xy       # tracking where seen, recovered video points where it was missing

    core_diag: dict = {}
    try:
        core = run_core(
            players, td.team_of_slot, ball, play_blocks=play_blocks, dead_blocks=dead_blocks, fps=td.fps,
            pitch=tuple(td.pitch), ball_channels={k: _fit_length(v, n) for k, v in td.ball_channels.items()},
            ball_channel_codes=td.ball_channel_codes, team_ids=td.team_ids, team_names=td.team_names,
            smart_tagging=smart, config_path=cfg_path, log=st.log, specialists=specialists,
            homography=homography)
        codes, core_diag = core.codes, core.diagnostics
        route = "heuristic"
    except Exception as exc:  # noqa: BLE001 - last resort: still deliver a valid file
        st.log(f"heuristic route failed ({type(exc).__name__}: {exc}); "
               "falling back to the nearest-player baseline")
        codes = np.full(n, X, dtype=np.int8)
        base = baseline_labels(players, td.team_of_slot, ball)
        for a, b in play_blocks:
            codes[a:b + 1] = base[a:b + 1]
        route = "baseline-fallback"

    labels = st.run("contract", lambda: enforce_submission_contract(codes, match), required=True)
    diagnostics = {
        "route": route, "loader_stages": dict(st.info), "core": core_diag,
        "n_frames": n, "team_names": list(td.team_names),
        "vision": dict(vision.stats) if vision is not None else None,
        "broadcast": broadcast,
        "final": summarize_labels(labels, play_blocks, td.fps),
        "seconds_total": round(time.perf_counter() - t_total, 3),
    }
    return PipelineResult(labels=labels, diagnostics=diagnostics)
