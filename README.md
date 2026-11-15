# mtah2027-possession-engine

Deterministic frame-level **ball possession engine** for *More than a Hack 2027* (FC Barcelona x Metrica
Sports), phase 1 -- team **Opus Operandi**.  Given the tactical Automatic Tracking Data (ATD) and the Smart
Tagging of the BAR-ATH match, it writes one label per video frame (`H` Barcelona in possession, `A`
Athletic Club in possession, `D` ball dead, `X` abstain / not evaluated) in the format validated by the
official `evaluation/validate_submission.py`.

The ranking metric is **spellF1** (possession spells matched by temporal IoU, a `D` splits a possession),
so the engine optimises the *number* of possessions and the precision of their boundaries.  Full technical
description: [`docs/TECHNICAL_REPORT.md`](docs/TECHNICAL_REPORT.md).  Status of every issue:
[`docs/ISSUES_TRACKER.md`](docs/ISSUES_TRACKER.md).

## Status

Milestones 1-11 (issues #1-#72; #41 does not exist) are complete.  **Milestone 9** ("intelligent analysis over the
provided data and invariance to player dropout") adopted three changes: the set-piece anchor fallback and period-start
clamp (#50, correctness fix), the isolation of the front end per graded play block (#51, correctness fix) and the
re-trained default model in the estimated "gaps over dead ball" world (#48, +0.017 spellF1 on B'(33)).  #42, #43, #44
and #49 were measured and left off by configuration; #47 is the audit against the official repository and #52 makes the
submission depend only on the committed model.  **Milestone 10** (#55-#66, post-freeze experiments in separate
branches: more PFF matches, context windows, learned turnover gate, real-video ball detectors, latency bias, seed
bagging, boundary snap, expected-spellF1 decoding, pre-trained ball model, pressure micro-geometry, hidden-ball phase
prior, tracklet-gated vision) adopted nothing: every change stays in the code, off by configuration, and the submission
is unchanged.  **Milestone 11** (#67-#72: oracle ceiling of the hidden ball, self-supervised ball detector, integration decision, value research,
world selection, sequence model) adopted one change: #72, a non-causal TCN over the specialists' per-frame features whose probabilities replace the
hidden-ball emissions (variant E100, mean of two training seeds, `models/sequence_tcn.json`, `sequence_model.enabled: true`).  The detector (#68/#69) was
not adopted.  The submission changed: SHA256 `a31c0a3c864fa6ce16bc39c1f3d3b51a45b9bd2abc1d2bfbbb6cc223b08b3a6a`
(273 possessions, mean 12.5 s, `D` 45.5 % of graded play frames; generated twice with identical hash on Python 3.13.14, Windows 11).  The previous submission
(SHA256 `cac8927c2adc1b1a6778d0045a2fae0ba4437aee8ea10a889de02db7552d0c7b`) is reproduced with `sequence_model.enabled: false` in `configs/default.yaml`.
Results: `docs/Opus Operandi.md` (sections 6 to 8) and `docs/TECHNICAL_REPORT.md`.  The submission document for the competition form is
[`docs/Opus Operandi.md`](<docs/Opus Operandi.md>).

### Architecture

```
ATD (x, y, s) + Smart Tagging + homography
  -> ingest (io/)  -> clean + kinematics + orientation (preprocessing/)
  -> features: dead-ball signals, set-piece anchors t*, throw-in resolver, ball kinematics / touches,
               collective physics, ball memory (features/)
  -> emissions log_probs[N,3]: Specialist A (ball visible) + Specialist B (ball absent) fused by ball visibility
               (models/); heuristic emissions as fallback route
  -> time-dependent transitions log A_t[N,3,3] (decoding/transition_matrix.py)
  -> Viterbi per graded half (numba kernel, bit-identical NumPy fallback; decoding/dynamic_viterbi.py)
  -> spell optimiser (micro-possession pruning, D bridge; postprocessing/spell_optimizer.py)
  -> submission contract (X / D forced outside the graded blocks, no internal X; postprocessing/submission_contract.py)
  -> outputs/<team>.csv  (+ KPI dashboard, validation, SHA256)
```

Optional stages that were built, measured on other matches and left **off** by configuration: broadcast (TV) sync
(#10, #39), attention crops / YOLO / multi-frame detector (#11, #36), aerial filter (#12), pseudo-labelling of the
target match (#21), MBR consensus (#26, #32), semi-Markov decoder (#34), specialist-opened transitions (#30 A2).

**Route decision (KPI 7).**  The learned route (Specialists A/B) is preferred over the training-free heuristic route
only if its leave-one-match-out spellF1 on degraded external matches beats the heuristic one by >= 0.04
(`reports/external_evidence.json`, or `--external-lomo`); the public 5 minutes are never used to choose.

**Selectable models** (`generate_submission.py --model-variant`, implies `--specialists load`):

| variant | file | training data |
|---|---|---|
| `default` (same as `gap`) | `models/specialists.json` | IDSSE degraded with the calibrated degrader at hidden<->dead affinity 33 (#48) + 20 PFF matches at weight 0.5 |
| `gap` | `models/specialists_gap.json` | = default |
| `mixed` | `models/specialists_mixed.json` | IDSSE (degraded, legacy degrader, blind to the play state) + 20 PFF matches at weight 0.5 (the default before #48) |
| `idsse` | `models/specialists_idsse.json` | IDSSE only |
| `pff` | `models/specialists_pff.json` | PFF only |
| `calib` | `models/specialists_calib.json` | IDSSE degraded with the calibrated degrader (#38, affinity 1) + PFF 0.5 |
| `ensemble` | `models/specialists_ensemble.json` | Issue #44 manifest: linear per-frame mix, 0.75 `gap` + 0.25 `mixed` (committed files only; measured +0.002 over the default on B'(33) without the decoder block, not adopted) |

Choose between them with label-free input statistics (`scripts/compare_input_stats.py --data-dir data`), never with
labels of the provided game.

**Tests.**  520 tests are collected on `develop` at the time of writing (`pytest --collect-only -q`).

### Issues by milestone

| Milestone | Issues | Topic |
|---|---|---|
| 1 Infrastructure, CI/CD, fixtures, evaluation bridge | #1-#4 | scaffold, CI + security gate, synthetic match, official-scorer bridge |
| 2 Streaming ingest, slot cleaning, TV sync | #5-#10 | ATD / Smart Tagging / homography loaders, cleaning, kinematics, broadcast sync (off) |
| 3 Vision and aerial ball | #11-#12 | attention crops + detectors (off), aerial filter (off) |
| 4 Dead ball, set-piece anchors, collective physics | #13-#18 | dead-ball signals, `t*` anchors, throw-in resolver, ball kinematics, collective physics, ball memory |
| 5 Degrader, self-supervision, specialists | #19-#22 | realistic degrader, Specialist A, block-masking pseudo-labels, Specialist B + fusion |
| 6 Dynamic Viterbi, spellF1 post-processing, KPIs, delivery | #23-#28 | `A_t`, numba Viterbi, spell optimiser, MBR + contract, KPI dashboard, CLI + report |
| 7 Post-Optuna improvements | #29-#36 | route overrides, occlusion turnovers, MBR, semi-Markov, extra training matches, multi-frame detector + audit video |
| 8 Real data (BAR-ATH ingestion, calibration, model selection) | #37-#40 | aggregate sanity report, calibrated degrader, TV sync decision, Smart Tagging on the real match |
| 9 Intelligent analysis and dropout invariance | #42-#52 | adopted: #48 model re-trained with gaps over dead ball, #50 set-piece anchor fallback + period-start clamp, #51 play-block isolation, #52 reproducible model loading; measured and off: #42 dropout invariance, #43 Smart Tagging alive prior, #44 decoder re-optimisation + ensemble, #49 restart-team rules; #45 docs, #46 release, #47 official audit (there is no #41) |
| 10 Post-freeze experiments (adopt only if >= +0.010) | #55-#66 | all measured on external matches and left off: #55 56 PFF matches, #56 multi-scale context + goalkeeper depth, #57 learned turnover gate, #58 real-video ball detectors (aggregates), #59 latency bias, #60 seed bagging, #61 boundary snap, #62 expected-spellF1 decoding, #63 pre-trained ball model, #64 pressure micro-geometry, #65 hidden-ball phase prior, #66 tracklet-gated vision |
| 11 Remaining margin: oracle ceiling, detector, sequence model | #67-#72 | adopted: #72 TCN hidden-ball emissions (two-seed ensemble, B'(33) +0.0130); not adopted: #68/#69 self-supervised ball detector (about 15 % within 0.3 m); #67, #70, #71 diagnostics and research; synthetic SHA256 `4224d1ca...` |
## Install

Python >= 3.10 (developed on 3.13, CI on 3.11).  Dependencies are pinned to exact versions.

```bash
python -m venv .venv
source .venv/bin/activate              # Windows: .venv\Scripts\activate
pip install -r requirements.txt        # numpy, pandas, scipy, scikit-learn, PyYAML, numba, lightgbm (+ pytest, ruff)
pip install -e . --no-deps             # makes `mtah2027` importable
# numba and lightgbm are optional at run time: without numba the NumPy fallbacks are bit-identical (the
# full-size hull fallback is slow, ~1 min); without lightgbm the specialists use scikit-learn when they are
# trained in-process, and only the heuristic emissions are available from the committed model file.
```

## Data (never committed)

The match assets are gated by the organisers and are **not** part of this repository (`data/`, `cache/`, videos,
`*_atd.*`, `*.npy`/`*.npz` and prediction CSVs are git-ignored and blocked by a CI security gate).  Unpack the
bundle into `data/` as described in `data/README.md`, or point `MTAH_DATA_DIR` at it.  The match data is only
ever processed locally; tests and CI use a deterministic **synthetic match** generated by
`scripts/generate_synthetic_fixtures.py`.

## Reproduce the submission

```bash
python -m venv .venv && source .venv/bin/activate           # Windows: .venv\Scripts\activate
pip install -r requirements.txt && pip install -e . --no-deps
python scripts/generate_submission.py --data-dir data --team "Opus Operandi"
# -> "outputs/Opus Operandi.csv"   (199,319 rows + header, validated with the official format check)
python evaluation/validate_submission.py "outputs/Opus Operandi.csv"
python scripts/generate_submission.py --data-dir data --team "Opus Operandi" --kpi      # + seven-KPI dashboard
python scripts/generate_submission.py --data-dir data --model-variant idsse          # another specialists file
```

* The script prints only aggregates (rows, label shares, number / mean duration of possessions, SHA256,
  runtime), never per-frame rows.
* Determinism: `PYTHONHASHSEED=0` is enforced (the script re-executes itself once if needed), global seeds are
  fixed (`seed: 0` in `configs/default.yaml`) and every decoder input is rounded to 1e-6.  Two runs give the
  same SHA256; the numba and the fallback decoders give the same file.
* All parameters live in `configs/default.yaml` (one section per module); `--config` selects another file.
* Self-check without the real data (what CI does): `python scripts/generate_submission.py --synthetic --out
  /tmp/sub_run1.csv` generates the full-length synthetic match with a fixed seed and also reports its score
  against the synthetic ground truth.
* Learned emissions (`--specialists auto|train|load|off`).  `auto` (default) loads the committed
  `models/specialists.json` whenever it exists (Issue #52: the submission depends only on the repository, as
  RULES.md requires); only if it is missing does it train from the external open matches on disk (`--bench-dir`,
  `$MTAH_BENCH_DIR` or the sibling `mtah2027-external-benchmarks/cache`), otherwise it uses the heuristic emissions.
  `train` always retrains.  With `--synthetic`, `auto` never trains (deterministic and fast).  The training
  is deterministic (LightGBM `deterministic=True`, `num_threads=1`, `seed=0`) and reproducible with
  `python scripts/train_specialists.py` (the committed file is its output).  See "Training data" in the
  technical report: only external open data, labelled algorithmically; nothing from BAR-ATH.
* Progressive fallback: every optional stage (cleaning, kinematics, Smart Tagging anchors, ...) is skipped
  and logged if its input or module is missing; if the heuristic route itself fails, the nearest-player
  baseline is written, so a valid file always exists.

## Quality control: the seven-KPI dashboard

```bash
python scripts/run_kpi_dashboard.py --data-dir data --submission "outputs/Opus Operandi.csv"
python scripts/run_kpi_dashboard.py --data-dir data --external-lomo        # + leave-one-match-out specialists
python scripts/run_kpi_dashboard.py --synthetic                            # full-length synthetic self-check
python scripts/generate_submission.py --data-dir data --kpi               # submission + dashboard in one go
```

Options: `--data-dir`, `--synthetic`, `--submission`, `--bench-dir` (default: the sibling
`mtah2027-external-benchmarks/cache` if it exists), `--skip-external`, `--external-lomo`.  Output: a table and
`outputs/kpi_report.json` with aggregate numbers only (safe to paste), plus `outputs/audit_transitions.csv` (20
transitions for the visual audit; git-ignored, never commit it, only its count is printed).  The seven KPIs are
sanity alarms, artificial blindness, degraded external evaluation, a smoke test on the public 5 minutes ("pipeline
check only - not for model selection", real data only), the audit export, the format check with SHA256, and the
specialists-vs-heuristic route comparison.

**Choosing the route without ground truth.**  The learned route lost to the heuristic one on the synthetic match
(domain shift), so on the real match decide with `--external-lomo`: submit the specialists only if their
leave-one-match-out spellF1 on the degraded external matches beats the heuristic route by >= 0.04; otherwise
submit `--specialists off`.  The agreement between the two routes on the real match is a diagnostic (low
agreement = domain-shift warning), never the criterion, and the public GT is never used to choose.  Details:
"Quality control" in `docs/TECHNICAL_REPORT.md`.

## Evaluate on other matches

`scripts/eval_external.py` scores the array-level core of the pipeline on open tracking data of **other**
matches (IDSSE Bundesliga / Metrica sample games, prepared by the sister repository
`mtah2027-external-benchmarks`) with the official metric, next to the official baseline recipe, with an
optional crude ball-hiding degradation:

```bash
python scripts/eval_external.py --bench-dir ../mtah2027-external-benchmarks/cache --only-complete-ball
python scripts/eval_external.py --only-complete-ball --hide-ball-frac 0.41 --seed 0     # crude hiding
python scripts/eval_external.py --only-complete-ball --degrade --seeds 0 1 2            # realistic (Issue #19)
```

`--degrade` uses `src/mtah2027/models/external_degrader.py`: 41 % of the alive frames lose the ball in
contiguous blocks (5-20 s 33 %, 20-60 s 48 %, > 60 s 19 %) placed preferentially where the ball is crowded,
plus correlated tracking noise, identity cuts, duplicate tracks and dropped players.  On the 7 IDSSE matches
(3 seeds) the wave-1 heuristic engine scores spellF1 0.435 / macroF1 0.773 (with its tuned overrides) and the wave-2 specialists
(leave-one-match-out) **0.642 / 0.910**, against 0.215 / 0.440 for the official baseline recipe; with a complete
ball 0.691 -> 0.761 (baseline 0.310).

`--specialists lomo` additionally scores the learned route **leave-one-match-out** (the specialists of every
held-out match are trained on the other six; the held-out match is the unlabelled target, so its pseudo-labels
come only from its own degraded tracking):

```bash
python scripts/eval_external.py --only-complete-ball --degrade --seeds 0 1 2 --specialists lomo
python scripts/eval_external.py --only-complete-ball --degrade --seeds 0 1 2 --specialists lomo --no-pseudo
```

`scripts/explore_occlusion_team.py` checks how well the team in possession can be predicted while the ball is
hidden from the collective-physics (#17) and ball-memory (#18) features alone (leave-one-match-out, degraded
external matches); its dataset builder is reused by the specialist B work.

Hyper-parameters are only ever tuned on such external data, never on the public 5 minutes of BAR-ATH.
The decoder parameters were swept with Optuna in `mtah2027-external-benchmarks` (`scripts/run_optuna_sweep.py`,
nested leave-one-match-out estimate + an unused degradation seed).  The heuristic route adopts its tuned values
through `route_overrides.heuristic` in `configs/default.yaml` (degraded IDSSE spellF1 0.396 -> 0.439 nested,
0.384 -> 0.427 on the unused seed); the specialists keep the shared defaults (+0.0195, under the +0.02 rule).
`reports/external_evidence.json` was regenerated with the final defaults (Issue #33): degraded 0.435 (heuristic, with its
overrides) vs 0.642 (specialists), complete ball 0.691 vs 0.761; the KPI 7 decision is unchanged (specialists).

### Improvement round after the Optuna sweep

Because the learned route under-segments (2467 vs 3061 possessions, 81 % of the hidden-ball turnovers missed), a
round of changes was measured with the LOMO protocol above (seeds 0-2) under one adoption rule: mean spellF1 gain
>= +0.01 and no seed worse than -0.003, else the code stays but is off by configuration.  Issue #30: past / future
window features (+-50 / +-100 frames; non-causal smoothing is legitimate offline) for Specialist B: **adopted**,
0.6059 -> 0.6307 spellF1 (every seed about +0.025), macroF1 0.9003 -> 0.9089, hidden-ball turnovers missed 81 -> 71 %;
opening H<->A / H/A->D inside occlusions at the specialists' logit sign changes or P(D) peaks: no gain (+-0.003),
**off** (`transition_matrix.spec_open`).  Details, negative results and the other items (specialists overrides, MBR)
Issue #31: `route_overrides.specialists` from the refreshed Optuna sweep (benchmarks repo), applied by `run_core` to the
learned route only: **adopted**, 0.6307 -> 0.6415 spellF1 through the real LOMO pipeline, better on every seed (the
search used seeds 0-1; the unused seed 2 gains +0.006).  Issue #32: MBR (median-boundary consensus, `mbr.enabled`) over tempering / `ha_occluded` variants of the
learned route only: no gain (+-0.0004), **off**.  Issue #34: explicit-duration (semi-Markov) decoder with per-state
duration tables learned from the external labels (`duration.enabled`): possessions 2409 -> 2503 of 3061 but spellF1
0.6415 -> 0.6390-0.6409, **off**.  Details and negative results in `docs/TECHNICAL_REPORT.md`
section 5.x.
Issue #35: extra training matches.  Validation A = held-out PFF matches with real occlusions (4 folds), validation B =
the 7 degraded IDSSE leave-one-match-out: IDSSE + 20 PFF matches at weight 0.5 is **adopted** (A 0.4659 -> 0.5231, B
0.6415 -> 0.6321); SkillCorner (-0.03 on B) is off.  The IDSSE-only and PFF-only models stay selectable
(`generate_submission.py --model-variant idsse|mixed|pff`, choose with `scripts/compare_input_stats.py`).
Issue #38: the degrader was calibrated to the label-free gap statistics of the competition ATD (`external_degrader_target`
in the configuration; `compare_input_stats.py` prints target vs degraded; the Issue #19 profile stays in
`external_degrader_legacy`).  Re-selection under it (validation B' = 7 degraded IDSSE LOMO, seeds 0-2; A = PFF held-out):
no variant gains the required +0.01 on B', so the default (IDSSE + PFF 0.5, trained with the legacy degrader) is kept;
`--model-variant calib` (calibrated-degrader training) is selectable.
Issue #48: the degrader gained a hidden-ball <-> dead-ball affinity (`dead_affinity`).  A label-free estimator (ball-free
player features + prior-shift EM, validated on IDSSE with planted affinity, PFF and Metrica; `scripts/estimate_gap_affinity.py`)
puts the D share of the BAR-ATH ball gaps at >= ~80 % (point estimate 94 %, aggregates only).  Training the specialists
with the calibrated degrader at affinity 33 gains +0.0174 on B' in that world (every seed >= +0.013) and passes the PFF guard
(A +0.005): **adopted as the default** (`models/specialists.json` = `--model-variant gap`); the previous default stays as
`--model-variant mixed`.  Details and the per-specialist PFF ablation: `docs/TECHNICAL_REPORT.md`, Item 7.
Issue #44: bounded decoder / fusion / ensemble sweep under the calibrated degrader (B'(33) seeds 0-1 selection, seed 2 blind, guards B'(1) and PFF): best +0.0091 on B'(33) (needed +0.01), not adopted.  `--model-variant ensemble` (`models/specialists_ensemble.json`, linear 0.75 `gap` + 0.25 `mixed`, committed files only) and the block `decoder_candidates.issue_44` of `configs/default.yaml` stay available, OFF; tools `scripts/decoder_sweep.py`, `scripts/train_cv_models.py`; Item 8 of the report.
Issue #42 (player-dropout invariance): `eval_external.py --strata` / `diagnose_possessions.py` break the results down by tracked
players (< 16, < 20, >= 20); on the calibrated degrader the model does not over-predict `D` under dropout.  Sample-size-robust
shape features (`specialist_features.normalise_shape`, `shape_min_field`) gain +0.002/+0.003 on B' and the native speed channel
(`pipeline.use_native_speed`) cannot be validated externally, so all stay OFF; the loader now maps the ATD channel `s` to
speed and checks its unit against the positions (the competition `s` is in m/s although the XML says km/h).

### Possession count and the KPI 7 evidence

On the degraded external matches the learned route predicts fewer possessions than the truth (2467 vs 3061 over
the 7 IDSSE matches, seed 0) while the heuristic route matches the count (3057).  `scripts/diagnose_possessions.py`
shows why: 43 % of the true possessions last < 3 s, and 976 of the 2351 true H<->A turnovers happen while the ball is
hidden, where 81 % are missed (37 % with the ball visible; 71 % after the context features of Issue #30).  Sweeping the transition gates (`ha_closed`, `ha_open`,
`ha_occluded`, touch gates), the spell optimiser (`micro_poss_s`, `min_dead_s`, `keep_min_touches`) and the fusion
temperature moves spellF1 by <= 0.003 and every extra flip is spurious, so the gap is the F1-optimal conservatism of
an evidence-limited emission model, not a gating bug; no default was changed.

`scripts/make_external_evidence.py` stores the aggregate leave-one-match-out result in the committed
`reports/external_evidence.json` (external open data only, with seeds, date and git sha).  The KPI dashboard uses it
for the KPI 7 route decision whenever `--external-lomo` is not run.

## Real-format check (Metrica EPTS exports)

The loaders are not tied to our synthetic imitation.  `load_atd_files(meta, raw)` parses any FIFA/EPTS pair,
including Metrica's open Sample Game 3 (eleven `DataFormatSpecification` blocks, one per substitution window, 35
player slots, no speed channel, `NaN` ball, normalised coordinates, `FieldSize` 105 x 68, periods in the provider
parameters); positions were cross-checked against `kloppy.metrica.load_tracking_epts` (3.18.0, test-only extra
`pip install -e .[crosscheck]`).  `scripts/check_real_format.py` runs the whole pipeline on such a pair and prints
aggregates only; Smart Tagging and the homography are optional:

```bash
python scripts/check_real_format.py --meta Sample_Game_3_metadata.xml --raw Sample_Game_3_tracking.txt
python scripts/check_real_format.py --meta atd.xml --raw atd.txt --smart-tagging st.xml --homography h.csv
```

`tests/test_29_real_format.py` + `tests/fixtures/metrica_sample3_structure_*` keep that structure covered in CI.

## Vision recovery (optional, off by default)

`vision/attention_crops.py`, `detectors.py`, `recovery.py` (Issue #11) recover the ball from the **tactical video**
where the tracking lost it: for every ball-missing graded frame (one in `vision.every_k`) a search point is
predicted (interpolation between the last/next seen ball pulled towards the pressure centroid), projected to
pixels with the tactical homography, and a 640x640 crop is cut and sent to a detector; detections are
reprojected with `H^-1`, filtered (confidence, inside the pitch, near the search point, plausible speed) and
injected into `ball_xy` only where it is NaN (`ball_source`: 1 tracking, 2 vision, 3 vision interpolation).
Detections are cached in `cache/vision/*.npz` (git-ignored; key = video size/mtime + plan + detector).
`run_pipeline` merges the cache when `vision.enabled: true`; it never decodes the video itself.

```bash
pip install -e ".[vision]"            # OpenCV headless 4.13.0.92 (reads the mp4, Hough fallback detector)
pip install -e ".[yolo]"              # + ultralytics 8.3.253 for the YOLO detector (COCO "sports ball")
python scripts/run_vision_recovery.py --data-dir data --benchmark 2000         # time the first 2000 frames
python scripts/run_vision_recovery.py --data-dir data --detector dryrun        # chain check, no model
python scripts/run_vision_recovery.py --data-dir data --detector yolo --weights yolov8n.pt [--every-k 5]
# then vision.enabled: true in configs/default.yaml and generate_submission.py as usual
python scripts/simulate_vision_value.py --fracs 0 0.2 0.4 0.6 --noise 1.0   # value estimate on external data
```

Only aggregates are printed (crops scanned, detections kept, share of hidden frames/gaps filled, runtime).
**Licences:** OpenCV is Apache-2.0.  `ultralytics` (and its pretrained weights, downloaded by the library) is
**AGPL-3.0**: it is imported only by `--detector yolo`, is a separate optional extra, and nothing in the
submission path needs it.  It stays off until validated locally on the real video.

**Multi-frame motion detector (Issue #36, `--detector motion`).**  A TrackNet-style temporal cue with OpenCV only:
the reader supplies the frame triplet (t-1, t, t+1) of each 640x640 crop, the camera pan is compensated
(phase correlation + sub-pixel warp), `min(|I_t - W(I_t-1)|, |I_t - W(I_t+1)|)` keeps what moves in both,
a morphological filter keeps blobs of the apparent ball size (taken from the homography scale, `vision.ball_diameter_m`),
the positions at t-1 / t+1 are checked in metres with `H^-1` against `vision.max_speed_mps`, and the centroid is
refined to sub-pixel.  Static lines and boots cancel out; a ball that does not move is not detectable.  Selectable
with `vision.detector: motion`; `vision.enabled` stays `false`.

## Audit video (local debugging only)

```bash
python scripts/render_audit_video.py --synthetic                                   # no data needed
python scripts/render_audit_video.py --data-dir data --start-frame 11185 --end-frame 11500
python scripts/render_audit_video.py --data-dir data --from-audit-csv outputs/audit_transitions.csv   # KPI 5 export
```

`vision/overlay_renderer.py` + `scripts/render_audit_video.py` (Issue #36) draw, over the tactical mp4 (or a plain
pitch when there is none): the pitch lines projected with `H_t`, a minimap (22 players, ball, possessor, collective-coherence
arrows), a 25-frame ball trail, a halo on the possessor and a HUD with the predicted H/A/D, P(H)/P(A)/P(D) and ball
visible/hidden.  Output: `outputs/audit_video.mp4` (`*.mp4` and `outputs/` are git-ignored).  **The audit video of
BAR-ATH is for local pipeline debugging only (alignment, format, bugs): never to label frames by hand or tune
parameters, and never uploaded.**  The script prints only aggregates and paths.

## Broadcast sync (optional, off by default)

`preprocessing/broadcast_sync.py` (Issue #10) places the TV tracking (`data/broadcast/*_broadcast_atd.*` +
`*_broadcast_homography.csv`, **not aligned** to the tactical clock) on the tactical timeline and injects its ball
where the tactical ball is NaN (`ball_source` 4).  Continuous blocks (homography `valid` and error < 5, >= 75
frames, split every 90 s) get a crop-invariant fingerprint (mean speed of the visible players, median distance to
the nearest visible opponent); the offset comes from a z-normalised cross-correlation against the tactical
fingerprint (accepted at corr >= 0.65), refined and verified with the player positions (label-free nearest
same-team distance) and a ball-agreement test; blocks that fail the strict global test are retried in a narrow
window around the offset of their nearest synchronised neighbour.  Without broadcast files the stage is skipped
and logged.

```bash
# 1. set broadcast_sync.enabled: true in configs/default.yaml (needs data/broadcast/ next to data/tactical/)
python scripts/generate_submission.py --data-dir data --out "outputs/Opus Operandi.csv"
#    the log prints "broadcast: N/M blocks synchronised, K hidden frames filled"; diagnostics["broadcast"] has
#    only aggregates (blocks, accepted, offset/corr/residual medians, fill rate, seconds)
# 2. value estimate on other matches with a simulated camera (never BAR-ATH):
python scripts/simulate_broadcast_value.py --seed 0 --specialists lomo
```

## Real-data sanity report (aggregates only)

```bash
python scripts/real_data_sanity.py --data-dir data            # JSON of counts / shares, never per-frame rows
python scripts/real_data_sanity.py --synthetic                # same report on the synthetic match
python scripts/external_d_diagnostic.py --bench-dir ../mtah2027-external-benchmarks/cache --groups pff idsse --degrade
```

`real_data_sanity.py` (Issue #37) reports the declared ATD channels, frames parsed vs expected, ball-missing share
per graded half, on-pitch players per team, homography validity (tactical / broadcast), Smart Tagging counts by
code / team / set-piece type, which pipeline stages ran or were skipped, and the share of predicted `D` cut by ball
visibility, tracked players (< 20) and Smart Tagging set-piece windows.  `external_d_diagnostic.py` produces the
same cuts on external matches next to their true `D`, to tell a bug from a property of the data.

## Smart Tagging on the real match (Issue #40)

```bash
python scripts/smart_tagging_check.py --data-dir data       # anchors, throw-ins, phase-prior consistency (aggregates)
python scripts/eval_anchor_accuracy.py --bench-dir ../mtah2027-external-benchmarks/cache --groups idsse pff metrica --collective-only-incomplete
```

`Team` of a phase is "the team performing the event" (`docs/official/OVERVIEW.md`), so the four defensive codes
(`Defending in Attacking / Middle / Defensive Third`, `Defensive Transition`) name the *defender*: the loader credits
the opponent in `phase_prior_home` (`smart_tagging_loader.DEFENDING_CODES`, `defending_codes=()` restores the raw
tag).  On BAR-ATH every attacking phase has a paired defending phase of the opponent over exactly the same frames, so
without the flip they cancelled each other; with it 67 % of the graded frames carry a prior and the predicted
possessor agrees with the possession side in 65-97 % of the alive frames per code (and 3-26 % with the raw defending
tag).  The collective restart cue (ball hidden around the set piece) is now gated on recent dead-ball evidence
(`setpiece_anchor.collective_dead_lag_s`, external pseudo-window check, three sources: mean |t* error| 208 -> 126,
247 -> 181, 259 -> 143 frames).

## Smart Tagging alive-ball prior (Issue #43, off by default)

```bash
python scripts/smart_tagging_stats.py --data-dir data          # structure of the real tagging (aggregates only)
python scripts/eval_phase_alive.py --protocol b --noise none medium high \
    --variants off a1:heuristic_emissions.w_phase_alive=1.0 --training-pickle t.pkl \
    --sources idsse pff --source-weight pff=0.5 --model-dir models_cache --json out.json
python scripts/eval_external.py --degrade --sim-tags medium    # any external run with simulated tags
```

`heuristic_emissions.w_phase_alive` lowers the `D` logit on hidden-ball frames covered by a live Smart Tagging phase
outside the pre-`t*` / post-goal windows (`pipeline.live_phase_mask`); `transition_matrix.phase_cue` opens `H<->A`
around `Attacking` / `Defensive Transition` starts; `pipeline.alive_exclude_dead_evidence` drops frames with dead-ball
evidence from the mask.  `io/smart_tagging_sim.py` derives a Smart Tagging from the true labels of an external match
(BAR-ATH structure, jitter / missed / spurious tags, dead-time bleed; levels low / medium / high) because the external
matches have none -- optimistic by construction.  Measured on B' and A at three noise levels the best gain is +0.002
spellF1, far from the +0.01 rule, so everything stays **off** (`docs/TECHNICAL_REPORT.md`, Issue #43).

## Restart-team rules and Long Ball flags (Issue #49, off by default)

```bash
python scripts/eval_phase_alive.py --protocol b --sim-restart --noise medium high     --variants off rules:restart_rules.enabled=true     wide:transition_matrix.restart_team_before_s=4.0,transition_matrix.restart_team_after_s=2.0     offset:setpiece_anchor.offset_prior=true --training-pickle t.pkl --sources idsse pff --source-weight pff=0.5
```

`features/restart_rules.py`: the team of every restart is voted among the tag, the rules of the game (`Kick Off (After
Goal)` = team that conceded the preceding `GOALS` tag; second-half `Kick Off (Start)` = opposite of the first-half
one) and the tracking (first clear controller after `t*`); dissent lowers `team_conf`, which scales
`transition_matrix.restart_other_team`; `restart_team_before_s / after_s` widen the window of that penalty beyond
`t* +- 3` frames; `setpiece_anchor.offset_prior` restricts the collective-cue search to the `t* - tag_start` band learned
from the ball-method anchors of the same match; `restart_rules.calibrate_weight` and `auto_long_ball_weight` take the
weights from Specialist A's confident states (never the ground truth); `heuristic_emissions.w_long_ball` lowers the `D`
logit on hidden-ball frames of `Long Ball` / `Ball in the Box` phases.  `--sim-restart` makes the simulated tags
name the kick-off teams and flip 5 / 10 / 20 % of the set-piece teams.  Decision and numbers:
`docs/TECHNICAL_REPORT.md`, Issue #49.

## Broadcast sync on the real bundle (Issue #39)

`python scripts/real_broadcast_check.py --data-dir data [--effect]` prints aggregates only (blocks synchronised,
correlation / residual / offset distributions, share of the hidden tactical frames filled, physical consistency of
the injected ball, label change with injection).  Decision (adoption rule: +0.01 spellF1 on the external simulation,
no seed worse than -0.003): **stays off** (`broadcast_sync.enabled: false`), see `docs/TECHNICAL_REPORT.md`.

## Repository layout

```
configs/default.yaml                 all parameters, one section per module
src/mtah2027/
  config.py                          typed config, match constants, bridge to games.py, seeds
  io/                                atd_loader (#5), smart_tagging_loader (#6), homography_loader (#7)
  preprocessing/                     clean_tracks (#8), kinematics_and_orientation (#9), broadcast_sync (#10, off)
  vision/                            attention_crops, detectors (+ motion, #36), recovery (#11, off), overlay_renderer (#36); aerial_filter (#12, off by default: -0.006..-0.016 spellF1 on degraded external)
  features/                          dead_ball_signals (#13), setpiece_anchor (#14), ball_kinematics (#16),
                                     throwin_resolver (#15), collective_physics (#17), ball_memory (#18)
  models/                            heuristic_emissions (#28 wave 1), baseline_labels, external_degrader (#19),
                                     frame_classifier + specialist_features / _data (shared), specialist_ball_present
                                     (#20), pseudo_labeling (#21), specialist_ball_absent + specialist_fusion (#22)
  decoding/                          transition_matrix (#23), dynamic_viterbi (#24), mbr_consensus (#26),
                                     duration_model + semi_markov (#34, off)
  postprocessing/                    spell_optimizer (#25), submission_contract (#26)
  validation/eval_bridge.py          in-memory bridge to the official scorer (#4)
  validation/kpi_dashboard.py        the seven quality KPIs (#27); CLI scripts/run_kpi_dashboard.py
  pipeline.py                        run_core (arrays) and run_pipeline (files -> contract-compliant labels)
models/specialists*.json             Specialists A/B trained on external open data (default, idsse, mixed, pff, calib; reproduced by train_specialists.py)
scripts/                             generate_submission.py, train_specialists.py, eval_external.py,
                                     generate_synthetic_fixtures.py, ...
tests/                               one test module per issue + shared synthetic-match fixtures
evaluation/  games.py  starter/      vendored official scorer, constants and starter kit (unmodified)
docs/                                Opus Operandi.md (submission document), TECHNICAL_REPORT.md, ISSUES_TRACKER.md, official/
```

## Tests and CI

```bash
ruff check src/ tests/ scripts/
pytest -v                    # full suite; `-m "not slow"` skips the full-length (199,319 frames) runs
```

GitHub Actions (`.github/workflows/ci.yml`) runs the security gate (no match data in Git), ruff, the test suite
and the end-to-end determinism check (two full-length synthetic runs must give the same SHA256).

## Third-party code

`games.py`, `evaluation/` (the official scorer `possession_metrics.py`, `validate_submission.py`,
`run_evaluation.py`, public ground-truth metadata and baseline results) and `starter/` are vendored
**unmodified** from <https://github.com/metrica-sports/more-than-a-hack-2027> at commit
`f8f2629ff1e0985f81d7e59d8413678514ee1530` (MIT; licence in `THIRD_PARTY_LICENSES/`).  The engine imports
them -- through `mtah2027.validation.eval_bridge` and `mtah2027.config` -- and never re-implements them.

## External data

No external data is redistributed in this repository; the matches are downloaded and prepared by
`mtah2027-external-benchmarks`.

* **IDSSE / DFL Sportec open data** (7 Bundesliga matches; training, tuning and validation), licensed **CC BY 4.0**.
  Citation: Bassek, M., Rein, R., Weber, H., & Memmert, D. (2025). An integrated dataset of spatiotemporal and event
  data in elite soccer. *Scientific Data*, 12, 195. <https://doi.org/10.1038/s41597-025-04505-y>.  Data: Figshare
  <https://doi.org/10.6084/m9.figshare.28196177> and Hugging Face `pysport/idsse-data`
  (<https://huggingface.co/datasets/pysport/idsse-data>).
* **PFF FC World Cup 2022 Open Data** (training and validation under real occlusions): free public access under PFF's
  terms at <https://data.pff.com>; not redistributed.
* **Metrica Sports sample data** (evaluation only): <https://github.com/metrica-sports/sample-data>.
