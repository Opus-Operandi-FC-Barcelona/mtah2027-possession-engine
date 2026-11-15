# Opus Operandi -- technical document (More than a Hack 2027, phase 1: ball possession)

Team: **Opus Operandi**.  Code: <https://github.com/Opus-Operandi-FC-Barcelona/mtah2027-possession-engine>.
Companion repository (other-match benchmarks and tuning): `mtah2027-external-benchmarks`.  Longer write-up with
every experiment: `docs/TECHNICAL_REPORT.md`.  The sections follow the points requested in `SUBMISSION.md`.

## 1. Approach and why

The ranking metric, spellF1, matches same-team possession spells by temporal IoU (0.3 ... 0.7); a `D` splits a
possession and `X` is forward-filled.  A one-frame offset is free, a one-second offset costs about 0.35 spellF1.  The
system is therefore built around *the right number of possessions* and *boundaries within a few frames*, not around
per-frame accuracy.  Possession in the ground truth is event-like: the team of the player that last touched the ball
agrees with the labelled possession on about 92 % of the alive frames of the external matches, against about 78 % for the
player nearest to the ball (the baseline's rule), and about 99 % of the labelled turnovers happen within 10 frames of a
detected touch.  The pipeline is built on that.

Pipeline (`src/mtah2027/pipeline.py`), every stage deterministic and vectorised:

1. **Ingest** the tactical ATD (every channel declared in `atd.xml` is parsed; here `x, y, s`), the Smart Tagging XML and
   the homography.
2. **Clean and orient**: drop off-pitch staff and duplicate tracks (< 0.6 m), interpolate gaps of <= 5 frames,
   Savitzky-Golay velocity and acceleration, mirror each half so that the home team attacks +x.
3. **Features**: continuous dead-ball score (ball beyond the lines, ball at rest, slow or walking players, set-piece
   geometry), distances and Link-and-Hoernig-style control weights, **touch detection**, exact restart frames `t*` of
   every tagged set piece found in the tracking (never taken from the tag window), throw-in team resolution from the
   tracking where the tag says `N/A`, dynamic goalkeepers and collective physics (spectral and convex-hull areas and
   their derivatives, vector coherence), ball memory for the frames where the ball is not tracked.
4. **Emissions** `log_probs[N, 3]`: **Specialist A** (ball visible) and **Specialist B** (ball absent), calibrated
   LightGBM classifiers fused per frame by ball visibility, with Smart Tagging phase as a soft prior.  A heuristic route
   (last toucher, dead-ball score) is the fallback and the comparison point.
5. **Time-dependent transitions** `log A_t[N, 3, 3]`: a possession changes hands only around a touch of the gaining team
   (cost -1.4, otherwise -11.5) or, moderately, inside a long ball occlusion; `H/A -> D` opens with dead-ball evidence;
   `D -> H/A` stays closed until the dead evidence drops or a restart frame is reached.
6. **Viterbi** per graded half, independently (numba kernel, bit-identical NumPy fallback).
7. **Spell optimiser**: micro-possessions shorter than about 1 s (0.98 s on the learned route, 1.75 s on the heuristic
   route, both tuned on other matches) between the other team's runs are absorbed unless the ball is visible
   and there is a double touch; long possessions get a `D` bridge over >= 4 s dead-evidence valleys that coincide with a
   touchline or set piece; forced-`D` intervals before restarts are applied.
8. **Contract**: `X` outside the graded blocks, `D` in the second-half pre-kick-off interval, no internal `X`, validated
   with the official `validate_submission.py`.

Why a decoder rather than per-frame thresholds: the transition costs encode the football prior that changes of
possession are rare and happen at touches, which removes the flicker that destroys spellF1 for per-frame classifiers.
Why learned specialists on top: while the ball is hidden (about 41 % of the graded frames of the provided tracking) the
team can still be inferred from collective movement (width and advance of the attacking block); on other matches with
the ball hidden, a model using only those cues predicts the team in 0.85 of the hidden-ball alive frames, against 0.62
for the "last touch" rule.

## 2. Assets used, tried and dropped

| Asset | Status | How |
|---|---|---|
| Tactical ATD (`x, y, s`, 46 slots, ball) | **used** | all geometry, kinematics, features |
| Smart Tagging XML | **used** | set-piece and goal windows -> exact restart frames and forced `D`; throw-in teams; weak phase prior.  The four defending codes name the *defender* (`Team` = team performing the event), so the loader credits the opponent with the possession of those phases |
| Tactical homography | loaded and validity-checked | only consumed by the optional stages below, which are off |
| Broadcast ATD + homography | **tried, off** | block-wise synchronisation (`preprocessing/broadcast_sync.py`): 63 % of the blocks synchronised but the TV ball fills only 7.4 % of the hidden tactical-ball frames; +0.005 spellF1 on the leave-one-match-out simulation, below the +0.01 adoption rule |
| Tactical / broadcast video: YOLO and multi-frame ball detector, attention crops | **built, off** | `vision/`; no video was processed for the submission; simulation shows a gain only with a ~0.3 m accurate detector.  Measured label-free on the real tactical video (Issue #58, 1,000 frames with the tracked ball visible, ball ~4.5 px): the most confident motion detection is within 2 m of the tracked ball in ~10 % of the frames, with ~3.7 false positives per frame, so the route stays off  The AGPL `ultralytics` package is an optional extra that nothing in the submission path needs |
| Aerial-ball filter | **built, off** | lowered spellF1 on degraded external matches (touches at kicks/landings are real boundaries) |
| Explicit-duration (semi-Markov) decoder | **built, off** | possessions 2409 -> 2503 of 3061 but spellF1 -0.001 ... -0.004 |
| MBR boundary consensus | **built, off** | +-0.0004 |
| Specialist-opened transitions inside occlusions | **built, off** | within +-0.003 |
| Block-masking pseudo-labelling of the provided match (#21) | **built, off** | algorithmic labels on the provided game are allowed, but on external matches it lowered spellF1 (0.606 -> 0.590) |
| SkillCorner as training data | **tried, dropped** | -0.026 ... -0.031 spellF1 on the degraded validation |
| Smart Tagging phases as an alive-ball prior / transition cue while the ball is hidden (#43) | **built, off** | measured with a Smart Tagging simulator (tags derived from true labels with low/medium/high noise): best +0.002 |
| Smart Tagging restart rules (post-goal and second-half kick-off team, wider restart window, `t*` offset prior, Long Ball flags) (#49) | **built, off** | +0.0003 at best: restarts were already resolved by the tracking anchors |
| Robust collective-shape features under player dropout, native speed channel (#42) | **built, off** | +0.002 ... +0.003 |
| Decoder re-optimisation and ensemble of two models trained in different worlds (#44) | **built, off** | best +0.009 (all guards positive) after about 700 candidates; selectable as `--model-variant ensemble` |
| Official scorer, `games.py`, starter | **used, vendored unmodified** | metric, constants, format check |
| Public 5 minutes of ground truth | pipeline check only | alignment and format; never used to choose between variants or tune |

Rule of the project: every optional stage is adopted only if it gains >= +0.01 mean spellF1 on the other-match
validation with no seed worse than -0.003; otherwise the code and tests stay and it is switched off by configuration.

## 3. Training data

**Nothing from the provided game is used as a label or for fitting.**  The learned models (Specialists A/B) and every
tuned hyper-parameter come from other matches, labelled **by algorithm**, never by hand:

| Dataset | Source / who labelled | Matches used | Licence | Role |
|---|---|---|---|---|
| IDSSE / DFL Sportec open data (Bassek et al., *Scientific Data* 12, 195, 2025, <https://doi.org/10.1038/s41597-025-04505-y>; data <https://doi.org/10.6084/m9.figshare.28196177>, also Hugging Face `pysport/idsse-data`; ids J03WMX, J03WN1, J03WOH, J03WOY, J03WPY, J03WQQ, J03WR9) | DFL-provided ball-status and possession channels; labels derived mechanically from them (ball dead -> `D`; alive -> the team of the possession channel) | 7 Bundesliga matches, 25 fps | CC BY 4.0 | training of the specialists (degraded), all tuning and validation |
| PFF FC World Cup 2022 Open Data (<https://data.pff.com>, free public access under PFF's terms) | PFF event data (`OTB` owner, `OUT` dead, set-piece restarts); labels derived by replaying the events | 64 converted; first 20 by name for training at weight 0.5, 8 never-used ones for validation | PFF terms accepted by the team; not redistributed | training and validation under real occlusions (ball missing in about a third of the frames) |
| Metrica Sports sample data, Games 1-3 | events -> labels (control events give possession, stoppages give `D`) | 3 | no licence file in the repository; used locally only | evaluation only (their ball channel disappears while the ball is dead, which would leak `D`) |
| SkillCorner Open Data, A-League 2024/25 | `phases_of_play` (gaps between phases = `D`) | 20 converted | MIT | tried for training, dropped; 10 fps broadcast tracking |

Rules for the labels are in `mtah2027-external-benchmarks/docs/LABELLING_RULES.md`.  For the IDSSE matches they mirror
how the competition derives its own ground truth from tracking channels.

* **Degradation.**  Each full-view match is degraded so that the models see competition-like inputs: the ball hidden in
  contiguous blocks, correlated tracking noise, identity cuts, duplicate and dropped tracks.  Two profiles exist: the
  first one built from the task description (`external_degrader_legacy`), and one calibrated against **label-free
  aggregate statistics of the provided tracking** (gap-length distribution, tracked players per frame;
  `external_degrader_target`); no labels of the provided game are involved.  Since Issue #48 the calibrated profile
  also places the hidden-ball blocks preferentially over dead-ball stretches (`dead_affinity`): a label-free estimator
  (ball-free player features, a classifier trained on other matches, prior-shift EM per tracked-player stratum),
  validated on IDSSE with planted affinities and on PFF and Metrica, estimates that at least about 80 % of the
  hidden-ball graded frames of the provided tracking are dead, against 41 % if gaps were placed at random.  The default
  model is trained in that estimated world (`dead_affinity` 33, realised odds ratio about 12-19).
* **Team symmetry.**  Each match is also used with the home/away roles swapped, so the models never see a team identity.
* **Calibration** by out-of-fold temperature on whole held-out matches.
* **Reproducibility.**  `python scripts/train_specialists.py --bench-dir <external cache>` rebuilds `models/specialists*.json`
  (a LightGBM text dump plus metadata; no match data); training is deterministic.
* A deterministic **synthetic match** (`scripts/generate_synthetic_fixtures.py`) in the exact layout of the data bundle
  is used for tests and CI.

## 4. Possession rules implemented

* **Dead ball (`D`).**  Declared from evidence: ball beyond the lines, ball at rest, players slowing or walking,
  set-piece geometry, and the exact interval before each restart `t*` found in the tracking; also between a goal and the
  next kick-off.  A `D` always splits a possession (`H D H` = two possessions).
* **Possession and loose balls.**  A team is in possession from the touch that wins the ball until the other team's
  winning touch.  While the ball is in flight or loose the state is held (the last toucher keeps the ball).
* **Challenges and duels.**  A change of possession requires a touch of the gaining team; simultaneous contact by both
  teams does not flip by itself, and micro-possessions shorter than about 1 s between the other team's runs are absorbed unless the ball
  is visible and shows a double touch.
* **Restarts.**  The restart frame is anchored in the tracking (the ball leaves rest), not at the tag window (the
  `Kick Off (Start)` windows open about 3.5 s before the kick-off).  Throw-ins tagged `Team = N/A` are resolved from the
  tracking (the isolated player within 1.5 m of the touchline, confirmed by the first controller); when the vote is
  below the confidence floor they stay unresolved and both `D -> H` and `D -> A` remain open.
* **Hidden ball.**  The last and next toucher are carried over the gap with decaying confidence; inside long
  occlusions the specialists combine evidence from both sides and the change lands where the collective movement
  reverses.
* **`X`.**  `X` outside the graded blocks and never inside them (the scorer forward-fills it for spellF1 and counts it
  as an error for macroF1).  Graded blocks come from `games.py`: first half from the kick-off to the end of the half,
  second half from the kick-off to the final whistle; the interval between the start of the second half and its
  kick-off is graded and fully dead, hence `D`.  Nothing crosses a half boundary.
* Frame alignment: video frame = ATD `frame_id - 1`.

## 5. Results and limitations

All numbers are on **other matches** with the official metric (`evaluation/possession_metrics.py`), mean over the
7 IDSSE matches, leave-one-match-out (the held-out match is an unlabelled target), seeds 0-2.  The score of the
provided game is not known to the authors.

| Setting | spellF1 | macroF1 |
|---|---|---|
| Official-style baseline (nearest player, majority vote), degraded | 0.215 | 0.440 |
| Heuristic route, degraded (first degrader) | 0.435 | 0.773 |
| Specialists (IDSSE only), degraded (first degrader) | 0.642 | 0.910 |
| Specialists (default, IDSSE + PFF), degraded (first degrader) | 0.632 | 0.907 |
| Specialists (Milestone 8 default), calibrated degrader | 0.604 | 0.902 |
| Specialists (Milestone 8 default), calibrated degrader with gaps over dead ball (B'(33)) | 0.657 | 0.914 |
| **Specialists (final default, trained at `dead_affinity` 33), B'(33)** | **0.675** | **0.922** |
| Specialists (final default), calibrated degrader with gaps placed at random (B'(1)) | 0.600 | -- |
| Specialists (final default), complete ball | 0.750 | 0.948 |
| Specialists (final default), 20 held-out PFF matches with real occlusions (4-fold) | 0.526 | 0.864 |

B'(33) and B'(1) differ in the evaluation world, not in the model: the same model scores higher when the hidden-ball
blocks fall over dead ball, because those frames are easier.  What matters is the difference between models measured in
the same world.

**Limitations (honest list).**

* **Player dropout.**  The provided tracking has on average 20.5 tracked players per frame and 14.2 % of the frames with
  fewer than 20; external matches have <= 1.2 % of such frames, so this regime has no external counterpart.  On the
  provided match nearly all of those frames are predicted `D`.  Issue #42 showed that player dropout alone does not
  cause this on external matches (fewer than 20 players: predicted `D` 45.7 % against 43.6 % true), and the label-free
  estimator of Issue #48 finds that 91 % of those frames lie inside ball gaps, i.e. joint tracking outages that most
  likely coincide with stoppages; the high `D` share there is therefore plausible but cannot be validated externally.
* **Hidden-ball turnovers.**  About 70 % of the H<->A turnovers that happen while the ball is hidden are still missed
  (81 % before the past/future-window features), so the learned route predicts fewer possessions than the truth
  (2409 against 3061 on the 7 degraded IDSSE matches).  Every extra flip tried is as often wrong as right.
* **Domain shift.**  Specialists are trained on Bundesliga and World Cup data degraded by our own model; the tracking
  noise and the gap structure of the provided match are known only through label-free aggregates.  On the synthetic
  match, whose physics is not football, the learned route scores below the heuristic one; the choice of route is made
  with external evidence only (KPI 7), never with the public ground truth.
* The validation measures are two proxies (degraded IDSSE and real-occlusion PFF) that disagree on the size of some
  trade-offs; spellF1 noise across matches is about 0.04.
* Smart Tagging weights cannot be tuned (no external match has Smart Tagging).  Video-based ball recovery and TV
  synchronisation were not validated on the real video beyond the aggregates above.
* With more time: a detector trained for the tactical camera, and a longer-context model of collective play for the
  hidden-ball turnovers.

## 6. Milestone 9 results

Milestone 9 ("intelligent analysis over the provided data and invariance to player dropout") was measured under a
common protocol fixed after the audit against the official repository (#47): the ground truth is derived from tracking
flags, and only the IDSSE labels share that semantics, so the primary criterion is **B'** = 7 IDSSE leave-one-match-out
with the calibrated degrader, seeds 0-2 (adopt at >= +0.01 mean spellF1, no seed below -0.003); PFF held-out (event
labels) is a guard only (reject if it drops more than 0.03).  From #48 on, B' is evaluated in the estimated world where
gaps fall over dead ball (B'(33)), with the random-placement world B'(1) as a second guard.

<!-- BEGIN MILESTONE 9 RESULTS -->
| Issue | Change | Decision | Result (spellF1, other matches) |
|---|---|---|---|
| #42 | Player-dropout invariance: strata by tracked players, robust shape features, native speed channel | off | best +0.004; dropout alone does not over-predict `D` (< 20 players: 45.7 % predicted vs 43.6 % true) |
| #43 | Smart Tagging alive-ball prior and transition cue while the ball is hidden (tag simulator) | off | best +0.002 at medium / high tag noise |
| #44 | Decoder re-optimisation (about 700 candidates) and ensemble of the two models trained in different worlds | off (`--model-variant ensemble` selectable) | best +0.009 on B'(33), all guards positive, below the bar |
| #47 | Audit against the official repository (`docs/OFFICIAL_AUDIT.md`) | -- | protocol above |
| #48 | Hidden-ball / dead-ball affinity in the degrader, label-free estimator, model re-trained at affinity 33 | **adopted** | B'(33) 0.657 -> **0.675** (+0.017, all seeds), PFF 0.521 -> 0.526, B'(1) -0.008 |
| #49 | Restart-team rules (post-goal / second-half kick-off, tracking vote), `t*` offset prior, Long Ball flags | off | +0.0003 |
| #50 | Set-piece anchor: bounded tag-window fallback and period-start clamp | **adopted** (correctness) | +0.0085 on B' (40 of 42 runs up) |
| #51 | Front end isolated per graded play block (no leak across half-time) | **adopted** (correctness) | +0.0004 |
| #52 | Submission depends only on the committed model (`auto` loads it; retraining reproduces it) | **adopted** | identical SHA256 for train and load |

Provided match, label-free aggregates of the submission (no ground truth involved; not a selection criterion):

| | Milestone 8 submission | Final submission |
|---|---|---|
| Possessions | 243 | 266 |
| Mean possession | 12.8 s | 12.3 s |
| `D` share of graded play frames | 50.1 % | 47.5 % |
| `H` / `A` share of graded play frames | -- | 33.0 % / 19.5 % |
| `D` share of hidden-ball graded frames | -- | 91.7 % (label-free estimate of the truth: >= about 80 %) |
| Runtime (load model, full match) | -- | 17.5 s |
<!-- END MILESTONE 9 RESULTS -->

## 7. Milestone 10: post-freeze experiments

Nine further ideas were each built in a separate branch and measured under the same protocol (B'(33), seeds 0-2, adopt at
>= +0.010 mean with no seed below -0.003; guards B'(1) and PFF).  None passed, so the submission is unchanged.

| Issue | Change | Result (spellF1 vs default on B'(33)) | Decision |
|---|---|---|---|
| #55 | 56 PFF training matches instead of 20 | -0.012 (PFF guard +0.03) | off |
| #56 | Multi-scale context windows and goalkeeper depth | -0.002 | off |
| #57 | Learned H<->A turnover gate (third specialist) | -0.001 ... -0.006 (+0.011 / +0.015 on real PFF occlusions) | off |
| #58 | Classic ball detectors on the real tactical video (aggregates only) | top hit within 2 m in 10 % of frames | vision off |
| #59 | Latency bias of the boundaries | about 1 frame early, below the 3-frame threshold | no shift |
| #60 | Bag of three specialist seeds | -0.001 | off |
| #61 | Boundary snap to the touch / restart frame | +0.0002 | off |
| #62 | Expected-spellF1 (minimum Bayes risk) decoding | 0.000 ... -0.001 | off |
| #63 | Pre-trained football ball model (roboflow/sports + supervision) on the real video | top hit within 2 m in 60 %, median 1.1 m; simulated -0.074 | off |
| #65 | Smart Tagging phase prior on the H-vs-A odds of hidden-ball frames (fixed 0.8-2.5 and label-free auto) | B'(33) +0.0009 at best, -0.0073 at high noise; A down to -0.034 | off |

The decoder-side changes all target the same missed turnovers, and turnover recall did not move with any of them: the
remaining error sits in the specialists' emissions, not in the decoder.  Details: `docs/TECHNICAL_REPORT.md`, Milestone 10.

## 8. Reproduction

```bash
python -m venv .venv && source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -r requirements.txt && pip install -e . --no-deps
# unpack the organisers' bundle into data/ (tactical/ and broadcast/), see data/README.md
python scripts/generate_submission.py --data-dir data --specialists load --team "Opus Operandi"   # -> "outputs/Opus Operandi.csv"
python evaluation/validate_submission.py "outputs/Opus Operandi.csv"
python scripts/generate_submission.py --data-dir data --specialists load --kpi                 # + KPI quality dashboard
python scripts/generate_submission.py --data-dir data --model-variant mixed  # alternatives: gap (= default) | mixed | calib | idsse | pff | ensemble
python scripts/generate_submission.py --synthetic --out /tmp/sub.csv         # self-check without the data (what CI does)
```

All dependencies are pinned to exact versions (`requirements.txt`, `pyproject.toml`; Python >= 3.10, developed on 3.13,
CI on 3.11).  The submission uses only the committed `models/specialists.json` (Issue #52: `--specialists auto` and
`load` both load it; retraining with `train_specialists.py` reproduces it byte for byte); numba and LightGBM are optional at run time with identical-result fallbacks.  `PYTHONHASHSEED=0` is
enforced, seeds are fixed (`seed: 0`), LightGBM runs with `deterministic=True, num_threads=1`, and every decoder input
is rounded to 1e-6, so two runs (and the numba and fallback decoders) give the same SHA256.  CI generates the full-length
synthetic match twice and compares the hashes.  `data/`, `cache/`, videos and prediction CSVs are git-ignored and
blocked by a CI security gate; the match data is processed only locally and never leaves the machine.
