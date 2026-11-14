# Technical report -- Opus Operandi (More than a Hack 2027, phase 1: ball possession)

Repository: `Opus-Operandi-FC-Barcelona/mtah2027-possession-engine`.  This report follows the points requested
in the competition's `SUBMISSION.md`; the short submission document is `docs/OpusOperandi.md`.  **Implemented in this
version = Milestones 1-8 (issues #1-#40)**: a complete, deterministic end-to-end route whose emissions come from
learned Specialists A/B trained on external open data (IDSSE + PFF), with the heuristic emissions as the fallback
route, plus optional stages (broadcast sync, vision recovery, aerial filter, MBR, semi-Markov) that were built and
measured and are **off** by configuration.  **Milestone 9** (issues #42-#46: player-dropout invariance, Smart Tagging
alive prior, bounded decoder re-optimisation, documentation, release) is **in progress**; its results are not part of
this report yet (see "Milestone 9" before section 6).  Sections 1-4 below were written for waves 1-2 and the later
items are documented in 5.x; where they differ the later sections and `docs/OpusOperandi.md` prevail.
## 1. Summary of the approach and why

**What is scored.**  spellF1 matches same-team possession spells between prediction and ground truth by temporal
IoU (0.3 ... 0.7), a `D` splits a possession, `X` is forward-filled.  A one-frame offset is free, a one-second
offset costs about 0.35 of spellF1.  Hence the system is built around two things: *the right number of
possessions* and *boundaries within a few frames*, rather than per-frame accuracy.

**Pipeline** (`src/mtah2027/pipeline.py`):

1. **Ingest** the ATD (streaming parser; every channel declared in `atd.xml` is read, including a possible
   ball `z`, owning-team or alive/dead channel) and the Smart Tagging XML.
2. **Clean** the 46 slots (off-pitch staff, duplicate tracks < 0.6 m), interpolate micro-gaps (<= 5 frames),
   Savitzky-Golay velocities/accelerations and mirror each half so that the home team always attacks +x.
3. **Features**: continuous dead-ball score (ball out of bounds, ball at rest, slow/walking players, set-piece
   geometry, status channel if present), ball-to-player distances and Link-and-Hoernig-style control weights,
   **touch detection** (ball acceleration peak or direction change with a player within 1.8 m).
4. **Restart anchoring**: the exact restart frame `t*` of every tagged set piece / kick-off after a goal is found
   in the tracking (ball leaves rest), never from the tag window; the dead interval before `t*` becomes a
   forced-`D` interval.
5. **Emissions** `log_probs[N, 3]` (wave 2, `--specialists`): **Specialist A** (ball visible, Issue #20) and
   **Specialist B** (ball absent, Issue #22) -- calibrated LightGBM classifiers over the dead-ball signals (#13),
   ball kinematics (#16), collective physics (#17: dynamic goalkeepers, spectral / convex-hull areas and their
   2 s derivatives), phase coherence and ball memory (#18) -- are fused per frame by ball visibility
   (tempered x0.5, floored at 1e-3, rounded to 1e-6); Smart Tagging phase, ATD owner and status channels enter
   as soft log-odds priors when they exist.  If no model is available the wave-1 heuristic below is used:
   calibrated-ish `p(D)` from the dead score and an H-vs-A logit dominated by the
   **last toucher** (see below), a weak instantaneous nearest-player term (weight 0.5 vs 3 for the last toucher), a weak Smart Tagging phase prior and a
   strong term if the ATD exposes an owning-team ball channel.  When the ball is missing the last (and the next)
   toucher is carried over the gap with decaying confidence.
6. **Time-dependent transition matrix** `log A_t[N, 3, 3]`: a possession can change hands only around a touch of
   the gaining team (open -1.4, otherwise -11.5) or, moderately (-6), inside a long ball occlusion; `H/A -> D`
   opens with dead evidence; `D -> H/A` stays closed until the dead evidence drops or a restart frame `t*` is
   reached (biased to the restarting team when the tag names it).
7. **Viterbi** (numba kernel with a bit-identical pure-Python fallback), each graded block decoded independently.
8. **spellF1-oriented post-processing**: micro-possessions < 1.2 s between the other team's runs are absorbed
   unless the ball is visible and there is a double touch; micro-`D` and isolated micro-possessions between `D`
   runs are removed; long possessions get a `D` bridge over >= 4 s dead-evidence valleys with touchline /
   set-piece support; forced-`D` intervals from step 4 are applied.  Nothing crosses a graded-block boundary.
9. **Contract**: `X` outside the graded blocks, `D` for 110808..113741, no internal `X` in graded blocks
   (forward-fill), `frame,label` CSV with LF line ends, validated with the official format check.

**Why last toucher.**  On the external matches the team of the player that last touched the ball agrees with the
labelled possession on ~92% of the alive frames, against ~78% for the player instantaneously nearest to the ball
(the baseline's rule); and ~99% of the labelled turnovers happen within 10 frames of a detected touch, with the
ball at a median 0.56 m from the nearest player.  Possession in the ground truth is therefore event-like, and the
decoder is built to switch exactly at touches and to hold the state in between (flight of the ball, occlusions).

**Why a decoder instead of per-frame thresholds.**  The transition costs encode the football prior that
possession changes are rare and happen at touches, which removes the flicker that destroys spellF1 for
per-frame classifiers (the official baselines show it: raw 0.04 vs smoothed 0.33 on the public 5 minutes).

## 2. Assets used and dropped

| Asset | Status | Use |
|---|---|---|
| Tactical ATD (`*_tactical_atd.txt/.xml`) | **used** | positions, speeds, ball; all declared channels are parsed |
| Tactical Smart Tagging XML | **used** | set-piece/goal windows -> restart frames; weak phase prior for H/A |
| External open tracking matches (IDSSE 7, PFF 20 for training; Metrica 3, PFF held-out for validation; SkillCorner tried and dropped) | **used for training and tuning only** | training data of Specialists A/B and all hyper-parameter choices; never BAR-ATH (section 3) |
| Tactical homography | loaded and validity-checked (#7, #37) | consumed only by the optional vision / broadcast stages, which are off |
| Broadcast ATD / homography | **not used** in wave 1 | `preprocessing/broadcast_sync.py` (#10): block-wise sync + ball injection, OFF by default (see Limitations) |
| Videos (tactical, broadcast) | **not used** | YOLO / multi-frame attention crops (#11, #36) are built and off; the video was not processed for the submission |
| Official scorer / `games.py` / starter | **used (vendored, unmodified)** | metric, constants, format check |
| Public 5-minute ground truth | **not used for any decision** | only reserved as a format/alignment smoke test |

## 3. Training data

**Nothing from the provided BAR-ATH match is used as a label or for fitting, and the public 5 minutes are never
used to choose between variants** (`RULES.md`: no hand labels on the provided game).  What is learned, and from
what:

| Model | Trained on | Labels | Amount |
|---|---|---|---|
| Specialist A (ball visible, #20) | **external** open tracking, degraded | algorithmic (below) | 365,664 rows (every 0.4 s) |
| Specialist B (ball absent, #22) | **external** open tracking, degraded | algorithmic (below) | 232,354 rows |
| B adaptation to the target match (#21) | block-masked copy of the competition match itself | algorithmic: A's ultra-confident spans | implemented, **off by default** (see below) |
| everything else (transition costs, dead-ball thresholds, ...) | generic football priors, examined on the external matches | -- | -- |

* **External dataset.**  Seven Bundesliga matches of the DFL / IDSSE open dataset (Bassek et al. 2025; ids
  `J03WMX, J03WN1, J03WOH, J03WOY, J03WPY, J03WQQ, J03WR9`), converted to metres / 25 fps by the sister repository
  `mtah2027-external-benchmarks` (not redistributed here).  Their per-frame labels are produced **algorithmically,
  never by hand**, following that repository's `docs/LABELLING_RULES.md`: ball dead per the provider's
  ball-status channel -> `D`; ball alive -> the team in the provider's possession channel; frames outside the
  periods -> `X`.  This mirrors how the competition derives its own ground truth from tracking channels.  The
  three Metrica sample games (labels derived from events; their ball channel disappears exactly while the ball is
  dead, which would leak `D`) are not used for training.
* **Degradation.**  Each match is degraded with the realistic degrader of Issue #19 (41% of the alive frames lose the
  ball in crowding-weighted 5-60+ s blocks, correlated tracking noise, identity cuts, duplicate and dropped
  tracks) with 3 seeds, so that the models see the conditions of the competition tracking.  The degraded tracking
  goes through exactly the same front end as the competition match (cleaning, orientation so that the home team
  attacks +x, #13, #16, #17, #18).
* **Team symmetry.**  Barcelona and Athletic differ from the Bundesliga teams, so every degraded match is
  also used with the home / away roles swapped (the orientation stage mirrors the pitch so that the new "home"
  attacks +x; labels `H <-> A`): 42 match variants in total.  The models never see a team identity, only roles.
* **Reproducibility.**  `python scripts/train_specialists.py --bench-dir <external cache>` rebuilds
  `models/specialists.json` (1.6 MB, a LightGBM text dump + metadata; **no match data**).  Training is
  deterministic (`deterministic=True`, `num_threads=1`, `seed=0`; retraining gives identical models).
  `generate_submission.py` retrains at run time when the external matches are on disk (`auto`), otherwise loads
  the committed file, otherwise falls back to the heuristic emissions.
* **Calibration.**  A temperature fitted on out-of-fold predictions of whole held-out matches (A: 1.01, B: 0.93;
  out-of-fold log-loss A 0.126, B 0.401) so that the probabilities reflect generalisation to an unseen match.
* **Self-supervision on the target match (#21).**  A's ultra-confident spans (P > 0.98 for >= 2 s) are
  algorithmic pseudo-labels of the competition match; blocks of 5-45 s are masked in them, the whole front end
  (including the ball memory from the start of each mask) is recomputed on the masked copy and B is adapted with
  a few extra boosting rounds.  This is allowed (algorithm-produced labels on the provided game) but, evaluated
  leave-one-match-out on the external matches, it did **not** help (section 5), so it is disabled in
  `configs/default.yaml` (`specialist_fusion.use_pseudo_labels: false`) and kept as a documented option.
* A deterministic **synthetic match** (`scripts/generate_synthetic_fixtures.py`, fixed seed) in the exact file
  layout of the real bundle is used for tests, CI and the end-to-end determinism check.

## 4. Possession rules implemented

* `D` (ball dead) is declared from dead-ball evidence (ball beyond the lines, ball at rest, players slowing down /
  walking, set-piece geometry, alive/dead status channel when declared); a `D` always splits a possession
  (`H D H` is two possessions); `D` is forced in the pre-restart interval of every anchored set piece and between
  a goal and the next kick-off, and in 110808..113741 (second-half start before the kick-off).
* A team is in possession from the touch that wins the ball until the other team's winning touch; a change of
  possession requires a touch by the gaining team (the decoder's transition costs), except inside long ball
  occlusions where evidence from both sides of the gap is combined and the change lands near the middle.
* Micro-possessions shorter than 1.2 s are not reported unless the ball is visible and shows a double touch.
* Labels for frames outside the graded blocks are `X`; inside graded blocks there is never an `X` (the scorer
  forward-fills `X` for spellF1 and counts it as an error for macroF1).
* Frame alignment: video frame = ATD `frame_id - 1`, per-half offsets taken from `games.py`.

## 5. Results (reproducible numbers)

All numbers use the **official metric** (`evaluation/possession_metrics.py`) through `eval_bridge`; graded
frames are those where the ground truth is not `X`.  No number below was measured on BAR-ATH.

**External matches** (`scripts/eval_external.py`, array-level core: features -> emissions -> transitions ->
Viterbi -> spell optimiser; no Smart Tagging, no file loaders).  "Baseline" = the official recipe (nearest player
to the ball + majority vote over +-50 frames, `D` only where nobody has the ball in the window) on the same arrays.

| Setting (mean over matches) | engine spellF1 | engine macroF1 | baseline spellF1 | baseline macroF1 |
|---|---|---|---|---|
| 7 IDSSE matches, complete ball | 0.691 (wave 2: **0.761**) | 0.916 (0.950) | 0.310 | 0.394 |
| 7 IDSSE matches, 41% of the ball hidden in random 5-60 s blocks (seed 0) | **0.487** | 0.814 | 0.218 | 0.453 |
| 7 IDSSE matches, realistic degrader (`--degrade`, Issue #19: crowding-weighted 41% occlusion blocks + tracking noise, identity cuts, duplicate / dropped tracks; seeds 0-2) | 0.435 (wave 2: **0.642**) | 0.773 (0.910) | 0.215 | 0.440 |
| 3 Metrica sample games (their own ball channel is missing exactly while the ball is dead, which leaks `D`) | 0.778 | 0.889 | 0.437 | 0.833 |
| all 10 matches as they are | 0.717 | 0.920 | 0.348 | 0.526 |
| all 10 matches, 41% of the ball additionally hidden | 0.505 | 0.802 | 0.247 | 0.517 |

**Wave 2 -- learned specialists, leave-one-match-out** (`eval_external.py --only-complete-ball [--degrade --seeds 0 1 2]
--specialists lomo`: for each of the 7 IDSSE matches the specialists are trained on the other six, degraded with
3 seeds and home/away-swapped, and the held-out match is processed as an unlabelled target; same arrays and metric
as the rows above; hyper-parameters examined only this way, never on BAR-ATH):

| Setting (mean over 7 matches) | wave-1 heuristic spellF1 / macroF1 | wave-2 specialists spellF1 / macroF1 | official-style baseline |
|---|---|---|---|
| complete ball | 0.691 / 0.916 | **0.761 / 0.950** | 0.310 / 0.394 |
| `--degrade`, seeds 0-2 (21 runs) | 0.435 / 0.773 | **0.642 / 0.910** (0.606 / 0.900 before 5.x) | 0.215 / 0.440 |

Every one of the 21 degraded runs improves (spellF1 +0.11 ... +0.30 per run; the learned route finds fewer, longer
possessions: 352 vs 466 predicted for 511 true on the first match).  Choices examined leave-one-match-out on the
degraded matches (spellF1 / macroF1): tempering of the log-probabilities x1.5 0.594, x1.0 0.600 / 0.900,
**x0.5 0.606 / 0.900** (kept; all numbers of this paragraph predate 5.x, where the route overrides set x0.80 / x0.87), x0.3 0.608; blending 30% of the heuristic emission back in 0.574 / 0.892 (worse, so
0%); relaxing the H<->A transition cost inside long occlusions (`ha_occluded` -6 -> -3 or -9) or at rest
(`ha_closed` -11.5 -> -8) changes the score by < 0.002, so the touch-based gating of Issue #23 is kept as it is
(B's confident team changes inside long gaps already pass the -6 occlusion gate).

**Optuna sweep of the decoder parameters** (`mtah2027-external-benchmarks/scripts/run_optuna_sweep.py`, reports in
its `reports/optuna_*.json`; TPE seed 0, 300 trials per route, the 7 IDSSE matches degraded with seeds 0-1, the
cached leave-one-match-out emissions so each trial only rebuilds transitions + Viterbi + spell optimiser; search
space = `transition_matrix`, `spell_optimizer`, `heuristic_emissions` / specialist tempering-blend).  The honest
estimate is nested (for each held-out match the parameters are chosen on the other six) and is checked again on
an unused degradation seed 2.  Adoption rule fixed before the sweep: nested gain >= +0.02 spellF1 and a gain on the
unused seed.

| Route | defaults | best | nested honest | unused seed 2: defaults -> tuned |
|---|---|---|---|---|
| heuristic | 0.396 | 0.439 | 0.439 (+0.043) | 0.384 -> 0.427 (7/7 matches up) |
| specialists (first sweep, shared features) | 0.605 | 0.628 | 0.624 (+0.0195) | 0.608 -> 0.626 |
| specialists (re-run after the window features, Issue #31) | 0.630 | 0.642 | 0.640 (+0.0102) | 0.633 -> 0.639 |

The heuristic parameters are adopted as `route_overrides.heuristic` in `configs/default.yaml` (merged over the shared
sections only when the heuristic route decodes; `config.route_config`); re-measured through the full array-level
pipeline (`eval_external.py --degrade --seeds 2 --only-complete-ball`) they give 0.4266 / 0.7734, matching the
sweep.  The first specialists sweep was under the rule (+0.0195) and kept the shared defaults; the re-run of Issue #31 is adopted after the real-pipeline check of 5.x.  On the toy synthetic match the tuned
heuristic parameters score lower than the shared defaults (short match 0.348 vs baseline 0.458), another reminder
that the synthetic physics is not football; the synthetic sanity test therefore runs on the shared defaults.

**Pseudo-labelling ablation (Issue #21)** (same protocol, degraded, seeds 0-2; B adapted with the block-masked
pseudo-labels of the held-out match's own degraded tracking, never its ground truth): without 0.606 / 0.900,
with the specified settings (anchors P > 0.98 sustained 2 s, blocks 5-45 s) **0.590 / 0.878**.  A variant that
drops the `D` pseudo-rows scored 0.574 (tempering x1; 0.600 without adaptation), and lighter weights / fewer
adaptation rounds did not beat no adaptation on the folds that were run.  Reason: with a calibrated P > 0.98
threshold the anchors are overwhelmingly `D` spans (about 4,900 of ~5,000 pseudo rows) and few H/A possessions
(~400-1,800 rows), so adaptation mostly shifts B towards `D`.  The module is complete and tested, but
`use_pseudo_labels` is **off** by default; whether it helps on the real, unseen BAR-ATH is not measurable
without labels.

**Occlusion-team learnability** (`scripts/explore_occlusion_team.py`, 7 degraded IDSSE matches x 3 seeds, leave-one-match-out
HistGradientBoosting on the collective-physics (#17) and ball-memory (#18) features only, accuracy of the team on the
alive frames whose ball is hidden): model **0.850**; physics only 0.847; memory only 0.796; constant majority team 0.529; current rule (last/next
touch, change point in the middle) 0.621; engine decoded output 0.582.  The learned signal is mostly team shape
(width of the attacking block, line advance), which is why the specialist B of Issue #22 should be trained on it.

Diagnostic on two IDSSE matches (complete ball): giving the engine the ground-truth team with its own `D` raises
spellF1 to ~0.96 (to ~0.87-0.89 with 41% of the ball hidden), whereas giving it the ground-truth `D` with its
own team changes almost nothing (0.64-0.73 clean, ~0.50 hidden).  Team assignment, not dead-ball detection, limits
the score, and under occlusion it is the part that degrades.

**Synthetic match** (fixed seed 0, ground truth known; the generator hides 41% of the ball during alive play):

| Match | engine spellF1 / macroF1 | possessions (pred/gt) | nearest-player +-50 baseline spellF1 / macroF1 |
|---|---|---|---|
| short, 12,000 frames (heuristic emissions, shared defaults) | 0.545 / 0.716 | 24 / 23 | 0.458 / 0.479 |
| full length, 199,319 frames (heuristic emissions, shared defaults, before `route_overrides`) | 0.564 / 0.762 | 346 / 293 | 0.434 / 0.532 |
| full length, 199,319 frames (specialists from `models/specialists.json`, the CI default) | 0.446 / 0.684 | 383 / 293 | 0.434 / 0.532 |

The synthetic generator is not real football (its dead-ball and team-shape physics differ from the Bundesliga
data the specialists are trained on), so the learned route scores below the heuristic on it; this is a
domain-shift warning, not a quality measure -- its purpose is the determinism / contract check.

**Runtime** of the full-length synthetic end-to-end run (`generate_submission.py --synthetic`, generation of the
match included, 16 cores laptop, numba + LightGBM): 50-56 s with the committed specialists (generation 24 s, model
loading 0.2 s, engine ~30 s including the extra collective-physics / ball-memory features); two runs give the
same SHA256.  Wave-1 heuristic route: 35 s.  Training from scratch (`scripts/train_specialists.py`: 42 degraded
variants + calibration folds + final fit, single-threaded) takes ~10-17 min.  Without numba the convex-hull
fallback alone needs ~58 s at full size, which is why CI installs numba.

The published floor for the real match is `baseline_smooth50` spellF1 0.2433; the real-match score of this
version is not known to the authors at the time of writing (the data is processed only locally).

### 5.x Improvement round after the Optuna sweep (occlusion turnovers, specialists overrides, MBR)

**Motivation.**  On the degraded external matches the learned route under-segments: 2467 predicted vs 3061 true
possessions (seed 0) and 81 % of the H<->A turnovers that happen while the ball is hidden are missed (37 % with the
ball visible).  That is the largest remaining source of lost spellF1.  The adoption rule used for every change in
this round: the mean leave-one-match-out (LOMO) spellF1 over seeds 0-2 must gain at least +0.01 and no seed may be
worse beyond -0.003 (improving on every seed, or on two with the third within -0.003); otherwise the code stays
but is **off** by configuration and the result is reported.  Protocol: the 7 IDSSE matches with a complete ball,
`--degrade`, seeds 0-2, LOMO specialists, official metric; reference = 0.6059 / 0.9003 (seeds 0/1/2: 0.6105 / 0.5994 /
0.6079).  External matches only, never BAR-ATH.

**Item 3 (Issue #30): occlusion turnovers.**  Past / future windows are legitimate because the match is processed
offline as a whole (non-causal smoothing is allowed).  `specialist_features.py` adds, for 21 ball-free signals
(collective physics #17: centroid speed, depth, width, spectral area of both teams; coherence / line advance #18; the
player-speed signals and `dead_score` of #13; home-minus-away differences), the centred mean over +-W frames and the
difference `mean(t..t+W) - mean(t-W..t)` for W = 50 and 100 frames (vectorised cumulative sums, NaN-safe, computed per
graded block so nothing crosses a half or block boundary): 84 extra columns that let Specialist B see the reversal of
the collective movement and short stops without the ball (`specialist_features.context_windows`, empty = off).

| LOMO, 7 IDSSE matches, `--degrade` | seed 0 | seed 1 | seed 2 | mean spellF1 / macroF1 | possessions (seed 0, true 3061) | hidden-ball turnovers missed (seed 0) |
|---|---|---|---|---|---|---|
| reference (develop) | 0.6105 | 0.5994 | 0.6079 | 0.6059 / 0.9003 | 2467 | 81.0 % |
| A1: context windows (**adopted**) | 0.6355 | 0.6240 | 0.6326 | **0.6307 / 0.9089** (+0.0248) | 2458 | 71.1 % |

A1 improves every seed by about +0.025 and removes ten points of missed hidden-ball turnovers, while the number of
predicted possessions hardly changes (the gain comes from better-placed, not more, flips).  `models/specialists.json`
was retrained with the new layout.

A2 (`transition_matrix.spec_*`, learned route only): inside occlusions H<->A is opened around the frames where the
specialists' H-vs-A logit changes sign (Schmitt trigger) towards the team the new sign names, H/A->D around P(D) peaks,
and both at collective-speed accelerations.  It is risky because earlier sweeps relaxing those gates only added
spurious flips, so it is measured and kept **off** (`spec_open: false`):

| variant (opening strength, window) | mean spellF1 / macroF1 | possessions (seed 0) | hidden turnovers missed |
|---|---|---|---|
| on the reference model: -4 / -4 / -5 (defaults) | 0.6061 / 0.9001 | 2484 | 80.3 % |
| -2 / -3 / -3, accel 0.7 | 0.6054 / 0.9002 | 2533 | 78.8 % |
| logit only, -1.4, +-25 frames | 0.6062 / 0.9008 | 2537 | 78.9 % |
| logit only, -2.5, hysteresis 1.0 | 0.6038 / 0.8996 | 2478 | 81.7 % |
| logit only, -0.7, +-40 frames | 0.6047 / 0.9009 | 2573 | 77.9 % |
| on A1: defaults | 0.6305 / 0.9089 | 2478 | 70.5 % |
| on A1: logit only, -1.4, +-25 frames | 0.6292 / 0.9091 | 2515 | 69.3 % |

Every extra flip is as likely wrong as right: the possession count rises towards the truth but spellF1 stays within
+-0.003 (noise ~0.04 across matches, ~0.002 across seeds), so A2 fails the rule and stays off.

**Item 1 (Issue #31): `route_overrides.specialists`.**  The earlier Optuna specialists gain (+0.0195 nested,
+0.018 on the unused seed 2) was consistent but under the pre-set +0.02 rule.  Because the features changed
(Issue #30) the sweep was refreshed first (`run_optuna_sweep.py --route specialists`, 300 trials, search seeds 0-1,
LOMO emissions recomputed with the new model; reports `optuna_specialists*.{json,yaml}` of the benchmarks repo): default
0.6297, best 0.6415 on the search seeds, nested honest 0.6399 (+0.0102), unused seed 2: default 0.6326 -> best 0.6389
(nested-selected 0.6391).  Since seeds 0-1 were used by the search, adoption was re-checked through the real
pipeline (`diagnose_possessions.py`, LOMO specialists, seeds 0-2), which now applies `route_config("specialists")`
(decoder sections and `specialist_fusion`) in `run_core` whenever the learned emissions are used (the heuristic
route is unaffected):

| LOMO, `--degrade` | seed 0 | seed 1 | seed 2 (unused by the search) | mean spellF1 / macroF1 | possessions (seed 0) | hidden turnovers missed (seed 0) |
|---|---|---|---|---|---|---|
| shared defaults (after #30) | 0.6355 | 0.6240 | 0.6326 | 0.6307 / 0.9089 | 2458 | 71.1 % |
| `route_overrides.specialists` (**adopted**) | 0.6475 | 0.6381 | 0.6389 | **0.6415 / 0.9099** (+0.0108) | 2409 | 70.5 % |

The gain is positive on all three seeds (+0.012, +0.014, +0.006), so the overrides pass the rule (mean >= +0.01, no seed
worse) and are committed in `configs/default.yaml` (`route_overrides.specialists`); the biggest changes are a looser
`ha_occluded` (-7.57), `d_exit` / `dead_ramp` ramps and tempering x0.80 / x0.87 with 15 % heuristic blend on visible
frames.  The seed-2 gain (+0.006) is smaller than on the search seeds, which is the expected optimism of the search.

**Item 2 (Issue #32): MBR over variants of the learned route.**  `run_core` can decode several variants of the
**specialists route only** (never mixing the heuristic route, 0.43, with the specialists, 0.64): variant `i`
multiplies the tempered emission log-probabilities by `mbr.scales[i]` and shifts `transition_matrix.ha_occluded` by
`mbr.ha_occluded_shifts[i]`; the decoded variants are merged by the median-boundary consensus of Issue #26
(`consensus_labels`, same-team IoU >= 0.3 matching, support >= 50 %) and the spell optimiser runs once on the result
(`mbr.enabled`, off by default; the heuristic route logs "skipped").  Expected gain is small because variants of one
model make correlated errors, and it was:

| MBR variants (around the adopted optimum) | seed 0 | seed 1 | seed 2 | mean spellF1 / macroF1 |
|---|---|---|---|---|
| no MBR (`route_overrides.specialists`) | 0.6475 | 0.6381 | 0.6389 | 0.6415 / 0.9099 |
| tempering x0.8 / x1 / x1.25 | 0.6475 | 0.6381 | 0.6389 | 0.6415 / 0.9099 |
| `ha_occluded` -1.5 / 0 / +1.5 | 0.6475 | 0.6381 | 0.6389 | 0.6415 / 0.9099 |
| tempering x0.8 / x1 / x1.25 and `ha_occluded` -1.5 / 0 / +1.5 | 0.6474 | 0.6381 | 0.6389 | 0.6415 / 0.9099 |
| five tempering variants x0.7 ... x1.4 | 0.6473 | 0.6380 | 0.6389 | 0.6414 / 0.9098 |
| tempering x0.5 / x1 / x2 | 0.6473 | 0.6378 | 0.6385 | 0.6412 / 0.9098 |
| `ha_occluded` 0 / +2.5 / +5 | 0.6488 | 0.6369 | 0.6377 | 0.6411 / 0.9100 |

The variants differ in at most a handful of boundary frames (the transition penalties dominate the emission scale), so
the median reproduces the central decode and the gain is zero within +-0.0004.  Fails the rule: MBR is **kept in the
code, off by default**.

**Item 4 (Issue #34): duration-aware (semi-Markov) decoder.**  43 % of the true possessions last < 3 s while the
HMM has an implicit flat duration, so an explicit-duration decoder was built and measured.
`decoding/duration_model.py` fits, from the **ground-truth labels of the external training matches only** (interior
runs of H, A and D per graded block, border runs dropped), a histogram per state on a 23-bin grid up to 60 s plus a
geometric tail (add-0.5 smoothing); the table (1523 H, 1510 A, 696 D runs from the 7 IDSSE matches) is stored in
`models/specialists.json` (`durations`) and refitted without the held-out match in every LOMO model.
`decoding/semi_markov.py` maximises, per graded block, entry transition `log A_t(s' -> s)` (touch / restart gates of
#23) + `w` * duration log-pmf + emissions, with the first and last segment censored (survival function) and durations
up to the cap explicit; longer ones use the geometric tail through a running maximum.  A numba kernel and a NumPy
fallback give identical paths (tested, and exact against brute-force enumeration); cost is O(N * cap * 3), about 0.3 s
for a 76 000-frame half with numba (2 s without), far below the 30 s budget.  Switch: `duration.enabled` (off).

| LOMO, 7 IDSSE matches, `--degrade` | seed 0 | seed 1 | seed 2 | mean spellF1 / macroF1 | possessions (seed 0, true 3061) | hidden turnovers missed (seed 0) |
|---|---|---|---|---|---|---|
| reference (HMM, `route_overrides.specialists`) | 0.6475 | 0.6381 | 0.6389 | 0.6415 / 0.9099 | 2409 | 70.5 % |
| HSMM, weight 0.5, centred | 0.6470 | 0.6369 | 0.6387 | 0.6409 / 0.9099 | 2427 | 70.1 % |
| HSMM, weight 1.0, centred | 0.6475 | 0.6368 | 0.6374 | 0.6406 / 0.9099 | 2450 | 69.8 % |
| HSMM, weight 2.0, centred | 0.6468 | 0.6346 | 0.6357 | 0.6390 / 0.9096 | 2503 | 69.1 % |
| HSMM, weight 1.0, not centred | 0.6417 | 0.6333 | 0.6369 | 0.6373 / 0.9094 | 2267 | 71.6 % |
| heuristic route: HMM -> HSMM weight 1.0 | 0.4417 -> 0.4414 | 0.4362 -> 0.4356 | 0.4266 -> 0.4276 | 0.4348 -> 0.4349 | 2225 -> 2264 | 97.1 -> 97.4 % |

The decoder raises the possession count towards the truth (2409 -> 2503) but the extra possessions are as often wrong
as right, so spellF1 moves by -0.0009 ... -0.0042 (heuristic: +0.0001): the transition matrix, tuned for the HMM,
already encodes the rarity of switching and the duration prior carries no evidence about *which* frame flips while the
ball is hidden.  Fails the adoption rule (needs +0.01): code, table and tests are kept, **off**.

**Item 5 (Issue #35): more training matches (SkillCorner, PFF).**  The specialists were trained only on the 7 IDSSE
matches degraded by our own degrader (Issue #19).  Two extra sources of the benchmarks repository were added to the
*training* rows (`specialist_data.extra_sources`; no synthetic degradation on top, both are already occluded):
SkillCorner A-League broadcast tracking (20 matches; frames with `observed == 0` are `X`; all its graded frames have a
visible ball, so it only feeds Specialist A; its `D` is the gap between `phases_of_play`, not the ball status) and PFF
FC World Cup 2022 (64 converted matches, labels from the provider's events; `ball_xy` contains *estimated* positions,
so the ball is NaN where `ball_visible` is False and the occlusions are real: 29-54 % of the frames).  Training uses
the first 20 PFF matches by name (deterministic).

*Revised adoption rule* (decided with the user): the IDSSE protocol relies on our own degrader and could be biased
against PFF, which is computer-vision tracking with real occlusions and probably closer to BAR-ATH.  Therefore
**Validation A** = held-out PFF matches with real occlusions (4-fold cross-validation over the 20 PFF matches by sorted
index modulo 4, models trained on IDSSE + the 15 other PFF matches, never on the evaluated match; reference = the
IDSSE-only specialists, which never saw PFF); **Validation B** = the usual 7 degraded IDSSE leave-one-match-out,
seeds 0-2.  Adopt the mixed model if A gains >= +0.01 spellF1 and B loses at most 0.01; the alternative models are kept
as selectable files.

| training set (specialists) | A: PFF spellF1 / macroF1 | A: possessions (true 7069) | B: seed 0 / 1 / 2 | B: mean spellF1 / macroF1 | B: possessions (seed 0, true 3061) | B: hidden turnovers missed |
|---|---|---|---|---|---|---|
| reference: IDSSE only | 0.4659 / 0.8405 | 7897 | 0.6475 / 0.6381 / 0.6389 | 0.6415 / 0.9099 | 2409 | 70.5 % |
| IDSSE + SkillCorner (all rows) | not run | | 0.6138 / 0.6059 / 0.6115 | 0.6104 / 0.9037 | 2184 | 70.7 % |
| IDSSE + SkillCorner (H/A rows only) | not run | | 0.6197 / 0.6107 / 0.6152 | 0.6152 / 0.9057 | 2189 | 70.6 % |
| IDSSE + PFF (weight 1) | 0.5355 / 0.8658 | 6128 | 0.6336 / 0.6189 / 0.6230 | 0.6252 / 0.9042 | 2251 | 72.1 % |
| IDSSE + PFF (PFF weight 0.5) -- **adopted** | **0.5231 / 0.8626** (+0.057) | 6395 | 0.6374 / 0.6263 / 0.6326 | **0.6321 / 0.9071** (-0.0094) | 2310 | 71.5 % |
| IDSSE (weight 0.5) + PFF | 0.5432 / 0.8673 | 5881 | not run | | | |
| PFF only | 0.5541 / 0.8704 | 5367 | 0.4996 / 0.4839 / 0.4916 | 0.4917 / 0.7918 | 1764 | 87.2 % |

Findings.  (1) SkillCorner hurts on B by -0.026 / -0.031 whether its `D` rows are kept or dropped (its dead frames and
its always-visible ball do not resemble the degraded matches); since both fail the rule on B, their A runs and the
down-weighted run (`skillcorner_weight`) were stopped and the code stays available but off.  (2) Real occlusions help a
lot where they exist: every PFF-containing model gains +0.057 ... +0.088 on A and loses 0.009 ... 0.016 on the synthetic
B, i.e. the two validations disagree on the size and the direction of the trade-off; the more PFF weight the larger A
and the lower B.  (3) PFF only is best on A (0.5541) but collapses on B (-0.15, only 1764 possessions predicted, 87 %
of hidden turnovers missed): PFF labels (events) define possession and dead ball differently from IDSSE's ball status
and from our degrader, so a model that only knows PFF does not transfer.  (4) IDSSE + PFF at weight 0.5 is the only
variant with A >= +0.01 and B >= -0.01, hence **adopted as the default** (`models/specialists.json`, trained by
`scripts/train_specialists.py` with the defaults of `specialist_data`; the run-time training of `generate_submission.py`
uses the same rows).  `reports/external_evidence.json` was regenerated with the adopted default (degraded LOMO 0.6321 / 0.9071, complete ball 0.7448 / 0.9458; heuristic unchanged 0.4348 / 0.6912).  Check of the committed files on the 8 PFF matches never used in any training (last 8 by name,
real occlusions): IDSSE-only 0.4657 / 0.8452, mixed (default) 0.5257 / 0.8648, PFF-only 0.5593 / 0.8737 spellF1 /
macroF1 (heuristic route 0.2954).  The decision is a trade between two proxies; the other two models stay selectable:
`models/specialists_idsse.json` (IDSSE only, the previous default), `models/specialists_mixed.json` (= the default) and
`models/specialists_pff.json` (PFF only), via `generate_submission.py --model-variant {idsse,mixed,pff}` (implies
`--specialists load`).  To choose on the competition input without any label, `scripts/compare_input_stats.py
--data-dir data` prints aggregate ball-gap statistics next to those of the external groups (never per-frame rows):

| group (7 matches each) | ball missing | gaps per 10 graded min | gap p50 / p90 / p99 (s) | hidden time in gaps <5 / 5-20 / 20-60 / >60 s (%) |
|---|---|---|---|---|
| IDSSE raw | 0 % | 0 | - | - |
| IDSSE degraded (seed 0) | 40.2 % | 5.7 | 35.1 / 82.5 / 118.4 | 0.1 / 9.0 / 43.4 / 47.5 |
| PFF (real occlusions) | 54.5 % | 71.7 | 0.9 / 11.2 / 57.4 | 20.5 / 26.2 / 36.9 / 16.4 |

(The degrader's blocks are long because the sampled 5-60 s blocks overlap and merge; PFF has many short gaps.  The
tracked players per frame are 21.8 (degraded IDSSE) and 22.0 (PFF).)  Rule of thumb for Monday: if the competition
ATD has long, few gaps close to the degraded-IDSSE profile, use `idsse`; if it has many short gaps like PFF, use the
default `mixed`.

**Item 6 (Issue #38): degrader calibrated to the real ATD gap statistics, model re-selection.**  Label-free aggregates of
the competition tactical ATD (`scripts/compare_input_stats.py --data-dir data`; tracking only, no labels, no per-frame
output) showed that the Issue #19 degrader had the right amount of hidden ball but the wrong shape.  The degrader was
calibrated (random search + 8-seed pooled check on the 7 IDSSE matches; nothing from BAR-ATH beyond those aggregates,
stored in `external_degrader_target`): log-normal block durations (`gap_model: lognormal`, median 12.5 s, sigma 0.95,
5-240 s) separated by >= 9 s of visible ball (`guard_s`, otherwise adjacent blocks merge into long gaps), `hide_frac`
0.424, more single-player dropouts (2.4 per min, 5-30 s) and a new group-dropout stream (0.5 per min, 3-6 players for
5-20 s).  The Issue #19 profile is the section `external_degrader_legacy` (`gap_model: classes`) and reproduces the old
random stream exactly.

| (7 IDSSE, 8 seeds pooled) | ball missing | gaps / 10 min | gap p50 / p90 / p99 (s) | hidden time <5 / 5-20 / 20-60 / >60 s (%) | players / frame | frames < 20 players |
|---|---|---|---|---|---|---|
| target (BAR-ATH ATD) | 40.7 % | 12.94 | 11.5 / 37.9 / 113.7 | 0.3 / 32.7 / 48.0 / 19.0 | 20.49 | 14.9 % |
| Issue #19 degrader (seed 0) | 40.2 % | 5.73 | 35.1 / 82.5 / 118.4 | 0.1 / 9.0 / 43.4 / 47.5 | 21.82 | 0.8 % |
| calibrated | 40.6 % | 13.38 | 12.2 / 38.6 / 97.3 | 0.0 / 37.2 / 42.9 / 19.9 | 20.77 | 15.4 % |

(The tail of the real gaps, p99, and the 5-20 s vs 20-60 s split are only approximated by a single log-normal.)

Re-selection, same protocol as #35.  **B'** = 7 degraded IDSSE leave-one-match-out with the calibrated degrader, seeds
0-2 (primary criterion now); **A** = PFF held-out, 4 folds (secondary guard: must not drop more than 0.01 vs mixed).
"Train degrader" = degrader of the IDSSE training rows.

| training set | train degrader | B' seed 0 / 1 / 2 | B' mean spellF1 / macroF1 | B' possessions (true 3061) | A spellF1 / macroF1 | A possessions (true 7069) |
|---|---|---|---|---|---|---|
| IDSSE + PFF 0.5 (**current default**) | legacy | 0.6008 / 0.6053 / 0.6069 | **0.6044** / 0.9016 | 2239 | 0.5231 / 0.8626 | 6395 |
| IDSSE only | legacy | 0.6091 / 0.6100 / 0.6086 | 0.6092 / 0.9026 (+0.0048) | 2425 | 0.4659 / 0.8405 | 7897 |
| PFF only | legacy | 0.5100 / 0.4938 / 0.4921 | 0.4986 / 0.8261 (-0.106) | 1858 | 0.5534 / 0.8707 | 5319 |
| IDSSE only | calibrated | 0.6107 / 0.6133 / 0.6136 | 0.6125 / 0.9056 (+0.0081) | 2329 | 0.4790 / 0.8412 (-0.044) | 7377 |
| IDSSE + PFF 0.5 | calibrated | 0.6084 / 0.6071 / 0.6099 | 0.6085 / 0.9035 (+0.0041) | 2219 | 0.5278 / 0.8639 (+0.005) | 6196 |

For reference, the same models scored with the old degrader (#35): mixed B 0.6321, IDSSE-only 0.6415; the calibrated
degrader is harder (-0.03 ... -0.04) because its gaps are less forgiving (shorter visible stretches, many player dropouts).
Decision rule: adopt a variant if it beats the default by >= +0.01 on the B' mean with no seed below -0.003 and A not
worse than -0.01.  The best B' gain is +0.0081 (IDSSE only, calibrated) and it also loses 0.044 on A; the calibrated
mixed model gains +0.004 on B' (every seed better) and +0.005 on A.  **No variant passes: the default
(`models/specialists.json` = IDSSE + PFF 0.5 trained with the legacy degrader) is kept**; `specialists.json` is unchanged.
The calibrated mixed model is stored as `models/specialists_calib.json` (`generate_submission.py --model-variant calib`;
training: `scripts/train_specialists.py --degrader-section external_degrader`).  `specialist_data.degrader_section`
(default `external_degrader_legacy`) keeps the run-time training equal to the committed default.  Tools:
`scripts/eval_pff_heldout.py` (validation A), `--degrader-section` of `eval_external.py` / `train_specialists.py`.
Decoder overrides under the calibrated degrader (`diagnose_possessions.py`, cached LOMO models, seeds 0-2, 21 runs):
with `route_overrides.specialists` 0.5997 / 0.8999, without 0.5954 / 0.8999 (+0.0043 from the tuned overrides), i.e. the
Issue #31 parameters are not broken by the calibration (absolute values differ slightly from the table above because
`diagnose_possessions.py` uses another match/seed offset).

### Real-data ingestion check (Issue #37, aggregates only)

`scripts/real_data_sanity.py` on the BAR-ATH bundle (nothing per-frame is printed).  Ingestion is correct: 199,319
frames parsed (all present), 46 slots, channels `x, y, s` for players and ball (`s` = `speed_kph`; it is not
recognised as the canonical `speed` channel, which is harmless because speeds are re-derived from positions), the
tactical homography is valid on 100 % of the graded frames (99.85 % with error < 5), the broadcast one on 55 %.
The ball is missing on 41.1 % of the graded frames (40.7 % / 41.6 % per half); on average 20.5 players per frame are
on the pitch, 14.2 % of the frames have fewer than 20 and 6.3 % fewer than 16 (episodes: median 2.4 s, 84 % of those
frames in episodes of > 5 s).  Smart Tagging: 638 instances parsed of 638 in the XML, 59 set pieces (26 throw-ins
with Team = N/A, 12 corners, 10 goal kicks, 5 free kicks, 2 + 4 kick-offs), 4 goals; all 14 codes and the six
label groups are recognised.  All stages ran (clean, kinematics, dead-ball signals, ball kinematics, set-piece
anchors, throw-in resolver, collective physics, ball memory, specialists, transitions, Viterbi, optimiser); the
aerial filter, broadcast sync and vision recovery are off by configuration.

`D` over-prediction diagnostic (specialists / heuristic emissions): the predicted `D` share is 24 % / 29 % with the
ball visible and 92 % / 83 % with it hidden, 94 % / 85 % when fewer than 20 players are tracked.  Checked for a bug:
the tracked players in the predicted-`D` frames are slow (median mean speed 0.8-1.2 m/s vs 2.0-2.2 m/s in the
frames labelled alive), so the `D` comes from the speed physics and not from NaN handling.  On external matches
with real occlusions the same pattern exists (PFF: 61 % of hidden-ball frames are truly `D`, 8 % with the ball
visible) and the engine reproduces the true `D` share per cell within a few points (degraded IDSSE and PFF, with
the specialists: 46.96 vs 46.62 % overall).  The heavy player dropouts of BAR-ATH (up to 14 % of the frames) have no
external counterpart (<= 1.2 %), so that cell cannot be validated; no bug was found and nothing was tuned.

### Smart Tagging on the real match (Issue #40, aggregates only)

*Parsing.*  638 / 638 instances, codes and label groups as documented (`Team`: 314 Barcelona, 288 Atletic Club, 36 N/A;
no phase code is unknown).  *Semantics bug fixed.*  `Team` is "the team performing the event", which for
`Defending in Attacking / Middle / Defensive Third` (118 instances) and `Defensive Transition` (24) is the defender, so
the possession of those phases belongs to the opponent.  The old loader credited the tagged team: every attacking phase
has a paired defending phase of the other team over the same frames (the alive-frame counts of `Progression` and
`Defending in Middle Third` are identical on BAR-ATH), so the two cancelled in `phase_prior_home`.  Evidence: the
official definitions plus the pairing structure; the predicted possessor agrees with the *possession side* in 74-97 %
of the alive frames of the defending codes (3-26 % with the raw tag).  Fix: `DEFENDING_CODES` in
`io/smart_tagging_loader.py`; the raw `TagEvent.team` is untouched.  Effect on the labels is small by design (the prior
weighs `w_phase = 0.3`): 0.4 % (specialists) / 1.1 % (heuristic) of the graded frames change, 245 -> 238 possessions.
The weight itself cannot be tuned (no external match has Smart Tagging).

*Anchors.*  59 set pieces, 57 anchored (96.6 %; before the gate below 58): 9 + 3 corners ball/collective, 17 + 9
throw-ins, `t*` inside the tag window for 93 %, 12 % of the graded frames forced to `D`.  All set-piece windows are
35 s long (5 s for the kick-offs).  The two `Kick Off (Start)` anchors fall 10 and 38 frames before the published
kick-off (the second with the collective cue).  Throw-ins with Team = N/A: 20 / 26 resolved (14 taker + controller,
3 + 3 one cue, 6 left unresolved on purpose).  *Anchor accuracy on external matches* (`scripts/eval_anchor_accuracy.py`:
pseudo 35 s windows around the true restarts of IDSSE / PFF / Metrica, restart at a random offset 3-33 s): ball cue
median error -2 frames, 85 % within 3 frames; collective cue (ball hidden) was poor (median +53 frames, 10 % within 12
frames).  Gating the collective jump on dead-ball evidence in the previous second (`collective_dead_lag_s = 1.0`)
reduces the mean |error| from 208 to 126 (IDSSE), 247 to 181 (PFF) and 259 to 143 (Metrica) frames at the cost of more
unresolved anchors (nothing is invented); adopted, since it is better on all three sources.  It cannot be scored in
spellF1 (the external matches carry no Smart Tagging).

### Set-piece anchor fallback and period starts (Issue #50, aggregates only)

*Measurement on BAR-ATH* (`Smart Tagging`, 59 set pieces; before the fix).  Dead-onset method per anchor: dead-score 40,
tag-window 17, none 2 (unresolved).  Frames forced to `D` inside the graded blocks: 10,465 by dead-score evidence and
**6,870 by the tag-window fallback** (4.4 % of the graded frames; 12.12 % forced in total).  Of the 6,870 tag-window frames
58.9 % have the ball visible inside the pitch, **35.3 % have the ball above 4 m/s** and 36.1 % have a collective player speed
above the median of the alive frames (mean `dead_score` 0.34): the 35 s windows open while play is alive, so back-filling
`D` to `tag_start` forced live play to `D`.  The labels of the full pipeline (production specialists) were 50.1 % `D`,
243 possessions.

*Fix.*  (1) Without dead-score evidence only `fallback_dead_s = 0.5 s` right before `t*` is forced (never before the tag
start); 0 forces nothing and leaves it to Viterbi.  The old behaviour is `fallback_tag_window: true`.  (2) A
`Kick Off (Start)` anchor whose search window contains the first frame of a play block (taken from `play_blocks`, that is
from the configuration / `games.py` constants, never hardcoded) is clamped to that frame, and a kick-off the tracking
cannot resolve gets `method = 'period-start'`; `clamp_period_starts: false` disables it.  Post-goal kick-offs are untouched.
The restart opening `D -> team` of the transition matrix now happens at 11185 / 113742 and not inside `HALF2_DEAD`.

*BAR-ATH effect (same specialists, aggregates).*  Frames forced to `D` in the graded blocks 18,904 (12.12 %) -> 12,618
(8.09 %); of those, tag-window/fallback frames 6,870 -> 192 (all adjacent to `t*`, 0 % above 4 m/s); `D` share 50.1 % ->
47.5 %, `H` 31.2 -> 33.1 %, `A` 18.7 -> 19.4 %; possessions 243 -> 265.

*B' (simulated tags; the external matches have no Smart Tagging and #43's simulator is not merged yet, so
`set_piece` windows were generated by a stand-in: 35 s windows that contain a true restart (dead run >= 5 s followed by
play) at a random offset of 3-33 s, subsampled to the BAR-ATH density of 5.7 per 10 min, types/teams from the labels,
throw-ins `Team = N/A`, plus the 5 s `Kick Off (Start)` window; medium = 1 s window jitter, high = 3 s jitter and 25 %
spurious windows).*  7 IDSSE matches, default calibrated degrader, leave-one-match-out specialists (mixed IDSSE + PFF), seeds 0-2,
spellF1 old fallback -> new:

| noise | seed 0 | seed 1 | seed 2 | mean |
|---|---|---|---|---|
| medium | 0.5860 -> 0.5934 (+0.0074) | 0.5825 -> 0.5902 (+0.0077) | 0.5850 -> 0.5912 (+0.0062) | 0.5845 -> 0.5916 (**+0.0071**) |
| high | 0.5814 -> 0.5907 (+0.0093) | 0.5774 -> 0.5888 (+0.0114) | 0.5806 -> 0.5899 (+0.0093) | 0.5798 -> 0.5898 (**+0.0100**) |

Over the 42 match-seed-noise runs: mean +0.0085 spellF1 (+0.0135 macroF1), 40 up, none below -0.003 (worst -0.0010), no seed
worse than +0.006.  PFF held-out (`eval_pff_heldout.py`) is unaffected by construction (no Smart Tagging on external
matches, the anchor stage is skipped).  *Decision:* the mean (+0.0085) is just below the +0.01 adoption bar of the plan, but
the sign is the same in all 6 noise x seed cells and in 40 of 42 runs, and the old behaviour is a bug on the real match
(35 % of its forced frames have a fast ball); adopted as a correctness fix, reversible by config.  The simulated tags
share the structure of the fix (windows that open in live play), so the size of the gain is optimistic.

### Broadcast synchronisation on the real match (Issue #39, aggregates only)

`scripts/real_broadcast_check.py` on `data/broadcast` (deterministic: two runs give the same injected ball).  117
continuous blocks (median 26 s), 74 synchronised (63 %; 39 rejected by the position test, 2 by fingerprint, 2 by ball);
fingerprint correlation median 0.78 (accepted blocks 0.62-0.93), median nearest-same-team residual 0.29 m, median
tactical-vs-TV ball disagreement 0.78 m (p90 1.5 m); the accepted blocks share essentially one offset (median 160.6 s
broadcast-to-tactical, 10-90 % range 160-1077 s) and are in broadcast order 100 % of the time.  They cover 58.8 % of
the graded frames, but the TV ball only fills **7.4 %** of the hidden tactical-ball frames (4,720 of 64,079; 88 runs, median
0.7 s): ball missing 41.1 % -> 38.1 %.  Physical consistency: no injected point off the pitch, in-run speed p90 9 m/s,
0.3 % of steps > 35 m/s; the jump between the last tactical ball and the first injected one has median 1.5 m but p90
13 m and p99 53 m (about 39 % of the borders imply > 35 m/s, mostly 1-frame gaps with ~1.5 m disagreement).  Switching
the injection on changes 2.0 % of the graded labels (243 -> 257 possessions, `D` 50.1 -> 49.7 %).

Value on external matches (`simulate_broadcast_value.py`, simulated TV view, 3 seeds x the 7 IDSSE + 3 Metrica matches,
spellF1 without -> with injection).  Two *non-leave-one-out* setups were also run and are **not** the production route:
`simf` (PFF-only specialists `models/specialists_pff.json`, which have never seen IDSSE / Metrica and are weak there,
0.51): +0.045 to +0.053, 30/30 runs up; `simm` (the production file `models/specialists.json`, mixed IDSSE + PFF, loaded):
+0.001 / +0.004 (seeds 0 / 1), but 7 of the 10 target matches are in its training set, so its baseline (0.68) is
inflated.  The setup that corresponds to production (mixed specialists applied to a match they were *not* trained
on) is leave-one-match-out: specialists trained on all other matches, the target as unlabelled input, 30 runs: **+0.0050**
mean (per seed +0.0030 / +0.0075 / +0.0046; 19 of 30 runs up, worst -0.016), 0.6198 -> 0.6283 on the first four IDSSE
matches and 0.6802 -> 0.6828 on the other six; heuristic route +0.012.  The adoption rule (mean >= +0.01, no seed
worse than -0.003) is **not met** for the specialists route, and the real fill (7.4 %) is half the simulated one (14-15
%), so `broadcast_sync.enabled` stays `false`.  Code, report and simulation remain available.
## Milestone 9 (in progress)

Issues #42 (invariance to player dropout: stratified diagnostics by tracked players, removal of spurious shortcuts to
`D`, dropout training variants) and #43 (alive prior from Smart Tagging phases while the ball is hidden, validated with
a deterministic Smart Tagging simulator on external matches) are being implemented; #44 (bounded decoder sweep under the
calibrated degrader, run in `mtah2027-external-benchmarks` issue #5) and #46 (release) follow.  Common protocol: B' =
7 IDSSE leave-one-match-out with the calibrated degrader (#38), seeds 0-2, with the breakdown by tracked players
(< 20, < 16, >= 20); A = PFF held-out 4-fold plus 8 unseen PFF matches; adopt if B' gains >= +0.01 spellF1 (no seed
worse than -0.003) and A drops by at most 0.01, or if the over-prediction of `D` in the < 20-player stratum is corrected
on all three seeds without losing more than 0.01 on A; otherwise code and tests stay and the change is off.
Results: pending (filled at #46).
## 6. Limitations (honest list)

* **Ball occlusions** (41% of graded frames) remain the largest source of lost spellF1: Specialist B recovers
  most of the team information from the collective physics and the ball memory (0.43 -> 0.64 degraded) but still
  leaves a gap to the complete-ball score (0.74); the learned route also predicts fewer possessions than the truth.
* **Domain shift.**  The specialists are trained on Bundesliga open data degraded by our own model of the ATD gaps
  (blocks hide alive play; the real gap statistics and the real tracking noise of BAR-ATH are known only through
  the task description).  If the real ATD loses the ball differently (e.g. mostly while it is dead) the learned
  `D`/team behaviour while the ball is absent may transfer worse; the synthetic match (different physics) already
  shows a drop for the learned route.  The heuristic route is one flag away (`--specialists off`) and goes through
  the same hash-checked pipeline.
* The block-masking pseudo-labelling (#21) did not help on external matches and is off by default.
* Touch detection depends on the ball track quality; noisy or interpolated ball positions produce missed or
  spurious touches.  Aerial balls: `vision/aerial_filter.py` (#12) derives `is_aerial` from the ball track alone (apparent speed > 35 m/s, reprojection jumps with homographies, straight flights > 18 m without grass deceleration, optional ATD z channel; the IDSSE benchmark has no z, so only the 2-D cues were measured) and suppresses touches there. It is OFF by default (`pipeline.use_aerial_filter`): on the 7 degraded IDSSE matches it lowered spellF1 (heuristic route 0.3968 -> 0.3806, leave-one-match-out specialists 0.6105 -> 0.6045), because the touches it removes at kicks/landings are real boundaries.
* Throw-ins with `Team = N/A` get their team from the tracking (`features/throwin_resolver.py`, #15): the player
  within 1.5 m of a touchline who is most isolated from opponents (and holds the ball) in `[start, t*]`, and the
  first clear controller in `[t*, t*+35]`, vote; below the confidence floor the throw-in stays unresolved and
  the decoder keeps both `D -> H` / `D -> A` open.  The team feeds the transition matrix (`D -> team` bias at
  `t*`) and an optional emission prior (`throwin_resolver.emission_bias`, off: no gain measured).  On the
  synthetic match 55 of 56 gradable throw-ins are right and none is wrong (one abstains); its effect on spellF1
  is nil there (the transition columns were already open), so the resolver is a safety net, not a gain claim.
* Multi-frame ball detector and audit video (Issue #36).  `MultiFrameMotionBallDetector` (`vision/detectors.py`,
  `detector: motion`, OFF) takes the frame triplet (t-1, t, t+1) of each attention crop, compensates the camera pan by
  phase correlation, keeps what moves in both temporal differences, filters blobs by the apparent ball size from the
  homography scale, validates the t-1 / t+1 positions in metres through `H^-1` (max 40 m/s) and refines the centroid to
  sub-pixel.  On synthetic video with a panning camera, static pitch lines, a centre circle and ball-sized static
  "boots" it returns only the moving ball with a mean error of 0.4 px (max 0.6 px) and nothing for a static scene, a
  pure pan or a player-sized mover; a ball slower than about half its own diameter per frame (or standing) is not
  detectable by construction.  It has NOT been run on the real video and no spellF1 value is claimed.
  `vision/overlay_renderer.py` + `scripts/render_audit_video.py` render an audit video (projected pitch lines,
  minimap with possessor and collective-coherence arrows, 25-frame ball trail, HUD with the predicted label and
  P(H/A/D), ball visible/hidden) from the pipeline's own outputs (`run_pipeline(keep_features=True)` exposes the
  emission probabilities without changing the labels).  Rule: the audit video of BAR-ATH is for local pipeline
  debugging only (alignment, format, bugs); it is never used to label frames by hand or to tune parameters, is
  git-ignored (`*.mp4`, `outputs/`) and is never uploaded.
* Video ball recovery (`vision/`, #11) is built and tested on synthetic video only and is OFF by default
  (`vision.enabled: false`); the real tactical video has not been processed.  Simulated value on the 7 degraded
  IDSSE matches (a detector that recovers the true ball on X % of the planned crops, one per 5 frames of each
  gap, then the same interpolating injection; 77 % of the hidden frames end up filled at X = 40 %), mean
  spellF1 (paired, seed 0): with 1 m noise the heuristic route gains +0.021 (X = 20), +0.010 (40), -0.020 (60)
  and the learned route LOSES 0.058 / 0.086 / 0.087 (its training data never had a noisy "recovered" ball);
  with 0.3 m noise at X = 40 the heuristic route gains +0.061 (0.3968 -> 0.4575) and the learned one +0.014
  (0.6105 -> 0.6248).  So it only pays with an accurate detector (about 0.3 m) and is no use at 1 m; the
  AGPL-3.0 `ultralytics` dependency is an optional extra and nothing in the submission path needs it.
* Broadcast sync (`preprocessing/broadcast_sync.py`, #10) is built and tested on synthetic data only and is OFF by
  default (`broadcast_sync.enabled: false`); the real broadcast files have not been processed.  Simulated value
  (`scripts/simulate_broadcast_value.py`, seed 0, the 10 external matches degraded with the #19 degrader; the TV
  view follows the ball with a 35 m window, shots of 3-12 s, cuts, replay-induced clock drift, permuted slots, 0.3 m
  player / 0.5 m ball noise): 38 % of the blocks are accepted, 94.5 % of them within 2 frames of the true offset
  (median error 0 frames; the other 5.5 % are wrong placements), 13.9 % of the hidden graded frames get a ball
  (median error 0.60 m, i.e. the injected noise).  Heuristic-route spellF1 0.4127 -> 0.4224 (+0.010; 7 matches up,
  3 down by up to 0.012), the learned route on the first two matches only (leave-one-match-out training is slow)
  0.5621 -> 0.5592 (-0.003).  A +0.01 is within the spellF1 noise (~0.04), so it stays off until a local run on the
  real files shows a high accepted share and a small position residual.
* Parameters are generic priors validated on other competitions' data; the tracking noise of BAR-ATH may be
  different.  No parameter was fitted to BAR-ATH.
* The KPI dashboard (#27) only flags alarms; its ranges and targets are never used to tune parameters.

## 7. Quality control: the seven KPIs and the route decision

`python scripts/run_kpi_dashboard.py --data-dir data` (or `generate_submission.py --kpi`) prints a table and writes
`outputs/kpi_report.json`; both contain only aggregate numbers (counts, shares, scores, SHA256, timings), so they
can be shared without exposing match data.  The thresholds live in `configs/default.yaml` (`kpi_alarms`,
`kpi_dashboard`).

| KPI | What | Gate |
|---|---|---|
| 1 | Macroscopic sanity alarms: possessions, mean duration, %D of graded, %H of alive | WARN outside the generic ranges (alarms, never targets) |
| 2 | Artificial blindness, no ground truth: on the first 20 graded minutes hide the ball in extra synthetic blocks, rerun, compare with the unmasked run | spellF1 >= 0.75 |
| 3 | Degraded external evaluation (other matches, Issue #19 degrader), heuristic route and, with `--external-lomo`, the leave-one-match-out specialists | spellF1 >= 0.65, median transition latency < 8 frames |
| 4 | Smoke test on the public 5 minutes, only with real data, block-bootstrap interval | printed as "pipeline check only - not for model selection" |
| 5 | 20 transitions (frame, time, from/to) for the visual audit, written to the git-ignored `outputs/audit_transitions.csv` | only the count is printed |
| 6 | Official `validate_submission` and SHA256 of the file | OK / FAIL |
| 7 | Route comparison without ground truth: specialists vs heuristic on the target match (spellF1 of one vs the other, possession counts, frame agreement) | decision rule below |

**Which route to submit (decision procedure).**  The learned route scored lower than the heuristic one on the
synthetic match (domain shift), so on the real match the choice is made without looking at the public ground truth:

1. Run the external benchmark with `--external-lomo`: the specialists are trained leave-one-match-out on the other
   matches and scored on the held-out degraded match, next to the heuristic route (KPI 3).
2. Prefer the specialists only if their mean spellF1 beats the heuristic route by at least
   `kpi_dashboard.route_margin_spellf1` (0.04, the metric's noise level); a smaller difference, or a loss,
   selects the training-free heuristic route.  Without external evidence the rule answers `undetermined`.
3. KPI 7 on the target match (agreement of the two routes, possession counts) is a *diagnostic*: low agreement
   (spellF1 of one vs the other below 0.5) warns of a domain shift and means "trust only the external evidence";
   it never selects a route by itself.  The public 5 minutes (KPI 4) check alignment and format only.
4. The chosen route is submitted with `--specialists auto|load` (learned) or `--specialists off` (heuristic).

Example (`run_kpi_dashboard.py --synthetic`, full-length synthetic match, 108 s; WARN lines are expected there:
the synthetic match has a different physics and more dead time than the generic priors): sanity alarms 383
possessions / 6.7 s / 59.6 % D / 51.6 % H; blindness spellF1 0.695; degraded external (heuristic route, 10 matches)
spellF1 0.413, median latency 5.0 frames; format OK; route agreement spellF1 0.597 (heuristic 346 vs specialists
383 possessions); against the synthetic truth heuristic 0.564 vs specialists 0.446.  `--external-lomo` retrains the
specialists once per held-out match and takes tens of minutes.

## 8. Reproduction

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt && pip install -e . --no-deps       # numba / lightgbm pinned, optional at run time
python scripts/train_specialists.py --bench-dir <external cache>      # optional: regenerates models/specialists.json
python scripts/generate_submission.py --data-dir data --team OpusOperandi
python evaluation/validate_submission.py outputs/OpusOperandi.csv
```

All dependencies are pinned (`requirements.txt`, `pyproject.toml`), seeds are fixed (`seed: 0`),
`PYTHONHASHSEED=0` is enforced, every decoder input is rounded to 1e-6 and ties are broken by state index, so
two runs (and the numba and fallback decoders) give the same SHA256.  CI repeats the run on the synthetic
full-length match twice and compares the hashes.  Full repository layout and commands: `README.md`.
