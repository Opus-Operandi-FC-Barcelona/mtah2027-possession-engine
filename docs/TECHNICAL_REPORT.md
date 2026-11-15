# Technical report -- Opus Operandi (More than a Hack 2027, phase 1: ball possession)

Repository: `Opus-Operandi-FC-Barcelona/mtah2027-possession-engine`.  This report follows the points requested
in the competition's `SUBMISSION.md`; the short submission document is `docs/Opus Operandi.md`.  **Implemented in this
version = Milestones 1-9 (issues #1-#52)**: a complete, deterministic end-to-end route whose emissions come from
learned Specialists A/B trained on external open data (IDSSE + PFF), with the heuristic emissions as the fallback
route, plus optional stages (broadcast sync, vision recovery, aerial filter, MBR, semi-Markov) that were built and
measured and are **off** by configuration.  **Milestone 9** (issues #42-#52) is complete: it adopted the model
re-trained with gaps over dead ball (#48), the set-piece anchor fallback (#50), play-block isolation (#51) and
reproducible model loading (#52); #42, #43, #44 and #49 were measured and are off (see "Milestone 9" before section 6).  Sections 1-4 below were written for waves 1-2 and the later
items are documented in 5.x; where they differ the later sections and `docs/Opus Operandi.md` prevail.
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
(`models/specialists.json` = IDSSE + PFF 0.5 trained with the legacy degrader) was kept at the time**; `specialists.json` was unchanged (superseded by Item 7 / Issue #48).
The calibrated mixed model is stored as `models/specialists_calib.json` (`generate_submission.py --model-variant calib`;
training: `scripts/train_specialists.py --degrader-section external_degrader`).  `specialist_data.degrader_section`
(default `external_degrader_legacy`) keeps the run-time training equal to the committed default.  Tools:
`scripts/eval_pff_heldout.py` (validation A), `--degrader-section` of `eval_external.py` / `train_specialists.py`.
Decoder overrides under the calibrated degrader (`diagnose_possessions.py`, cached LOMO models, seeds 0-2, 21 runs):
with `route_overrides.specialists` 0.5997 / 0.8999, without 0.5954 / 0.8999 (+0.0043 from the tuned overrides), i.e. the
Issue #31 parameters are not broken by the calibration (absolute values differ slightly from the table above because
`diagnose_possessions.py` uses another match/seed offset).

**Item 7 (Issue #48): hidden-ball <-> dead-ball affinity of the degrader, label-free estimate for BAR-ATH, model re-selection.**
The calibrated degrader (#38) placed ball gaps blind to the play state (odds ratio OR = odds(D | hidden) /
odds(D | visible) of about 1), whereas real trackers hide the ball much more often while it is dead (PFF: 61-64 % of
the hidden frames are `D`, 8 % of the visible ones, OR ~ 19).  A model fitted on blind gaps learns "hidden tells
nothing about D"; one fitted on coupled gaps learns the opposite.  Which one fits BAR-ATH had never been measured.

*Degrader.*  `DegraderParams.dead_affinity` (default 1.0 = old behaviour, same random stream, config key
`external_degrader.dead_affinity`).  Above 1 the start weight of a block is multiplied by
`dead_affinity ** (share of D frames inside the block)` and a block that mostly overlaps a dead-ball stretch is
aligned with it (ball leaves play, not seen until the restart; it lasts at least as long as the stretch); the hidden
target is counted on all graded frames (x 0.972) so the missing share stays at 41 % (pooled realised OR with
`artefacts=False`, 7 IDSSE matches, seed 0: affinity 1 -> 0.9, 1.45 -> 3.0, 2 -> 3.2, 5 -> 5.8, 8 -> 7.7, 12 -> 10.4,
20 -> 15.0, 35 -> 18.8, 60 -> 24.1, 120 -> 30.3).  `DegradeReport` now holds `hidden_dead_share`,
`visible_dead_share` and `realised_odds_ratio`.  Limitation, measured: aligning gaps with dead stretches lengthens them
(8.1 gaps / 10 min and p50 / p90 / p99 = 23.6 / 58.8 / 140.6 s at affinity 33 vs the target 12.9 and 11.5 / 37.9 / 113.7;
affinity 1: 14.4 and 11.9 / 32.4 / 73.7), so the coupling and the gap-length shape cannot both be matched with this
mechanism; missing share (41.2 %) and players per frame (20.8, 14.9 % of the frames < 20) stay on target.

*Label-free estimator* (`mtah2027.validation.gap_affinity`, `scripts/estimate_gap_affinity.py`).  Ball-free features of
the players only (tracked players, mean / p80 speed, shares of slow / fast players, spread, 10 s means), a gradient
boosting classifier P(D | features) trained on other matches (IDSSE degraded by the calibrated degrader, so its
dropout is the BAR-ATH-like one), prior-shift EM (Saerens et al. 2002) on the frames of the gaps >= 1 s and, separately,
on the visible frames, **per player-count stratum (< 20 / >= 20 tracked players, each with its own reference prior)**.
The class-conditional histograms are used in discriminative form (the classifier).  No BAR-ATH labels, no public GT.
Pre-stated tolerance: estimated D share in the hidden frames within 10 points of the truth for every case (pooled
5 points); OR within a factor 2 when the visible D share is >= 15 %.  Validation (leave-one-match-out for IDSSE):

| validation case | est. D in hidden | true | est. D in visible | true | est. OR | true OR | worst match (hidden D) |
|---|---|---|---|---|---|---|---|
| IDSSE planted, affinity 1 (7 matches) | 40.3 % | 40.2 % | 47.4 % | 47.1 % | 0.74 | 0.75 | 10.0 pts |
| IDSSE planted, affinity 4.4 | 69.6 % | 68.0 % | 26.9 % | 28.1 % | 6.67 | 5.54 | 9.3 pts |
| IDSSE planted, affinity 33 | 79.2 % | 77.0 % | 19.6 % | 21.7 % | 17.4 | 12.4 | 7.8 pts |
| PFF (10 matches, real occlusions, event labels) | 65.1 % | 64.3 % | 5.6 % | 8.6 % | 31.7 | 19.2 | - |
| Metrica Games 1-3 (ball dropped while dead) | 100 % | 94.0 % | 6.0 % | 2.4 % | saturated | 651 | - |

All five pass the tolerance on the hidden D share (pooled error <= 2.2 points on IDSSE, 0.8 on PFF, 6.0 on Metrica
where it saturates); the OR is right within x1.2-1.4 on IDSSE but over-estimated x1.65 on PFF because the visible D share
is small (5.6 vs 8.6 %) and is meaningless when a share saturates.  So the **D share in the hidden frames is the
reliable quantity; the OR is only indicative.**  The estimator overshoots slightly at the top of the range.

*BAR-ATH estimate* (tactical ATD, graded blocks without the known-dead start of half 2; aggregates only; 64,028
hidden-gap frames, 91,844 visible frames): D share in the hidden gaps **94.3 %** (stratum >= 20 players 91.8 %, n =
44,612; stratum < 20 players 100 %, n = 19,416), D share in the visible frames 17.1 % (15.3 % for >= 20, 75.6 % for
< 20, n = 2,751), OR 80 (62 in the >= 20 stratum).  Gap borders (131 / 129 gaps): ball within 1.5 m of a line at the
last-seen frame in 29.8 % (first-seen 27.9 %), slow (< 1 m/s) in 29.8 % (7.0 %).  Caveats: (i) the value lies above the
planted range validated here (<= 80 %), where the estimator tends to overshoot, so the defensible reading is "at least
~80 %", i.e. at least the strongest coupling the degrader can produce; (ii) 91 % of the frames with < 20 players are
inside gaps, i.e. joint tracking outages, which no external match has, so the < 20 stratum (100 %) is not used;
(iii) a model trained at affinity 33 therefore errs on the safe side of what the estimator found.  The estimator passes
its stated tolerance on all validation cases, so the estimate is used; the training degrader is set to affinity 33
(realised OR ~ 12-19 depending on the artefacts; hidden D 77-80 %).

*Curve* (B' = 7 IDSSE leave-one-match-out, learned route only, seeds 0-2, held-out matches degraded with the given
affinity = how coupled the evaluation world is; these runs predate #50 / #51 / #43).  Mean spellF1 (true hidden-frame
D share in brackets):

| held-out affinity (true D % in hidden) | 1 (41 %) | 1.45 (59 %) | 7 (70-71 %) | 33 (79-80 %) | 120 (82-84 %) |
|---|---|---|---|---|---|
| IDSSE only, legacy (blind) training | 0.6092 | 0.6259 | 0.6389 | 0.6491 | 0.6540 |
| IDSSE + PFF 0.5, legacy training (previous default) | 0.6075 | - | 0.6442 | 0.6556 | 0.6633 |
| IDSSE + PFF 0.5, trained at affinity 33 | 0.6000 | - | - | 0.6737 | - |
| IDSSE only, trained at affinity 33 | 0.5998 | - | - | 0.6759 | - |

The curve is **not flat** (+0.04 from affinity 1 to 33 for the same model), but that rise is mostly the evaluation
world getting easier (gaps aligned with stoppages are easy to decode); what matters for the choice is the difference
between models in the same world.  In the coupled world (affinity 33) training with the matching degrader gains
+0.018 / +0.027 (mixed / IDSSE-only) over training blind; in the blind world it loses 0.008 / 0.009.  Models trained at
affinity 7 or 120 were not run (compute); the blind calibrated degrader (#38, affinity 1) is Item 6.

*Specialist A / B ablation* (trained at affinity 33; PFF weight per specialist; B' at affinity 33 / 1; pre-#51):

| PFF weight A : B | B'(33) spellF1 | macroF1 | possessions (true 3061) | D % in hidden, predicted (true 80.0) | B'(1) spellF1 |
|---|---|---|---|---|---|
| 0 : 0 (IDSSE only) | 0.6759 | 0.9241 | 2533 | 80.4 | 0.5998 |
| 0.5 : 0.5 | 0.6737 | 0.9218 | 2416 | 80.9 | 0.6000 |
| 0.5 : 0 | 0.6733 | 0.9234 | 2449 | 80.4 | 0.5979 |
| 0 : 0.5 | 0.6763 | 0.9225 | 2499 | 80.9 | 0.6015 |
| 0.25 : 0 | 0.6719 | 0.9232 | 2469 | 80.4 | 0.5977 |
| 0.15 : 0 | 0.6742 | 0.9234 | 2504 | 80.4 | 0.6004 |
| 0.5 : 0.25 | 0.6752 | 0.9229 | 2417 | 80.7 | 0.6022 |
| 0.5 : 0.15 | 0.6746 | 0.9231 | 2428 | 80.6 | 0.6036 |
| 0.25 : 0.25 | 0.6739 | 0.9227 | 2438 | 80.7 | 0.6018 |

All nine are within 0.005 (seed noise 0.002-0.004).  The hypothesis "PFF helps A but poisons P(D) of B" is
**not supported** once the training degrader is coupled: B's hidden-frame D share is already on target (80.4-81.0 vs
80.0 %), so no D temperature / bias on the hidden frames was needed or tested.  `specialist_data.pff_weight_b` and
the `--source-weight-b` flags of `eval_external.py`, `eval_pff_heldout.py`, `train_specialists.py` and
`gap_curve.py` keep the option (training rows of B can take another PFF weight than those of A).

*Re-selection on top of develop (#50, #51, #43 included; training rows and LOMO rebuilt)*; B' at affinity 33 is the
primary criterion (estimated world), B' at affinity 1 the blind-world check, A = PFF held-out guard (reject if A drops by
more than 0.03 vs the default):

| model (training degrader) | B'(33) spellF1 / macroF1 | seeds 0 / 1 / 2 | vs default | possessions | D % hidden, predicted (true 80.0) | B'(1) | A spellF1 / macroF1 |
|---|---|---|---|---|---|---|---|
| default: IDSSE + PFF 0.5, legacy (**previous**) | 0.6571 / 0.9139 | 0.6539 / 0.6607 / 0.6567 | - | 2473 | 77.5 | 0.6084 | 0.5212 / 0.8624 |
| IDSSE only, affinity 33 | 0.6761 / 0.9240 | 0.6750 / 0.6795 / 0.6737 | +0.0190 | 2526 | 80.4 | 0.6005 | 0.4692 / 0.8400 (-0.052): **rejected by the guard** |
| IDSSE + PFF 0.5, affinity 33 (**adopted**) | 0.6745 / 0.9221 | 0.6721 / 0.6815 / 0.6700 | **+0.0174** (+0.0182 / +0.0208 / +0.0133) | 2410 | 81.0 | 0.6004 | 0.5263 / 0.8636 (+0.005) |
| PFF 0.5 (A) / 0.15 (B), affinity 33 | 0.6743 / 0.9232 | 0.6704 / 0.6829 / 0.6696 | +0.0172 | 2416 | 80.6 | 0.6022 | 0.5174 / 0.8598 (-0.004) |

Decision: **adopt IDSSE + PFF 0.5 trained with the calibrated degrader at `dead_affinity` 33** (+0.0174 on B'(33),
every seed above -0.003, A guard passed with +0.005).  `models/specialists.json` is regenerated
(`configs/default.yaml`: `specialist_data.degrader_section: external_degrader`, `dead_affinity: 33`), the same file is
`models/specialists_gap.json` (`--model-variant gap`), and the previous default stays selectable as
`--model-variant mixed` (`models/specialists_mixed.json`, legacy degrader).  Cost accepted knowingly: -0.008 spellF1
in the blind world (B'(1) 0.6084 -> 0.6004).  The gain is evidence about the *estimated* world, not about BAR-ATH
labels, which we do not have.
**Item 8 (Issue #44): bounded re-optimisation of the decoder and ensembles under the calibrated degrader (not adopted; code, model variant and block kept OFF).**
Sweep in `mtah2027-external-benchmarks` (issue #5, `reports/issue44/`), harness `scripts/decoder_sweep.py` (front end, model
probabilities and heuristic emissions of a held-out cell are computed once; each candidate re-runs fusion priors / scaling,
transitions, Viterbi and the spell optimiser; checked equal to an uncached `run_core`) with LOMO / 4-fold models from
`scripts/train_cv_models.py`.  Protocol: selection on B'(33) (7 IDSSE LOMO, `dead_affinity` 33) seeds 0-1, seed 2 blind;
guards B'(1) (reject > 0.01 drop) and A (20 PFF 4-fold + 8 unseen PFF, reject > 0.03 drop); baseline = current default
(B'(33) 0.6693, B'(1) 0.6017, A 0.5263 / 0.5263).  Swept: `route_overrides.specialists` (`specialist_fusion` scales and blends;
`transition_matrix` ha_closed / ha_open / ha_occluded / dead_open / dead_closed / d_exit / dead_ramp / alive_ramp / touch;
`spell_optimizer` micro_poss / min_dead / min_isolated / bridge) in four stages (60 one-at-a-time, 234 random, 81 grid, 320 joint random) plus
47 ensemble candidates.  Ensemble: `--model-variant ensemble` loads the manifest `models/specialists_ensemble.json`
(committed model files only, linear or log-linear mix of the per-frame class probabilities before the soft priors, then the
usual floor / scaling / `1e-6` quantisation; nothing is trained at submission time; synthetic SHA identical across runs).

| candidate | B'(33) mean (seeds 0 / 1 / 2 blind) | B'(1) | A 4-fold (20) | A unseen (8) |
|---|---|---|---|---|
| default | 0.6693 (0.6709 / 0.6675 / 0.6695) | 0.6017 | 0.5263 | 0.5263 |
| decoder block D1 (gap) | 0.6759 (+0.0066; +0.0071 / +0.0095 / +0.0033) | +0.0029 | +0.0059 | +0.0063 |
| decoder block `gap_r11` (gap) | 0.6757 (+0.0064) | +0.0024 | +0.0070 | +0.0083 |
| **lin 0.75 gap + 0.25 mixed, block `lin75_r06`** | **0.6784 (+0.0091; +0.0113 / +0.0068 / +0.0091)** | +0.0068 | +0.0097 | +0.0087 |
| lin 0.75 gap + 0.25 mixed + D1 | 0.6764 (+0.0071) | +0.0071 | +0.0071 | +0.0075 |
| lin 0.5 gap + 0.5 mixed + D1 | 0.6746 (+0.0052) | +0.0078 | +0.0062 | +0.0086 |
| lin 0.75 gap + 0.25 calib + D1 | 0.6748 (+0.0054) | +0.0061 | +0.0088 | +0.0046 |
| `mixed` alone | 0.6512 (-0.0181) | +0.0030 | -0.0032 | -0.0006 |
| `calib` alone | 0.6510 (-0.0184) | +0.0056 | +0.0037 | +0.0040 |

Findings.  (i) The decoder is on a plateau: single parameters move B'(33) by < 0.002; the gain comes from a joint move (lower
`scale_visible`, much more expensive H<->A switches `ha_closed`, shorter `micro_poss_s`, `blend_hidden` 0) and saturates at +0.008.
(ii) The ensemble is worth about +0.001 to +0.002 on B'(33) on top of the decoder block: `mixed` / `calib` alone lose 0.018 on B'(33) (trained
for another world), so mixing them in only helps in the blind world B'(1) and slightly on PFF.  Best on B'(33) is a 0.75 / 0.25 linear mix.
(iii) Best candidate +0.0091 (selection seeds +0.0090, blind +0.0091, no seed negative, A +0.0097 / +0.0087, B'(1) +0.0068, all guards
pass) but **below the +0.01 bar**, and the gain is of the size of the selection effect (max over ~700 candidates), so nothing is applied:
the block is stored as `decoder_candidates.issue_44` in `configs/default.yaml` (copy into `route_overrides.specialists` to use it).
(iv) Strata (B'(33), pooled): predicted D equals the true D in every stratum (< 16 players 50.1 / 48.1 %, < 20 46.1 / 45.1 %, >= 20 44.7 / 44.4 %),
so the second adoption path (correcting D over-prediction under dropout) does not apply.  Spell-utility MBR over the ensemble members was not run
(the existing MBR of Issue #32 stays off).

### Invariance to player dropout (Issue #42; B' by stratum, not adopted)

*Context.*  On the competition ATD (aggregates) 14.2-14.9 % of the graded frames have fewer than 20 tracked players
(6.3 % fewer than 16, episodes of median 2.4 s) and the model predicts `D` on 94 % of them, while external matches had
at most 1.2 % such frames, so the behaviour was unvalidated.  The calibrated degrader (#38) reproduces the dropout
(15.8 % of the graded frames below 20 players, 1.6 % below 16), so the strata can be measured.

*Tools.*  `validation/dropout_strata.py` (additive counters: macroF1, true / predicted `D`, `D` with the ball hidden /
visible, H/A accuracy on alive frames, missed `H<->A` turnovers) wired into `eval_external.py --strata`,
`diagnose_possessions.py` (always) and `eval_pff_heldout.py` (`--unseen-model` scores a committed model on the last 8
PFF matches).  `eval_external.py --lomo-cache DIR` caches the leave-one-match-out models.

*Reference B'* (7 IDSSE LOMO, calibrated degrader, seeds 0-2, current default mixed model, pooled over 21 runs):

| stratum | frames | macroF1 | true D % | predicted D % | pred. D %, ball hidden / visible | H/A acc. alive | turnovers missed |
|---|---|---|---|---|---|---|---|
| < 16 players | 1.6 % | 0.8688 | 41.7 | 44.4 | 43.2 / 45.1 | 81.2 % | 63.6 % (of 129) |
| < 20 players | 15.8 % | 0.8858 | 43.6 | 45.7 | 45.1 / 46.1 | 83.6 % | 61.0 % (of 1145) |
| >= 20 players | 84.2 % | 0.9077 | 44.7 | 45.1 | 42.7 / 46.8 | 87.6 % | 56.1 % (of 5908) |

**Finding: on external matches degraded like BAR-ATH the model does *not* over-predict `D` under dropout** (+2.1 points
in the < 20 stratum, +2.7 in < 16, vs +0.4 with all players; macroF1 is 0.02-0.04 lower because H/A accuracy falls
81-84 % vs 88 %, not because of `D`).  The 94 % `D` seen on BAR-ATH is therefore not reproduced by dropout alone: either the
real dropouts are different (slow players, stoppages: they walk at 0.8-1.2 m/s) or it is real dead ball.  Feature
importance (gain, committed default): the count columns carry nothing (`n_field_*`, `diff:n_field`, `n_active`:
A 0.1 %, B 0.03 %), so neutralising them (variant (a)) is a no-op and was stopped without a full run; the shape columns
(width, hull, spectral area, their `diff:` and `ctx:` versions) carry 8 % (A) and 23 % (B).  These do depend on the
sample size: the mean fraction of the complete outfield shape kept by n random players of 10 (clean IDSSE frames) is
hull 0.28 / 0.40 / 0.54 / 0.67 / 0.79 / 0.90 for n = 4 ... 9, width 0.69 ... 0.97, so a team with 6 players looks 46 %
less spread.

*Variants* (`specialist_features`: `normalise_shape` divides area / extent by that table and re-derives the area
derivatives; `shape_min_field` blanks a team's shape below that many players; training rows rebuilt for each variant,
same IDSSE + PFF 0.5 mix, legacy training degrader unless stated):

| variant | B' seed 0 / 1 / 2 | B' mean spellF1 / macroF1 | vs default | < 20 stratum: predicted D % / macroF1 / turnovers missed | < 16 stratum: predicted D % / macroF1 |
|---|---|---|---|---|---|
| default (reference) | 0.6008 / 0.6053 / 0.6069 | 0.6044 / 0.9016 | | 45.7 / 0.8858 / 61.0 % | 44.4 / 0.8688 |
| (b) `normalise_shape` | 0.6038 / 0.6067 / 0.6085 | 0.6063 / 0.9019 | +0.0019 | 45.5 / 0.8861 / 60.5 % | 44.0 / 0.8700 |
| (b) `normalise_shape` + `shape_min_field: 6` | 0.6080 / 0.6057 / 0.6074 | 0.6070 / 0.9021 | +0.0026 | 45.6 / 0.8852 / 61.1 % | 45.1 / 0.8540 |
| (d) trained with the calibrated degrader (= `calib`) | 0.6084 / 0.6071 / 0.6099 | 0.6085 / 0.9035 | +0.0041 | 44.6 / 0.8911 / 61.2 % | 42.7 / 0.8823 |
| (a) `neutralise_counts` | not run (columns carry < 0.1 % gain) | | | | |

*Native speed channel (c).*  The competition XML declares `s` with unit `speed_kph`; the loader did not recognise `s`.
`io/atd_loader.py` now maps channel `s` (or `spd`, ... or any non-coordinate channel with a km/h unit) to `speed`, and
verifies the unit against the positions (`_check_speed_unit`, players and ball separately, only applied when the median
ratio confirms a 3.6 factor).  `scripts/check_speed_unit.py` on BAR-ATH (aggregate): players, median native / derived
ratio 0.993 after the correction (0.276 with the declared km/h division), i.e. **the player `s` values are already m/s
(DATA.md is right, the XML unit label is wrong)** and the loader applies factor 3.6 to undo the division; the ball
channel does not agree with its positions under either unit (median(s)/median(derived) 0.056, 8,717 usable pairs) so no ball
correction is applied.  `fill_speed_from_native` (in `compute_kinematics_and_orientation`) uses the native speed where the
Savitzky-Golay speed is missing or within 3 frames of a track edge, behind `pipeline.use_native_speed` (OFF): the external
matches have no speed channel, so it cannot be evaluated on B'.  The ATD cache version was bumped (5).

*Decision (rule: B' >= +0.01 mean with no seed below -0.003, or fix of the < 20 `D` over-prediction winning on 3 seeds with
A within the guard of -0.03).*  Best gain +0.0041 (calibrated training, all three seeds better); the robust-shape
variants give +0.0019 / +0.0026; there is no `D` over-prediction in the < 20 stratum to fix.  **Nothing is adopted**:
`models/specialists.json` and `reports/external_evidence.json` are unchanged, the new options are OFF by config
(`specialist_features.normalise_shape / shape_min_field / neutralise_counts`, `pipeline.use_native_speed`).  A
(PFF held-out, 4 folds, default model): spellF1 0.5231 / macroF1 0.8626, 8 never-seen PFF matches 0.5257 / 0.8648; PFF
has no frames below 20 players (all in the >= 20 stratum: predicted D 41.9 % vs true 41.1 %), and A was not rerun for the
non-adopted variants.  The combination of (b) with calibrated training (one more 3.5 h LOMO run) was not run: the
two gains are +0.004 and +0.002-0.003 and the criterion is +0.010.  Reading for the real match: the 94 % `D` in the
< 20 stratum cannot be attributed to the sample-size sensitivity of the features; only an external dataset with long real
dropouts would validate or refute it.

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
## Milestone 9

Protocol (fixed after the official audit, #47): B' = 7 IDSSE leave-one-match-out with the calibrated degrader (#38),
seeds 0-2, with the breakdown by tracked players (< 20, < 16, >= 20); A = PFF held-out 4-fold plus 8 unseen PFF
matches as a guard.  Adopt if B' gains >= +0.01 spellF1 with no seed worse than -0.003 and A drops by at most 0.03, or if
the over-prediction of `D` in the < 20-player stratum is corrected on all three seeds; correctness fixes (#50, #51, #52)
are adopted when they do not lose on B'.  From #48 on, B' is evaluated in the estimated world B'(33) (gaps over dead
ball) with B'(1) as a second guard.  Outcome: adopted #48 (+0.017 on B'(33)), #50 (+0.0085), #51 (+0.0004), #52
(reproducibility); off #42 (+0.004), #43 (+0.002), #44 (+0.009), #49 (+0.0003).  Each item below has the full tables.

**Release (Issue #46).**  Final submission generated with `--specialists load` (default model = `specialists_gap.json`)
twice with identical SHA256 `cac8927c2adc1b1a6778d0045a2fae0ba4437aee8ea10a889de02db7552d0c7b`, valid for
`validate_submission.py`; label-free aggregates on the provided match: 266 possessions, mean 12.3 s, `D` 47.5 % of the
graded play frames (Milestone 8 submission: 243, 12.8 s, 50.1 %).

### Smart Tagging alive-ball prior and transition cue (Issue #43; B' / A on simulated tags, not adopted)

**Idea.**  With the ball hidden the specialists predict `D` for 92 % of the hidden graded frames of BAR-ATH (Issue #37), while
the hidden-ball frames are mostly live play (`FAQ.md` / `DATA.md`).  A frame covered by a live Smart Tagging phase
(any phase code; `SmartTagging.phase_cover`) and outside the pre-`t*` windows of the set pieces, the unresolved set-piece
windows, the forced-dead intervals and the post-goal windows (`pipeline.live_phase_mask`) is weak evidence of an alive
ball.  `heuristic_emissions.w_phase_alive` (default 0 = off) subtracts that many logits from the `D` logit on hidden-ball
frames of such a mask (`apply_priors` for the specialists, `compute_emissions` for the heuristic route).  Optional
extras, all off: `transition_matrix.phase_cue` opens `H<->A` (`phase_cue_open`, -4) from 4 s before to 8 s after the start
of an `Attacking` / `Defensive Transition` phase, towards the team that gains the ball and only inside occlusions; and
`pipeline.alive_exclude_dead_evidence` additionally drops from the mask the frames with physical dead-ball evidence
(ball out, `dead_score >= 0.6`) and the 3 s after them (the phases lag when the ball leaves the pitch).

**BAR-ATH structure (aggregates, `scripts/smart_tagging_stats.py`).**  575 phase instances in 12 codes (`Players in the
Box` 97, `Ball in Final Third` 93, `Ball in the Box` 72, `Progression` 66 = `Defending in Middle Third` 66, `Creating
Chances` 30 = `Defending in Defensive Third` 30 with a fixed 30 s duration, `Build Up` 22 = `Defending in Attacking Third` 22,
`Long Ball` 29, the two Transition codes 24 each), 55 per 10 graded minutes, durations p10-p90 9-62 s, every covered frame
lies under 3.1 phases on average, **union coverage 69.4 % of the graded frames** (67.3 % with a non-zero H/A prior), uncovered
gaps median 13 s (p10-p90 3-56 s), 59 set-piece windows of 29-35 s covering 28 % of the graded frames.  Phases are not
possessions: they cover far more than the alive share (about 50-60 %), so a good part of the covered frames is dead.

**Simulator** (`io/smart_tagging_sim.simulate_smart_tagging`, deterministic per seed): possession runs of the true labels
become paired attacking / defending phases (defender tagged, as in the XML), extra unpaired phases, transitions at 10 % of
the possession changes, `Kick Off (Start)` windows, and only 5.7 set pieces per 10 graded minutes (the BAR-ATH rate:
most restarts are *untagged*; throw-ins are `Team = N/A`, plus 2 % N/A phases); then noise: Gaussian boundary jitter, missed
tags (FN), spurious tags (FP, random team and place) and a *dead-time bleed* (phase ends lengthened until the union
coverage reaches 69 %, which also reproduces the lag after the ball goes out).  Noise levels low / medium / high =
jitter 1 / 3 / 6 s, FN 10 / 25 / 40 %, FP 5 / 15 / 30 %.  At medium noise 92-93 % of the alive and 32-37 % of the dead
graded frames are covered.  **The clean tagging is derived from the truth, so every gain below is optimistic.**  SkillCorner
phases were not available in the benchmark cache (no SkillCorner match); the PFF and IDSSE tags are simulated.

**Protocol.**  B' = 7 IDSSE leave-one-match-out with the calibrated degrader, seeds 0-2 (21 runs per cell; specialists of
the production recipe, IDSSE + PFF 0.5, legacy training degrader); A = PFF held-out, 4 folds, 20 matches, real occlusions.
`scripts/eval_phase_alive.py`, same models in every cell (trained once per held-out group).  `off` = `w_phase_alive` 0
**with the same simulated tags** (the tags alone move spellF1 through the set-piece anchors and the H/A phase prior:
B' none 0.6044 -> medium 0.5941 / high 0.5891; A none 0.5231 -> 0.5109 / 0.5051).  The `none` rows reproduce the
Issue #38 numbers (0.6044 and 0.5231).  The last two columns are the predicted / true `D` share of the hidden-ball frames
with fewer than 16 / 20 tracked players (the #42 per-stratum spellF1 had not landed).  d = difference to `off` with the
same tags; worst seed = lowest of the three seed-mean differences.

B' (7 IDSSE, seeds 0-2):

| noise | variant | spellF1 | d | worst seed | macroF1 | possessions | hidden D pred / true % | < 16 pl. | < 20 pl. |
|---|---|---|---|---|---|---|---|---|---|
| none | off (no tags) | 0.6044 | | | 0.9016 | 6798 | 43.1 / 41.3 | 43.2 / 36.9 | 45.1 / 40.3 |
| low | off | 0.5955 | | | 0.8944 | 6514 | 45.6 / 41.3 | 45.2 / 36.9 | 47.4 / 40.3 |
| low | alive 0.5 | 0.5964 | +0.0009 | -0.0006 | 0.8955 | 6526 | 44.1 / 41.3 | 42.2 / 36.9 | 45.3 / 40.3 |
| low | alive 1.0 | 0.5958 | +0.0003 | +0.0002 | 0.8953 | 6551 | 42.9 / 41.3 | 40.4 / 36.9 | 43.5 / 40.3 |
| low | alive 2.0 | 0.5921 | -0.0034 | -0.0039 | 0.8926 | 6573 | 40.7 / 41.3 | 37.9 / 36.9 | 40.7 / 40.3 |
| low | cue | 0.5958 | +0.0003 | -0.0002 | 0.8944 | 6526 | 45.6 / 41.3 | 45.1 / 36.9 | 47.4 / 40.3 |
| low | alive 1.0 + cue | 0.5960 | +0.0005 | -0.0001 | 0.8953 | 6559 | 42.9 / 41.3 | 40.4 / 36.9 | 43.5 / 40.3 |
| low | alive 1.0 + dead-evidence mask | 0.5954 | -0.0001 | -0.0005 | 0.8951 | 6545 | 43.1 / 41.3 | 40.5 / 36.9 | 43.7 / 40.3 |
| low | alive 2.0 + dead-evidence mask | 0.5916 | -0.0039 | -0.0044 | 0.8924 | 6561 | 41.1 / 41.3 | 38.3 / 36.9 | 41.1 / 40.3 |
| medium | off | 0.5941 | | | 0.8919 | 6535 | 45.9 / 41.3 | 45.1 / 36.9 | 47.4 / 40.3 |
| medium | alive 0.5 | 0.5960 | +0.0019 | +0.0011 | 0.8933 | 6543 | 44.4 / 41.3 | 43.2 / 36.9 | 45.4 / 40.3 |
| medium | alive 1.0 | 0.5952 | +0.0011 | +0.0006 | 0.8933 | 6571 | 43.3 / 41.3 | 41.8 / 36.9 | 43.7 / 40.3 |
| medium | alive 2.0 | 0.5907 | -0.0034 | -0.0046 | 0.8907 | 6608 | 41.1 / 41.3 | 40.1 / 36.9 | 41.0 / 40.3 |
| medium | cue | 0.5943 | +0.0002 | -0.0001 | 0.8919 | 6547 | 45.9 / 41.3 | 45.1 / 36.9 | 47.4 / 40.3 |
| medium | alive 1.0 + cue | 0.5953 | +0.0012 | +0.0008 | 0.8933 | 6580 | 43.3 / 41.3 | 41.7 / 36.9 | 43.7 / 40.3 |
| medium | alive 1.0 + dead-evidence mask | 0.5947 | +0.0005 | +0.0001 | 0.8932 | 6565 | 43.6 / 41.3 | 41.9 / 36.9 | 44.0 / 40.3 |
| medium | alive 2.0 + dead-evidence mask | 0.5904 | -0.0037 | -0.0042 | 0.8909 | 6597 | 41.7 / 41.3 | 40.4 / 36.9 | 41.5 / 40.3 |
| high | off | 0.5891 | | | 0.8876 | 6593 | 46.2 / 41.3 | 48.6 / 36.9 | 48.6 / 40.3 |
| high | alive 0.5 | 0.5903 | +0.0013 | -0.0000 | 0.8886 | 6607 | 44.7 / 41.3 | 45.2 / 36.9 | 46.5 / 40.3 |
| high | alive 1.0 | 0.5891 | +0.0000 | -0.0003 | 0.8878 | 6634 | 43.4 / 41.3 | 44.0 / 36.9 | 44.8 / 40.3 |
| high | alive 2.0 | 0.5853 | -0.0038 | -0.0045 | 0.8843 | 6680 | 41.0 / 41.3 | 41.5 / 36.9 | 41.8 / 40.3 |
| high | cue | 0.5892 | +0.0002 | -0.0001 | 0.8876 | 6606 | 46.2 / 41.3 | 48.6 / 36.9 | 48.6 / 40.3 |
| high | alive 1.0 + cue | 0.5892 | +0.0002 | -0.0003 | 0.8878 | 6643 | 43.4 / 41.3 | 44.0 / 36.9 | 44.8 / 40.3 |
| high | alive 1.0 + dead-evidence mask | 0.5888 | -0.0003 | -0.0006 | 0.8877 | 6624 | 43.7 / 41.3 | 44.3 / 36.9 | 45.2 / 40.3 |
| high | alive 2.0 + dead-evidence mask | 0.5851 | -0.0039 | -0.0044 | 0.8846 | 6670 | 41.7 / 41.3 | 42.4 / 36.9 | 42.5 / 40.3 |

A (PFF held-out, 20 matches, real occlusions; validation only: reject if the drop exceeds 0.03):

| noise | variant | spellF1 | d | macroF1 | possessions | hidden D pred / true % |
|---|---|---|---|---|---|---|
| none | off (no tags) | 0.5231 | | 0.8626 | 6395 | 66.3 / 64.0 |
| medium | off | 0.5109 | | 0.8485 | 6019 | 68.2 / 64.0 |
| medium | alive 1.0 | 0.5109 | -0.0000 | 0.8491 | 6131 | 65.6 / 64.0 |
| medium | alive 2.0 | 0.5026 | -0.0083 | 0.8441 | 6299 | 62.9 / 64.0 |
| medium | alive 1.0 + dead-evidence mask | 0.5106 | -0.0003 | 0.8487 | 6106 | 66.4 / 64.0 |
| medium | alive 2.0 + dead-evidence mask | 0.5039 | -0.0070 | 0.8458 | 6229 | 64.8 / 64.0 |
| high | off | 0.5051 | | 0.8453 | 6149 | 68.0 / 64.0 |
| high | alive 1.0 | 0.5039 | -0.0011 | 0.8434 | 6265 | 65.1 / 64.0 |
| high | alive 2.0 | 0.4948 | -0.0103 | 0.8352 | 6443 | 61.7 / 64.0 |
| high | alive 1.0 + dead-evidence mask | 0.5033 | -0.0018 | 0.8440 | 6240 | 66.1 / 64.0 |
| high | alive 2.0 + dead-evidence mask | 0.4970 | -0.0081 | 0.8390 | 6371 | 64.0 / 64.0 |

**Decision: not adopted, `w_phase_alive = 0`, `phase_cue: false`, `alive_exclude_dead_evidence: false`.**  Adoption rule
(coordinator update): B' mean >= +0.01 at medium **and** high noise with no seed below -0.003 (A only a guard, -0.03), or a
demonstrated fix of the `D` over-prediction winning all three seeds at medium and high.  The best B' gain, with optimistic
tags, is +0.0019 (alive 0.5, medium) and +0.0013 (high); the weight that corrects the `D` over-prediction (alive 2.0:
predicted 41.0-41.1 % vs true 41.3 %) loses 0.003-0.004 spellF1 on every seed, and the intermediate weight 1.0 gains
+0.0011 at medium and nothing at high (worst seed -0.0003).  The transition cue adds +0.0002 (neutral), and excluding the
frames with dead-ball evidence from the mask does not help (it only weakens the prior).  Why so little: on degraded IDSSE
the specialists predict `D` for 43-46 % of the hidden frames vs 41 % true (the 92 % of BAR-ATH is not reproduced by this
recipe on external matches), and with 69 % phase coverage 30-55 % of the covered frames are dead, so the prior pushes dead
frames towards alive about as often as it helps.  Code, tests and the simulator stay
(`tests/test_43_smart_tagging_alive_prior.py`), switchable by configuration.  No BAR-ATH label or output was used and
nothing was tuned on BAR-ATH; the effect of the prior on BAR-ATH is not reported because it stays off.

**#51 -- the front end of a graded block only sees its own block.**  The warm-up (`0..11184`), half-time
(`90652..110807`), the dead start of the second half (`110808..113741`, forced `D` by the contract) and the post-match are
never graded, but the tracking there used to flow into the graded frames: the 5-frame gap interpolation, the
Savitzky-Golay runs (a valid run crossing a block edge is filtered as one), the cleaning (duplicate slots keep the
*longest* trajectory over the whole file), the visible-ball distances (`frames_since/until_visible`), the ball memory
(`last/next visible`, interpolation, last/next touch team) and the carry exponentials of the heuristic emissions.
BAR-ATH has 21,644 ball frames outside the two play blocks.  `pipeline.compute_front_end` now masks (NaN) players, ball and
float ball channels outside `play_blocks` before stage 2 (`pipeline.isolate_play_blocks: true`, `mask_outside_blocks`), so every
accumulator starts empty at the block edge and nothing is interpolated or filtered across it.  The second half starts at
`HALF2_PLAY` (`113742`), the same block the pipeline passes to the decoder and to the orientation detector; the `HALF2_DEAD`
frames are therefore blank for the front end (they are forced to `D` after decoding, so they never needed features).
Measured with `scripts/audit_block_leak.py` (isolation off vs on, committed specialists, aggregates only): on BAR-ATH
2,927 of 155,923 graded frames (1.9 %) change at least one feature, 2,309 of them within 60 s of a block edge (largest
distance 84.7 s; `bm:*` 2,346, `heur:*` 1,920, `sig:*` 643, `ctx:*` 808, kinematics 17-42, `bxy` 0), 1,923 emission rows,
and **0.006 % of the final labels** (D share 50.10 % -> 50.11 %, 243 possessions both).  On the 10 external matches
(0.05-2 % of frames between periods) 4-13 % of the graded frames change a feature (mostly `ctx:*`, 375-1,196 of them within
60 s of an edge; the rest, up to ~1,500 s from an edge, comes from the file-wide choice of the duplicate slot to keep).
B' (7 IDSSE leave-one-match-out, calibrated degrader, seeds 0-2, the cached legacy-mixed models, same models both arms):
spellF1 0.5997 -> 0.6001 (+0.0004), macroF1 0.8999 -> 0.9000, possessions 2,217 -> 2,215 (seed means 0.6017/0.5972/0.6003 ->
0.6024/0.5975/0.6005).  A correctness fix, adopted (gate: B' not worse than -0.003).  The committed `models/specialists*.json`
were trained with the old front end and still load; `#48` regenerates them.  Invariance test:
`tests/test_51_play_block_isolation.py` fills every frame outside the play blocks with random players/balls/channels and
asserts identical feature matrix, emissions and final labels (heuristic and specialist routes) in `HALF1` and `HALF2_PLAY`.
### Restart-team rules, wider restart window, t* offset prior and Long Ball flags (Issue #49; B' / A on simulated tags, not adopted)

**What was built** (everything behind config flags, default off, `restart_rules` section and three keys elsewhere).
(1) `features/restart_rules.apply_team_rules`: the team of every resolved restart is voted among the tag, the rules of the
game (`Kick Off (After Goal)` = team that conceded the preceding `GOALS` tag, second-half `Kick Off (Start)` = opposite of
the first-half one, taken from the order of the two tags, never from a frame number) and the tracking (first clear
controller within 1.4 s after `t*`); any dissent lowers the reliability `team_conf` (0.5) that scales
`transition_matrix.restart_other_team`; an N/A team is filled by the vote.  (2) `transition_matrix.restart_team_before_s /
restart_team_after_s`: the other-team penalty also holds in `[t* - before, t* + after]` and not only in `t* +- 3` frames.
(3) `setpiece_anchor.offset_prior`: the collective-cue search is restricted to the band (5-95 % quantiles +- 1 s) of
`t* - tag_start` learned, per type or pooled, from the ball-method anchors of the same match (a tracking measurement, no
label); nothing is invented when the band excludes the jump.  (4) `heuristic_emissions.w_long_ball` /
`restart_rules.auto_long_ball_weight`: `Long Ball` / `Ball in the Box` phases lower the `D` logit on hidden-ball frames,
the weight being `clip(logit(alive | in phase) - logit(alive | outside), 0, 1.5)` measured on the visible frames where
Specialist A is ultra-confident (0 with fewer than 250 such frames); `restart_rules.calibrate_weight` likewise scales the
team penalty by the agreement of the tagged teams with Specialist A's first confident state after each `t*`.  No ground
truth enters any weight.  The simulator (`io/smart_tagging_sim`) gained `kickoff_teams` (the `Kick Off (Start)` tags name the
first-possession team) and `wrong_team_rate` (set-piece teams flipped; own random stream, so the earlier draws are
unchanged); `scripts/eval_phase_alive.py --sim-restart` switches them on (wrong teams 10 % medium, 20 % high).  The
simulator has no `GOALS` / `Kick Off (After Goal)` tags (the external labels do not mark goals), so the post-goal rule is
covered by unit tests only.

**Protocol.**  As #43: B' = 7 IDSSE leave-one-match-out, calibrated degrader, seeds 0-2 (21 runs per cell), specialists
of the production recipe (IDSSE + PFF 0.5, legacy training degrader); A = PFF held-out, 4 folds, 20 matches, seed 0,
medium tags.  `off` = the same simulated tags with the feature disabled; d = difference to `off`; worst seed = lowest of
the three seed-mean differences.  Variants: `rules` (`restart_rules.enabled`), `wide` (4 s before, 2 s after), `rules_wide`,
`offset`, `lb` (rules on + `auto_long_ball_weight`).

| set | tags | off spellF1 | variant | spellF1 | d | worst seed |
|---|---|---|---|---|---|---|
| B' | medium | 0.5993 | rules | 0.5993 | +0.0000 | +0.0000 |
| B' | medium | | wide | 0.5995 | +0.0001 | +0.0000 |
| B' | medium | | rules_wide | 0.5995 | +0.0001 | +0.0000 |
| B' | medium | | offset | 0.5996 | +0.0002 | +0.0001 |
| B' | medium | | lb | 0.5996 | +0.0003 | -0.0004 |
| B' | high | 0.5960 | rules | 0.5960 | +0.0000 | +0.0000 |
| B' | high | | wide | 0.5959 | -0.0001 | -0.0003 |
| B' | high | | rules_wide | 0.5960 | -0.0000 | -0.0000 |
| B' | high | | offset | 0.5961 | +0.0001 | -0.0000 |
| B' | high | | lb | 0.5952 | -0.0008 | -0.0021 |
| A | medium | 0.5154 | rules | 0.5154 | +0.0000 | |
| A | medium | | wide | 0.5154 | +0.0000 | |
| A | medium | | rules_wide | 0.5154 | -0.0000 | |
| A | medium | | offset | 0.5153 | -0.0001 | |
| A | medium | | lb | 0.5147 | -0.0007 | |

**Why nothing moves.**  On a single match with the sampled tags (diagnostic, aggregates) 43 of 50 anchors resolve; the
tracking vote agrees with the tagged team in 26 of 34 comparable restarts (76 %, against 90 % correct simulated tags), so
the vote mostly ends in a tie and only lowers the confidence.  The decoder decides the exit of `D` with the
emissions: with the penalty `restart_other_team` raised from -3 to -12 only 7 of the graded frames change; the restart
team, however well known, is therefore worth far less than 0.001 spellF1 here, and the exact `t*` of the ball-method
anchors (85 % within 3 frames on external windows, #40) leaves the widening nothing to repair.  The `offset` band
only acts on collective-cue anchors (a third of the throw-ins / corners of #40), and the `Long Ball` flag is the
#43 alive prior restricted to two codes (same sign, same size).

**Decision.**  The adoption rule (B' mean >= +0.01 with no seed below -0.003 on both noise levels; A not below -0.03) is
**not met**: the best cell is +0.0003, A changes by at most -0.0007 (guard satisfied), so everything stays **off**.  No
correctness fix is applied either: none of the variants is a fix of a measured defect (unlike #50), they are optional
priors.  Code, tests and the simulator options remain available; for the real match they could only be judged on the
label-free aggregates of `apply_team_rules` (`diagnostics["restart_rules"]`: restarts, changed teams, unreliable ones,
tracking agreement) and of the calibration (`calibrated_weight`, `long_ball.weight`), which are printed with the other
diagnostics and contain no per-frame data.

### Reproducibility of the learned route (Issue #52)

RULES.md requires that running the repository against the provided data reproduces the submission byte for byte.
`--specialists auto` used to retrain at run time whenever the sibling benchmark cache was on disk, so the file depended
on the local machine rather than on the repository.  `auto` now loads the committed `models/specialists.json` first and
trains only when no committed file exists (`train` keeps retraining; synthetic CI runs never train).
`tests/test_52_reproducible_specialists.py` checks the order, that the committed model was trained with the sources and
weights declared in `configs/default.yaml`, and that the target-match adaptation (`use_pseudo_labels`) stays off
(official issue #2).  The pipeline diagnostics now report the emissions actually used (`specialists` or `heuristic`)
instead of a fixed `heuristic`.

## Milestone 10: post-freeze experiments (separate branches, adopted only if >= +0.010)

Protocol (fixed before any run, same harness as Item 8, `scripts/decoder_sweep.py`): B'(33) = 7 IDSSE leave-one-match-out
with the calibrated degrader at `dead_affinity` 33, seeds 0-2; adopt only if the mean gains >= +0.010 spellF1 with no seed
below -0.003; guards B'(1) (reject a drop > 0.01) and A (PFF 4-fold + 8 unseen PFF, reject a drop > 0.03).  Baseline = the
submitted default: B'(33) 0.6693 (0.6709 / 0.6675 / 0.6695), B'(1) 0.6017, A 0.5263 / 0.5263.  External matches only;
on BAR-ATH only label-free aggregates (#58, #63).  Each experiment lives in its own branch, merged with its code OFF.

| issue | change | B'(33) vs default | other evidence | decision |
|---|---|---|---|---|
| #55 | 56 PFF training matches instead of 20 | -0.0115 | B'(1) -0.0069; A +0.025 to +0.030 | not adopted |
| #56 | multi-scale context windows + goalkeeper depth | -0.0019 to -0.0025 | ~48 % of turnovers still missed | not adopted |
| #57 | learned H<->A turnover gate (Specialist C) | -0.0008 (max) to -0.0060 (hard) | hard: A +0.011 / +0.015; turnover recall unchanged (0.517) | not adopted, OFF |
| #58 | classic ball detectors on the real video | - | motion: top hit within 2 m 10.4 %, 3.75 false positives / frame | vision stays OFF |
| #59 | latency bias of the boundaries | - | H<->A about 1 frame early, below the 3-frame threshold | no shift applied |
| #60 | bag of 3 specialist seeds | -0.0013 | B'(1) -0.0012; A +0.0000 / +0.0003 | not adopted |
| #61 | boundary snap to touch / restart | +0.0002 | within seed noise (~0.002) | not adopted, OFF |
| #62 | expected-spellF1 (MBR) decoding | 0.0000 to -0.0007 | - | not adopted, OFF |
| #63 | supervision + roboflow/sports ball model | -0.074 (simulated, 50 % recovery) | real video: top hit within 2 m 60 %, median 1.12 m, 1.18 false positives / frame | not adopted, OFF |
| #64 | implicit carrier + 22v22 pressure micro-geometry features | +0.0002 | B'(1) +0.0023; A +0.0003 / +0.0014; turnover recall 0.518 (unchanged) | not adopted, OFF |
| #65 | Smart Tagging phase prior on the H-vs-A odds of hidden frames (0.8 / 1.5 / 2.5 / auto, with and without cue) | +0.0009 (medium, best) to -0.0073 (high, 2.5) | B'(1) -0.001 to -0.021; A -0.007 to -0.034; turnover recall falls | not adopted, OFF |
| #66 | tracklet gate / emission tempering on the #63 detections | gate 3: -0.0048; gate 5: +0.0005 (simulated, 50 %) | gate keeps 6-25 % of the points; B'(1) gate 3 +0.0089 | not adopted, OFF |

Outcome: nothing is adopted and the submission is unchanged (SHA256 `cac8927c2adc1b1a...`).  The positive cells (#61 +0.0002,
and #44 +0.009 of Milestone 9) do not add up to the bar: every decoder-side change targets the same missed turnovers, which come
from the specialists' emissions (turnover recall stays at 0.517 whatever the transition gate does), not from the decoder.  The
one result against the trend is the hard gate of #57 on real PFF occlusions (+0.011 / +0.015), which loses on every B'(33) seed;
it is recorded as future work, not applied.


### All available PFF matches in training, 56 instead of 20 (Issue #55; not adopted)

Training rows: the 7 IDSSE matches (calibrated degrader, `dead_affinity` 33) plus the first 56 PFF matches by name at weight 0.5
(`pff_max_matches: 56`); the 8 last PFF matches by name stay out of training (they are the unseen validation set).
Same harness and models as Item 8 (`scripts/train_cv_models.py`: LOMO over the 7 IDSSE matches, PFF 4-fold over the 56 matches,
`--full` all-data model scored on the 8 unseen; `scripts/decoder_sweep.py run --n-pff 56`).  The baseline was recomputed with the
same harness and reproduces Item 8 exactly.  Candidate: spellF1, seeds 0 / 1 / 2.

| | default (20 PFF) | 56 PFF | delta |
|---|---|---|---|
| B'(33) mean | 0.6693 (0.6709 / 0.6675 / 0.6695) | 0.6578 (0.6613 / 0.6514 / 0.6608) | **-0.0115** (-0.0096 / -0.0161 / -0.0087) |
| B'(33) macroF1 | 0.9206 | 0.9176 | -0.0030 |
| B'(1) guard | 0.6017 (0.6019 / 0.5991 / 0.6041) | 0.5948 (0.6014 / 0.5905 / 0.5923) | -0.0069 |
| A, first 20 PFF, 4-fold | 0.5263 (macroF1 0.8636) | 0.5497 (0.8696) | +0.0235 |
| A, 56 PFF, 4-fold over the 56 | 0.5311 (0.8706) | 0.5560 (0.8773) | +0.0249 |
| A, 8 unseen PFF | 0.5263 (0.8652) | 0.5559 (0.8722) | +0.0296 |

The 56-match fold is index modulo 4 over the first 56 matches (14 held out per fold, so the first-20 subset keeps its folds); the
default models never saw matches 21-56, which makes that row conservative for the default.  Rule: B'(33) >= +0.010 and no seed below
-0.003; reject if B'(1) drops > 0.01 or A drops > 0.03.  **Rejected: B'(33) is lower in all three seeds (-0.0115 on average).**  The guards
pass (B'(1) -0.0069, A improves), so the cause is not a broken model but a shift of the mixture: the PFF rows grow 2.8x at the same
weight 0.5, so the model fits the world of real, visible-ball occlusions (A +0.025 to +0.030) at the cost of the degraded-IDSSE world that
mimics the competition ATD (B').  Not run (would be a selection on the same B' world): re-weighting PFF to keep its total mass
(about 0.18 per match).  `configs/default.yaml` and `models/specialists.json` are unchanged; `train_cv_models.py --full` and
`decoder_sweep.py --n-pff / unseen_<variant>.json` stay as tools.

### Multi-scale context windows and goalkeeper depth for the hidden-ball specialist (Issue #56; not adopted)

*Hypothesis.*  After a turnover while the ball is hidden the team shape and the goalkeepers (the recovering team's keeper steps up
8-15 m in 5-8 s, the other drops) need 6-10 s to settle, so context windows of 25 / 75 / 175 / 250 frames (instead of 50 / 100) and
the goalkeeper depth could expose it.  *Code* (`models/specialist_features.py`, all behind `specialist_features`, default unchanged):
`context_windows` accepts any window list; `gk_context: true` adds, for the orientation-normalised keeper depth of each team
(`cp:gk_xy_home_x` / `cp:gk_xy_away_x`, Issue #17 goalkeeper identification) the centred mean and the change
`mean(t..t+W) - mean(t-W..t)` with `W = gk_span` (150 frames = 6 s); block-local, NaN-safe, vectorised, deterministic
(`tests/test_56_multiscale_gk_context.py`).

*Protocol* as Item 8 (Issue #44): 7 IDSSE leave-one-match-out with the calibrated degrader at `dead_affinity` 33 (B'(33), seeds 0 / 1 / 2),
B'(1) and A (20 PFF, 4 folds) as guards; training rows rebuilt for every variant with the current default recipe (calibrated degrader
affinity 33, IDSSE + PFF 0.5, `pff_max_matches` 20); models from `scripts/train_cv_models.py`, evaluation with `scripts/decoder_sweep.py`
(default decoder).  The default baseline was recomputed with the same harness and reproduces the reported numbers exactly.

| variant | B'(33) mean (seeds 0 / 1 / 2) | vs default (per seed) | B'(1) | A 4-fold (20) |
|---|---|---|---|---|
| default (windows 50 / 100) | 0.6693 (0.6709 / 0.6675 / 0.6695), macro 0.9206 | | 0.6017 | 0.5263 |
| windows only (25, 75, 175, 250) | 0.6670 (0.6642 / 0.6649 / 0.6718) | -0.0023 (-0.0067 / -0.0026 / +0.0023) | 0.6012 (-0.0005) | 0.5304 (+0.0041) |
| goalkeeper only (windows 50 / 100 + GK +-6 s) | 0.6674 (0.6654 / 0.6631 / 0.6736) | -0.0019 (-0.0055 / -0.0044 / +0.0041) | 0.6012 (-0.0005) | 0.5242 (-0.0021) |
| both | 0.6668 (0.6694 / 0.6597 / 0.6712) | -0.0025 (-0.0015 / -0.0078 / +0.0017) | 0.6019 (+0.0002) | 0.5252 (-0.0011) |

*Hidden-turnover diagnostic* (`dropout_strata`, B'(33) pooled over the three seeds, share of the true `H<->A` changes -- 7053 in
the strata < 20 and >= 20 players -- that the decoded sequence misses): default 48.0 %, windows 48.5 %, goalkeeper 48.7 %, both 48.4 %.
Recall of the turnovers is therefore unchanged (52.0 / 51.5 / 51.3 / 51.6 %): the slow settling of the shape and of the keepers is
already captured by the existing 50 / 100-frame context, or is too noisy per frame to separate a turnover from the normal
reorganisation of the block.

**Decision: not adopted** (needed +0.010 on B'(33) with no seed below -0.003).  Every variant is slightly *below* the default on B'(33)
(-0.0019 to -0.0025, with at least one seed below -0.003 in every variant; only seed 2 gains),
so the defaults stay (`context_windows: [50, 100]`, `gk_context: false`).  The code stays available, OFF.  The all-data
model of each variant (unseen-PFF guard) was not trained because the primary criterion already fails.

### Specialist C, a learned H<->A turnover gate (Issue #57; not adopted, OFF)

`src/mtah2027/models/specialist_c.py`, `scripts/train_specialist_c.py`, config blocks `specialist_c` and
`transition_matrix.learned_gate_*` (`learned_gate_mode: off`).  C is a LightGBM (`deterministic=True`, `num_threads=1`,
`seed=0`; scikit-learn fallback through `FrameClassifier`) on the 206-column feature matrix of the specialists (so it works with
a visible and a hidden ball: distances of the nearest players, closing speeds, touch score, collective signals and their
+-50 / 100 frame context, coherence R, goalkeepers, time since the ball was lost) that predicts the class "no change / H -> A /
A -> H within +-5 frames of the frame", the target being derived from the ground truth of external matches (direct H <-> A
change between two live frames).  Rows are those of the default training pickle (calibrated degrader, dead_affinity 33, stride
10, home / away swapped copies; 598k rows, 1.3 % positives per direction), every 4th negative kept with weight 4; a fitted
temperature was 0.99-1.00 so the calibration folds are off.  Leave-one-match-out: the C of a held-out IDSSE match never sees it
(PFF flavour: IDSSE + 20 PFF at weight 0.5; for the PFF 4-fold guard the held-out fold's matches are excluded; for the 8 unseen
PFF matches the C is trained on all rows).  The gate maps P_C to the transition: `log A_t = closed + (open - closed) * g`, with three
variants fixed beforehand: `hard` (g = 1 where P_C >= 0.5, replaces the touch rule), `blend` (g = 0.5 rule + 0.5 P_C),
`max` (g = max(rule, P_C)); the occlusion floor `ha_occluded` is kept.  Harness `scripts/decoder_sweep.py` (cached front end and
default A/B models, only the transitions change; candidates attach C with `"c": "<flavour>"`; the pooled turnover recall is
printed with `agg --turnover`: share of the true direct H<->A changes matched by a predicted change of the same direction within
25 frames).  spellF1 (delta vs the default 0.6693 / 0.6017 / 0.5263 / 0.5263):

| world | default | hard | blend | max |
|---|---|---|---|---|
| B'(33), 7 IDSSE LOMO, seeds 0-2, IDSSE-only C | 0.6693 | 0.6633 (-0.0060) | 0.6654 (-0.0039) | 0.6685 (-0.0008) |
| per seed delta (0 / 1 / 2) | | -0.0093 / -0.0053 / -0.0033 | -0.0080 / -0.0007 / -0.0029 | -0.0011 / -0.0014 / 0.0000 |
| B'(33), IDSSE + PFF C | 0.6693 | 0.6620 (-0.0073) | 0.6646 (-0.0047) | 0.6687 (-0.0006) |
| per seed delta (0 / 1 / 2) | | -0.0103 / -0.0078 / -0.0038 | -0.0080 / -0.0032 / -0.0031 | -0.0007 / -0.0012 / +0.0001 |
| B'(1) (guard, seed 0; IDSSE / +PFF C) | 0.6019 | 0.5991 / 0.5996 | 0.6005 / 0.6005 | 0.6014 / 0.6020 |
| A PFF 4-fold, 20 matches (guard; IDSSE / +PFF C) | 0.5263 | 0.5374 / 0.5374 (+0.0111) | 0.5330 / 0.5334 | 0.5249 / 0.5253 |
| A unseen, 8 matches (guard; IDSSE / +PFF C) | 0.5263 | 0.5413 / 0.5421 (+0.0150 / +0.0158) | 0.5356 / 0.5366 | 0.5263 / 0.5265 |

Turnover recall (matched / true changes): B'(33) default 0.517 (3647 / 7053, 6297 predicted changes), `hard` 0.493, `blend` 0.500,
`max` 0.517; B'(1) 0.407 -> 0.389 / 0.396 / 0.405; PFF 4-fold 0.352 -> 0.344 / 0.346 / 0.353; unseen PFF 0.371 -> 0.360 / 0.364 / 0.375.
The B'(33) default reproduces 0.6693 (0.6709 / 0.6675 / 0.6695) exactly.

Why it does not help: a gate can only allow a change, it cannot make the emissions prefer it.  The variants that open the gate
wider (`max`) leave the recall unchanged (0.517), so the missed turnovers are not blocked by the transition matrix; the emissions
(specialists' probabilities) never swing to the other team.  The variants that close the gate where P_C is low (`hard`,
`blend`) remove true changes together with the false ones (recall -2.4 points on B'(33)).  On the real PFF occlusions the
hard gate removes 7 % of the predicted changes and gains +0.011 / +0.015 (fewer spurious spells), but the adoption rule is set
on B'(33), the world that imitates the competition ATD, where it loses 0.006-0.007 on every seed.  Rule (B'(33) mean >= +0.010,
no seed below -0.003, B'(1) not down by more than 0.01, A not down by more than 0.03) fails on the primary criterion; the
variant picked on seeds 0-1 (`max`, -0.0013) is -0.0000 / +0.0001 on the blind seed 2.  Not run: a C trained on stride-1 rows
(positives are about one row per event at stride 10), and the full-data C model (not trained: no variant is adopted).  Default OFF;
the submission path and the synthetic SHA256 (`2a31a913...`) are unchanged in two runs.

### Ball detectors on the real tactical video (Issue #58; aggregates only)

Purpose: the broadcast route was measured label-free on the real match (Issue #39) but the ball detectors of the
vision route (motion #36, optional YOLO) had only been tested on synthetic video.  `scripts/real_video_ball_check.py`
samples (seeded, default 1000) video frames of the evaluated halves where the ATD ball is visible and the
homography is usable, cuts the 640x640 attention crop around the projected ATD ball (`--search-offset-m` shifts
it by a seeded random offset to mimic a real search point), runs the available detectors, reprojects the detections
with H^-1 and compares them with the ATD ball.  It never reads possession labels, never writes crops or per-frame
rows, and refuses to print anything that is not an aggregate.  A detector that cannot be built (no ultralytics or
weights) is skipped and listed under `skipped_detectors`.  The submission path and defaults are not changed.

```bash
python scripts/real_video_ball_check.py --data-dir data --sample 1000 --json reports/real_video_ball_check.json
```

Numbers printed, per detector: detection rate (% of sampled frames with a detection within 1 / 2 / 5 m of the ATD
ball, any candidate and highest-confidence candidate), median and p90 error in metres of the nearest detection,
false positives per frame (detections farther than 5 m), frames skipped (invalid homography / off-image, video
read failures), detections per frame, runtime; plus the median apparent ball size in px.  The crop is centred on
the ATD ball, so the rates are an upper bound of the gap-recovery use.

Result on the real BAR-ATH tactical video (run locally; aggregates only; seed 0, 1000 sampled frames out of 91,247
eligible, 91,844 graded frames with the ATD ball visible, 597 skipped for invalid homography / off-image, 0 read
failures; YOLO skipped because `ultralytics` is not installed):

| detector, crop | any detection within 1 / 2 / 5 m | top-confidence within 1 / 2 / 5 m | nearest error median / p90 | false positives per frame | detections per frame | time |
|---|---|---|---|---|---|---|
| motion, centred on the ATD ball | 15.5 / 27.9 / 51.0 % | 5.1 / 10.4 / 21.6 % | 4.7 / 15.5 m | 3.75 | 4.67 | 214 s |
| motion, crop shifted 3 m | 15.4 / 27.2 / 50.4 % | 5.2 / 10.8 / 22.0 % | 4.8 / 16.5 m | 3.73 | 4.65 | 212 s |
| hough, centred on the ATD ball | 7.2 / 13.8 / 25.3 % | 5.8 / 9.8 / 14.7 % | 13.9 / 40.8 m | 3.92 | 4.24 | 285 s |
| hough, crop shifted 3 m | 7.0 / 13.7 / 25.9 % | 5.8 / 9.7 / 14.8 % | 13.7 / 40.2 m | 3.93 | 4.26 | 215 s |

Median apparent ball size: 4.5 px.  Reading: on the tactical camera of the stadium the ball is about 4-5 pixels wide
and the available detectors do not find it.  Even with the crop centred on the true ball, the most confident
detection lies within 2 m of it in only ~10 % of the frames, there are ~3.7 false positives per frame, and the
numbers barely change when the crop is shifted by 3 m, which means most "hits" are other blobs (players, lines)
that happen to be close to the ball rather than the ball itself.  The simulation of Issue #36 needs a ~0.3 m
accurate detector to gain anything, so this confirms with real measurements the decision to keep `vision.enabled:
false` in the submission.  A detector trained for this camera remains future work.

### Signed latency of the boundaries on other matches (Issue #59; diagnostic, no shift applied)

Diagnostic added to `scripts/eval_external.py` (`lat_by_type` per run, `latency_summary`): signed latency in frames of the
transitions matched by the official scorer (`evaluate()["transitions"][25]`, tolerance 25 frames, `pred - gt` as in
`possession_metrics.transition_scores`, matching on the destination label), split into H<->A, alive -> D and D -> alive
(restarts).  Models: the cached leave-one-match-out models of the default recipe (`models/specialists.json` is byte
identical to the `gap` variant; Issue #44 `lomo_gap`, `pff_gap`), no retraining.  Protocol of Items 7-8 (B'(33): 7 IDSSE
LOMO, calibrated degrader `dead_affinity` 33, seeds 0 1 2, harness `scripts/decoder_sweep.py`); the run reproduces the
default of the #44 harness exactly (spellF1 0.6693 = 0.6709 / 0.6675 / 0.6695).  A PFF (20 matches, 4-fold models) and A-unseen
(8 matches, committed models) as checks (both 0.5263, as in Item 8).  "Runs" = (match, seed) cells; median per run.

| world | transitions | runs | median of run medians | run medians min / max | runs early / late | pooled median | pooled early / exact / late | pooled IQR |
|---|---|---|---|---|---|---|---|---|
| B'(33) | all (7133 matched) | 21 | 0.0 | -1 / 0 | 29 % / 0 % | 0 | 49 % / 28 % / 23 % | -2 .. 0 |
| B'(33) | H<->A (3667) | 21 | -1.0 | -1 / 0 | 90 % / 0 % | -1 | 55 % / 21 % / 24 % | -3 .. 0 |
| B'(33) | alive -> D (1738) | 21 | -1.0 | -1 / 0 | 86 % / 0 % | -1 | 60 % / 26 % / 14 % | -2 .. 0 |
| B'(33) | D -> alive (1728) | 21 | 0.0 | 0 / 0 | 0 % / 0 % | 0 | 25 % / 44 % / 31 % | 0 .. +1 |
| PFF 4-fold | all (3721) | 20 | -1.5 | -3 / 0 | 85 % / 0 % | -1 | 53 % / 3 % / 43 % | -8 .. +9 |
| PFF 4-fold | H<->A (1837) | 20 | -1.8 | -3 / +4 | 75 % / 20 % | -1 | 52 % / 3 % / 44 % | -8 .. +9 |
| PFF 4-fold | alive -> D (1013) | 20 | -0.2 | -4 / +3 | 50 % / 20 % | -1 | 51 % / 4 % / 45 % | -8 .. +9 |
| PFF 4-fold | D -> alive (871) | 20 | -3.5 | -7 / +5 | 90 % / 10 % | -3 | 58 % / 3 % / 39 % | -8 .. +9 |
| PFF unseen | all (1413) | 8 | -2.0 | -4 / 0 | 88 % / 0 % | -2 | 56 % / 3 % / 41 % | -10 .. +7 |

Decision rule (fixed beforehand): stable bias = at least 80 % of the runs with the same sign **and** |median of medians| >= 3
frames on B'(33).  On B'(33) the sign is consistent for H<->A (90 % of the runs early) but the magnitude is one frame
(median of medians -1 for H<->A and alive -> D, 0 for restarts, 0 overall), far below 3 frames, and one frame is free in
the metric.  PFF (real tracker) shows a small early tendency (median -1 to -2) with a spread of about +-9 frames, so it
does not contradict "no bias worth correcting" either, but it is not evidence for a +3 shift: on B'(33) a shift of the
bulk would move the exact matches (28 %, 44 % for restarts) away from zero.  **Not adopted**: `boundary_shift_frames` was
not implemented (Task 2 only applied with a stable bias); the submission path and the synthetic SHA are untouched.

### Seed bagging of the specialists (Issue #60; not adopted)

Same world, same training rows (IDSSE calibrated degrader `dead_affinity` 33 + the first 20 PFF at weight 0.5, training pickle
fixed), two extra LightGBM seeds (1, 2; `deterministic=True`, `num_threads=1`; the seed-0 models are the first member), one model per
LOMO fold (7 IDSSE), per PFF fold (4) and the all-data model for the unseen PFF.  The bag is a `--model-variant` style ensemble
manifest (`format: ensemble`, `mode: linear`, equal weights; `specialist_fusion.save_bag_manifest`): class probabilities are averaged
per frame before the soft priors, floor, scaling and 1e-6 quantisation, so it is deterministic.  Tools: `train_cv_models.py --seed
N [--full]`, `train_specialists.py --seed N`, `train_specialists(seed=...)`; evaluation with `scripts/decoder_sweep.py` (the baseline
reproduces 0.6693 / 0.6017 / 0.5263 / 0.5263 exactly).  spellF1:

| | default (seed 0) | bag of 3 | delta | seed 1 alone | seed 2 alone |
|---|---|---|---|---|---|
| B'(33) mean (degradation seeds 0 / 1 / 2) | 0.6693 (0.6709 / 0.6675 / 0.6695) | 0.6680 (0.6694 / 0.6667 / 0.6680) | **-0.0013** (-0.0015 / -0.0008 / -0.0015) | 0.6681 | 0.6676 |
| B'(33) macroF1 | 0.9206 | 0.9207 | +0.0001 | 0.9201 | 0.9206 |
| B'(1) guard | 0.6017 | 0.6005 | -0.0012 | 0.5999 | 0.5977 |
| A, first 20 PFF, 4-fold | 0.5263 | 0.5263 | -0.0000 | 0.5258 | 0.5246 |
| A, 8 unseen PFF | 0.5263 | 0.5266 | +0.0003 | 0.5260 | 0.5287 |

The single extra seeds are within -0.0020 of the default on B'(33), i.e. the seed-to-seed spread is about 0.001-0.002, an order of
magnitude below the +0.010 bar, and averaging three members removes none of the gap to the default (the members agree on the frames
that decide the spells; the residual error is bias of the training world, not variance of the booster).  **Rejected**: B'(33) is
-0.0013 (rule: mean >= +0.010; the "no seed below -0.003" condition holds but is moot), guards pass (B'(1) -0.0012, A +-0.000).  Not run: the
bag over degradation seeds (a second training pickle with another degrader seed), planned only if the seed bag showed >= +0.005.
`configs/default.yaml` and `models/specialists*.json` are unchanged; no `specialists_bag.json` is committed (the submission and the
synthetic SHA256 do not change).

### Boundary snap to the touch / restart frame (Issue #61; not adopted, OFF)

**Idea.**  After Viterbi and the spell optimiser (`pipeline` step 10b, `postprocessing/boundary_snap.py`,
`boundary_snap.enabled`, default **false**) each `H<->A` boundary moves to the nearest touch peak (`touch_score > 0`,
ball visible, not aerial) of the gaining team (`nearest_team`) within +-N frames, and each `D -> alive` boundary to the
restart frame `t*` of an unused resolved set-piece anchor within +-N; `alive -> D`, block edges, moves that would relabel
forced-dead frames, empty a run or cross the neighbouring boundary are never applied.  N in {6, 12} fixed a priori
(12 chosen on seeds 0-1, seed 2 blind).  Harness `scripts/decoder_sweep.py` (cached front end; candidates via
`route: {"boundary_snap.enabled": true, ...}`), cached LOMO `gap` models, no retraining; `--verify` cached == uncached.

| B'(33) 7 IDSSE LOMO | seed 0 | seed 1 | seed 2 | mean | delta |
|---|---|---|---|---|---|
| default | 0.6709 | 0.6675 | 0.6695 | 0.6693 | - |
| snap N=6 | 0.6707 | 0.6671 | 0.6699 | 0.6692 | -0.0001 |
| snap N=12 | 0.6708 | 0.6678 | 0.6699 | 0.6695 | +0.0002 |

Decision: **not adopted** (needed >= +0.010).  The Viterbi boundaries are already within a few frames of the touch, and
frame-level moves inside the +-25-frame scoring tolerance barely change the matching.  IDSSE runs carry no Smart
Tagging, so the restart branch (unit-tested only) is not exercised there.  Guards B'(1) and A PFF were not run (the
primary rule already fails).  Default config and synthetic submission unchanged.

### Decoding by expected spellF1, minimum Bayes risk over posterior samples (Issue #62; not adopted, OFF)

`src/mtah2027/decoding/expected_spellf1.py`, config block `expected_spellf1` (`enabled: false`).  Per graded block: log-space
forward pass over the same quantised emissions and time-varying transitions `A_t` as Viterbi, `k` posterior paths by backward
sampling (fixed seed, NumPy-precomputed conditional probabilities so the njit and the pure-Python kernels agree), the Viterbi
path added as candidate 0, and the candidate with the highest mean spellF1 against the others is kept.  The pair score is a
vectorised re-implementation of the official `spell_scores` (verified equal to `possession_metrics.spell_scores` to 1e-12 in
`tests/test_62_expected_spellf1.py`; the forward-backward sampler is verified against brute force on a 5-frame chain).
Protocol of Items 7-9 (harness `scripts/decoder_sweep.py`, cached default models, no retraining; candidates fixed beforehand:
`k=100 T=1`, `k=50 T=1`, `k=100 T=2`, T = sampling temperature).  spellF1 means:

| world | default | k=100 T=1 | k=50 T=1 | k=100 T=2 |
|---|---|---|---|---|
| B'(33), 7 IDSSE LOMO, seeds 0-2 | 0.6693 | 0.6690 (-0.0003) | 0.6686 (-0.0007) | 0.6693 (+0.0000) |
| B'(33) per seed delta (0 / 1 / 2) | | -0.0002 / +0.0006 / -0.0014 | +0.0020 / 0.0000 / -0.0042 | 0 / 0 / 0 |
| B'(1) (guard) | 0.6017 | 0.5994 (-0.0023) | 0.5994 (-0.0023) | 0.6017 (+0.0000) |
| A PFF 4-fold, 20 matches (guard) | 0.5263 | 0.5193 (-0.0070) | 0.5192 (-0.0071) | 0.5231 (-0.0031) |
| A unseen, 8 matches (guard) | 0.5263 | 0.5231 (-0.0032) | 0.5216 (-0.0047) | 0.5263 (+0.0000) |

Why it does not help: the emissions plus the transition matrix make the posterior very concentrated around the Viterbi path;
the samples differ from it by flicker (about 210 spells per half against 189 for Viterbi in the cell inspected), so the
Viterbi path is the most central candidate (it is chosen in practically every block) and the selection returns it or a
near-identical neighbour.  With T = 2 the decoder collapses to Viterbi exactly on the IDSSE worlds.  The rule (B'(33) mean
>= +0.010, no seed below -0.003) fails by a factor of 10 or more; no candidate was selected on a single seed.  Cost when
on: about 0.3 s per 80k-frame block for `k = 100` (0.1 s for `k = 50`) with numba, i.e. about 1 s per match, far below the
+60 s budget.  Default OFF; the submission path and the synthetic SHA are unchanged.

### Ball detector with roboflow/supervision on the real tactical video (Issue #63; not adopted, OFF)

`supervision` (MIT) is a toolkit, not a detector.  `src/mtah2027/vision/sv_ball.py` combines it with the
single-class ball model of roboflow/sports (`football-ball-detection.pt`, YOLOv8, loaded only from a local file
through `ultralytics.YOLO(path)`; no hosted inference, no API key, no frame leaves the machine).  Two fixed
configurations, chosen a priori: `sv_ball` (`InferenceSlicer`, 320 px tiles, 64 px overlap) and `sv_ball640` (the
whole 640x640 attention crop, no slicing); confidence floor 0.05.  Optional temporal filter (`--sv-clip 5`): the 5
consecutive frames around each sampled frame are scanned, candidates are reprojected with `H^-1` and the centre
frame candidates are rescored by the support they find in the neighbour frames (speed gate 38 m/s, bonus when
t-1 / t+1 agree with a constant-velocity motion).  ByteTrack is not used: supervision >= 0.31 moved it to a separate
package.  CPU only, deterministic (same output on repeated runs, checked).

Licence notes.  `supervision` and `roboflow/sports` code: MIT.  `ultralytics` (YOLOv8 runtime): AGPL-3.0 (already
an optional extra, never needed for the submission).  The weights file has no licence of its own in the
repository; the roboflow/sports README says the ball dataset (Roboflow Universe, `football-ball-detection-rejhg`)
was derived from the Kaggle "DFL - Bundesliga Data Shootout" broadcast footage, so the model was trained on
broadcast views and its use must follow the YOLOv8/AGPL terms plus whatever terms that dataset carries.  Weights
and videos are never committed; torch/ultralytics/supervision live in a separate environment and the default
path and CI do not need them (tests skip with `pytest.importorskip`).

Sanity check on a public roboflow/sports broadcast sample (`2e57b9_0.mp4`, 8 frames, aggregates only): the model
returns a ball candidate in 8/8 full frames (median top confidence 0.85).  Cost on CPU for one 640x640 crop:
about 4 s (`sv_ball`) and 0.9 s (`sv_ball640`) per frame, times the clip length with `--sv-clip`.

Run (aggregates only, same guard as Issue #58):

```bash
<venv>/Scripts/python scripts/real_video_ball_check.py --data-dir data --detectors sv_ball640,sv_ball     --sv-weights <path>/football-ball-detection.pt --sample 1000 --json <outside the repo>/sv_ball.json
```

**Result on the real tactical video** (run locally by the lead; aggregates only; seed 0, 300 frames with the ATD ball
visible, ball ~4.5 px, crop centred on the ATD ball = ceiling):

| detector | any detection within 1 / 2 / 5 m | top-confidence within 1 / 2 / 5 m | nearest error median / p90 | FP per frame | s per frame (CPU) |
|---|---|---|---|---|---|
| motion (#58, 1000 frames) | 15.5 / 27.9 / 51.0 % | 5.1 / 10.4 / 21.6 % | 4.7 / 15.5 m | 3.75 | 0.2 |
| `sv_ball640` (YOLO roboflow, 640 crop) | 43.3 / 63.0 / 78.7 % | 41.0 / 59.7 / 75.7 % | 1.12 / 8.5 m | 1.18 | 1.27 |
| `sv_ball` (InferenceSlicer 320) | 46.3 / 67.0 / 83.0 % | 42.7 / 61.7 / 78.0 % | 1.07 / 6.8 m | 1.47 | 5.19 |

`sv_ball640`: frames with any detection 95 %, 2.1 detections per frame; crop shifted 3 m: top-confidence within 1 / 2 / 5
m 42.7 / 61.0 / 77.0 % (robust to the search offset); 5-frame temporal filter: no change.  These frames are ones where
the tracking *sees* the ball; in tracking gaps the rate is probably lower (occlusion).

**Value in the hidden-ball gaps (simulation, `scripts/simulate_sv_ball_value.py`).**  A simulated detector
(`vision/sim_detector.py`) reproduces the measured top-confidence CDF: per planned crop (one every `vision.every_k` = 5
frames of each hidden gap, as #36) it returns the true ball plus an error drawn from the mixture <= 1 m 41.0 % /
1-2 m 18.7 % / 2-5 m 16.0 % / far 19.3 % (a false positive, uniform 5-30 m from the ball) / no detection 5 %
(uniform radius inside a bin, uniform angle).  Scenarios fixed a priori: gap detection rate 100 / 50 / 25 % of the
measured rate (frames removed, nested per seed), and the measured `motion` profile as a control.  The points go
through the same `inject_points` as the real chain (no speed gate, no pitch filter: pessimistic about the real
`select_detections`, which would drop some far false positives), then the production specialist route (`lomo_gap`
models, default config, vision off elsewhere), scored with the official metric on the degraded matches: B'(33) = 7 IDSSE
leave-one-match-out, B'(1), seeds 0-2.  The `none` rows reproduce the references (B'(33) 0.6693 / 0.6709 / 0.6675 /
0.6695, B'(1) 0.6017).

B'(33) (hidden-ball frames filled by detections or interpolation: 76 % at 100 / 50 %, 42 % at 25 %, 34 % motion):

| scenario | spellF1 | d | macroF1 | d per seed (0 / 1 / 2) |
|---|---|---|---|---|
| none | 0.6693 | | 0.9206 | |
| sv_ball640 100 % | 0.5756 | -0.0938 | 0.8395 | -0.0900 / -0.0984 / -0.0928 |
| sv_ball640 50 % | 0.5950 | -0.0744 | 0.8625 | -0.0694 / -0.0762 / -0.0775 |
| sv_ball640 25 % | 0.6279 | -0.0415 | 0.9015 | -0.0421 / -0.0419 / -0.0404 |
| motion (measured) | 0.5242 | -0.1451 | 0.8218 | -0.1450 / -0.1480 / -0.1422 |

B'(1) guard: none 0.6017; 100 % 0.5678 (-0.0339); 50 % 0.5800 (-0.0217); 25 % 0.5943 (-0.0073); motion 0.5026 (-0.0991).
The PFF guard was not run (B'(33) failed the rule).

**Decision: not adopted.**  The rule (B'(33) mean >= +0.010, no seed < -0.003 at 50 %) is missed by a wide margin:
at every rate the simulated sv_ball640 *lowers* spellF1 (-0.074 at 50 %), and the loss is monotone in the rate.  With
41 % of the points within 1 m but 24 % off by more than 5 m (or missing) and another 35 % off by 1-5 m, the injected
track misleads the possession model more than the empty gap does, which the tracking-free specialists already handle
well.  The 1.27 s per crop on CPU would also cost about 11.7k planned crops per simulated match; on the real match,
with the label-free 40.7 % of the graded frames hidden (~63.5k frames, one crop every 5) that is ~12.7k crops, about
4.5 h of CPU for `sv_ball640` (about 18 h for `sv_ball`), plus the AGPL-3.0 (ultralytics) and weights /
dataset-terms caveat above.  `vision.enabled` stays `false` and `sv_ball` is not wired into the submission route.
It would only be worth revisiting with a detector whose top-confidence candidate is within ~0.3 m in most
frames, or with a confidence-gated variant (use only candidates that agree over several frames), which the
measured aggregates do not support.

Reproduce: `python scripts/simulate_sv_ball_value.py run --world b33 --seeds 0 1 2 --models-root <lomo models> --out-dir D`
(and `--world b1`), then `agg --out-dir D`.

### Implicit carrier and 22-v-22 pressure micro-geometry for the hidden-ball specialist (Issue #64; not adopted, OFF)

*Hypothesis.*  Without the ball, the team that has it is the one whose moving player is being converged on by the rivals.
`features/carrier_swarm.py`, block `specialist_features.carrier_swarm` (default **false**; off = identical feature matrix and
synthetic SHA256).  Per team, outfield players only (dynamic goalkeepers excluded): `swarm_score` = max over the moving
(`|v| >= 1.5 m/s`) outfield players of the sum over outfield rivals within 3.5 m of `exp(-d_ij / 2.0) * max(closing_speed_ij, 0)`;
`carrier_x` / `carrier_vx` of the argmax player; team-independent `min_interteam_dist` and `duel_density_3m` (home-away outfield
pairs below 3.0 m); `diff:` (home minus away) of swarm / carrier columns; `ctx:` windows (50 / 100, as `context_windows`) of all of
them.  Vectorised in 4096-frame chunks, NaN-safe, deterministic (`tests/test_64_carrier_swarm_features.py`, hand-built scenes).

*Protocol* as Item 8: training rows rebuilt with the flag on (IDSSE calibrated degrader `dead_affinity` 33 + first 20 PFF at
0.5), 7 LOMO models, 4 PFF folds and the all-data model (unseen PFF); `scripts/decoder_sweep.py`, default decoder.  Baseline
= the cached default (reproduces 0.6693 / 0.6017 / 0.5263 / 0.5263 as in #56).

| | default | carrier_swarm | delta |
|---|---|---|---|
| B'(33) mean (seeds 0 / 1 / 2) | 0.6693 (0.6709 / 0.6675 / 0.6695) | 0.6695 (0.6685 / 0.6665 / 0.6734) | **+0.0002** (-0.0024 / -0.0010 / +0.0039) |
| B'(33) macroF1 | 0.9206 | 0.9209 | +0.0003 |
| B'(1) guard | 0.6017 | 0.6040 | +0.0023 |
| A, first 20 PFF, 4-fold | 0.5263 | 0.5266 | +0.0003 |
| A, 8 unseen PFF | 0.5263 | 0.5277 | +0.0014 |

Turnover recall (`agg --turnover`, B'(33) pooled, 7053 true H<->A changes): 0.518 (3653 matched, 6332 predicted); the default is
0.520 in #56, i.e. unchanged: the missed turnovers do not become visible through the pressure geometry either.

**Rejected**: B'(33) +0.0002 against the +0.010 bar (the seed spread alone is -0.0024 .. +0.0039).  Guards pass but are moot.
`configs/default.yaml` and `models/specialists*.json` unchanged; the flag stays false.
### Phase prior on the H-vs-A odds of hidden-ball frames (Issue #65; not adopted, OFF)

*Hypothesis.*  #43 only lowered the `D` logit under a live phase; the dominant error on hidden frames is the *team* of the
possession (turnover recall 0.51 at B'(33)), and the Smart Tagging phase prior (`w_phase = 0.3`, `phase_prior_home` in
[-1, 1]) is weak.  Giving it more weight only where the ball is hidden, and only on the `H`-vs-`A` log-odds, cannot touch the
`D` decision.

*Code* (learned route, `specialist_fusion.apply_priors`; default off, outputs and the synthetic SHA unchanged):
`heuristic_emissions.w_phase_hidden` adds `w * phase_prior_home` to `z = log(pH / pA)` on hidden-ball frames (optionally only
inside hidden runs of at least `phase_hidden_min_s`, `hidden_run_mask`); `p_dead = sigmoid(zd)` is exactly unchanged.
`auto_calibrate_phase_hidden: true` replaces the weight by a label-free estimate (`auto_phase_hidden_weight`): on the visible
frames where Specialist A is ultra-confident, `q` is the `|phase|`-weighted agreement of the prior sign with A's team, and
`w = clip(logit(q) - w_phase, 0, phase_hidden_cap = 2.5)` (0 with fewer than 250 usable frames); logged as
`report["phase_hidden"]` (weight, usable frames, agreement; aggregates only).  `scripts/decoder_sweep.py run --sim-noise
{medium,high}` simulates the tags of #43 per cell (seed = cell seed) and candidates accept `"config": {"section.key": v}`
overrides.  Tests: `tests/test_65_hidden_phase_prior.py`.

*Protocol* (Milestone 10, same models and harness as Item 8; tags simulated as in #43 / #49, no BAR-ATH data): B'(33) seeds
0-2, B'(1) seeds 0-2, A = PFF 4-fold (20 matches) and 8 unseen PFF.  Fixed beforehand: `w_phase_hidden` in {0.8, 1.5, 2.5, auto},
each with and without `transition_matrix.phase_cue`.  Baseline = same tags, weight 0.  Without tags the harness reproduces
the Milestone 10 baselines exactly (B'(33) 0.6693, B'(1) 0.6017, A 0.5263 / 0.5263).

spellF1 (d = difference to the same-tag baseline; B'(33) seed deltas 0 / 1 / 2):

| noise | variant | B'(33) | d (seeds) | B'(1) d | A 4-fold d | A unseen d |
|---|---|---|---|---|---|---|
| medium | baseline | 0.6610 | | 0.5942 | 0.5161 | 0.5193 |
| medium | cue only | 0.6610 | +0.0000 | +0.0003 | +0.0001 | +0.0001 |
| medium | 0.8 | 0.6619 | +0.0009 (+0.0023 / -0.0016 / +0.0019) | -0.0010 | -0.0069 | -0.0071 |
| medium | 1.5 | 0.6609 | -0.0000 (+0.0009 / -0.0030 / +0.0021) | -0.0041 | -0.0102 | -0.0103 |
| medium | 2.5 | 0.6585 | -0.0025 (-0.0023 / -0.0055 / +0.0004) | -0.0099 | -0.0163 | -0.0180 |
| medium | auto | 0.6585 | -0.0025 (same as 2.5: the estimate sits at the cap) | -0.0099 | -0.0161 | -0.0180 |
| high | baseline | 0.6589 | | 0.5917 | 0.5131 | 0.5174 |
| high | cue only | 0.6591 | +0.0002 | -0.0002 | -0.0003 | +0.0003 |
| high | 0.8 | 0.6577 | -0.0012 (-0.0012 / -0.0005 / -0.0018) | -0.0034 | -0.0129 | -0.0019 |
| high | 1.5 | 0.6569 | -0.0020 (-0.0016 / -0.0022 / -0.0022) | -0.0106 | -0.0213 | -0.0104 |
| high | 2.5 | 0.6516 | -0.0073 (-0.0087 / -0.0085 / -0.0047) | -0.0214 | -0.0336 | -0.0239 |
| high | auto | 0.6525 | -0.0064 (-0.0072 / -0.0076 / -0.0044) | -0.0181 | -0.0309 | -0.0219 |

The `+ phase_cue` rows differ from the rows above by at most 0.0005 in any world (the cue is neutral, as in #43).  Turnover recall (true `H<->A` changes matched, pooled; baseline at medium / high B'(33) 0.505 / 0.507,
B'(1) 0.408, PFF 0.347, unseen 0.373): it falls with the weight in every world (B'(33) 0.503 / 0.501 / 0.497 at 0.8 / 1.5 / 2.5,
PFF 0.328 / 0.315 / 0.309, unseen 0.362 / 0.352 / 0.332), i.e. the extra phase prior recovers none of the missed turnovers.

**Decision: not adopted** (`w_phase_hidden: 0.0`, `phase_hidden_min_s: 0.0`, `auto_calibrate_phase_hidden: false`).  The rule
(B'(33) >= +0.010 at medium **and** high with no seed below -0.003; B'(1) not down > 0.01; A not down > 0.03) fails: the best
B'(33) cell is +0.0009 (medium, 0.8) and every weight is negative at high noise; at 2.5 / auto B'(1) is down 0.010-0.021 and A
down 0.016-0.034.  The phases are coarse (the sampled tags jitter 3-6 s and 25-40 % are missing), and where the prior is
wrong it overrules the specialist's turnover evidence, so the `H<->A` boundaries get later and fewer.  The calibration is
agreement on *visible* frames, where the ball already decides the team, so it overestimates the value on hidden frames and
lands on the cap at medium noise.  Code, tests and the harness options stay; the real match was not touched.
### Tracklet-gated vision and emission tempering (Issue #66; not adopted, OFF)

Question left open by #63: if the `sv_ball640` detections are accepted only when they form a smooth run over several
consecutive crops (or if the specialists are less confident on frames whose ball was injected), is the loss of -0.074 gone
and does it turn into a gain?  Two OFF switches in the `vision` section (defaults change nothing; synthetic SHA256 unchanged):

- `vision.tracklet_gate` (`src/mtah2027/vision/tracklet_gate.py`, wired in `recover_ball` after `select_detections`): two
  detections of consecutive planned crops (<= `tracklet_max_link_frames` = 10 frames apart) are linked when the step speed is
  <= `tracklet_max_speed_mps` (35 m/s, plus 1 m of position slack); a run is kept only if it has >= `tracklet_min_len`
  detections (3; also tested 5) and every point is within `tracklet_max_residual_m` (1.0 m) of a least-squares fit of the run
  (constant velocity, quadratic from 6 points); the worst point is dropped and the run re-checked until stable.  Isolated
  detections never pass.
- `vision.injected_emission_scale` (default 1.0): multiplies the fused log-emissions by this factor on frames with
  `ball_source` 2 or 3 (video detection / interpolation); `run_core(..., ball_source=...)` receives the source from the
  recovered ball.  The tempered rows are rounded to 1e-6.

**Simulator with temporal structure** (`vision/sim_detector.simulate_detections_temporal`, `scripts/simulate_tracklet_gate_value.py`).
The per-crop marginals are exactly those of #63 (top candidate within 1 / 2 / 5 m 41.0 / 59.7 / 75.7 %, 19.3 % false positive,
5 % no detection), but the state of consecutive planned crops (every 5 frames of a gap) is a Markov chain whose stationary
distribution is that mixture.  Assumptions, none measured (the real video was only scored per frame, #63): a true hit is kept
by the next crop with probability 0.6 (position noise independent per crop, drawn from the 1 / 2 / 5 m bins: the "noise of
about 1 m that persists over consecutive crops" is the persistence of the *hit*, not of its error); a false positive persists
with probability 0.2 and, when it does, it is a moving object (a boot, a line mark) that keeps its identity and drifts with 3 m/s
Gaussian steps, so a persistent false positive looks like a smooth track and passes the gate; a missing detection persists with
0.3.  Sensitivity variant `pf`: false positives persist with 0.6.  The detection-rate scenarios (25 / 50 / 100 %) then remove
crops independently, as in #63.  The `iid` row is the #63 generator and reproduces its numbers exactly (B'(33) -0.0744, B'(1)
-0.0226 at 50 %, same seeds), which also checks that the `ball_source` plumbing with scale 1.0 changes nothing.

Protocol as #63: production specialist route (`lomo_gap` models, default config otherwise), B'(33) seeds 0-2 (7 IDSSE
matches each) and B'(1) seed 0, spellF1 against the no-injection default (B'(33) 0.6693, B'(1) 0.6019 for seed 0).

B'(33), scenario 50 % (d = spellF1 minus no injection; "kept" = share of the returned points that pass the gate; error after
gating = median distance to the true ball of the kept points and share of kept points more than 5 m off):

| scenario | spellF1 | d | per seed d (0 / 1 / 2) | hidden frames filled | points kept | kept <= 1 m | kept > 5 m | median error kept |
|---|---|---|---|---|---|---|---|---|
| none | 0.6693 | | | 0 % | | | | |
| #63 iid detector, no gate | 0.5950 | -0.0744 | -0.0694 / -0.0762 / -0.0775 | 76 % | 100 % | 43 % | 20 % | 1.35 m |
| temporal, no gate | 0.5891 | -0.0803 | -0.0800 / -0.0838 / -0.0771 | 77 % | 100 % | 43 % | 20 % | 1.35 m |
| temporal, gate 3 | 0.6645 | -0.0048 | -0.0077 / -0.0070 / +0.0003 | 14 % | 25 % | 70 % | 3.0 % | 0.70 m |
| temporal, gate 5 | 0.6698 | +0.0005 | +0.0012 / -0.0002 / +0.0004 | 3 % | 6 % | 78 % | 0.5 % | 0.63 m |
| temporal, scale 0.5 | 0.6046 | -0.0647 | -0.0649 / -0.0692 / -0.0600 | 77 % | 100 % | 43 % | 20 % | 1.35 m |
| temporal, gate 3 + scale 0.5 | 0.6655 | -0.0038 | -0.0059 / -0.0067 / +0.0011 | 14 % | 25 % | 70 % | 3.0 % | 0.70 m |
| persistent FP, no gate | 0.5954 | -0.0739 | -0.0768 / -0.0780 / -0.0671 | 81 % | 100 % | 43 % | 20 % | 1.34 m |
| persistent FP, gate 3 | 0.6603 | -0.0090 | -0.0104 / -0.0119 / -0.0046 | 19 % | 32 % | 59 % | 19 % | 0.84 m |
| persistent FP, gate 5 | 0.6692 | -0.0001 | -0.0005 / -0.0031 / +0.0031 | 5 % | 9 % | 65 % | 16 % | 0.74 m |

Other detection rates (temporal detector):

| scenario | B'(33) d | B'(1) d | points kept (B'(33)) |
|---|---|---|---|
| 25 %, no gate | -0.0413 | -0.0102 | 100 % |
| 25 %, gate 3 | -0.0012 | -0.0008 | 10 % |
| 25 %, gate 3 + scale 0.5 | -0.0009 | -0.0007 | 10 % |
| 100 %, no gate | -0.0910 | -0.0328 | 100 % |
| 100 %, gate 3 | -0.0272 | -0.0004 | 43 % |
| 100 %, gate 3 + scale 0.5 | -0.0214 | +0.0062 | 43 % |

B'(1) at 50 % (seed 0): no gate -0.0209, gate 3 +0.0089, gate 5 +0.0018, scale 0.5 alone -0.0101, gate 3 + scale 0.5 +0.0106,
persistent FP gate 3 +0.0028 / gate 5 -0.0030.  It is a guard (one seed): it never drops by more than 0.01 with a gate, but it
is not the adoption criterion.

**Decision: not adopted.**  Adoption rule: B'(33) mean >= +0.010 with no seed below -0.003, B'(1) not down more than 0.01.
At 50 % no variant gets near it: the best are gate 5 (+0.0005, one seed at -0.0002) and gate 3 (-0.0048, two seeds below
-0.003).  The gate does what was hoped: it removes the false positives (kept points more than 5 m off fall from 20 % to 3 %
and the median error from 1.35 m to 0.70 m; with persistent false positives 19 %, so the gate is only as good as their
isolation) and the loss of -0.074 shrinks to about -0.005 or 0.  But it does so by discarding 75-94 % of the points, so what
remains is little information: the hidden-ball gap is still decided by the tracking-free specialists, and the ~0.7 m median
error of the kept points (hits with ~1 m noise) is not enough to move spellF1 above the no-injection baseline.  Tempering
the emissions alone (scale 0.5) recovers only 0.015 of the 0.080 lost, because the wrong positions still feed the ball
features of the specialists; combined with the gate it is within seed noise of the gate alone (+0.001).  The loss is
monotone in the number of injected points (100 % hurts most).  The real runtime stays what #63 measured (about 4.5 h of CPU
for `sv_ball640` at one crop every 5 hidden frames, an AGPL-3.0 ultralytics runtime and the weights / dataset-terms caveat),
so even a small simulated gain would not have justified enabling it.  `vision.enabled` and `vision.tracklet_gate` stay
`false`, `vision.injected_emission_scale` stays 1.0.

Reproduce: `python scripts/simulate_tracklet_gate_value.py run --world b33 --seeds 0 1 2 --models-root <lomo models> --out-dir D`
(and `--world b1 --seeds 0`), then `agg --out-dir D`.

## Milestone 11: remaining margin in tactical video

### Oracle ceiling of the hidden-ball value, accuracy curve, error decomposition and soft evidence (Issue #67; diagnostic, everything OFF)

Question: how much spellF1 is left in the hidden-ball frames, as an upper bound for any video ball detector, and how accurate
and clean must a detector be to reach +0.010?  External data only (never BAR-ATH): B'(33) = 7 IDSSE leave-one-match-out with the
calibrated degrader at `dead_affinity` 33 (seeds 0-2), guard B'(1) (seed 0), production specialist route (`lomo_gap` models,
default config), official metric.  `scripts/oracle_ceiling.py` extends the machinery of #63 / #66; the `none` rows reproduce the
references exactly (B'(33) 0.6693 = 0.6709 / 0.6675 / 0.6695, B'(1) 0.6019).  Hidden frame = graded frame without ball in the
degraded match; "restored" frames are written through the same `inject_points` as the real chain (cadence `k`: one point
every `vision.every_k` = 5 frames of each gap, interpolated in between) or at every hidden frame (cadence `all`).  Coverage is
nested per seed.  Run time: 69 scenarios on each seed-0 cell, 16 key rows on seeds 1-2; B'(1) has 13 key rows (seed 0); the 4
refinement rows (sigma 0.2 / false positives 2 and 5 %) and all soft rows are seed 0 only (stated per row).

**1. Oracle (true ball restored in the hidden frames).**  B'(33), d = spellF1 minus `none`:

| coverage | cadence k=5: spellF1 | d | macroF1 d | per seed d (0 / 1 / 2) | every frame: spellF1 | d | per seed d |
|---|---|---|---|---|---|---|---|
| 100 % | 0.7104 | +0.0410 | +0.0188 | +0.0392 / +0.0406 / +0.0434 | 0.7154 | +0.0460 | +0.0452 / +0.0451 / +0.0478 |
| 75 % (seed 0) | 0.7067 | +0.0358 | +0.0171 | | 0.7166 | +0.0457 | |
| 50 % | 0.6977 | +0.0283 | +0.0155 | +0.0262 / +0.0278 / +0.0310 | 0.7123 | +0.0430 | +0.0430 / +0.0409 / +0.0450 |
| 25 % | 0.6846 | +0.0153 | +0.0096 | +0.0143 / +0.0118 / +0.0198 | 0.7074 (seed 0) | +0.0365 | |

B'(1) (seed 0, guard): 100 % k +0.1085, every frame +0.1128; 50 % k +0.0701, every frame +0.1071; 25 % k +0.0336.  So a perfect
detector would be worth +0.041 to +0.046 on B'(33) (about 0.04 above the noise of spellF1) and +0.11 on B'(1), where the holes
are mostly live-ball.  The cadence costs little at 100 % (-0.005) but matters at low coverage (the interpolation fills 49 % of the
hidden frames at 25 % / k against 99.8 % at every frame).  Even with the true ball in every hidden frame the score stays
0.715: the hidden frames are not where most of the spellF1 is lost (item 3).

**2. Accuracy curve (true ball + isotropic Gaussian error sigma + false positives uniform 5-30 m away, injected as the ball,
cadence k).**  B'(33) d; cells with 3 seeds give the mean, the others are seed 0 only (`s0`):

| sigma | fp 0 %, cov 100 | fp 0 %, cov 50 | fp 10 %, cov 100 | fp 10 %, cov 50 | fp 25 %, cov 100 | fp 25 %, cov 50 |
|---|---|---|---|---|---|---|
| 0.1 m | +0.0229 | +0.0251 | -0.0188 (s0) | -0.0070 (s0) | -0.0607 (s0) | -0.0450 (s0) |
| 0.3 m | -0.0119 (s0) | -0.0026 | -0.0407 (s0) | -0.0279 (s0) | -0.0711 (s0) | -0.0576 (s0) |
| 0.5 m | -0.0320 | -0.0235 | -0.0528 (s0) | -0.0462 (s0) | -0.0790 (s0) | -0.0690 (s0) |
| 1 m | -0.0648 (s0) | -0.0533 | -0.0745 | -0.0632 | -0.0986 (s0) | -0.0796 (s0) |
| 2 m | -0.0922 (s0) | -0.0866 | -0.1017 (s0) | -0.0911 (s0) | -0.1172 (s0) | -0.0993 |

Refinement (cov 50 %, seed 0): sigma 0.2 m, fp 0 -> +0.0113; sigma 0.1 m, fp 2 % -> +0.0151; sigma 0.1 m, fp 5 % -> +0.0061;
sigma 0.2 m, fp 2 % -> +0.0020.  B'(1) guard (seed 0): sigma 0.5 m, fp 0, cov 100 +0.0317; sigma 0.3 m, fp 0, cov 50 +0.0471;
sigma 1 m + fp 10 % at 50 % -0.0061 (at 100 % -0.0254).

**Crossing +0.010 on B'(33): a detector has to put the ball within about 0.2 m (1 sigma) in the covered frames AND have fewer than
about 2-3 % false positives; 50 % coverage is enough (the loss with sigma is steeper than the gain from coverage).**  The
measured `sv_ball640` (41 % within 1 m, 24 % beyond 5 m or missing) sits in the sigma >= 1 m, fp ~ 20 % region of the table,
between -0.06 and -0.10, consistent with #63 (-0.074).  Accuracy is not the whole story: a 0.5 m error with no false positives
already loses -0.024 (50 %), because the injected ball feeds the ball-present specialist, which was trained on tracking-precision
positions and reads contact / carrier geometry at sub-metre scale.

**3. Error decomposition of the current route on B'(33)** (mean of 21 cells; official `possession_metrics` functions; hidden =
graded frames without ball, 41 % of the graded frames).  "fix hidden / fix visible" = spellF1 if the prediction were replaced by
the truth in the hidden (resp. visible) graded frames (not additive, a bound on where the loss lives):

| | B'(33) | B'(1) (7 cells, seed 0) |
|---|---|---|
| spellF1 now | 0.6693 | 0.6019 |
| fix hidden frames only | 0.7574 (+0.088) | 0.8367 (+0.235) |
| fix visible frames only | 0.9215 (+0.252) | 0.7774 (+0.176) |
| frame accuracy hidden / visible | 0.936 / 0.929 | 0.841 / 0.952 |
| recall of H / A in hidden frames | 0.756 / 0.727 | 0.758 / 0.767 |
| recall of H / A in visible frames | 0.913 / 0.914 | 0.922 / 0.917 |
| recall of D in hidden / visible | 0.983 / 0.981 | 0.949 / 0.986 |
| possessions: truth / predicted | 437 / 350 | 437 / 313 |

In B'(33) the hidden holes sit largely in dead-ball stretches (`dead_affinity` 33), so most of what the current route loses is
in frames where the ball IS visible.  Fate of the 437 true possessions at IoU 0.5 (B'(33); in brackets those mostly in hidden
frames): matched 268 (23), merged into a predicted spell that also covers a neighbour 50 (10), boundary error 37 (8), split into
>= 2 predicted spells 3 (1), no same-team overlap (missed) 80 (26).  Possession count errors are almost all merges / misses
(the route predicts 350 spells against 437), not splits: it under-segments.  H<->A turnovers (matched within +-1 s): 336 true,
174 matched, 161 missed (39 of them in hidden frames), 75 spurious; stops (to D): 99 true, 17 missed, 16 spurious; restarts
(from D): 99 true, 17 missed, 16 spurious.  Median absolute latency of the matched boundaries: 1.4 frames for H<->A and for stops,
0.9 for restarts, i.e. boundary timing is not the problem; the missing turnovers are.  With the oracle ball in every hidden
frame (`orc100_all`) hidden-frame accuracy rises 0.936 -> 0.982, matched possessions 268 -> 293, missed turnovers in hidden
frames 39 -> 21, but missed turnovers in visible frames stay at about 121 (total missed 161 -> 142) and spellF1 stops at
0.715.  On B'(1) the picture is the opposite (live-ball holes: 158 of the 336 turnovers are in hidden frames, 120 of the 196 missed
ones; fixing the hidden frames would give +0.235, the oracle ball +0.113).

**4. Soft evidence** (`vision.soft_evidence_weight`, default 0 = off; `src/mtah2027/vision/soft_evidence.py`, wired in
`run_core(..., soft_evidence=(frames, xy, conf))`).  The detection does not become the ball (the ball features stay empty); it
only tilts the H / A emission by `weight * conf * tanh((d_A - d_H) / 3 m)` nats (d = distance of the detection to the nearest
player of each team, applied within 5 frames of the detection; D untouched).  Same detections and error models as item 2, conf = 1,
seed 0, B'(33) d:

| detections | w = 1 | w = 2 | w = 4 |
|---|---|---|---|
| perfect, cov 100 | +0.0014 | -0.0020 | -0.0192 |
| perfect, cov 50 | +0.0017 | -0.0007 | -0.0141 |
| sigma 0.5 m, fp 10 %, cov 100 | +0.0029 | -0.0029 | -0.0203 |
| sigma 0.5 m, fp 10 %, cov 50 | +0.0031 | -0.0007 | -0.0149 |
| sigma 1 m, fp 10 %, cov 50 | +0.0025 | -0.0019 | -0.0168 |
| sigma 2 m, fp 25 %, cov 100 | +0.0015 | -0.0030 | -0.0241 |
| sigma 2 m, fp 25 %, cov 50 | +0.0024 | -0.0039 | -0.0205 |

(sigma 0.3 m rows equal sigma 0 within 0.0005.)  B'(1) (seed 0): w = 1, perfect, cov 100 +0.0050; sigma 1 m, fp 10 %, cov 50 +0.0030.
Soft evidence is far more tolerant of detector error than hard injection (it never loses more than 0.004 at w <= 2 with 2 m / 25 %
false positives, where injection loses 0.10), but it also gains nothing: the best row is +0.0031, because a nearest-team tilt only
repeats what the specialists already infer from the player geometry.  It is not a route to +0.010.

**Conclusions.**  (a) The ceiling of a perfect hidden-ball detector is +0.041 (k = 5) to +0.046 (every frame) on B'(33) and +0.11 on
B'(1); 25 % coverage already gives +0.015.  (b) Reaching +0.010 needs a median error <= ~0.2 m and < ~3 % false positives; a
detector of the `sv_ball640` class (1 m, ~20 % far) cannot, and gating (#66) cannot fix that without a much more precise detector.
(c) The main spellF1 losses of the current route on B'(33) are under-segmentation (350 predicted spells against 437) and missed
H<->A turnovers in frames where the ball is visible (about 120 of 336 per match), not boundary latency and not the hidden-ball
frames; the oracle leaves 0.715.  On B'(1)-like holes (live-ball occlusions) the video would matter much more.  (d) Soft evidence is
safe but worth about +0.002.  Nothing is enabled: `vision.soft_evidence_weight` stays 0.0; the synthetic SHA256 is unchanged
(2a31a913...).

Reproduce: `python scripts/oracle_ceiling.py run --world b33 --seeds 0 1 2 --models-root <lomo models> --out-dir D` (and `--world b1
--seeds 0 --rows <list>`), then `agg --out-dir D [D2 ...] --json reports/issue67_oracle_ceiling.json`.

### Self-supervised ball detector on the GPU (Issue #68; aggregates only, not adopted, OFF)

Question: can a detector fine-tuned on the match video itself (labels from the automatic tracking, no manual labels)
reach the ~0.3 m top-candidate accuracy that #63's simulation says the possession model needs?

Pipeline (all local; separate vision environment outside OneDrive: Python 3.11, torch 2.6.0+cu124, torchvision 0.21.0+cu124,
ultralytics 8.4.212 (AGPL-3.0, optional), opencv 5.0.0.93; pins in `requirements-vision.txt`, RTX 3060 12 GB):

1. `scripts/build_ball_dataset.py` (`vision/ball_dataset.py`): eligible frames = graded, ATD ball visible, usable homography,
   projection inside the image (91 247).  Temporal split: 2-min blocks, every 5th block validation, 10 s guard band (train
   frames near validation blocks dropped).  One frame in 4: 17 639 train positives (640x640 JPEG, the ball at a uniformly random
   position of the window) + 1 579 ball-free crops; 4 500 validation frames x 2 lossless crops: `atd` (window around the ATD ball
   plus a random offset <= 3 m, the ceiling) and `prior` (search point that does NOT use the ball: interpolation of a simulated gap
   whose length is drawn from the real tracking gaps, pulled to the pressure centroid as in `attention_crops`; the ball is inside
   the window in 70.5 % of them).  749 validation frames are within 2 s of a tracking gap.  25 min to build.
2. Label quality (aggregate check): the ATD ground projection is not pixel-aligned with the ball seen in the video (the ball is
   above the ground plane and the tracking has latency/noise); a first fine-tune on the raw ATD pixels did not learn at all
   (classification loss flat at the trivial value, mAP50 ~0.02; about 1.4 h of GPU discarded).  `scripts/refine_ball_labels.py`
   therefore relabels by agreement of two independent algorithmic sources: the public roboflow `football-ball-detection.pt`
   (YOLOv8x, run locally as teacher) must have a candidate of confidence >= 0.4 within 40 px of the ATD pixel; the label is the teacher
   box.  70.1 % of the positives agree (12 361 kept; median teacher-ATD distance 9.8 px, p90 28 px; the vertical bias is
   systematic: teacher median 3.3 px above the ATD point).  Validation labels/metrics stay ATD-based.
3. Training: yolov8s (COCO weights), imgsz 960, batch 16, AdamW lr 7e-4, no mosaic, 12 epochs = 75.5 min (7.5 min/epoch once the
   machine was not CPU-starved).  The weights are not committed (`*.pt` is gitignored); they are reproduced by the three scripts.
   mAP against the ATD boxes is uninformative (stays <= 0.01 because of the ATD offset), so epochs were judged by the metre metrics.

Result on the validation blocks (top-confidence candidate vs the ATD ball, pitch metres via H^-1; `reports/issue68_detector_profile.json`):

| detector / crops | within 0.3 / 0.5 / 1 / 2 / 5 m | median / p90 (m) | FP per crop (>5 m) | with detection |
|---|---|---|---|---|
| `sv_ball640` teacher, `atd` crops (4 500) | 14.1 / 26.0 / 44.2 / 60.4 / 75.6 % | 1.20 / 14.0 | 1.55 | 98.1 % |
| fine-tuned s960, `atd` crops | 14.6 / 27.0 / 46.3 / 63.2 / 78.5 % | 1.00 / 8.5 | 0.22 | 92.4 % |
| fine-tuned s960, `atd` near-gap (749) | 17.1 / 27.8 / 44.7 / 56.9 / 79.8 % | 1.12 / 6.8 | 0.23 | 93.9 % |
| `sv_ball640` teacher, `prior` crops containing the ball (3 171) | 15.2 / 27.6 / 45.4 / 61.1 / 76.4 % | 1.14 / 14.3 | 1.48 | 97.5 % |
| fine-tuned s960, `prior` crops | 15.8 / 28.5 / 47.6 / 64.2 / 79.8 % | 0.94 / 7.2 | 0.20 | 92.0 % |
| fine-tuned s960, `prior` near-gap (494) | 17.4 / 27.9 / 46.4 / 58.3 / 82.4 % | 1.06 / 5.6 | 0.20 | 93.7 % |

Inference at imgsz 1280 is no better (prior: 16.0 / 28.9 / 48.0 / 64.5 / 80.5 %, FP 0.31).  Crops centred on a prior that does not use the
ball behave like the ceiling crops (the detector is robust to the search point as long as the ball is inside the window, 70.5 % here).
The detector confidence is not informative: the share of correct top candidates is the same at every confidence gate (conf >= 0.7 still
keeps 88 % of the frames, with 30 % within 0.5 m).  Top-candidate error histogram (bins <= 0.15 / 0.3 / 0.5 / 1 / 2 / 5 / 10 / >10 m, atd
crops): 5.6 / 9.0 / 12.5 / 19.3 / 16.8 / 15.3 / 5.8 / 15.6 %.

Cost: 30.2 crops/s on the RTX 3060 at imgsz 960 (teacher 15 crops/s while sharing the GPU).  The plan of #63 (one crop every 5 frames of the
64 079 hidden graded frames = 12 687 crops) takes 7 min of GPU inference plus decoding; every hidden frame 35 min.  Decoding the tactical video
runs at ~200 frames/s.

**Conclusion: not adopted, no change to the submission route** (default pipeline and synthetic SHA unchanged; `vision.enabled` stays `false`).
Self-supervised fine-tuning removes most false positives (1.5 -> 0.2 per crop) and shifts 1-2 points at every radius, but the top candidate is within 0.3 m in
only ~15 % of the frames (the #63 simulation needs "most"), and the error profile (about 50 % within 1 m, 16 % beyond 10 m) is a clear
improvement over #58/#63 only in the tail.  The ceiling is set by the labels: the ATD ground projection and an image ball (aerial, above the
ground plane) differ by ~0.5-1 m, so a metre-accurate evaluation against the ATD needs a height model (`vision/aerial_filter.py`) that is not
used here.  The profile is exported for the simulation of #69.  `ft_ball.FineTunedBallDetector` (same interface as the other detectors, GPU, lazy
torch/ultralytics import) is available for running over the gap frames.

Reproduce (vision venv): `build_ball_dataset.py --data-dir <data> --out <cache>/ds`, `refine_ball_labels.py --ds <cache>/ds --out <cache>/ds_ref --teacher
<football-ball-detection.pt>`, `train_ball_detector.py --ds <cache>/ds --data <cache>/ds_ref/data.yaml --weights yolov8s.pt --imgsz 960 --epochs 12 --hours 1.7`,
`eval_ball_detector.py --ds <cache>/ds --model ft=<last.pt>:960 --model sv=<teacher.pt>:640 --json reports/issue68_detector_profile.json`.

### Integration decision for the self-supervised ball detector (Issue #69; decision only, not adopted, nothing wired)

Plan of #69: simulate the value of the #68 detector in B'(33) / B'(1) with its measured error profile (hard injection, tracklet filter of #66, soft
evidence of #67) and wire it behind `vision.*` only if a variant met the adoption rule (B'(33) >= +0.010, no seed below -0.003, B'(1) and A guards).

Decision: **not adopted, no code and no change to the submission route.**  The measured profile of the fine-tuned detector (#68) has the top
candidate within 0.3 m of the tracking ball in only ~15 % of the frames (about 50 % within 1 m, 16 % beyond 10 m), while the oracle ceiling of #67 and the
simulation of #63 / #66 need about 0.2 m accuracy before the injected ball helps more than it hurts; the detector confidence is uninformative (the
share of correct candidates does not rise with the confidence gate), so a gate cannot select the good detections.  Hard injection at this accuracy
was already negative in the #63 / #66 simulations, and the tracklet filter of #66 discards most of the points.  Running the simulation would only
re-confirm a value known to be below the rule, and the real run (GPU decoding of the tactical video) adds cost, an AGPL-3.0 dependency
(ultralytics) and a model file that cannot be committed.  `vision.enabled` stays `false`; the sequence model of #72 is the milestone's adopted change.

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
python scripts/generate_submission.py --data-dir data --team "Opus Operandi"
python evaluation/validate_submission.py "outputs/Opus Operandi.csv"
```

All dependencies are pinned (`requirements.txt`, `pyproject.toml`), seeds are fixed (`seed: 0`),
`PYTHONHASHSEED=0` is enforced, every decoder input is rounded to 1e-6 and ties are broken by state index, so
two runs (and the numba and fallback decoders) give the same SHA256.  CI repeats the run on the synthetic
full-length match twice and compares the hashes.  Full repository layout and commands: `README.md`.
