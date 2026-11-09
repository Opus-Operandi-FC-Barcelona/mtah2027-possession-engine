# Technical report -- Opus Operandi (More than a Hack 2027, phase 1: ball possession)

Repository: `Opus-Operandi-FC-Barcelona/mtah2027-possession-engine`.  This report follows the points requested
in the competition's `SUBMISSION.md`.  **Implemented in this version = wave 1** (a complete, deterministic,
heuristic end-to-end route).  Everything marked *planned* belongs to waves 2/3 and is not part of the
submitted file unless the tracker (`docs/ISSUES_TRACKER.md`) says it is completed.

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
5. **Emissions** `log_probs[N, 3]`: calibrated-ish `p(D)` from the dead score and an H-vs-A logit dominated by the
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
| Tactical homography | loader implemented (#7), **not used** in wave 1 | needed for the planned attention crops |
| Broadcast ATD / homography | **not used** in wave 1 | planned sync (#10, wave 3) to inject a ball where the tactical one is missing |
| Videos (tactical, broadcast) | **not used** | planned YOLO attention crops (#11, wave 3) |
| Official scorer / `games.py` / starter | **used (vendored, unmodified)** | metric, constants, format check |
| Public 5-minute ground truth | **not used for any decision** | only reserved as a format/alignment smoke test |

## 3. Training data

* **No learned model is trained in wave 1.**  All parameters are generic football priors in
  `configs/default.yaml`.
* **Nothing from the provided match is used for fitting**, and the public 5 minutes are never used to choose
  between variants.  Hyper-parameters were examined only on **open tracking data of other matches** (IDSSE
  Bundesliga and Metrica Sports sample games 1-3, with possession labels derived algorithmically from their
  events) kept in the sister repository `mtah2027-external-benchmarks` -- those files are not redistributed here.
* A deterministic **synthetic match** (`scripts/generate_synthetic_fixtures.py`, fixed seed) in the exact file
  layout of the real bundle is used for tests, CI and the end-to-end determinism check.
* *Planned (wave 2)*: a self-supervised specialist trained on high-confidence segments of the match itself with
  block masking (the ball is hidden artificially, ground truth comes from the confident ball-visible segments),
  plus the external matches degraded to the empirical gap structure.  No manual labelling is involved.

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
| 7 IDSSE matches, complete ball | **0.691** | 0.934 | 0.310 | 0.394 |
| 7 IDSSE matches, 41% of the ball hidden in random 5-60 s blocks (seed 0) | **0.487** | 0.814 | 0.218 | 0.453 |
| 3 Metrica sample games (their own ball channel is missing exactly while the ball is dead, which leaks `D`) | 0.778 | 0.889 | 0.437 | 0.833 |
| all 10 matches as they are | 0.717 | 0.920 | 0.348 | 0.526 |
| all 10 matches, 41% of the ball additionally hidden | 0.505 | 0.802 | 0.247 | 0.517 |

Diagnostic on two IDSSE matches (complete ball): giving the engine the ground-truth team with its own `D` raises
spellF1 to ~0.96 (to ~0.87-0.89 with 41% of the ball hidden), whereas giving it the ground-truth `D` with its
own team changes almost nothing (0.64-0.73 clean, ~0.50 hidden).  Team assignment, not dead-ball detection, limits
the score, and under occlusion it is the part that degrades.

**Synthetic match** (fixed seed 0, ground truth known; the generator hides 41% of the ball during alive play):

| Match | engine spellF1 / macroF1 | possessions (pred/gt) | nearest-player +-50 baseline spellF1 / macroF1 |
|---|---|---|---|
| short, 12,000 frames | 0.545 / 0.716 | 24 / 23 | 0.458 / 0.479 |
| full length, 199,319 frames | 0.564 / 0.762 | 346 / 293 | 0.434 / 0.532 |

**Runtime** of the full-length synthetic end-to-end run (`generate_submission.py --synthetic`, generation of the
match included, 16 cores laptop): 35 s with numba, 35 s with the pure-Python fallback decoder (identical SHA256);
the engine itself (loading, features, decoding, post-processing) takes ~18 s.  Viterbi: 0.03 s (numba), 0.6 s
(fallback).

The published floor for the real match is `baseline_smooth50` spellF1 0.2433; the real-match score of this
version is not known to the authors at the time of writing (the data is processed only locally).

## 6. Limitations (honest list)

* **Ball occlusions** (41% of graded frames) are handled only by carrying the last/next touch and by player
  speeds for `D`; real possession changes inside an occlusion are located at the middle of the gap.  This is the
  largest source of lost spellF1 (see section 5) and the main target of the planned specialist B.
* Touch detection depends on the ball track quality; noisy or interpolated ball positions produce missed or
  spurious touches.  Aerial balls are not modelled (`is_aerial` is all false until Issue #12).
* The throw-in team resolver (#15) and collective-physics / ball-memory features (#17, #18) are planned; throw-ins
  with `Team = N/A` are anchored in time but their team is inferred from the first touch only.
* Parameters are generic priors validated on other competitions' data; the tracking noise of BAR-ATH may be
  different.  No parameter was fitted to BAR-ATH.
* The KPI dashboard (#27, planned) will flag sanity alarms; in wave 1 only aggregate statistics are printed.

## 7. Reproduction

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt && pip install -e . --no-deps       # numba==0.68.0 optional
python scripts/generate_submission.py --data-dir data --team OpusOperandi
python evaluation/validate_submission.py outputs/OpusOperandi.csv
```

All dependencies are pinned (`requirements.txt`, `pyproject.toml`), seeds are fixed (`seed: 0`),
`PYTHONHASHSEED=0` is enforced, every decoder input is rounded to 1e-6 and ties are broken by state index, so
two runs (and the numba and fallback decoders) give the same SHA256.  CI repeats the run on the synthetic
full-length match twice and compares the hashes.  Full repository layout and commands: `README.md`.
