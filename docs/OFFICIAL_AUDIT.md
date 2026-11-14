# Audit of the engine against the OFFICIAL criteria (More than a Hack 2027, phase 1)

Auditor: read-only review. Date of review: 2026-11-02. Nothing in this file was committed.

## 0. Scope, rules observed, and what was (not) touched

* Read in full: `docs/official/*.md` (README, OVERVIEW, CHALLENGE, DATA, EVALUATION, FAQ, RULES, SUBMISSION),
  `games.py`, `evaluation/possession_metrics.py` (spell/fill/segment logic), `evaluation/run_evaluation.py`,
  `evaluation/validate_submission.py`, `starter/*.py` (baseline, load_atd, load_smart_tagging), and our
  `README.md`, `docs/TECHNICAL_REPORT.md`, `docs/OpusOperandi.md`, `configs/default.yaml` and the src modules cited below.
* Never opened: `data/`, `cache/`, `outputs/`, videos, any prediction CSV (including `evaluation/results/*.csv`).
* Public ground truth: I did **not** open `evaluation/gt/*.csv` and did not run the scorer. In section 2 I used
  one set of numbers that is **derived from the public 5-minute GT**: the 4x4 confusion matrix already stored by the
  official scorer in `evaluation/reports/baseline_raw.json` / `baseline_smooth50.json` (aggregate counts, produced
  earlier by `run_evaluation.py`). I used it **only to understand the label semantics of ball-hidden vs D** (the use
  CHALLENGE.md invites). It is n = 5 minutes, 17 possessions, the first minutes of the match; it must **not** be
  used to choose models or parameters (RULES.md, EVALUATION.md "public window is not a leaderboard"). I recommend
  nobody copies those numbers into configs.
* All metric experiments below use the official functions (`possession_metrics.py`) on synthetic arrays that I
  generated (`/tmp/aud/metric_demo.py`, not in the repo), never on BAR-ATH.

## 0.1 Upstream check: vendored copy vs https://github.com/metrica-sports/more-than-a-hack-2027

Cloned HEAD (commit `f8f2629`, 2026-10-23, "Clarifying the guidelines for annotating the provided video").
History: initial commit 2026-09-24, docs fixes 2026-10-05/15, `c073f29` (2026-10-22) and `f8f2629` (2026-10-23) are the
annotation clarifications.

* **Byte-for-byte diff (CRLF-normalised) of all 8 docs, `games.py`, `possession_metrics.py`, `run_evaluation.py`,
  `validate_submission.py`, `starter/{baseline,load_atd,load_smart_tagging,load_homography}.py`: NO differences.** The
  vendored copy is current as of today. (Notebooks `starter/0*.ipynb`, `utils.py`, `make_dummy_submission.py`,
  `requirements*.txt`, `CITATION.cff` exist upstream; not diffed, not relevant to scoring.)
* No discussions feature content found; the organiser answers are in the GitHub **issues** (18 open/closed). Relevant:

| # | state | Answer / content | Impact on us |
|---|---|---|---|
| 2 | closed | Weights of a network are fine **if trained on other data**, source documented; training on BAR-ATH (even using the provided tracking as labels for a video detector) is **not** allowed; runtime is not a requirement; GPU OK. | Our video stages are off and untrained; fine. |
| 8 | closed | Line = *which game*: no hand labels on BAR-ATH for any purpose; unlimited hand labels on other games, with access + documentation. | OK. |
| 10 | closed | Choosing/optimising **parameter values** against hand labels of BAR-ATH is **not** allowed ("fits the model to that match"). Fallback = values set by hand before labelling, stated in the doc. | Our hyperparameters come from other matches; but see section 6/compliance note on `external_degrader_target`. |
| 11 | closed | Training need not be exactly reproducible; **submission must be byte-for-byte** from the committed weights, CPU inference, pinned deps, fixed seeds. | We commit `models/specialists*.json`; **default `--specialists auto` retrains at run time if the benchmark cache is on disk** (see 0.2). |
| 14 | closed | Committing intermediate outputs with script/model/settings: yes. Watching video to understand: yes; turning it into labels/corrections/**rules and constants tied to specific frames**: no. | OK; audit video policy in report is consistent. |
| 15 | closed | Post-match task: using future frames is allowed. | Our non-causal windows are fine. |
| 16 | closed | IDSSE (CC BY 4.0) explicitly allowed; **cite the dataset**; keep all hyperparameters chosen on other games. | `OpusOperandi.md` cites Bassek et al. 2025, CC BY 4.0. Keep the citation in the PDF. |
| 17 | **open** | A team points out that in the **public GT** a Barca *won header* with no further control is labelled `H` for 1.32 s between two `A` spells (frames 15766-15798), contradicting the "aerial duel won without control" rule; organisers have not answered. | Important semantic evidence, see section 5. |
| 20, 22, 25, 26, 27, 28, 29 | open | Validation-only hand labels (unanswered; by #10 surely no), cloud GPU inference, GPU determinism, "is checking output against the video allowed", requests for another match from the same pipeline (#27, #28), running time when hosts re-run. | If a second Metrica-pipeline match is released (#28), it would be by far the best calibration source for gap placement; watch the repo until 17/11. |

## 0.2 Compliance-level observations (not about score)

1. **Reproduction path.** `README.md` / `scripts/generate_submission.py:53`: `--specialists auto` "trains at run time if the
   external matches are on disk, otherwise loads the committed file". The CSV a reviewer reproduces (no external cache)
   comes from `models/specialists.json`; the CSV we upload, if made with the sibling benchmark cache present, comes from
   run-time training. Training is declared deterministic, but issue #11 asks for byte equality *with the committed
   weights*. **Generate the final CSV with `--specialists load` (and an explicit `--model-variant`)** and put that exact
   command in the README. Cheap, removes a whole class of risk.
2. **Docs not final.** `docs/OpusOperandi.md` section 6 still contains "placeholders ... pending" for #42-#46 and says
   "in progress". `README.md` still says specialists "keep the shared defaults (+0.0195, under the +0.02 rule)" while the
   report (5.x item 1) says the overrides were adopted. `OpusOperandi.md` says micro-possessions "< 1.2 s" are absorbed but
   the active `route_overrides.specialists.spell_optimizer.micro_poss_s` is **0.9847** (`configs/default.yaml:~530`) and
   the heuristic route's is 1.7542. SUBMISSION.md asks the document to state "the possession rules you implemented";
   state the numbers actually in force.
3. **Accessibility of external data (RULES.md "organisers can access and inspect").** IDSSE is open (CC BY 4.0). Metrica
   sample-data is public (`OpusOperandi.md` says "no licence file in the repository"; check the licence before citing).
   PFF FC World Cup 2022 tracking is available on request, not openly; the document must say how to obtain it. SkillCorner
   is MIT. Training data section per SUBMISSION.md ("who labelled and how much") is mostly there; add "PFF events are
   human-annotated by PFF" (other game = allowed).
4. **Parameters tied to BAR-ATH aggregates.** `external_degrader_target` (`configs/default.yaml:250`) and the degrader
   calibration of #38 use label-free aggregates of the BAR-ATH ATD (gap lengths, players per frame). No labels are
   involved, so this is defensible, but state it explicitly as "tracking-only statistics, no labels". Do not extend it to
   BAR-ATH *predicted-label* statistics (counts, D share) as targets; those are alarms only (project rule 5).

---

## 1. Label-definition compatibility of the external training sets

### 1.1 What the GT is (official text)

CHALLENGE.md (97-103 of the vendored file): GT "derived from corrected tracking data ... a frame is `D` where the
corrected tracking marks the ball dead, and otherwise takes the team the tracking names as owning it. Nothing is
smoothed, interpolated or hand-adjusted afterwards." "In phases where the ball is contested or loose, possession stays
with the last team that had it." "The tracking names a team even while the ball is out of play ... the ground truth
throws that away." So GT = (provider ball-status flag) x (provider owner channel), per frame, no event layer.

### 1.2 Classification of the sources (code: `mtah2027-external-benchmarks/src/mtah_bench/labels.py`, `pff.py`, `skillcorner.py`; `docs/LABELLING_RULES.md`)

| Source | How labels are made | Same construction as GT? | Event / abstraction bias | Verdict |
|---|---|---|---|---|
| **IDSSE / DFL (Bassek 2025)** | `BallStatus` (dead -> `D`) + `BallPossession` owner (alive -> team; missing owner -> last owner), `labels.py:116-150`. Per-frame channels of the tracking file itself. | **Yes, same construction** (status flag x owner channel). Both are provider-operator "corrected tracking". | Residual: DFL operator rules for *when* ownership flips (touch vs control) are not Metrica's; ball always present in IDSSE (we degrade it ourselves). | **Best semantic match.** Use as the primary label source. |
| **Metrica sample Games 1-3** | Events (`PASS`, `RECOVERY`, `BALL OUT`, ...) replayed into state changes (`labels.py:43-114`). | **No**: event-derived. | Flips only at control events at event timestamps ("a few frames"); pass in flight stays with passer; `D` from `BALL OUT`/fault events. Plus the tracker *drops the ball while dead* (92-99 % of `D` frames ball-missing, 2-5 % of `H/A`) = a leak. | Correct to use for eval only, and only as a boundary-timing sanity check. Not GT-like for `D` placement. |
| **PFF FC World Cup 2022** | Human-annotated events replayed (`OTB` -> owner team, `OUT` -> `D`, restart `OTB` after a foul etc.) (`pff.py`, LABELLING_RULES "PFF"). | **No**: event-derived. | (i) Timing = event time stamps (a few frames, human). (ii) `OTB` = "a player on the ball" is a *contact* event: it flips the owner on any contact event the annotator logs, which is closer to the GT's *touch-flips* (see issue #17) than a "controlled possession" but at annotator granularity. (iii) `D` includes long non-play gaps (VAR, injuries) like GT but is placed by the annotator, not by a tracking flag. (iv) The PFF tracker is 29.97 fps CV with **estimated** ball positions; dead balls are mostly untracked (4-14 % of `D` frames have a visible ball), so *ball-hidden strongly predicts D* (61 % of hidden frames are `D`, 8 % of visible; TECHNICAL_REPORT #35/#37). | Right *kind* of noise (real CV occlusions), wrong *label* mechanism and a tracker-specific hidden<->D coupling. Use with caution and low weight; do not use as the only judge. |
| **SkillCorner A-League** | `phases_of_play`: team per phase; `D` = gap between phases (>= 1.8 s) (`skillcorner.py`, LABELLING_RULES). | **No**: phase-level abstraction. | Smooth by construction (no micro-possessions), 10 fps broadcast tracking, 25-40 % of frames `X`, ball always visible. Per-frame `possession.group` disagrees with the phase team in ~20 %. | Dropping it was right (-0.026 ... -0.031 on B' in #35). It carries a "phase-level" bias towards fewer, longer possessions. |

Summary: only **IDSSE** reproduces the GT mechanism. PFF and Metrica carry *event bias* (timestamps, flip rules,
`D` placement); SkillCorner carries *phase-abstraction bias*. The engine's own diagnosis "possession is event-like: last
toucher agrees on 92 % of alive frames" (`TECHNICAL_REPORT.md:53-57`) was measured on exactly this mixture and
mostly on IDSSE; it is a statement about the provider-owner convention, not necessarily about Metrica's.

Caveat on IDSSE itself: it is Bundesliga/Sportec optical+operator data. Its "dead" definition (operator) is analogous to,
not identical with, Metrica's corrected tracking. Nobody can verify more without BAR-ATH labels (and we must not).

### 1.3 Did "no loss on PFF" block a better model? (TECHNICAL_REPORT item 5/6, #35 and #38)

Numbers (`TECHNICAL_REPORT.md:366-372, 441-449`), spellF1 / macroF1:

| model | B' (IDSSE LOMO, calibrated degrader, seeds 0-2) | A (PFF held-out, 4-fold) |
|---|---|---|
| default = IDSSE+PFF0.5, legacy degrader | 0.6044 / 0.9016 | 0.5231 / 0.8626 |
| IDSSE only, calibrated degrader | **0.6125** / 0.9056 (+0.0081; per seed +0.0099, +0.0080, +0.0067) | 0.4790 / 0.8412 (**-0.0441**) |
| IDSSE+PFF0.5, calibrated degrader (`calib`) | 0.6085 / 0.9035 (+0.0041) | 0.5278 / 0.8639 (+0.005) |

What the table says honestly:

1. The IDSSE-only-calibrated candidate failed **two** gates, not only the PFF one: its B' gain (+0.0081) is below the
   +0.01 bar *and* A falls 0.044 (guard -0.01). Removing the PFF guard alone would not have adopted it
   (0.0081 < 0.01). The question "did the PFF guard block a better model" is therefore: it blocked a
   +0.008 improvement that is **0.2x the stated metric noise** (SPELL_NOISE_BAND 0.04; across-seed s.d. ~0.002 only
   measures the degrader seed, not match sampling), i.e. not distinguishable from zero.
2. The three B' means differ by <= 0.008 with every candidate inside noise; the only large effects in the whole
   exercise are the ones that change *label mechanism or domain*: PFF-only (-0.106 on B', +0.03 on A) and SkillCorner.
   So the model-selection machinery is operating below its resolution; the choice of default (`mixed` + legacy degrader)
   is not distinguishable from `calib` or `idsse` on these proxies. That is not a bug; it is a statement that the
   *proxies*, not the candidates, are the bottleneck.
3. The *direction* of the PFF guard is defensible (real occlusions) but the validation A uses **PFF labels** (event-derived,
   section 1.2). A gain on A may be partly "learned to imitate PFF's `OUT`/hidden coupling" rather than "better on BAR-ATH".
   Conversely B' uses IDSSE labels (GT-like) but a degrader that is **uncoupled from the dead/alive state** (section 2). Each
   proxy is wrong in a different way, and the adoption rule (A must not drop > 0.01 AND B' must gain > 0.01) selected the model
   that is the compromise between two wrong proxies. This is a risk-averse but not an evidence-based choice.
4. Recommendation: do not re-open this particular comparison now. Re-run it **after** fixing the gap-placement realism of the
   degrader (improvement 1 in section 6), when B' stops being blind to the thing that matters most for BAR-ATH.

---

## 2. Ball-missing semantics (hidden ball vs dead ball)

### 2.1 What the official material says

DATA.md: "the ball goes missing disproportionately on the frames that matter: occlusions, crowded boxes and balls in the
air"; FAQ: "disproportionately on the interesting ones - crowded boxes, aerial duels, tackles". 56.9 % of all frames, 59.3 %
of graded frames have the ball (41 % hidden).

### 2.2 What the engine does on BAR-ATH (TECHNICAL_REPORT:#37 section, "D over-prediction diagnostic")

Predicted `D` share: 24 % with ball visible, **92 %** with ball hidden (specialists route; heuristic 29 % / 83 %), 94 % when
< 20 players tracked; overall ~50 % `D` of graded frames (`D 50.1 -> 49.7 %` in #39), 238-257 possessions
(`TECHNICAL_REPORT.md` #39/#40). The report's check "tracked players in predicted-D frames are slow" shows the `D` comes from
speed physics, not NaN handling.

### 2.3 Three regimes of "hidden -> D" and where BAR-ATH sits

Define the odds ratio OR = [P(D|hidden)/P(alive|hidden)] / [P(D|visible)/P(alive|visible)] (how much more likely a hidden
frame is dead than a visible one).

| data | P(D|hidden) | P(D|visible) | OR |
|---|---|---|---|
| **Degraded IDSSE** (our degrader) | ~ the match's D share (blocks placed over the whole graded timeline, weight (0.25+crowding), no status coupling; `external_degrader.py:_hide_blocks`) | ~ same | **~1** (slightly above or below via crowding) |
| **PFF (real CV occlusions)** | 61 % | 8 % | **~18** |
| **Metrica sample (corrected, ball dropped when dead)** | ~92-99 % of D frames hidden vs 2-5 % of alive | -- | of order 10^2 |
| **BAR-ATH, first 5 min only** (from `evaluation/reports/baseline_raw.json`, baseline_raw predicts D exactly where the ball is hidden; confusion rows GT H/A/D x pred H/A/D): GT D 1469, of which 494 hidden (33.6 %); GT alive 6031, of which 5+386 = 391 hidden (6.5 %); hidden total 885 -> P(D\|hidden) = 55.8 %; visible: 975 / 6615 = 14.7 % | 55.8 % | 14.7 % | **~7** |
| **BAR-ATH engine output, full match** | 92 % | 24 % | **~36** |

Reading:

* The public window says hidden is **real D-enriched** on BAR-ATH (OR ~ 7, "5x more often hidden when dead than when
  alive"). So "DATA.md says the ball goes missing in live play" is only half the picture; at least in the first 5 minutes
  the ball also vanishes at dead-ball moments (ball off the pitch/at the stands, stationary at a restart spot). The
  hidden-ball share is only 11.8 % in that window versus 41 % over the match, so the window is **not representative
  of the regime that dominates the scoring**; extrapolating OR from it is unsafe (and forbidden as a tuning target).
* The degrader is the outlier: OR ~ 1 versus 7-18 in the two real CV trackers. A model fit only on degraded IDSSE learns
  "hidden tells you nothing about D", i.e. relies on player physics; PFF at weight 0.5 injects an OR of 18 through the
  *gap-structure* features (`since_seen_s`, `until_seen_s`, `gap_dist_m`, last/next seen position and velocity;
  `features/ball_memory.py` docstring), because in PFF a gap often starts when the ball leaves play and ends at the
  restart. Which mixture reaches BAR-ATH is unmeasured.
* The engine's implied OR of ~36 is ~2x above the highest real-tracker figure we have (PFF 18) and ~5x above the (small-n)
  BAR-ATH window. Implied overall `D` ~50 % is at the top of the external range (29-52 %); the generic alarm in
  `configs/default.yaml:21` is 15-55 %. These are alarms, not targets; I report them as a **sanity inconsistency**,
  not as proof of error.
* Coherent story that explains three symptoms at once (D 50 %, hidden-> D 92 %, 238-257 possessions where external
  matches have 314-511 and the public 5 minutes extrapolate to ~360): the slow-player physics inflated by dropped
  tracks (14.9 % of frames < 20 players, no external counterpart) pushes hidden stretches to `D`, a `D` cuts a
  possession, the possession count falls, and recall of spells is capped. The report's own LOMO figure (2409 predicted of
  3061 true = 79 %) already shows under-segmentation; on BAR-ATH it is 66-71 % by the crude extrapolation. Hypothesis, not a
  finding: it needs the test in 2.5.

### 2.4 How `external_degrader.py` places gaps (code evidence)

* `_hide_blocks` (`external_degrader.py:~143-185`): target = `hide_frac * alive.sum()` (covers 42.4 % of **alive** frames);
  block start sampled from all graded frames with weight `(density_floor + mean_crowding)**gamma`; blocks cross dead stretches
  freely; `guard_s = 9` s of visible ball between blocks (lognormal model, median 12.5 s, sigma 0.95, capped 240 s).
* No term depends on `gt` being `D`, ball speed, ball leaving the pitch, restart geometry or time since last stoppage. Any
  coupling to D comes only from crowding (set pieces crowd as well), so OR ~ 1 as stated.
* Calibration #38 matched **only gap-length statistics and players per frame** of BAR-ATH (`external_degrader_target`),
  not the *position of gaps relative to play state*; its own success criterion ("reproduces the label-free aggregates") is
  silent on this axis.
* Block durations (lognormal, median 12.5 s) are measured on BAR-ATH where hidden = (live occlusions) U (dead-ball
  disappearances). If a part of those 12.5-s blocks are dead-ball disappearances, the degrader then pastes them over live
  play: duration shape right, state composition wrong.

### 2.5 Proposed label-free calibration of gap **placement** (needs no BAR-ATH labels)

Goal: estimate lambda = P(a gap frame is dead) on BAR-ATH, or equivalently the OR, then make the degrader reproduce it.
Use only the ATD of BAR-ATH (algorithmic features of the data, as in #38) and class-conditional statistics from **other**
matches with known status.

1. *Gap-boundary statistics (model-free).* For every gap (ball hidden >= 1 s) on BAR-ATH take the **last visible and first
   visible** frames: ball speed, distance to the nearest line/goal area/corner, whether the last position is outside or
   within 1.5 m of a line, nearest-player distance and crowding (<= 3 m), collective speed of both teams in the 3 s before and
   after. Compute the same statistics on (a) degraded IDSSE where status is known, split by "gap starts in D / in alive",
   (b) PFF (known). Fit the mixture weight pi_start (and pi_end) so the BAR-ATH histogram equals pi*h_D + (1-pi)*h_alive of the
   reference class-conditionals (EM / moment matching on a 2-component known-density mixture, i.e. prior-shift estimation,
   Saerens et al. 2002; or black-box shift estimation with a confusion matrix of a **ball-free** classifier; Lipton et al. 2018).
2. *Interior statistics.* Same on **collective speed and crowding inside gaps** (the quantity you suggested): histogram of
   mean outfield speed (and of "all 22 within the middle third at < 1 m/s") in hidden frames vs visible frames of BAR-ATH,
   matched against the class-conditional histograms of IDSSE (and PFF) *stratified by tracked players* (< 20, >= 20) so that
   dropout does not masquerade as dead-ball slowness. Because player dropout removes the fast runners, stratify or
   use team-level speed percentiles of the players that are present.
3. *Degrader change.* Add to `external_degrader.py` a coupling parameter `dead_affinity` (relative weight of placing a block
   start in a `D` stretch, and `dead_gap_frac` for gaps that coincide with a whole `D` stretch from ball-out to restart),
   fitted so that (1) and (2) on degraded IDSSE equal BAR-ATH's. Keep the crowding weight for live gaps.
4. *Honest validation of the estimator (no BAR-ATH labels):* (i) on PFF where the true OR is known (18), the estimator
   from steps 1-2 should recover it within +-25 %; (ii) on IDSSE with **planted** placement (OR in {1, 5, 18}), recover the
   plant; (iii) on the three Metrica games (OR ~10^2). If the estimator fails (i)-(iii) it is not used.
5. *Use.* (a) retrain B' on the re-placed degrader; (b) report the LOMO score as a function of OR (a sensitivity curve);
   if spellF1 is flat across OR in [3, 30] the whole issue is cosmetic and we stop. (c) Optional runtime variant: a class-prior
   shift of B's `D` logit on hidden frames, computed by EM on the BAR-ATH hidden frames (no labels), validated the same way
   by (i)-(iii).

Success criteria must be set *before* the run (project rule: +0.01 on B'-new with no seed below -0.003; no PFF drop beyond
0.01) and the BAR-ATH aggregate (`P(D|hidden)` predicted) is reported but never used to accept a variant.

---

## 3. Exact alignment checks (code vs `games.py` / `DATA.md`)

| Check | Official | Our code | Verdict |
|---|---|---|---|
| `video_frame = frame_id - 1` | DATA.md "Frame alignment"; `starter/load_atd.py::video_frame_of` | `io/atd_loader.py:~540` `rows = ids[sel] - 1`; the id is read from the data column (not from the line number); duplicates counted; `docstring:22` | **OK**. Uses the file's own `frame_id`, robust to gaps. |
| `round(seconds * 25)` | DATA.md; `starter/load_smart_tagging.py:64-65` `int(round(start_s * fps))` | `io/smart_tagging_loader.py:206` `int(round(start_s * fps))`, end likewise (exclusive) | **OK, identical** (same Python banker's rounding as the official starter, so 0.5-frame ties resolve identically). |
| Pitch 105.3 x 68.0 (metadata 105 x 68) | DATA.md: "games.py carries 105.3 x 68.0 ... the 30 cm matters only for distances" | `config.py:24` `PITCH = (105.3, 68.0)`; `atd_loader.py:676-686` passes the `games.py` pitch **explicitly** into `load_atd_files(pitch=...)`, which overrides the XML `FieldSize` (`atd_loader.py:611-612`); homography projections use the same constant (`homography_loader.py:96-107`) | **OK**. Note: `load_atd_files` called directly without `pitch=` would silently use the XML 105x68 (only tests/other-matches); fine. `cross_check_games()` runs at import (`config.py:215`). |
| Half blocks | `games.half_video_blocks`: (1, 11185, 90651), (2, 110808, 190197) | `config.py:27-30`: `HALF1=(11185,90651)`, `HALF2_DEAD=(110808,113741)`, `HALF2_PLAY=(113742,190197)`; cross-checked against `games.half_video_blocks` at import (`config.py:196-200`) | **OK** (verified: blocks print `[(1, 11185, 90651), (2, 110808, 190197)]`). |
| 2,934 dead frames | FAQ: graded and all `D`: 110808..113741 | `submission_contract.py::enforce_submission_contract`: `codes[half2_dead] = D`, then `fill_run` across the whole half-2 run (matches the scorer's `valid_runs`) | **OK**; 113741-110808+1 = 2934. |
| `X` outside graded blocks | ungraded -> ignored | `not_evaluated_blocks` -> `X`; no `X` inside graded runs | **OK** (see section 5 for why this is optimal). |
| Kick-off anchors in `games.py` | `kickoff` (7, 27, 11), `kickoff_2nd` (75, 49, 16); "observations about the video ... the ball is struck on this frame" | `anchor_video_frame` gives 11186 and 113741. Official graded starts: 11185 (half 1) and first live 113742 (half 2). So the anchors are 1 frame **after** the first graded frame in half 1 and 1 frame **before** the first live frame in half 2. We do not use the anchors; we use 11185 / 113742 from DATA.md (`check_time_base`, `smart_tagging_loader.py:275`). | **OK**; the +-1-frame difference is free on spellF1 ("perfect but 1 frame late = 1.0000"). Worth noting in the doc, and **do not** derive `t*` from the anchors. |
| Second-half kick-off in tag windows | tag windows are ~3.5-5 s around the restart (starter comment line 38) | TECHNICAL_REPORT #40: the two `Kick Off (Start)` `t*` fall **10 and 38 frames before** the published kick-off | The 38-frame one (collective cue) is harmless in half 2 because the contract forces `D` to 113741; in half 1 it is before the graded start. If the same cue misfires on a mid-match restart it costs ~1.5 s at that boundary (about 0.35 spellF1 on that possession). |
| First graded frame of half 1 is alive | DATA/FAQ: "the first half is graded from the kickoff itself" (kick-off is at the first graded frame) | not forced anywhere (decoder decides); `dead_ball_signals` has a `geom_kickoff` cue | **Check**: with the ball at rest on the centre spot at 11185-11186 the engine may label the first 1-2 frames `D`; 1-2 frames are free on spellF1, but the first possession must carry the **kick-off team** (tag `Kick Off (Start)`, `Team`). See section 4. |
| **Speed channel units** | DATA.md: "`1_s` ... speed in m/s" | Report #37: channel declared `s`, "`s` = `speed_kph`" from the XML, "not recognised as the canonical `speed` channel, harmless because speeds are re-derived". Code: `atd_loader.py:52,623-636` only converts to m/s when the channel is named `speed` and its unit is kph; `s` goes to `players_extra`/`ball_channels['s']`; every feature uses Savitzky-Golay speeds (`kinematics_and_orientation.py:237-240`); `grep players_extra|ball_extra` outside the loader: no uses | **Conflict is real but inert.** The XML unit and DATA.md disagree (kph vs m/s) and the engine never consumes `s`. Close it with a one-line label-free check on the real file: `median(|s_declared|)/median(|dx/dt|)` should be ~1 (m/s) or ~3.6 (kph). Document the result. If `s` is a pre-smoothing provider speed, it may be a lower-noise ball-speed feature, but only worth it if the check says the units are consistent. |
| Home/away | DATA.md: home = Barcelona (first `<Team>`); `starter/baseline.py::team_of_each_track` | `smart_tagging_loader.py:40-43` maps Barcelona->HOME, Atletic/Athletic->AWAY; ATD teams from `<Team>` order (`atd_loader` `team_ids`, `local_team_id`) | OK; recommend a one-time assertion that `team_names[0]` contains "Barcelona" (report #37 prints names). |

---

## 4. Smart Tagging: what exists, what we use, what we ignore, safe exploitation

### 4.1 Per DATA.md / OVERVIEW.md (638 instances)

Codes (n): PLAYERS IN THE BOX 97, BALL IN FINAL THIRD 93, BALL IN THE BOX 72, PROGRESSION 66, DEFENDING IN MIDDLE THIRD 66,
SET PIECES 59, CREATING CHANCES 30, DEFENDING IN DEFENSIVE THIRD 30, LONG BALL 29, ATTACKING TRANSITION 24, DEFENSIVE
TRANSITION 24, BUILD UP 22, DEFENDING IN ATTACKING THIRD 22, GOALS 4. Tags: `Team` (Barcelona 314, Atletic Club 288, **N/A 36**,
including **every throw-in**), `Half`, `Type` (Throw-in 26, Corner Kick 12, Goal Kick 10, Free Kick 5, Kick Off (After Goal) 4,
Kick Off (Start) 2), `Side`, `Direction of ball entry`, `Max Players in the box` (string, `7+`). `Team` = "the team performing
the event". Not ground truth (false positives and negatives); phases are intervals, not possessions.

### 4.2 What the code does (`io/smart_tagging_loader.py`, `pipeline.py:291-295`, `models/specialist_fusion.py::apply_priors`)

* **Used.**
  * All non-set-piece, non-goal phases -> a signed per-frame side `phase_prior_home` = (#home - #away overlapping phases)/total
    (`smart_tagging_loader.py:231-261`), with the four *defending* codes (`DEFENDING_CODES`, 118 + 24 instances) flipped to the
    opponent because `Team` is the performer (`:36-38`, `possession_side`). Fed as a soft H-vs-A log-odds term with
    `w_phase = 0.3` (`configs/default.yaml:177`), in both the heuristic emissions and, after the specialists, in `apply_priors`.
    The flip is correct by the official wording and by the pairing argument in #40.
  * SET PIECES -> restart anchors `t*` (ball method or collective method), forced-`D` interval before `t*`, restart team bias
    in the transition matrix (`restart_open -1.0`, `restart_other_team -3.0`, `restart_halfwidth` = 3 frames), throw-in team
    resolution for `Team = N/A`.
  * GOALS -> forced `D` from goal to the next `Kick Off (After Goal)` anchor.
* **Ignored.** `Side`, `Direction of ball entry`, `Max Players in the box` (tag values), the **liveness meaning** of the 14-code
  phase windows (every one of the 12 non-set-piece, non-goal codes is by definition live play), the **phase start times** of
  ATTACKING/DEFENSIVE TRANSITION as turnover times, cross-consistency of restart teams (kick-off after goal = conceding team;
  second-half kick-off = opposite of first-half kicker), `LONG BALL` as an aerial/occlusion flag (a long ball in flight is
  exactly when the ball goes missing), `BALL IN THE BOX` / `PLAYERS IN THE BOX` as "ball is hidden in a crowded box but alive".
* **Open issues #42/#43** already cover the alive prior and transition hints; the list below separates what they cover.

### 4.3 Concrete, safe ways to exploit it (each label-free; weights fitted on BAR-ATH **visible-ball** frames against the engine's own ultra-confident Specialist-A states, or on a simulator, never on GT)

1. **Alive prior inside phase windows (alive-vs-D on hidden frames) -> #43.** A frame inside a non-set-piece, non-goal phase
   is alive with high probability. Make it a *soft* term on the D logit, applied to the **interior** of the window only
   (erode ~1.5-2 s each side because intervals are automatic windows), only on hidden-ball frames (visible frames already
   have kinematic evidence), and capped (false positives exist). Calibrate the weight by the empirical log-odds: on BAR-ATH
   visible frames, among frames inside phase windows what fraction has Specialist-A P(D) > 0.9? That fraction is the
   tag's false-alive rate (a measurement of the tag's reliability, using the algorithm's labels, which RULES.md allows).
2. **Dead evidence from the *gaps between phases*.** Complement: the frames covered by **no** phase and not inside a set-piece
   window are either the very beginning/end of a window or dead-ball time. Use "uncovered by any phase" as a weak D prior with
   the same self-calibration.
3. **Turnover timing from transitions (-> #43 "transitions").** `ATTACKING TRANSITION (Team = T)` starts when T recovers the
   ball; `DEFENSIVE TRANSITION (Team = T)` starts when T loses it (the opponent gains). That is 48 tagged turnover times
   (about 13 % of the ~360 possessions) with timestamps from an automatic engine that sees the same video. Use them as soft
   *hazard* boosts in `log A_t(H<->A)` within +-2 s of the start, for gaining team only, **only inside ball-hidden stretches**
   (where our own flip timing is worst); weight from the same visible-ball self-calibration (hit rate of tag start vs our
   confident touch-based flip).
4. **Phase-side change as a turnover hint.** Because every `Progression(T)` is paired with `Defending in Middle Third(opp)` over
   the same frames, a change of the *possession side* between consecutive phases marks a turnover window without needing the
   transition codes. Same use as 3.
5. **Restart team as a hard-ish constraint where the tag names it.** 12 corners + 10 goal kicks + 5 free kicks + 6 kick-offs = 33
   named restarts (plus 26 throw-ins resolved from tracking). The first possession after `t*` belongs to `Team`. Currently a
   soft bias for **3 frames** around `t*`. If `t*` is off by more than 3 frames (collective cue: median 53 frames before
   gating, 10 % within 12), the bias is not in force when the Viterbi chooses the exit. Widen the bias to the whole interval
   `[t* - slack, t* + 2 s]` (slack from the anchor's confidence) and increase `restart_other_team` for ball-method anchors only.
6. **Free rule-based cross-checks on restart teams (zero risk, label-free).** Second-half kick-off team = the opposite of the
   first-half kick-off team; `Kick Off (After Goal)` team = the team that *conceded* (= the opposite of the preceding `GOALS`
   `Team`). Run as an assertion over the tags: any violation marks that tag unreliable and lowers its weight. This also
   gives the first-possession team in each half (`11185`, `113742`) from the tags, independent of the tracker.
7. **Set-piece restart position prior.** From the ball-method anchors (report #40: corners 9 ball / 3 collective, throw-ins 17 / 9; other types not broken
   down) compute, per `Type`, the distribution of `t* - start_frame` (and `end_frame - t*`). For the anchors resolved by the weak
   collective cue, restrict the argmax-J search to
   that learned offset band. It is a learned property of the tagging engine, not a frame-specific constant, so it satisfies
   issue #14's wording ("rules and constants tied to specific frames of the game" are the forbidden ones).
8. **`LONG BALL` / `BALL IN THE BOX` windows as an "alive but hidden" feature** (and the opposite of D): they identify the
   aerial/crowded occlusions that the engine currently reads as stoppage when tracked players are missing.

Risks to manage: tags are false-positive-prone; a tag prior must never exceed the evidence of the clear ball track (cap the
log-odds at about +-1.5); sensitivity must be tested with the simulator already built for #43 (random tag deletion,
jittered boundaries, wrong team, 20-30 % FP).

---

## 5. Metric subtleties and whether the post-processing is optimal

All numbers in this section are from the official `possession_metrics` on **synthetic** sequences (360 possessions, mean
~10 s, 43 % D, lognormal durations; `/tmp/aud/metric_demo.py`, seed 0).

### 5.1 The exact rules that matter (code lines in `possession_metrics.py`)

* `possession_spells` = maximal runs of `H` or `A` per graded run; `D` and `X` are not possessions; **runs never straddle the
  halftime hole** (`valid_runs`).
* `spell_overlaps` (`:472-489`): only **same-team** overlapping pairs, IoU = intersection / union, sorted by `(-IoU, gt_idx,
  pred_idx)`. `matched_gt_flags` (`:492-508`): greedy one-to-one, stops when IoU < threshold. Per threshold in
  (0.3, 0.4, 0.5, 0.6, 0.7): P = TP/n_pred, R = TP/n_gt, F1 = 2PR/(P+R); spellF1 = mean over the five.
* `fill_abstentions` (`:274-293`): forward-fill, back-fill at the head, **per valid run**; the 2,934 D frames at the start of
  half 2 belong to the same run as the second half, so a leading `X` at 110808 would back-fill from the first known label.
* macroF1 is computed on the raw array: every `X` on a graded frame is a miss for all three classes.

### 5.2 Cost of splitting vs merging (synthetic, one effect at a time, fraction of GT possessions affected)

| Perturbation | 10 % | 20 % | 30 % | n_pred / n_gt at 20 % |
|---|---|---|---|---|
| **Split** a possession in the middle with a 1-s `D` (`H D H`) | 0.905 | 0.820 | 0.742 | 431 / 360 |
| Split with a 1.2-s opposite-team flip (`H A H`) | 0.868 | 0.755 | 0.662 | 499 / 360 |
| Split with a 0.8-s opposite flip | 0.863 | 0.751 | 0.656 | 504 / 360 |
| **Miss** a possession by merging it into the previous run | 0.910 | 0.823 | 0.718 | 281 / 360 |
| Miss it by turning it into `D` | 0.947 | 0.889 | 0.824 | 288 / 360 |
| Merge `H D H` across all D < 14 s (all 360 -> 318) | 0.867 | | | 318 / 360 |

* Cost per wrong possession is ~0.9-1.0 spellF1 points per 1 % of possessions in every case; splitting and merging are
  **symmetrical to first order**, so there is no free lunch in biasing the count either way. A split at 10 % of the length
  costs less (0.893 at 20 %) than at 50 % (0.820) because the long piece keeps IoU > 0.7 in more thresholds.
* An opposite-team micro-flip hurts more than a `D` split (-0.065 more at 20 %): it adds a false spell *and* shreds the
  host possession *and* creates a bogus `A`. A spurious flip is therefore the most expensive single error type, which is why
  the engine's reluctance (`ha_closed`, touch gating) is right for visible-ball frames.
* Timing: a shift of 1 / 3 frames costs 0 (as published); 8: -0.007; 12: -0.022; 25 (1 s): -0.131 on these long
  possessions (the official table gives -0.35 on real, shorter possessions).
* **Absorbing real micro-possessions.** If 12 % of GT possessions are < 1.2 s H-A-H carve-outs (the engine's own duration
  table says 12.6 % of IDSSE/PFF possessions are < 1.2 s and 25.9 % < 2 s; `models/specialists.json` `durations`), a prediction
  that is perfect except for absorbing all of them scores **0.907**, a loss of ~0.09. So `micro_poss_s` is not free: it
  concedes up to ~0.09 of ceiling in exchange for fewer spurious flips. It is only justified if the micro-possessions are
  not detectable (hidden-ball frames, yes; visible with a double touch, no, and that is exactly the `keep_min_touches` rule).
  The active specialist-route value is 0.98 s, the heuristic route's 1.75 s.
* **Is the public GT consistent with "absorb micro-possessions"?** Issue #17 reports a 1.32-s `H` between two `A` spells after a
  won header with no control. 33 frames is above 0.98 s (kept) but below 1.2-1.75 s (killed). The mechanical GT flips on
  *contact by the nearest player*, not on "controlled possession" as CHALLENGE.md describes; the organisers have not
  answered. Our rule "a flip requires a touch of the gaining team" is consistent with that; "absorb < 1.2 s unless a double
  touch is visible" is partially inconsistent for header duels. Keep 0.98, do **not** move it towards 1.75 for the learned route.

### 5.3 `X` handling

Synthetic: write `X` for 100 H/A hand-overs.

| `X` block | spellF1 | macroF1 | note |
|---|---|---|---|
| none | 1.000 | 1.000 | |
| 1 s | 0.997-1.000 | 0.992 | |
| 3 s centred on the hand-over | 0.949 | 0.977 | spellF1 loses 5 points |
| 6 s centred on the hand-over | 0.871 | 0.955 | |
| 3 s inside a possession | 0.984 | 0.977 | almost free on spellF1 |

`X` can only be a tie among "carry the previous label" outcomes for spellF1 and always costs macroF1, so **abstaining never
beats the best of {hold, flip}** for the engine; the contract's "never `X` inside graded blocks" is optimal for spellF1 +
macroF1. The only genuine use of `X` is the leading frames of half 1 if no restart team is known; negligible.

### 5.4 Loose balls and deflections

Official text: possession stays with the last team that had it; a deflection that does not intercept, a touch without
control and an aerial duel won without control are not changes. Public GT (issue #17) contradicts this for one header, so
the mechanical rule is "owner = provider's channel". The engine's rule (hold the last toucher's team through flight and
loose phases; flip at a gaining touch) matches the text and most of the GT. The residual disagreement is concentrated in
duels/headers with 0.5-2 s possessions, which the engine absorbs or misses. Nothing to fix without evidence; but do not
spend effort on rules that enforce the text more strictly.

### 5.5 Where the current post-processing may not be optimal

1. **MAP vs F1-optimal decisions.** Adding one predicted spell with probability p of becoming a TP raises F1 iff
   `2p > F1_current` (from F1 = 2TP/(P+G)). At spellF1 ~0.5-0.6 the break-even is p ~ 0.25-0.3, not 0.5. Viterbi is MAP: it
   suppresses a flip or an alive-instead-of-D decision unless it is > 50 % likely. Where a flip does not damage an
   already-matched neighbour (new spell in a `D`-vs-alive ambiguity, or in an otherwise un-matched region) the correct rule
   is looser than MAP. The fact that "every extra flip is as likely wrong as right but spellF1 stays flat" (report 5.x A2)
   shows the second-order damage to neighbours matters; a real decision-theoretic decoder (sample paths from the HMM
   posterior and choose the segmentation with the highest expected spellF1) can optimise this properly. See improvement 2.
2. **`bridge_requires_geometry: false` in `route_overrides.specialists`** (`configs/default.yaml:~539`): the `D` bridge over a
   >= 2.85-s dead-score valley inside a >= 31.7-s possession is inserted without touchline/set-piece support. In the dropout
   regime the smoothed `dead_score` is inflated by missing runners; this knob creates extra `D` and splits long possessions.
   Test it inside the < 20-player stratum of #42.
3. **Per-half independence is optimal**, nothing crosses blocks; correct. The contract and the micro_dead/isolated rules
   (min_dead 1.86 s; D runs < 2 s are 1.1 % of training D runs) are consistent with the data.
4. **KPI/route selection (KPI 3 and 7) is blind to the placement issue** (section 2); it decides specialists vs heuristic on
   degraded IDSSE where OR = 1.

---

## 6. Ranked top-5 improvements for BAR-ATH spellF1

Impact figures are my **estimates with uncertainty**, not measurements; none can be measured on BAR-ATH. "Overlap" marks the
running issues #42 (player-dropout invariance), #43 (Smart Tagging alive prior), and the planned #44 (decoder re-opt under
the calibrated degrader).

### 1. Make the degrader's gap **placement** realistic (hidden <-> dead coupling) and re-validate B, then re-pick the model
* Why: evidence in section 2 (OR 1 vs 7-18 vs engine 36; D 92 % when hidden; 66-71 % of the extrapolated possession count).
  It is the one proxy mismatch that directly drives the largest observed output anomaly.
* Expected: if the over-`D` is real, +0.03 to +0.08 spellF1 (every `D`->alive hidden recovery is a recovered possession);
  if BAR-ATH really has a very high hidden-D share, ~0 (and we learn the sensitivity). Wide uncertainty.
* Risk: medium. A wrong coupling pushes the other way; mitigated by the estimator validation (PFF/Metrica/planted) and a
  sensitivity curve (accept only if the choice is robust across OR in the plausible range).
* Validate honestly: OR-parameterised degrader on the 7 IDSSE matches + PFF held-out (known OR 18) + Metrica games; show LOMO spellF1
  and `P(D|hidden)` vs OR; estimator recovery tests (section 2.5). The BAR-ATH label-free estimate is an input to the
  *degrader*, not a score.
* Files: `models/external_degrader.py` (`_hide_blocks`, new params), `configs/default.yaml` (`external_degrader*`,
  `external_degrader_target`), `scripts/compare_input_stats.py` (boundary + interior statistics), `models/specialist_data.py`,
  `scripts/train_specialists.py`, benchmarks `run_optuna_sweep.py`.
* Overlap: **precondition for #44** (tuning the decoder on a degrader with the wrong placement tunes the wrong thing);
  complements **#42** (same symptom, different cause: players vs gap structure), no direct conflict.

### 2. F1-aware (posterior-sampling MBR) decoding to fix under-segmentation
* Why: 66-71 % of the extrapolated possession count (79 % LOMO), 70 % of hidden-ball turnovers missed; break-even `p > F1/2`
  analysis (5.5). The existing MBR (variants of one model, +-0.0004) cannot help because the variants are not diverse; sampling
  from the forward-backward posterior with a spell-F1 utility is a different mechanism.
* Expected: +0.01 to +0.03 (published MBR-for-F1 gains are modest). Safe floor: it can fall back to the Viterbi path when the
  expected-utility gain is below a margin.
* Risk: the emission probabilities are tempered/overconfident; sampling needs a calibration pass (LOMO) or the samples are too
  similar. Runtime fine (numba FFBS + matching of ~300 spells x ~100 samples).
* Validate: B' (calibrated degrader, seeds 0-2) + A (PFF held-out + 8 unseen). Adoption rule as in the project (>= +0.01, no seed
  below -0.003).
* Files: `decoding/dynamic_viterbi.py` (sampler), `decoding/mbr_consensus.py` (utility-based selection), `pipeline.py::run_core`,
  `configs/default.yaml` (`mbr`).
* Overlap: **inside #44's scope** (decoder re-optimisation); do it after item 1 so both use the same degrader.

### 3. Extract more from Smart Tagging beyond the signed phase prior
* Why: section 4.3 items 1-8: alive prior, turnover timing from 48 transition tags, restart-team constraints (33 named restarts,
  kick-off rule cross-checks), t* offset prior by `Type`, self-calibration of weights from visible-ball frames.
* Expected: +0.01 to +0.04 (more if tags are as reliable as the 74-97 % agreement of #40 suggests; transition tags alone can
  re-time ~10 % of possessions).
* Risk: false-positive tags; weights not tunable on external matches. Mitigations: interior erosion, cap on log-odds, weights
  from the visible-ball self-consistency, simulator sweep (already built).
* Validate: Smart Tagging simulator on IDSSE/PFF (deleted/jittered/wrong-team/FP tags), plus label-free self-consistency on
  BAR-ATH visible frames (engine A's confident states). No GT.
* Files: `io/smart_tagging_loader.py` (per-code liveness windows, transition events, restart-team asserts),
  `models/specialist_fusion.py::apply_priors`, `decoding/transition_matrix.py` (restart window width, hazard boosts),
  `features/setpiece_anchor.py` (offset prior), `configs/default.yaml`.
* Overlap: **items 1-4 are #43** (alive prior and transitions); **items 5-7 (restart-team constraints, rule cross-checks, offset
  prior) are NOT in #43** and are cheap, low-risk additions.

### 4. Player-dropout invariance of the emissions (14.9 % of the frames < 20 players)
* Why: no external counterpart (<= 1.2 %), `D` 94 % in that stratum, `bridge_requires_geometry: false` interacts with it.
* Expected: +0.02 to +0.05 (this regime is 15 % of the frames and the worst-behaved).
* Risk: low-medium; the calibrated degrader already injects single/group dropouts, so it is testable.
* Validate: B' stratified by tracked players (< 20, < 16, >= 20), A unchanged within 0.01.
* Files: `models/specialist_features.py` (count-normalised speeds, per-team percentiles of *present* players, drop raw counts),
  `models/specialist_data.py` (dropout training variants), `postprocessing/spell_optimizer.py` (bridge geometry in the stratum),
  `configs/default.yaml`.
* Overlap: **this is #42**.

### 5. Ensemble / diversify the emissions, and finalise the model choice after item 1
* Why: the candidates differ by <= 0.008 on B' and by 0.04 on A; averaging log-odds of `mixed` + `idsse` + `calib` (+ `pff` at
  low weight) reduces variance without choosing between wrong proxies; also removes the arbitrary PFF guard.
* Expected: +0.005 to +0.015 (a few GBM-ensemble points on log-loss usually translate to a small spell gain).
* Risk: low; extra files in `models/`, still deterministic.
* Validate: same B'/A protocol; compare to best single model.
* Files: `models/specialist_fusion.py`, `models/specialist_data.py`, `scripts/generate_submission.py` (`--model-variant`),
  `models/specialists_*.json`.
* Overlap: none (touches the same files as #42/#43, merge conflicts only).

### Not recommended now
* More aggressive `micro_poss_s` (costs up to 0.09 of ceiling on 12 % micro-possessions), more `X` use, video recovery or TV
  sync (measured ~+0.005 in simulation, real fill 7.4 %), pseudo-labelling of the target (0.606 -> 0.590 on externals), and
  tuning anything on the public window.

---

## 7. Short list of facts the coordinator may want to verify locally (aggregate-only checks)

1. Units: `median(s_declared)/median(speed_from_xy)` on the tactical ATD (m/s -> ~1, km/h -> ~3.6). Print one number.
2. `team_names[0]` contains "Barcelona" and the `TagEvent.team` mapping H/A agrees with the first `<Team>` of the XML.
3. At the first graded frame of each half, the label of the final CSV is `H`/`A` (not `D`), and equals the Smart Tagging
   `Kick Off (Start)` team (print only the letter agreement, not rows).
4. Fraction of BAR-ATH gaps whose **last-visible** ball position is outside or within 1.5 m of a line, versus degraded IDSSE and
   PFF (the feature behind section 2.5 step 1). Aggregate only.
5. Generate the final CSV with `--specialists load --model-variant <x>`, run `evaluation/validate_submission.py`, and record the
   SHA256 in the technical document.
