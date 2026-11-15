# Milestone 11 research: untapped sources of spellF1 beyond the tactical-video ball detector (Issue #70)

Scope: where can more spellF1 still come from, with evidence.  Out of scope (other agents, in progress): #67 (oracle ceiling of
a perfect ball in the hidden frames + error decomposition) and #68 (self-supervised GPU ball detector on the tactical video).
Rules respected: no `data/`, `cache/`, video or prediction file was opened; real-match numbers below are aggregates printed by
the scripts in `scripts/research/`; no hand labels; no variant was chosen on BAR-ATH; no default changed.

State of play (from `TECHNICAL_REPORT.md`, Milestones 7-10): production B'(33) 0.6693, B'(1) 0.6017, A (PFF held-out) 0.5263;
public 5-minute pipeline check 0.594 vs best official baseline 0.424.  41 % of graded frames have no ball in the real tracking.
Twelve Milestone-10 experiments (decoder, gates, priors, features, bagging, MBR) are all < +0.010; turnover recall sits at 0.517
whatever the transition does, so the loss is in the **emissions** of the hidden-ball frames, not in the decoder.

## 1. Ranked inventory

Gain = expected change of B'(33) spellF1 (my estimate, with its basis).  Cost in engineer-hours.

| # | Idea | Evidence | Expected gain | Cost | Risk | Verdict |
|---|---|---|---|---|---|---|
| 1 | **World selection by adversarial validation**: decide which training world (degraded IDSSE B'(33) vs real-occlusion PFF A, mixed) BAR-ATH resembles, label-free | #55: 56 PFF matches **A +0.025/+0.030 but B' -0.0115**; #57 hard gate **A +0.011/+0.015, B' -0.006**; #38/#44 dead_affinity 33 vs 1 moves B' by 0.07.  Every adoption rule is decided in a simulated world whose match to the real ATD is only checked on gap length and player count (section 2.5); stoppage proxy in hidden frames: BAR-ATH 50 %, PFF 24 %, degraded IDSSE 0.8 % (section 2.4) | decision value +0.01 to +0.03 (it picks between models that differ by that much in the two worlds); 0 if both worlds agree | 6 | low | **do (A)** |
| 2 | **Sequence emission model** (learned turnover timing from collective motion, trained with block masking on IDSSE + PFF): TCN / small transformer over the 22-player window, per-frame H/A/D logits; replaces the 206-column LightGBM only on hidden-ball frames | Literature: ball-free possession/event inference works with sequence models (Kim et al., KDD 2023, ball trajectory + possessor from player context; PathCRF 2026; Borghesi et al. TCN 2023); here every *feature*-level change gave 0 (#56 -0.002, #64 +0.0002, #65 +0.0009), turnover recall stuck at 0.517; learnability probe (section 2.3): simple team-shape GBM reaches 80.3 % team accuracy on alive hidden frames | +0.01 to +0.03 | 24-40 (GPU) | high | **do (B)**, gated by the same B'(33) rule |
| 3 | Real-occlusion hard gate (#57) only on the world the BAR-ATH resembles | A +0.011/+0.015 vs B' -0.006: the sign depends on the world, i.e. on idea 1 | -0.006 to +0.015 | 3 after (1) | medium | decide with (1), no separate work |
| 4 | Broadcast TV video: ball detector on the TV image (ball much larger, roboflow model was trained on broadcast footage) | Only **26.8 %** of the hidden tactical frames fall in a synchronised TV block, and the TV ATD already sees the ball in 31.8 % of those (= 8.5 % of all hidden frames); a TV detector could add at most ~18 % of the hidden frames; #63 simulation: an imperfect injected ball *lowers* spellF1 (-0.074 at 50 % rate) | 0 to +0.008 | 20-30 | high | defer; revisit only if #68 shows a detector with < 0.5 m error |
| 5 | Broadcast "live coverage" as a liveness cue (TV block present vs cut away) | Slow-player proxy 10.2 % in covered vs 50.3 % in uncovered frames, at block ends 7.6 % -> 16.1 % (section 2.2); but synchronisation needs moving players, so the cue is partly circular | 0 to +0.01, unmeasurable on external data (no external broadcast) | 6 | high (self-reference) | do not pursue |
| 6 | Broadcast players to repair tactical dropouts (14.4 % of graded frames < 20 players) | Only **12.8 %** of those frames are inside a synchronised block (3.2 % for < 16): at most 1.8 % of graded frames could be repaired | < +0.003 | 8 | low | drop |
| 7 | Camera look-at point (pitch point under the TV image centre) as a coarse ball proxy | Median distance to the true ball 16 m (|dx| 8 m, |dy| 10 m); information probe (section 2.3): oracle ball +3.2 points of team accuracy, 8 m noise +0.7, 16 m -0.1 | ~0 | 4 | low | drop |
| 8 | Native ball speed channel `s` of the ATD | Finite on 8.8 % of graded frames in only 29 long runs (median 163 frames, longest 7,042), same derived-speed distribution inside and outside, no link to stoppage (section 2.1) | 0 | done | - | dead end |
| 9 | Smart Tagging as hard boundary constraints | #43, #49, #65 on tags simulated from the truth (optimistic): best +0.0009 (B'), A -0.007 to -0.034; the XML windows are 29-35 s with 3-6 s jitter and 69 % union coverage | <= +0.002 | 8 | medium | drop |
| 10 | Per-stratum calibration, ensembles, test-time augmentation | #60 bagging -0.0013, #42 shape normalisation +0.002 to +0.004, MBR 0.0000; TTA is a weaker variance reduction than bagging and was not run | <= +0.003 | 4 | low | drop |
| 11 | Self-training on the match itself with the current model (#21 style) | #21 pseudo-labels -0.016 (anchors are 98 % `D`); a confidence-gated H/A-only variant has no evidence | -0.01 to +0.005 | 10 | medium | drop |
| 12 | Player-identity / tracklet stitching to recover the carrier | Slot team is known from the XML, so stitching adds no team information; #66 tracklet gate on detections -0.005 | ~0 | 12 | medium | drop |
| 13 | Physics of ball-free intervals (pass vs dribble) | Already inside #64 (implicit carrier, 22-v-22 pressure) +0.0002 and #56 windows | ~0 | - | - | covered |

## 2. Probes

All real-match probes: `scripts/research/*.py`, aggregates only, run locally on the bundle (graded frames = 155,923 on the broadcast
side after block isolation, 158,857 on the tactical side).  External probes use the committed benchmark caches only.

### 2.1 Native ball speed channel (`real_ball_channel_probe.py`, `real_ball_speed_semantics.py`)

* Channels declared for the ball: `x, y, speed`.  Position finite on 59.3 % of graded frames; `speed` finite on **8.8 %** (10.2 % of
  the hidden-position frames, 7.8 % of the visible ones), i.e. it is *not* a hidden-ball channel.
* The finite frames form 29 runs (length p10/p50/p90 = 47 / 163 / 601 frames, max 7,042; median gap between runs 964 frames).
* Inside the finite runs the position-derived ball speed is distributed like outside (<0.4 m/s 8.1 % vs 3.9 %, >=2 m/s 73 % vs 80 %,
  median 4.9 vs 5.1 m/s); the median player speed is 1.26 vs 1.36 m/s.  No stoppage or state information.  Dead end, as the
  Issue #42 unit check already suggested.
* Side finding: hidden-ball frames are mostly slow-player frames: median player speed 0.99 m/s with 50.3 % below 1 m/s, against
  1.66 m/s and 10.9 % when the ball is visible.  The real hidden ball is therefore strongly D-enriched or stoppage-enriched.

### 2.2 Broadcast ATD beyond the ball (`real_broadcast_beyond_ball.py`, `real_broadcast_liveness_cue.py`; the #39 sync, accepted blocks)

| quantity | value |
|---|---|
| graded frames covered by a synchronised block | 58.77 % |
| tactical ball hidden: overall / inside covered frames | 41.10 % / 18.71 % |
| hidden tactical frames covered by a block | **26.76 %** |
| TV ball visible given tactical visible / tactical hidden | 93.78 % / 31.84 % |
| TV ball visible among *all* hidden tactical frames | 8.52 % |
| tactical vs TV ball when both visible: distance p10/50/90 | 0.2 / 0.8 / 5.8 m (57 % within 1 m, 11.7 % beyond 5 m) |
| tactical frames with < 20 players: share of graded / covered by a block | 14.41 % / 12.83 % (< 16 players: 6.47 % / 3.21 %) |
| of those covered, with >= 1 extra TV player (same team, > 2 m from every tactical one) | 93.7 % (mean 3.0 extras) |
| TV field of view width p10/50/90 | 29 / 38 / 48 m (12 % of frames < 30 m, none < 20 m) |
| look-at point vs tactical ball (both available): p10/50/90 | 7.0 / 16.1 / 23.3 m; within 10 m 19 %, within 20 m 75 % |
| slow-player proxy (median speed < 1 m/s), covered / uncovered | 10.2 % / 50.3 % |
| slow-player proxy, covered & ball hidden / uncovered & ball hidden | 26.1 % / 60.1 % |
| slow-player proxy 5 s before -> 5 s after the end of a block (n = 69) | 7.6 % -> 16.1 % |
| uncovered stretches | 69, duration p10/50/90 4.7 / 19 / 87 s; 77 % of uncovered frames are in stretches > 30 s |

Reading.  (i) The TV is blind where the tactical camera is blind: the correlation of the two trackers' failures, plus the sync not
covering hidden frames (the fingerprint needs visible, moving players), keeps the useful TV ball to 8.5 % of the hidden frames; a
TV detector can reach at most the remaining ~18 % of hidden frames.  (ii) The synchronised blocks cannot repair player dropouts:
they rarely exist exactly when players are missing (12.8 % of the < 20-player frames).  (iii) The camera look-at point is a poor
ball proxy (median 16 m, mostly because of the perspective in *y*).  (iv) Block coverage is strongly associated with live play,
but sync success itself depends on player motion, so this is partly circular and cannot be validated without labels or an
external TV feed.  Caveat on "extra players": TV slots include referees or duplicates, so 3.0 is an upper bound.

### 2.3 Information content of a coarse ball proxy (`coarse_ball_proxy_curve.py`, external data only)

Leave-one-match-out gradient boosting, 4 IDSSE matches degraded with the calibrated degrader (dead_affinity 33, seed 0), 1,942
alive hidden-ball frames (stride 20), target = possessing team.  `base` = team-shape physics only (centroids, spread, mean
velocity, tracked counts); the others add features relative to a proxy point (true ball + 3 s-correlated Gaussian noise of s m).

| features | team accuracy, alive hidden frames |
|---|---|
| base (no ball) | 80.3 % |
| + oracle ball (s = 0, 1 s-smoothed) | 83.5 % (+3.2) |
| + proxy s = 8 m | 81.0 % (+0.7) |
| + proxy s = 16 m | 80.9 % (+0.6) |
| + proxy s = 25 m | 79.6 % (-0.7) |

Small sample (about +-1 point of noise), frame accuracy and not spellF1, and a simple model that does not use ball history (the
real specialists get more from the last touch).  Still, even the *exact* ball position adds little to the team on top of the
collective shape, and a 16 m proxy adds nothing, which retires ideas 5-7 as information sources.  This complements #67 (oracle
spellF1 ceiling): the gain of a perfect ball in hidden frames is in the *boundary timing and D*, not in a static team cue.

### 2.4 World match (`world_match_slow_share.py`)

Share of frames with the label-free stoppage proxy (median tracked-player speed over 1 s < 1 m/s), by ball visibility, graded frames:

| world | hidden-ball share | proxy, ball visible | proxy, ball hidden |
|---|---|---|---|
| IDSSE degraded, dead_affinity 1 (7 matches) | 39.6 % | 0.5 % | 0.3 % |
| IDSSE degraded, dead_affinity 33 (7 matches, B'(33)) | 41.2 % | 0.1 % | 0.8 % |
| PFF real occlusions (8 matches) | 55.3 % | 6.4 % | 23.5 % |
| **BAR-ATH real** (frames without a native ball speed, 91 % of graded) | 40.7 % | 10.9 % | 50.3 % |

The hidden-ball frames of BAR-ATH are about twice as "slow" as PFF's and two orders of magnitude slower than the degraded IDSSE
world used for B'(33) (whose slow frames are almost absent even though its true `D` share is 29-52 %).  Part of this is tracking
noise and dropouts rather than football (IDSSE tracking is cleaner, so its speeds rarely fall under 1 m/s), so the table does not
prove a wrong `D` rate; it does show that **no external world reproduces the real hidden-frame physics**, and that the one that is
closest in kind (PFF) is the one on which #55 and #57 gain.  This is the motivation of next issue A.


## 3. Literature used

* Kim et al., "Ball trajectory inference from multi-agent sports contexts using Set Transformer and hierarchical Bi-LSTM", KDD 2023
  (arXiv 2306.08206): ball and possessor inferred from player context only.
* Kim et al., "PathCRF" (arXiv 2602.12080, 2026): ball-free event detection as a possession path over a player graph.
* Borghesi et al., "Temporal convolutional networks for possession estimation in football", Expert Systems with Applications 2023.
* Link and Hoernig, "Individual ball possession in soccer", PLOS ONE 12(7), e0179953, 2017: Bayesian-network control model (the
  family of the current distance/last-toucher features).
* Vidal-Codina et al., "Automatic event detection in football using tracking data", Sports Engineering 2022: rule baselines above 90 %
  detection on complete tracking, which is the regime the current engine already sits in when the ball is visible.

## 4. Proposed next issues

**A. Label-free world selection (adversarial validation).**  Build a domain classifier that separates BAR-ATH hidden-ball frames
from (a) B'(33) degraded IDSSE hidden frames and (b) PFF real-occlusion hidden frames, on a feature set that does not contain the
ball (collective physics, tracked counts, gap structure: `since_seen`, `until_seen`, gap length, slow-player share, line advance).
Report only AUCs / two-sample statistics and the per-world nearest-neighbour share of BAR-ATH frames (aggregates), then fit the
mixture weight of the two worlds by classifier-based density-ratio (prior shift) so that a weighted external validation set
matches BAR-ATH.  Re-score the three adoption-rule contenders (#55 56-match training, #57 hard gate, dead_affinity) on that weighted
set.  Success: a preregistered weight and the resulting choice, with the sensitivity of spellF1 to the weight.  Cost ~6 h, risk low.

**B. Sequence emission model for hidden-ball frames.**  Train a causal+anticausal TCN (or a small transformer) over 10 s windows
of player tracks (positions relative to team centroids, velocities, counts, gap indicators) with block masking, targets H/A/D per
frame from IDSSE + PFF, in the B'(33) leave-one-match-out protocol with the same adoption rule (B'(33) >= +0.010, no seed below
-0.003, B'(1) and A guards).  Emissions go through the unchanged fusion, transitions and Viterbi; the existing LightGBM stays as
the fallback.  Include a turnover-recall readout (current 0.517).  Cost 24-40 h incl. GPU training, risk high; the only
candidate with a literature precedent and a mechanism (timing of turnovers inside gaps) that the 12 Milestone-10 experiments never touched.

**C. (conditional on #68) Fusion of a self-supervised tactical-video detector as a gated factor, not as an injected ball.**  Rather
than injecting points (which hurt in #63/#66 because the specialists were trained on exact balls), feed confidence-gated detections
as an additional emission factor and retrain the hidden-ball specialist with simulated detector errors taken from the measured #68
CDF.  Start only if #68 reports a top-confidence error below ~0.5 m on > 70 % of frames.

**D. (low priority) TV video detector restricted to synchronised hidden frames.**  Ceiling ~18 % of the hidden frames; revisit
only after C shows a gated detector pays on the tactical camera.
