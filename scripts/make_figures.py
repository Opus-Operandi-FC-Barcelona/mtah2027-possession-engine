# ruff: noqa: E501
"""Draw the figures of the technical document and the README (``docs/figures/``).

Two groups:

* ``--aggregates`` (default): figures drawn only from aggregate numbers measured on *other* matches
  (``docs/figures/figure_data.json``, copied from ``reports/*.json`` and ``docs/TECHNICAL_REPORT.md``) and
  from the official public scoreboard, plus the method diagrams.  No match data is read.
* ``--match``: three figures drawn locally from the provided match (ball presence and tracked players per
  30 s, the public window against the public ground truth, the full-match timeline).  They read the data
  bundle, the public ground truth and the submission CSV, write PNG files only and print nothing per frame.

Usage::

    pip install matplotlib==3.11.0
    python scripts/make_figures.py                                   # aggregates + diagrams
    python scripts/make_figures.py --match --data-dir data --submission "outputs/Opus Operandi.csv"
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "docs" / "figures"
DATA = OUT / "figure_data.json"

# palette: categorical slots 1-2 for the two teams, neutral grey for dead ball / "off"
H_C, A_C, D_C = "#2a78d6", "#eb6834", "#b8b7b1"
ADOPT, OFF = "#2a78d6", "#b8b7b1"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
SURF = "#fcfcfb"
BOX = {"in": "#dce9fb", "ext": "#fdf0d5", "learn": "#fde3d6", "dec": "#d9f2e8"}

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 9, "axes.edgecolor": INK2, "axes.labelcolor": INK2,
    "xtick.color": INK2, "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
    "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF, "axes.titleweight": "bold",
    "axes.titlesize": 10, "axes.titlecolor": INK, "legend.frameon": False,
})


def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / name, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("wrote", (OUT / name).relative_to(REPO))


def bar_labels(ax, bars, fmt="{:.3f}", dy=0.008):
    for b in bars:
        v = b.get_height()
        ax.text(b.get_x() + b.get_width() / 2, v + dy, fmt.format(v), ha="center", va="bottom", fontsize=8, color=INK)


# ----------------------------------------------------------------------------------------- aggregates
def fig_validation_ladder(d):
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), sharey=True)
    for ax, key, title in ((axes[0], "ladder_b33", "IDSSE, leave-one-match-out, B'(33), 3 seeds"),
                           (axes[1], "ladder_pff8", "8 never-used PFF matches, real occlusions")):
        steps = d[key]
        names = [s[0] for s in steps]
        vals = [s[1] for s in steps]
        cols = [OFF] * (len(vals) - 1) + [ADOPT]
        bars = ax.bar(range(len(vals)), vals, color=cols, width=0.62)
        bar_labels(ax, bars)
        ax.set_xticks(range(len(vals)), names, fontsize=7.5)
        ax.set_title(title)
        ax.set_ylim(0, 0.85)
        ax.grid(axis="x", visible=False)
    axes[0].set_ylabel("spellF1 (official metric)")
    save(fig, "validation_ladder.png")


def fig_validation_table(d):
    rows = d["validation_table"]
    cols = ["IDSSE B'(33)", "IDSSE B'(1)", "PFF A (20)", "PFF unseen (8)"]
    fig, ax = plt.subplots(figsize=(8.6, 3.2))
    x = np.arange(len(cols))
    w = 0.26
    shades = ["#9ec5f4", "#5598e7", H_C]
    for i, r in enumerate(rows):
        bars = ax.bar(x + (i - 1) * w, r["values"], w - 0.02, color=shades[i], label=r["name"])
        bar_labels(ax, bars, dy=0.006)
    ax.set_xticks(x, cols)
    ax.set_ylim(0, 0.85)
    ax.set_ylabel("spellF1")
    ax.legend(loc="upper center", ncol=3, bbox_to_anchor=(0.5, -0.12), fontsize=8)
    ax.grid(axis="x", visible=False)
    ax.set_title("Same ordering on every validation set")
    save(fig, "validation_sets.png")


def fig_grid73(d):
    g = d["grid73"]
    variants = list(g)
    cols = ["b33", "b1", "pff"]
    lab = ["B'(33)\nprimary", "B'(1)\nguard", "PFF A\nguard"]
    m = np.array([[g[v][c] for c in cols] for v in variants])
    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    im = ax.imshow(m, cmap="Blues", vmin=-0.005, vmax=0.065, aspect="auto")
    for i in range(m.shape[0]):
        for j in range(m.shape[1]):
            ax.text(j, i, f"{m[i, j]:+.4f}", ha="center", va="center", fontsize=8,
                    color="white" if m[i, j] > 0.035 else INK)
    ax.set_xticks(range(3), lab)
    names = [f"{v}  (w={g[v]['w']}, s={g[v]['s']})" for v in variants]
    ax.set_yticks(range(len(variants)), names, fontsize=8)
    ax.grid(False)
    k = variants.index("V50s")
    ax.add_patch(plt.Rectangle((-0.5, k - 0.5), 3, 1, fill=False, ec=A_C, lw=2))
    cb = fig.colorbar(im, ax=ax, fraction=0.05)
    cb.set_label("spellF1 change vs #72", color=INK2)
    ax.set_title("#73 preregistered grid (box = adopted)")
    save(fig, "grid73.png")


def fig_ledger(d):
    items = sorted(d["ledger"], key=lambda r: r[1])
    clip = -0.03
    fig, ax = plt.subplots(figsize=(8.6, 7.2))
    y = np.arange(len(items))
    vals = [max(r[1], clip) for r in items]
    cols = [ADOPT if r[2] else OFF for r in items]
    ax.barh(y, vals, color=cols, height=0.7)
    for i, r in enumerate(items):
        if r[1] < clip:
            ax.text(0.0012, i, f"{r[1]:+.4f} (bar clipped)", va="center", ha="left", fontsize=7, color=INK2)
            continue
        ax.text(vals[i] + (0.0012 if vals[i] >= 0 else -0.0012), i, f"{r[1]:+.4f}", va="center",
                ha="left" if vals[i] >= 0 else "right", fontsize=7, color=INK2)
    ax.set_yticks(y, [r[0] for r in items], fontsize=7.5)
    ax.axvline(0.01, color=A_C, ls="--", lw=1.2)
    ax.text(0.0115, 9, "adoption bar: +0.010 mean,\nno seed below -0.003", fontsize=7.5, color=A_C, va="center")
    ax.axvline(0, color=INK2, lw=0.8)
    ax.set_xlim(clip - 0.012, 0.072)
    ax.set_xlabel("spellF1 change on the primary external validation")
    ax.grid(axis="y", visible=False)
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=ADOPT, label="adopted"), Patch(color=OFF, label="built, measured, left off")],
              loc="lower right", fontsize=8)
    ax.set_title(f"Every idea measured on other matches: {sum(r[2] for r in items)} adopted, "
                 f"{sum(not r[2] for r in items)} left off")
    save(fig, "experiment_ledger.png")


def fig_world(d):
    w = d["world"]
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.3), gridspec_kw={"width_ratios": [1.1, 1]})
    ax = axes[0]
    names = ["gaps placed\nat random", "label-free estimate\n(lower bound)", "submission\n(Milestone 9)"]
    vals = [w["random"], w["estimate"], w["submission"]]
    bars = ax.bar(range(3), vals, color=[OFF, "#5598e7", H_C], width=0.6)
    bar_labels(ax, bars, fmt="{:.0%}", dy=0.015)
    ax.set_xticks(range(3), names, fontsize=8)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("dead-ball share of hidden-ball frames")
    ax.set_title("Where the ball goes missing in BAR-ATH")
    ax.grid(axis="x", visible=False)
    ax = axes[1]
    names = ["Milestone 8\nmodel", "re-trained at\naffinity 33 (#48)"]
    bars = ax.bar(range(2), [w["m8_b33"], w["m48_b33"]], color=[OFF, ADOPT], width=0.5)
    bar_labels(ax, bars, dy=0.004)
    ax.set_ylim(0.6, 0.7)
    ax.set_xticks(range(2), names, fontsize=8)
    ax.set_ylabel("spellF1, B'(33)")
    ax.set_title("Training in that world: +0.017, all seeds")
    ax.grid(axis="x", visible=False)
    save(fig, "hidden_ball_world.png")


def fig_turnovers(d):
    t = d["turnover_recall"]
    fig, ax = plt.subplots(figsize=(7.4, 3.0))
    labels = ["IDSSE B'(33)\nball visible", "IDSSE B'(33)\nball hidden", "PFF A\nball visible", "PFF A\nball hidden"]
    ref = [t["b33"][0][0], t["b33"][0][1], t["pff"][0][0], t["pff"][0][1]]
    new = [t["b33"][1][0], t["b33"][1][1], t["pff"][1][0], t["pff"][1][1]]
    x = np.arange(4)
    b1 = ax.bar(x - 0.18, ref, 0.34, color=OFF, label="before #73")
    b2 = ax.bar(x + 0.18, new, 0.34, color=ADOPT, label="submitted (#73)")
    bar_labels(ax, b1, dy=0.01)
    bar_labels(ax, b2, dy=0.01)
    ax.set_xticks(x, labels, fontsize=8)
    ax.set_ylim(0, 0.8)
    ax.set_ylabel("H<->A turnover recall (±25 frames)")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(axis="x", visible=False)
    ax.set_title("Turnovers found, by ball visibility at the true change")
    save(fig, "turnover_recall.png")


def fig_oracle(d):
    o = d["oracle"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.4), gridspec_kw={"width_ratios": [1.6, 1]})
    ax = axes[0]
    sig = o["sigma"]
    for key, lab, c in (("fp0", "no false detections", H_C), ("fp10", "10 % false detections", "#5598e7"),
                        ("fp25", "25 % false detections", "#9ec5f4")):
        ax.plot(sig, o[key], marker="o", ms=5, lw=2, color=c, label=lab)
    ax.axhline(o["perfect"], color=INK2, lw=1, ls=":")
    ax.text(2.05, o["perfect"], f"perfect ball +{o['perfect']:.3f}", va="center", ha="right", fontsize=7.5,
            color=INK2, bbox=dict(fc=SURF, ec="none", pad=1))
    ax.axhline(0.01, color=A_C, lw=1.2, ls="--")
    ax.text(0.105, 0.013, "adoption bar +0.010", fontsize=7.5, color=A_C)
    ax.axhline(0, color=INK2, lw=0.8)
    ax.axvspan(0.95, 2.1, color="#f3f2ee", zorder=0)
    ax.text(1.4, 0.027, "detectors measured on\nthe real tactical video", ha="center", fontsize=7.5, color=INK2)
    ax.set_xscale("log")
    ax.set_xticks(sig, [f"{s:g}" for s in sig])
    ax.set_xlim(0.09, 2.15)
    ax.set_xlabel("detector position error (m)")
    ax.set_ylabel("spellF1 change, B'(33)")
    ax.legend(fontsize=7.5, loc="lower left", bbox_to_anchor=(0.0, 0.08))
    ax.set_title("A video ball helps only if it is very precise")
    ax = axes[1]
    cov = o["coverage"]
    bars = ax.bar(range(len(cov)), [c[1] for c in cov], color=["#9ec5f4", "#5598e7", H_C], width=0.55)
    bar_labels(ax, bars, fmt="+{:.3f}", dy=0.001)
    ax.axhline(0.01, color=A_C, lw=1.2, ls="--")
    ax.set_xticks(range(len(cov)), [f"{c[0]} % of gaps" for c in cov], fontsize=8)
    ax.set_ylim(0, 0.05)
    ax.set_ylabel("spellF1 change (perfect ball)")
    ax.set_title("Ceiling by gap coverage")
    ax.grid(axis="x", visible=False)
    save(fig, "video_ceiling.png")


def fig_scoreboard(d):
    s = d["public_scoreboard"]
    names = [r[0] for r in s]
    spell = [r[1] for r in s]
    macro = [r[2] for r in s]
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.0), sharey=True)
    for ax, vals, title in ((axes[0], spell, "spellF1 (ranking)"), (axes[1], macro, "macroF1 (tie-break)")):
        cols = [H_C if n == "Opus Operandi" else OFF for n in names]
        bars = ax.barh(range(len(names)), vals, color=cols, height=0.6)
        for b, v in zip(bars, vals):
            ax.text(v + 0.01, b.get_y() + b.get_height() / 2, f"{v:.3f}", va="center", fontsize=8, color=INK)
        ax.set_xlim(0, 1.05)
        ax.set_title(title)
        ax.grid(axis="y", visible=False)
    ci = d["public_ci"]
    axes[0].errorbar(spell[0], 0, xerr=[[spell[0] - ci[0]], [ci[1] - spell[0]]], fmt="none", ecolor=INK, capsize=3, lw=1)
    axes[0].set_yticks(range(len(names)), names, fontsize=8)
    axes[0].invert_yaxis()
    fig.suptitle("Public 5 minutes, official scorer (pipeline check only; bar = bootstrap 5-95 %)", fontsize=9, color=INK2)
    save(fig, "public_scoreboard.png")


# ------------------------------------------------------------------------------------------- diagrams
def _box(ax, x, y, w, h, title, body, kind):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.012,rounding_size=0.02",
                                fc=BOX[kind], ec=INK2, lw=0.8))
    ax.text(x + w / 2, y + h - 0.035, title, ha="center", va="top", fontsize=8.2, weight="bold", color=INK)
    ax.text(x + w / 2, y + h - 0.085, body, ha="center", va="top", fontsize=6.9, color=INK2, linespacing=1.25)


def _arrow(ax, p, q):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=9, color=INK2, lw=0.9,
                                 connectionstyle="arc3,rad=0"))


def fig_pipeline():
    fig, ax = plt.subplots(figsize=(10, 4.3))
    ax.set_xlim(0, 1.005)
    ax.set_ylim(-0.08, 1.03)
    ax.axis("off")
    _box(ax, 0.01, 0.62, 0.17, 0.3, "Tactical ATD", "46 slots + ball, 25 fps\nball hidden on 41 %\nof graded frames", "in")
    _box(ax, 0.01, 0.25, 0.17, 0.3, "Smart Tagging", "set pieces, goals,\nthrow-ins -> restart\nwindows (tracking-anchored)", "in")
    _box(ax, 0.01, -0.05, 0.17, 0.24, "External labels", "IDSSE 7 + PFF 20,\ndegraded (#48 world)", "ext")
    _box(ax, 0.22, 0.47, 0.18, 0.45, "Features", "clean + orient + Savitzky-Golay\ndead-ball score\ntouches, Link-Hoernig control\nrestart frame t*, throw-in team\ncollective physics\nball memory over gaps", "learn")
    _box(ax, 0.44, 0.58, 0.18, 0.34, "Specialists A / B", "LightGBM, ball visible /\nball hidden, calibrated\n(OOF temperature)", "learn")
    _box(ax, 0.44, 0.13, 0.18, 0.36, "Sequence TCN", "8 residual blocks\nreceptive field ±20 s\n2 seeds, NumPy inference\nhidden 1.0 / visible 0.5", "learn")
    _box(ax, 0.66, 0.47, 0.15, 0.45, "Dynamic Viterbi", "log A_t[N,3,3]\nH<->A only at a touch\n(-1.4 vs -11.5)\nD gated by evidence\nper half, numba", "dec")
    _box(ax, 0.66, 0.05, 0.15, 0.34, "Spell optimiser", "absorb < 1 s flickers\nD bridge on stoppages\nforced D before t*", "dec")
    _box(ax, 0.85, 0.3, 0.14, 0.4, "Opus Operandi.csv", "199,319 frames\ncontract: X / D\nvalidated, SHA256\nbyte-identical", "dec")
    for p, q in (((0.18, 0.77), (0.22, 0.75)), ((0.18, 0.4), (0.22, 0.55)), ((0.18, 0.07), (0.44, 0.2)),
                 ((0.40, 0.75), (0.44, 0.75)), ((0.40, 0.55), (0.44, 0.4)), ((0.62, 0.75), (0.66, 0.72)),
                 ((0.62, 0.3), (0.66, 0.6)), ((0.735, 0.47), (0.735, 0.39)), ((0.81, 0.22), (0.85, 0.4))):
        _arrow(ax, p, q)
    ax.text(0.0, 1.0, "inputs", fontsize=7.5, style="italic", color=INK2)
    ax.text(0.31, 1.0, "learning", fontsize=7.5, style="italic", color=INK2)
    ax.text(0.7, 1.0, "decoding", fontsize=7.5, style="italic", color=INK2)
    save(fig, "pipeline.png")


def fig_decoder():
    fig, ax = plt.subplots(figsize=(5.6, 5.0))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.axis("off")
    for (x, y), s, c in (((0.18, 0.75), "H", H_C), ((0.82, 0.75), "A", A_C), ((0.5, 0.2), "D", D_C)):
        ax.add_patch(plt.Circle((x, y), 0.09, fc=c, ec=INK2, lw=0.8, alpha=0.9))
        ax.text(x, y, s, ha="center", va="center", fontsize=13, weight="bold", color="white")
    ax.add_patch(FancyArrowPatch((0.28, 0.79), (0.72, 0.79), arrowstyle="<|-|>", mutation_scale=10, color=INK2))
    ax.text(0.5, 0.85, "H<->A: -1.4 within ±5 frames of a touch of the\ngaining team, -11.5 elsewhere", ha="center",
            fontsize=7.3, color=INK)
    ax.add_patch(FancyArrowPatch((0.22, 0.66), (0.43, 0.27), arrowstyle="<|-|>", mutation_scale=10, color=INK2))
    ax.add_patch(FancyArrowPatch((0.78, 0.66), (0.57, 0.27), arrowstyle="<|-|>", mutation_scale=10, color=INK2))
    ax.text(0.05, 0.42, "-> D opens with\ndead-ball evidence", fontsize=7.3, color=INK)
    ax.text(0.7, 0.42, "D -> team opens at the\nrestart frame t* or when\nthe evidence drops", fontsize=7.3, color=INK)
    ax.text(0.5, 0.02, "emissions: Specialists A/B mixed with the sequence TCN; one Viterbi per graded half",
            ha="center", fontsize=7, color=INK2, style="italic")
    save(fig, "decoder_states.png")


def fig_degrader():
    fig, ax = plt.subplots(figsize=(10, 2.3))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    _box(ax, 0.01, 0.1, 0.26, 0.8, "Clean external match", "IDSSE (DFL channels) or PFF (events)\nall players, ball always seen\nlabels by algorithm, never by hand", "ext")
    _box(ax, 0.36, 0.02, 0.3, 0.96, "Degraded like BAR-ATH",
         "ball hidden in contiguous blocks\n(gap lengths calibrated label-free)\nblocks placed over dead ball\n(dead_affinity 33, estimator #48)\ncorrelated noise, identity cuts,\nduplicate and dropped tracks", "learn")
    _box(ax, 0.75, 0.1, 0.24, 0.8, "Looks like the target", "same gap structure and\ntracked-player counts\nhome/away swapped copies", "in")
    _arrow(ax, (0.27, 0.5), (0.36, 0.5))
    _arrow(ax, (0.66, 0.5), (0.75, 0.5))
    save(fig, "degradation.png")


def fig_rules():
    """The engine's decision path for a candidate change, as a flow chart (questions -> outcomes)."""
    from matplotlib.patches import Polygon

    fig, ax = plt.subplots(figsize=(10, 3.9))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 40)
    ax.axis("off")

    def question(x, y, text):
        w, h = 9.5, 6.5
        ax.add_patch(Polygon([(x - w, y), (x, y + h), (x + w, y), (x, y - h)], closed=True, fc="#fff4cc",
                             ec=INK2, lw=0.9))
        ax.text(x, y, text, ha="center", va="center", fontsize=7.2, color=INK, linespacing=1.2)

    def outcome(x, y, title, body, colour):
        w, h = 16, 8.5
        ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle="round,pad=0.3,rounding_size=1.2",
                                    fc=colour, ec=INK2, lw=0.8))
        ax.text(x, y + 2.2, title, ha="center", va="center", fontsize=7.8, weight="bold", color=INK)
        ax.text(x, y - 1.3, body, ha="center", va="center", fontsize=6.6, color=INK2, linespacing=1.2)

    def arrow(p, q, label=None, lx=0.0, ly=0.0):
        ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=9, color=INK2, lw=0.9))
        if label:
            ax.text((p[0] + q[0]) / 2 + lx, (p[1] + q[1]) / 2 + ly, label, fontsize=7, color=A_C, weight="bold",
                    ha="center", va="center")

    ax.add_patch(FancyBboxPatch((1, 27.5), 11, 7, boxstyle="round,pad=0.3,rounding_size=1.2", fc=BOX["in"],
                                ec=INK2, lw=0.8))
    ax.text(6.5, 31, "frame t\n(team X\nin control)", ha="center", va="center", fontsize=7.4, color=INK)
    question(25, 31, "dead-ball\nevidence?")
    question(47, 31, "touch by the\nother team\nwithin ±5 fr?")
    question(69, 31, "new run ≥ 1 s,\nor visible ball\n+ double touch?")
    arrow((12.3, 31), (15.5, 31))
    arrow((34.5, 31), (37.5, 31), "no", 0, 1.6)
    arrow((56.5, 31), (59.5, 31), "yes", 0, 1.6)
    arrow((78.5, 31), (82.5, 31), "yes", 0, 1.6)
    outcome(25, 12, "D, then restart", "D until the frame t* where\nthe ball leaves rest;\nthen the team taking it", "#e9e8e4")
    outcome(47, 12, "hold", "team X keeps the ball:\nloose balls, flights,\ndeflections", "#dce9fb")
    outcome(69, 12, "absorb", "flicker or failed\nchallenge: no change", "#dce9fb")
    outcome(91.5, 31, "change", "on the frame\nof the touch", "#fde3d6")
    arrow((25, 24.5), (25, 16.5), "yes", 2.2, 0)
    arrow((47, 24.5), (47, 16.5), "no", 2.0, 0)
    arrow((69, 24.5), (69, 16.5), "no", 2.0, 0)
    ax.text(50, 2.2, "Ball hidden: the last and next possessor are carried over the gap; in long gaps the sequence model "
            "places the change where the collective movement reverses.", ha="center", fontsize=7, color=INK2,
            style="italic")
    save(fig, "possession_rules.png")


# ------------------------------------------------------------------------------------- match figures
def _runs(lab):
    """Start, end (exclusive) and value of each run of equal labels."""
    lab = np.asarray(lab)
    cut = np.flatnonzero(lab[1:] != lab[:-1]) + 1
    s = np.r_[0, cut]
    e = np.r_[cut, len(lab)]
    return s, e, lab[s]


def _strip(ax, lab, x0, scale, y, h):
    s, e, v = _runs(lab)
    col = {"H": H_C, "A": A_C, "D": D_C}
    for a, b, c in zip(s, e, v):
        if c in col:
            ax.add_patch(plt.Rectangle((x0 + a * scale, y), (b - a) * scale, h, fc=col[c], ec="none"))


def match_figures(data_dir, submission, gt_path, cache_dir):
    import pandas as pd

    sys.path.insert(0, str(REPO / "src"))
    from mtah2027.config import HALF1, HALF2, HALF2_PLAY
    from mtah2027.io.atd_loader import load_atd

    n = 199319
    fps = 25
    win = 30 * fps
    td = load_atd(data_dir, cache_dir=cache_dir, use_cache=cache_dir is not None, n_frames=n)
    ball = (~np.isnan(td.ball_xy).any(axis=1)).astype(float)
    players = (~np.isnan(td.players_xy).any(axis=2)).sum(axis=1).astype(float)
    k = n // win
    bm = ball[:k * win].reshape(k, win).mean(1)
    pm = players[:k * win].reshape(k, win).mean(1)
    t = (np.arange(k) + 0.5) * win / fps / 60
    fig, axes = plt.subplots(2, 1, figsize=(10, 3.6), sharex=True)
    for ax, v, lab, c in ((axes[0], bm, "ball tracked\n(30 s mean)", H_C), (axes[1], pm, "tracked players\n(30 s mean)", A_C)):
        for a, b in (HALF1, HALF2):
            ax.axvspan(a / fps / 60, b / fps / 60, color="#efeeea", zorder=0)
        ax.axvspan(11185 / fps / 60, 18684 / fps / 60, color="#cfe3d8", zorder=0)
        ax.plot(t, v, color=c, lw=1.1)
        ax.set_ylabel(lab, fontsize=8)
        ax.grid(axis="x", visible=False)
    axes[0].set_ylim(0, 1.05)
    axes[1].set_xlabel("video minute")
    graded = np.zeros(n, bool)
    graded[HALF1[0]:HALF1[1] + 1] = True
    graded[HALF2[0]:HALF2[1] + 1] = True
    pub = np.zeros(n, bool)
    pub[11185:18685] = True
    axes[0].set_title(f"Ball tracked on {ball[graded].mean():.0%} of graded frames but {ball[pub].mean():.0%} "
                      "of the public window (green); grey = graded halves")
    save(fig, "ball_presence.png")

    sub = pd.read_csv(submission)["label"].astype(str).str.strip().str.upper().to_numpy()
    gt = pd.read_csv(gt_path)
    gt_lab = gt["label"].astype(str).str.strip().str.upper().to_numpy()
    gt_frames = gt["frame"].to_numpy()
    a, b = int(gt_frames.min()), int(gt_frames.max()) + 1
    full_gt = np.full(n, "X", dtype=object)
    full_gt[gt_frames] = gt_lab
    fig, ax = plt.subplots(figsize=(10, 2.2))
    _strip(ax, full_gt[a:b], 0, 1 / fps, 1.2, 0.8)
    _strip(ax, sub[a:b], 0, 1 / fps, 0.1, 0.8)
    ax.set_xlim(0, (b - a) / fps)
    ax.set_ylim(0, 2.1)
    ax.set_yticks([0.5, 1.6], ["Opus Operandi", "ground truth"])
    ax.set_xlabel(f"seconds into the public window (frames {a} to {b - 1})")
    ax.grid(False)
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=H_C, label="H Barcelona"), Patch(color=A_C, label="A Athletic"),
                       Patch(color=D_C, label="D dead ball")], ncol=3, loc="upper center",
              bbox_to_anchor=(0.5, 1.32), fontsize=8)
    save(fig, "public_window.png")

    fig = plt.figure(figsize=(10, 3.4))
    gs = fig.add_gridspec(2, 2, width_ratios=[3, 1], hspace=0.5)
    lengths = {"H": [], "A": []}
    for i, (h0, h1) in enumerate(((HALF1[0], HALF1[1] + 1), (HALF2[0], HALF2_PLAY[1] + 1))):
        ax = fig.add_subplot(gs[i, 0])
        seg = sub[h0:h1]
        _strip(ax, seg, 0, 1 / fps / 60, 0, 1)
        ax.set_xlim(0, (h1 - h0) / fps / 60)
        ax.set_ylim(0, 1)
        ax.set_yticks([0.5], [f"half {i + 1}"])
        ax.grid(False)
        s, e, v = _runs(seg)
        for x0, x1, c in zip(s, e, v):
            if c in lengths:
                lengths[c].append((x1 - x0) / fps)
    fig.axes[-1].set_xlabel("minutes from the start of the graded half")
    ax = fig.add_subplot(gs[:, 1])
    allp = lengths["H"] + lengths["A"]
    bins = np.arange(0, 80, 4)
    ax.hist([np.clip(lengths["H"], 0, 79), np.clip(lengths["A"], 0, 79)], bins=bins, stacked=True,
            color=[H_C, A_C], label=["Barcelona", "Athletic"], edgecolor=SURF, linewidth=0.6)
    ax.set_xlabel("possession length (s)")
    ax.set_title(f"{len(allp)} possessions, mean {np.mean(allp):.1f} s", fontsize=8.5)
    ax.legend(fontsize=7.5)
    ax.grid(axis="x", visible=False)
    fig.suptitle("Full-match prediction: possessions and stoppages in both halves", fontsize=9.5, weight="bold")
    save(fig, "full_match.png")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--match", action="store_true", help="also draw the three figures of the provided match")
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--cache-dir", default=str(REPO / "cache"))
    ap.add_argument("--submission", default=str(REPO / "outputs" / "Opus Operandi.csv"))
    ap.add_argument("--gt", default=str(REPO / "evaluation" / "gt" / "BAR-ATH_possession_gt_public.csv"))
    ap.add_argument("--only-match", action="store_true", help="skip the aggregate figures")
    a = ap.parse_args(argv)
    if not a.only_match:
        d = json.loads(DATA.read_text(encoding="utf-8"))
        fig_validation_ladder(d)
        fig_validation_table(d)
        fig_grid73(d)
        fig_ledger(d)
        fig_world(d)
        fig_turnovers(d)
        fig_oracle(d)
        fig_scoreboard(d)
        fig_pipeline()
        fig_decoder()
        fig_degrader()
        fig_rules()
    if a.match or a.only_match:
        match_figures(a.data_dir, a.submission, a.gt, a.cache_dir)


if __name__ == "__main__":
    main()
