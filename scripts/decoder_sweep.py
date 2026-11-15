#!/usr/bin/env python
"""Issue #44: decoder / fusion / ensemble sweep on external matches (aggregates only, never BAR-ATH).

A *cell* is one held-out evaluation: (world, match, seed).  Worlds:

* ``b33`` / ``b1``: B' = 7 IDSSE leave-one-match-out, held-out match degraded by the calibrated degrader at
  ``dead_affinity`` 33 / 1 (seed = ``seed + index`` of the match, as ``gap_curve.py``);
* ``pff``: A = the first 20 PFF matches, each scored with the model of its 4-fold split (real occlusions);
* ``unseen``: A-unseen = the last 8 PFF matches with the committed (all-data) models.

The front end, the model probabilities and the heuristic emissions of a cell do not depend on the decoder
parameters, so they are computed once per cell (monkey-patched caches around the unmodified ``run_core``) and
every *candidate* (``route_overrides.specialists`` overrides + optional ensemble) only re-runs the cheap tail
(fusion priors / scaling, transitions, Viterbi, spell optimiser).  ``--verify`` checks the cached path against
an uncached ``run_core`` for the first candidate.

    python scripts/decoder_sweep.py run --world b33 --seeds 0 1 --candidates cands.json --models-root W \
        --out-dir W/s1 --shard 0/4
    python scripts/decoder_sweep.py agg --out-dirs W/s1 [--baseline base] [--strata NAME]

``cands.json``: ``[{"name": "...", "route": {"transition_matrix.ha_occluded": -7,
"specialist_fusion.scale_hidden": 0.9}, "ensemble": {"members": [["gap", 0.5], ["mixed", 0.5]],
"mode": "loglinear"}}, ...]`` (no ensemble = ``gap`` alone).  Models: ``<models-root>/lomo_<variant>/<match>.json``
and ``pff_<variant>/fold<k>.json`` (``scripts/train_cv_models.py``); the committed ``models/specialists_<variant>.json``
for ``unseen``.
"""
# ruff: noqa: E501
from __future__ import annotations

import argparse
import copy
import dataclasses
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
for _extra in (REPO_ROOT / "src", REPO_ROOT, REPO_ROOT / "scripts"):
    if str(_extra) not in sys.path:
        sys.path.insert(0, str(_extra))

DEFAULT_BENCH = REPO_ROOT.parent / "mtah2027-external-benchmarks" / "cache"
WORLDS = ("b33", "b1", "pff", "unseen")


# ---------------------------------------------------------------------------------------------
# configuration, candidates, cells
# ---------------------------------------------------------------------------------------------
def candidate_config(raw: dict, route: dict) -> dict:
    """``raw`` (the parsed YAML) with ``route_overrides.specialists.<section>.<key>`` overridden."""
    cfg = copy.deepcopy(raw)
    spec = cfg.setdefault("route_overrides", {}).setdefault("specialists", {})
    for dotted, value in route.items():
        section, _, key = dotted.partition(".")
        spec.setdefault(section, {})[key] = list(value) if isinstance(value, (list, tuple)) else value
    return cfg


def cells_for(world: str, bench: Path, seeds, n_pff: int = 20, n_unseen: int = 8) -> list[dict]:
    if world in ("b33", "b1"):
        files = sorted(bench.glob("idsse_*.npz"))
        return [{"world": world, "match": f.stem, "seed": int(s), "index": i, "path": str(f)}
                for s in seeds for i, f in enumerate(files)]
    pff = sorted(bench.glob("pff_*.npz"))
    if world == "pff":
        return [{"world": world, "match": f.stem, "seed": 0, "index": i, "path": str(f), "fold": i % 4}
                for i, f in enumerate(pff[:n_pff])]
    return [{"world": world, "match": f.stem, "seed": 0, "index": i, "path": str(f)}
            for i, f in enumerate(pff[-n_unseen:])]


def cell_id(cell: dict) -> str:
    return f"{cell['world']}__{cell['match']}__s{cell['seed']}"


def load_cell_data(cell: dict, config_raw: dict):
    from mtah2027.models.external_degrader import DegraderParams, degrade_match
    from mtah2027.validation import eval_bridge as eb

    z = np.load(cell["path"], allow_pickle=True)
    players, team = z["players_xy"].astype(np.float32), z["team_of_slot"].astype(np.int8)
    ball = z["ball_xy"].astype(np.float32)
    fps, pitch = float(z["fps"]), tuple(float(v) for v in z["pitch"])
    periods = [(int(a), int(b)) for a, b in z["period_bounds"]]
    gt = eb.to_codes(z["gt"])
    if "ball_visible" in z.files:
        ball = np.where(z["ball_visible"][:, None], ball, np.float32(np.nan)).astype(np.float32)
    if cell["world"] in ("b33", "b1"):
        aff = 33.0 if cell["world"] == "b33" else 1.0
        base = DegraderParams.from_config(config_raw)
        dm = degrade_match(players, team, ball, gt, fps=fps, seed=cell["seed"] + cell["index"],
                           params=dataclasses.replace(base, dead_affinity=aff))
        players, ball = dm.players_xy, dm.ball_xy
    return players, team, ball, fps, pitch, periods, gt


def model_path(root: Path, cell: dict, variant: str) -> Path:
    if cell["world"] in ("b33", "b1"):
        return root / f"lomo_{variant}" / f"{cell['match']}.json"
    if cell["world"] == "pff":
        return root / f"pff_{variant}" / f"fold{cell['fold']}.json"
    own = root / f"unseen_{variant}.json"      # Issue #55: candidate all-data model (else the committed one)
    return own if own.is_file() else REPO_ROOT / "models" / f"specialists_{variant}.json"


def build_specialists(cand: dict, cell: dict, root: Path, store: dict, config_raw: dict):
    from mtah2027.models import specialist_fusion as sf

    ens = cand.get("ensemble")
    members = ens["members"] if ens else [["gap", 1.0]]
    sets = []
    for variant, weight in members:
        key = (variant, cell["match"], cell["world"])
        if key not in store:
            store[key] = sf.load_specialists(model_path(root, cell, variant), config_raw)
        sets.append((store[key], float(weight)))
    first = sets[0][0]
    if not ens:
        return first
    return dataclasses.replace(first, ensemble=tuple(sets), mix_mode=ens.get("mode", "loglinear"))


# ---------------------------------------------------------------------------------------------
# cell evaluation with cached front end / model probabilities
# ---------------------------------------------------------------------------------------------
class CellCache:
    """Monkey-patches the expensive, decoder-independent steps of ``run_core`` for one cell."""

    def __init__(self, ball: np.ndarray):
        from mtah2027 import pipeline as pl
        from mtah2027.models import specialist_fusion as sf

        self.ball, self.fe, self.heur, self.fm, self.members = ball, None, None, None, {}
        self.pl, self.sf = pl, sf
        self.orig = (pl.compute_front_end, pl.heuristic_emissions, sf.build_feature_matrix,
                     sf.member_probabilities)

    def install(self) -> None:
        o_fe, o_he, o_fm, o_mp = self.orig

        def fe(*a, **k):
            ball = a[2] if len(a) > 2 else k.get("ball_xy")
            if ball is not self.ball:
                return o_fe(*a, **k)
            if self.fe is None:
                self.fe = o_fe(*a, **k)
            return self.fe

        def he(fe_, *a, **k):
            if self.heur is None or self.heur[0] is not fe_:
                self.heur = (fe_, o_he(fe_, *a, **k))
            return self.heur[1]

        def fm(fe_, heuristic, **k):
            if self.fm is None or self.fm[0] is not fe_:
                self.fm = (fe_, o_fm(fe_, heuristic, **k))
            return self.fm[1]

        def mp(member, fm_, fe_, heuristic, **k):
            key = id(member)
            if key not in self.members:
                self.members[key] = (member, o_mp(member, fm_, fe_, heuristic, **k))
            return self.members[key][1]

        self.pl.compute_front_end, self.pl.heuristic_emissions = fe, he
        self.sf.build_feature_matrix, self.sf.member_probabilities = fm, mp

    def uninstall(self) -> None:
        self.pl.compute_front_end, self.pl.heuristic_emissions = self.orig[0], self.orig[1]
        self.sf.build_feature_matrix, self.sf.member_probabilities = self.orig[2], self.orig[3]


def run_cell(cell: dict, cands: list, root: Path, config_raw: dict, verify: bool, log) -> dict:
    from eval_external import score

    from mtah2027.pipeline import run_core
    from mtah2027.validation import dropout_strata as ds

    players, team, ball, fps, pitch, periods, gt = load_cell_data(cell, config_raw)
    visible = np.isfinite(ball).all(axis=1)
    n_tracked = ds.tracked_counts(players)
    store: dict = {}
    out: dict = {"cell": {k: v for k, v in cell.items() if k != "path"}, "results": {}}
    t0 = time.perf_counter()
    cache = CellCache(ball)
    cache.install()
    try:
        for ci, cand in enumerate(cands):
            cfg = candidate_config(config_raw, cand.get("route", {}))
            spec = build_specialists(cand, cell, root, store, config_raw)
            core = run_core(players, team, ball, play_blocks=periods, fps=fps, pitch=pitch, config_path=cfg,
                            log=lambda m: None, specialists=spec, durations=spec.durations)
            row = score(gt, core.codes)
            row["strata"] = ds.stratum_counts(gt, core.codes, n_tracked, visible)
            out["results"][cand["name"]] = row
            if verify and ci == 0:
                cache.uninstall()
                ref = run_core(players, team, ball, play_blocks=periods, fps=fps, pitch=pitch, config_path=cfg,
                               log=lambda m: None, specialists=spec, durations=spec.durations)
                cache.install()
                out["verified"] = bool(np.array_equal(ref.codes, core.codes))
                log(f"  verify {cell_id(cell)}: cached == uncached -> {out['verified']}")
    finally:
        cache.uninstall()
    out["seconds"] = round(time.perf_counter() - t0, 1)
    return out


def cmd_run(args) -> int:
    import yaml

    cfg_path = Path(args.config) if args.config else REPO_ROOT / "configs" / "default.yaml"
    config_raw = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    cands = json.loads(Path(args.candidates).read_text(encoding="utf-8"))
    names = [c["name"] for c in cands]
    if len(set(names)) != len(names):
        raise SystemExit("candidate names must be unique")
    cells = cells_for(args.world, Path(args.bench_dir), args.seeds, n_pff=args.n_pff)
    k, _, m = args.shard.partition("/")
    cells = [c for i, c in enumerate(cells) if i % int(m) == int(k)]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for cell in cells:
        path = out_dir / f"{cell_id(cell)}.json"
        have = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {"results": {}}
        todo = [c for c in cands if c["name"] not in have["results"]]
        if not todo:
            continue
        res = run_cell(cell, todo, Path(args.models_root), config_raw, args.verify, lambda s: print(s, flush=True))
        res["results"] = {**have["results"], **res["results"]}
        path.write_text(json.dumps(res), encoding="utf-8")
        print(f"{cell_id(cell)}: {len(todo)} candidates in {res['seconds']} s", flush=True)
    return 0


# ---------------------------------------------------------------------------------------------
# aggregation
# ---------------------------------------------------------------------------------------------
def load_results(out_dirs) -> dict:
    """``{(world, seed): {match: {candidate: row}}}`` from the cell files."""
    table: dict = {}
    for d in out_dirs:
        for p in sorted(Path(d).glob("*.json")):
            r = json.loads(p.read_text(encoding="utf-8"))
            c = r["cell"]
            table.setdefault((c["world"], c["seed"]), {}).setdefault(c["match"], {}).update(r["results"])
    return table


def summarise(table: dict, world: str, seeds, cand: str, key: str = "spell_f1") -> dict:
    per_seed = {}
    for s in seeds:
        rows = [m[cand][key] for m in table.get((world, s), {}).values() if cand in m]
        if rows:
            per_seed[s] = float(np.mean(rows))
    return {"mean": float(np.mean(list(per_seed.values()))) if per_seed else float("nan"), "per_seed": per_seed}


def candidates_in(table: dict) -> list:
    names: list = []
    for byseed in table.values():
        for byc in byseed.values():
            for n in byc:
                if n not in names:
                    names.append(n)
    return names


def build_rows(table: dict, names: list, base: str) -> tuple[list, list]:
    worlds = sorted({w for w, _ in table})
    seeds_by_world = {w: sorted({s for ww, s in table if ww == w}) for w in worlds}
    rows = []
    for n in names:
        row = {"name": n}
        for w in worlds:
            sm = summarise(table, w, seeds_by_world[w], n)
            bm = summarise(table, w, seeds_by_world[w], base)
            row[w] = {**sm, "delta": sm["mean"] - bm["mean"],
                      "seed_delta": {s: sm["per_seed"][s] - bm["per_seed"][s]
                                     for s in sm["per_seed"] if s in bm["per_seed"]},
                      "macro": summarise(table, w, seeds_by_world[w], n, "macro_f1")["mean"]}
        rows.append(row)
    return rows, worlds


def cmd_agg(args) -> int:
    from mtah2027.validation import dropout_strata as ds

    table = load_results(args.out_dirs)
    names = candidates_in(table)
    base = args.baseline or names[0]
    rows, worlds = build_rows(table, names, base)
    sort_world = args.sort_world or worlds[0]
    shown = sorted(rows, key=lambda r: -r[sort_world]["mean"])
    if args.top:
        shown = shown[:args.top]
    print(f"baseline: {base}   (spellF1; d = mean delta vs baseline)")
    print(f"{'candidate':<46}" + "".join(f"{w:>9}{'d':>8}" for w in worlds))
    for r in shown:
        print(f"{r['name']:<46}" + "".join(f"{r[w]['mean']:9.4f}{r[w]['delta']:+8.4f}" for w in worlds))
    for n in args.detail or []:
        r = next(x for x in rows if x["name"] == n)
        print(f"-- {n}")
        for w in worlds:
            ps = {s: round(v, 4) for s, v in r[w]["per_seed"].items()}
            ds_ = {s: round(v, 4) for s, v in r[w]["seed_delta"].items()}
            print(f"   {w}: mean {r[w]['mean']:.4f} macro {r[w]['macro']:.4f} per seed {ps} delta per seed {ds_}")
    if args.strata:
        for w in worlds:
            for n in dict.fromkeys([base, args.strata]):
                cells = [m[n] for (ww, _), by in table.items() if ww == w for m in by.values() if n in m]
                pooled = ds.pool([c["strata"] for c in cells])
                total = pooled["lt20"]["frames"] + pooled["ge20"]["frames"]
                print(f"strata [{w}] {n}:")
                for line in ds.format_table(ds.summarize(pooled), total):
                    print("  " + line)
    if args.json:
        Path(args.json).write_text(json.dumps(rows, indent=1), encoding="utf-8")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--world", choices=WORLDS, required=True)
    r.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
    r.add_argument("--candidates", required=True)
    r.add_argument("--models-root", required=True)
    r.add_argument("--out-dir", required=True)
    r.add_argument("--bench-dir", default=str(DEFAULT_BENCH))
    r.add_argument("--config", default=None)
    r.add_argument("--shard", default="0/1")
    r.add_argument("--n-pff", type=int, default=20, help="PFF matches of the 4-fold world (Issue #55: 56)")
    r.add_argument("--verify", action="store_true")
    a = sub.add_parser("agg")
    a.add_argument("--out-dirs", nargs="+", required=True)
    a.add_argument("--baseline", default=None)
    a.add_argument("--sort-world", default=None)
    a.add_argument("--top", type=int, default=0)
    a.add_argument("--detail", nargs="*", default=None)
    a.add_argument("--strata", default=None, help="candidate whose pooled strata are printed next to the baseline")
    a.add_argument("--json", default=None)
    args = ap.parse_args(argv)
    return cmd_run(args) if args.cmd == "run" else cmd_agg(args)


if __name__ == "__main__":
    raise SystemExit(main())
