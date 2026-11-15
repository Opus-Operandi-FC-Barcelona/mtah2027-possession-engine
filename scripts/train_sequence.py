#!/usr/bin/env python
"""Issue #72: train the sequence model (TCN, PyTorch) on the dense external sequences of ``sequence_data.py``.

Needs PyTorch (GPU venv); the trained weights are exported to ``.npz`` and used by NumPy only
(``mtah2027.models.sequence_model``).  Training data are degraded external matches (IDSSE worlds at
``dead_affinity`` 33 / 1 and PFF with real occlusions); window sampling follows the world mixture of
Issue #71 (``--mixture``).  ``--exclude`` leaves a match out (leave-one-match-out); ``--val`` holds out PFF
training matches for a dev check (never an evaluation cell).

    python scripts/train_sequence.py --data-dir D --out M/lomo/idsse_J03WMX.npz --exclude idsse_J03WMX

Issue #74 (both off by default, the #72 network is reproduced): ``--extra-dir`` appends the carrier / swarm
columns of Issue #64 (``sequence_data.py build --ref-dir``) to the input; ``--boundary-weight`` multiplies the
emission loss of the frames within ``--boundary-window`` frames of a true direct H<->A change.

Issue #78 (off by default, the #72 network is reproduced): ``--crf c1`` trains the emission head with the
negative log-likelihood of a linear-chain CRF (learned global 3x3 transition matrix, zero diagonal) plus
``--crf-ce-weight`` times the per-frame cross-entropy; ``--crf c2`` also makes the H<->A transition scores a learned
function of the TCN features at each frame (``-softplus(w . h_t + b)``, exported as ``w_ha`` / ``b_ha``).
"""
# ruff: noqa: E501
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
for _extra in (REPO_ROOT / "src", REPO_ROOT):
    if str(_extra) not in sys.path:
        sys.path.insert(0, str(_extra))

from mtah2027.models.sequence_model import SequenceModel, turnover_targets  # noqa: E402

DILATIONS = (1, 2, 4, 8, 16, 32, 64, 128)


NEG = -1.0e4           # "minus infinity" of the log-semiring (finite: no NaN gradients)
HA_BIAS_INIT = 6.0     # -softplus(6) = -6.0: the H<->A switch starts as expensive as the production occluded cost


def crf_global(raw):
    """Learned 3x3 transition matrix with a zero diagonal."""
    return raw * (1.0 - torch.eye(3, device=raw.device))


def crf_nll(unary, trans, y, valid):
    """Mean (per valid frame) negative log-likelihood of a linear chain.

    ``unary[B, T, 3]``, ``trans[B, T, 3, 3]`` (``trans[:, t, i, j]`` scores ``t-1 -> t``), ``y[B, T]`` and ``valid[B, T]``
    (a prefix of every row: padding at the end).  The partition function is a log-semiring product of the
    per-frame matrices ``M_t[i, j] = trans[t, i, j] + unary[t, j]`` reduced as a balanced tree (log2 T levels
    instead of T sequential steps); padded frames are identity matrices.  ``numpy`` mirror: ``models.sequence_crf``.
    """
    b, t, _ = unary.shape
    yy = y.clamp(max=2)
    vm = valid.float()
    gold = (unary.gather(-1, yy[..., None])[..., 0] * vm).sum(1)
    if t > 1:
        pair = trans[:, 1:].gather(-2, yy[:, :-1, None, None].expand(b, t - 1, 1, 3))[:, :, 0]
        gold = gold + (pair.gather(-1, yy[:, 1:, None])[..., 0] * vm[:, 1:]).sum(1)
    m = trans[:, 1:] + unary[:, 1:, None, :]                       # [B, T-1, 3, 3]
    eye = torch.where(torch.eye(3, dtype=torch.bool, device=unary.device), 0.0, NEG)
    m = torch.where(valid[:, 1:, None, None], m, eye.expand_as(m))
    length = m.shape[1]
    size = 1 << max(0, (length - 1).bit_length()) if length > 1 else 1
    if size > length:
        m = torch.cat([m, eye.expand(b, size - length, 3, 3)], dim=1)
    while m.shape[1] > 1:
        left, right = m[:, 0::2], m[:, 1::2]
        m = torch.logsumexp(left[..., :, :, None] + right[..., None, :, :], dim=-2)
    p = m[:, 0] if length >= 1 else eye.expand(b, 3, 3)
    logz = torch.logsumexp(unary[:, 0, :, None] + p, dim=(-2, -1))
    return ((logz - gold).sum() / vm.sum().clamp(min=1.0))


class TCN(nn.Module):
    def __init__(self, n_in: int, channels: int, dilations=DILATIONS, dropout: float = 0.1, in_drop: float = 0.1,
                 crf: str = "none"):
        super().__init__()
        self.crf = crf
        if crf != "none":
            self.crf_raw = nn.Parameter(torch.full((3, 3), -6.0))
        if crf == "c2":
            self.ha = nn.Linear(channels, 2)
            nn.init.zeros_(self.ha.weight)
            nn.init.constant_(self.ha.bias, HA_BIAS_INIT)
        self.inp = nn.Linear(n_in, channels)
        self.c1 = nn.ModuleList([nn.Conv1d(channels, channels, 3, dilation=d, padding=d) for d in dilations])
        self.c2 = nn.ModuleList([nn.Linear(channels, channels) for _ in dilations])
        self.drop = nn.Dropout(dropout)
        self.in_drop = nn.Dropout(in_drop)
        self.he = nn.Linear(channels, 3)
        self.ht = nn.Linear(channels, 3)

    def trunk(self, x):                         # x [B, T, F]
        h = self.inp(self.in_drop(x))
        for c1, c2 in zip(self.c1, self.c2):
            r = F.relu(c1(h.transpose(1, 2))).transpose(1, 2)
            h = h + c2(self.drop(r))
        return h

    def forward(self, x):
        h = self.trunk(x)
        return self.he(h), self.ht(h)

    def forward_crf(self, x):
        """Emission logits, turnover logits and the chain transitions ``[B, T, 3, 3]`` (C2: H<->A from the features)."""
        h = self.trunk(x)
        g = crf_global(self.crf_raw)
        trans = g.expand(x.shape[0], x.shape[1], 3, 3)
        if self.crf == "c2":
            ha = -F.softplus(self.ha(h))                                   # [B, T, 2]
            sel = torch.zeros(3, 3, device=x.device)
            sel[0, 1] = 1.0
            sel[1, 0] = 1.0
            a = torch.zeros(x.shape[0], x.shape[1], 3, 3, device=x.device)
            a[..., 0, 1] = ha[..., 0]
            a[..., 1, 0] = ha[..., 1]
            trans = trans * (1.0 - sel) + a
        return self.he(h), self.ht(h), trans

    def export(self, mean, std, names, dilations) -> SequenceModel:
        def np_(t):
            return t.detach().cpu().numpy()

        w = {"mean": mean, "std": std, "w_in": np_(self.inp.weight).T, "b_in": np_(self.inp.bias),
             "w_e": np_(self.he.weight).T, "b_e": np_(self.he.bias),
             "w_t": np_(self.ht.weight).T, "b_t": np_(self.ht.bias)}
        if self.crf != "none":
            w["crf_trans"] = np_(crf_global(self.crf_raw))
        if self.crf == "c2":
            w["w_ha"] = np_(self.ha.weight).T
            w["b_ha"] = np_(self.ha.bias)
        for i, (c1, c2) in enumerate(zip(self.c1, self.c2)):
            w[f"w1_{i}"] = np.stack([np_(c1.weight[:, :, k]).T for k in range(3)])
            w[f"b1_{i}"] = np_(c1.bias)
            w[f"w2_{i}"] = np_(c2.weight).T
            w[f"b2_{i}"] = np_(c2.bias)
        return SequenceModel(w, {"feature_names": list(names), "dilations": list(dilations),
                                 "channels": int(self.inp.out_features)})


def world_of(stem: str) -> str:
    return "pff" if stem.startswith("pff_") else ("idsse1" if "__a1_" in stem else "idsse33")


def load_variants(data_dir: Path, exclude, val_matches, extra_dir: Path | None = None):
    tr, va = [], []
    for p in sorted(data_dir.glob("*.npz")):
        match = p.stem.split("__")[0]
        if match == exclude:
            continue
        z = np.load(p)
        x = z["x"]
        if extra_dir is not None:
            x = np.concatenate([x, np.load(extra_dir / p.name)["x"]], axis=1)
        item = {"x": x, "y": z["y"], "vis": z["vis"], "blocks": z["blocks"], "world": world_of(p.stem),
                "match": match}
        (va if match in val_matches else tr).append(item)
    return tr, va


def windows_sampler(variants, mixture, win, rng):
    by_world = {w: [v for v in variants if v["world"] == w] for w in mixture}
    worlds = [w for w in mixture if by_world[w]]
    probs = np.array([mixture[w] for w in worlds], float)
    probs /= probs.sum()

    def one():
        w = worlds[rng.choice(len(worlds), p=probs)]
        v = by_world[w][rng.integers(len(by_world[w]))]
        a, b = v["blocks"][rng.integers(len(v["blocks"]))]
        start = int(rng.integers(a, max(a + 1, b - win + 2)))
        return v, slice(start, min(start + win, b + 1))
    return one


def batch(sample, bs, win, mean, std, device, hw, bw=0):
    xs, ys, ts, bs_ = [], [], [], []
    for _ in range(bs):
        v, s = sample()
        x = (v["x"][s].astype(np.float32) - mean) / std
        y = v["y"][s].astype(np.int64)
        n = len(y)
        if n < win:
            x = np.concatenate([x, np.zeros((win - n, x.shape[1]), np.float32)])
            y = np.concatenate([y, np.full(win - n, 3, np.int64)])
        t = turnover_targets(np.where(y == 3, 9, y), hw)
        xs.append(np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0))
        ys.append(y)
        ts.append(t)
        bs_.append(turnover_targets(np.where(y == 3, 9, y), bw) > 0 if bw else np.zeros(win, bool))
    return (torch.from_numpy(np.stack(xs)).to(device), torch.from_numpy(np.stack(ys)).to(device),
            torch.from_numpy(np.stack(ts)).to(device), torch.from_numpy(np.stack(bs_)).to(device))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--exclude", default=None)
    ap.add_argument("--val", nargs="*", default=[])
    ap.add_argument("--mixture", nargs=3, type=float, default=(0.320, 0.204, 0.476),
                    metavar=("IDSSE33", "IDSSE1", "PFF"))
    ap.add_argument("--channels", type=int, default=64)
    ap.add_argument("--steps", type=int, default=2500)
    ap.add_argument("--batch", type=int, default=12)
    ap.add_argument("--window", type=int, default=2048)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--turn-weight", type=float, default=6.0)
    ap.add_argument("--half-window", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--crf", choices=("none", "c1", "c2"), default="none", help="Issue #78: CRF NLL training")
    ap.add_argument("--crf-ce-weight", type=float, default=0.5, help="Issue #78: weight of the per-frame CE")
    ap.add_argument("--extra-dir", default=None, help="Issue #74: carrier / swarm columns appended to the input")
    ap.add_argument("--boundary-weight", type=float, default=1.0, help="Issue #74: emission loss weight near a change")
    ap.add_argument("--boundary-window", type=int, default=75, help="frames either side of a true H<->A change")
    args = ap.parse_args(argv)
    torch.set_num_threads(2)

    data_dir = Path(args.data_dir)
    names = json.loads((data_dir / "names.json").read_text(encoding="utf-8"))
    extra_dir = Path(args.extra_dir) if args.extra_dir else None
    if extra_dir is not None:
        names = names + json.loads((extra_dir / "names.json").read_text(encoding="utf-8"))
    tr, va = load_variants(data_dir, args.exclude, args.val, extra_dir)
    bw = args.boundary_window if args.boundary_weight != 1.0 else 0
    mixture = dict(zip(("idsse33", "idsse1", "pff"), args.mixture))
    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    sub = np.concatenate([v["x"][::50].astype(np.float32) for v in tr])
    mean = np.nan_to_num(np.nanmean(sub, axis=0)).astype(np.float32)
    std = np.nan_to_num(np.nanstd(sub, axis=0))
    std = np.where(std < 1e-6, 1.0, std).astype(np.float32)
    model = TCN(len(names), args.channels, crf=args.crf).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=args.steps, pct_start=0.1)
    sample = windows_sampler(tr, mixture, args.window, rng)
    tw = torch.tensor([1.0, args.turn_weight, args.turn_weight], device=device)
    t0 = time.perf_counter()
    for step in range(args.steps):
        model.train()
        x, y, t, nb = batch(sample, args.batch, args.window, mean, std, device, args.half_window, bw)
        if args.crf != "none":
            le, lt, trans = model.forward_crf(x)
        else:
            le, lt = model(x)
        m = y < 3
        if args.crf != "none":
            loss_e = crf_nll(le, trans, y, m) + args.crf_ce_weight * F.cross_entropy(le[m], y[m])
        elif bw:
            wgt = 1.0 + (args.boundary_weight - 1.0) * nb[m].float()
            loss_e = (F.cross_entropy(le[m], y[m], reduction="none") * wgt).mean()
        else:
            loss_e = F.cross_entropy(le[m], y[m])
        loss_t = F.cross_entropy(lt[m], t[m], weight=tw)
        loss = loss_e + loss_t
        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        if step % 250 == 0 or step == args.steps - 1:
            print(f"step {step}: emission {loss_e.item():.4f} turnover {loss_t.item():.4f} "
                  f"({time.perf_counter() - t0:.0f} s)", flush=True)
    sm = model.export(mean, std, names, DILATIONS)
    sm.meta.update({"steps": args.steps, "exclude": args.exclude, "mixture": list(args.mixture),
                    "seed": args.seed, "n_variants": len(tr),
                    "extra": bool(extra_dir), "boundary_weight": args.boundary_weight, "crf": args.crf})
    sm.save(args.out)
    if va:
        sv = windows_sampler(va, {"pff": 1.0, "idsse33": 1.0, "idsse1": 1.0}, args.window,
                             np.random.default_rng(99))
        model.eval()
        accs, ent, hid_acc = [], [], []
        with torch.no_grad():
            for _ in range(60):
                v, s = sv()
                x, y, t, _ = batch(lambda: (v, s), 1, args.window, mean, std, device, args.half_window)
                le, lt = model(x)
                m = y < 3
                accs.append((le.argmax(-1)[m] == y[m]).float().mean().item())
                vis = torch.from_numpy(v["vis"][s]).to(device)
                vis = F.pad(vis, (0, y.shape[1] - len(vis)))
                mh = m[0] & ~vis
                if mh.any():
                    hid_acc.append((le[0].argmax(-1)[mh] == y[0][mh]).float().mean().item())
                ent.append(F.cross_entropy(lt[m], t[m], weight=tw).item())
        print(f"val: acc {np.mean(accs):.4f} hidden acc {np.mean(hid_acc):.4f} turnover loss {np.mean(ent):.4f}")
    print(f"saved {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
