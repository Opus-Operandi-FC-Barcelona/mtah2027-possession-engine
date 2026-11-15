#!/usr/bin/env python
# ruff: noqa: E501
"""Issue #72: export trained TCN ``.npz`` file(s) as the committed ``models/sequence_tcn.json`` (no torch needed).

    python scripts/export_sequence_json.py --out models/sequence_tcn.json S0/full.npz S1/full.npz

One input -> a single model; several -> an ensemble (mean of the softmax probabilities).  Weights are stored as
shortest-repr float32 decimals, so loading gives back the exact trained float32 values.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from mtah2027.models.sequence_model import (  # noqa: E402
    SequenceEnsemble,
    SequenceModel,
    load_sequence_json,
    save_sequence_json,
)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("npz", nargs="+")
    ap.add_argument("--out", required=True)
    ap.add_argument("--note", default="", help="free-text provenance stored in the metadata")
    a = ap.parse_args(argv)
    members = [SequenceModel.load(p) for p in a.npz]
    meta = {"combine": "mean of softmax probabilities", "probability_decimals": 6, "emission_variant": "E100",
            "training_seeds": [m.meta.get("seed") for m in members], "note": a.note,
            "data": "degraded external matches only (IDSSE dead_affinity 33 / 1, PFF 20..55 with real occlusions); "
                    "no BAR-ATH data"}
    model = members[0] if len(members) == 1 else SequenceEnsemble(members, meta)
    out = save_sequence_json(model, a.out, meta if len(members) == 1 else None)
    back = load_sequence_json(out)
    got = back.members if isinstance(back, SequenceEnsemble) else [back]
    import numpy as np
    assert all(np.array_equal(x.w[k], y.w[k]) for x, y in zip(members, got) for k in x.w), "round trip mismatch"
    print(f"wrote {out} ({Path(out).stat().st_size / 1e6:.2f} MB, {len(got)} member(s)); round trip exact")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
