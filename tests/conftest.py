"""Shared fixtures: a deterministic synthetic BAR-ATH match (Issue #3).

Tests never touch the real match data (``data/``, ``cache/``); everything runs on this generator.
"""
from __future__ import annotations

import importlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
for extra in (ROOT / "scripts", ROOT):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

SYNTHETIC_SEED = 0
SYNTHETIC_FRAMES = 12000


@pytest.fixture(scope="session")
def synth_module():
    return importlib.import_module("generate_synthetic_fixtures")


@pytest.fixture(scope="session")
def synthetic_info(tmp_path_factory, synth_module):
    """Summary dict of a generated short match (also contains the paths and the true labels)."""
    out = tmp_path_factory.mktemp("synthetic_match")
    info = synth_module.generate_synthetic_match(out, seed=SYNTHETIC_SEED, n_frames=SYNTHETIC_FRAMES)
    info["dir"] = out
    return info


@pytest.fixture(scope="session")
def synthetic_dir(synthetic_info):
    """Bundle root usable as ``data_dir`` by ``games.asset_paths`` and the starter loaders."""
    return synthetic_info["dir"]


@pytest.fixture(scope="session")
def synthetic_spec(synthetic_info):
    return synthetic_info["spec"]


@pytest.fixture(scope="session")
def synthetic_gt(synthetic_dir):
    """Ground-truth labels of the synthetic match as an array of 'H'/'A'/'D'/'X' strings."""
    table = pd.read_csv(synthetic_dir / "synthetic_gt.csv", dtype={"label": "object"})
    return table["label"].to_numpy(dtype="<U1")


def parse_atd(path):
    """Parse an ATD raw file into (frame_ids, players[T, S, 3], ball[T, 3]) float arrays."""
    ids, players, balls = [], [], []
    with open(path, encoding="ascii") as handle:
        for line in handle:
            frame, slots, ball = line.rstrip("\n").split(":")
            ids.append(int(frame))
            players.append([[float(v) for v in g.split(",")] for g in slots.split(";")])
            balls.append([float(v) for v in ball.split(",")])
    return np.array(ids), np.array(players), np.array(balls)


@pytest.fixture(scope="session")
def tactical_atd(synthetic_info):
    return parse_atd(synthetic_info["paths"]["tactical_atd_txt"])


@pytest.fixture(scope="session")
def broadcast_atd(synthetic_info):
    return parse_atd(synthetic_info["paths"]["broadcast_atd_txt"])


@pytest.fixture(scope="session")
def synthetic_full_dir(tmp_path_factory, synth_module):
    """Full-length (199,319 frames) synthetic bundle; expensive, used only by ``slow`` tests."""
    out = tmp_path_factory.mktemp("synthetic_full")
    synth_module.generate_synthetic_match(out, seed=SYNTHETIC_SEED, full_length=True)
    return out
