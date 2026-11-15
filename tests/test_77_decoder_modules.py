"""Issue #77: decoder modules revisited with the sequence-model emissions (scoring itself needs external data)."""
# ruff: noqa: E501
from __future__ import annotations

import sys

import yaml

from mtah2027.config import REPO_ROOT, route_config
from mtah2027.decoding.duration_model import DurationParams
from mtah2027.decoding.expected_spellf1 import ExpectedSpellF1Params
from mtah2027.postprocessing.boundary_snap import BoundarySnapParams

sys.path.insert(0, str(REPO_ROOT / "scripts"))
import decoder_modules_eval as dm  # noqa: E402


def _raw():
    return yaml.safe_load((REPO_ROOT / "configs" / "default.yaml").read_text(encoding="utf-8"))


def test_variants_are_the_preregistered_ones():
    assert {"ship", "D1", "D05", "X", "S", "D1S", "D05S", "XS"} == set(dm.VARIANTS)
    assert dm.VARIANTS["D05"]["duration.weight"] == 0.5
    assert dm.VARIANTS["X"]["expected_spellf1.k"] == 100 and dm.VARIANTS["X"]["expected_spellf1.seed"] == 0
    assert dm.VARIANTS["XS"].items() >= dm.VARIANTS["S"].items()


def test_switches_are_read_by_the_learned_route():
    """duration / expected_spellf1 / boundary_snap are not overridden in route_overrides.specialists."""
    raw = _raw()
    assert not {"duration", "expected_spellf1", "boundary_snap"} & set(raw["route_overrides"]["specialists"])
    for name, check in (("D05", lambda c: DurationParams.from_config(c).enabled and DurationParams.from_config(c).weight == 0.5),
                        ("X", lambda c: ExpectedSpellF1Params.from_config(c).enabled),
                        ("S", lambda c: BoundarySnapParams.from_config(c).enabled)):
        learned = route_config("specialists", dm.variant_config(raw, name))
        assert check(learned), name
    base = route_config("specialists", dm.variant_config(raw, "ship"))
    assert not DurationParams.from_config(base).enabled and not BoundarySnapParams.from_config(base).enabled
    assert not raw["boundary_snap"]["enabled"]            # the input is not modified


def test_extra_overrides_parse():
    assert dm.parse_extra("sequence_model.visible_weight=0.5,duration.center=false") == {
        "sequence_model.visible_weight": 0.5, "duration.center": False}


def _table(scores):
    return {("b33", 0): {"m": {n: {"spell_f1": v} for n, v in scores.items()}}}


def test_selection_uses_better_duration_variant_for_the_combination():
    base = {"D1": 0.70, "D05": 0.71, "X": 0.70, "S": 0.70, "D1S": 0.99, "D05S": 0.72, "XS": 0.70}
    assert dm.select_variant(_table(base))[0] == "D05S"            # D1S is not a candidate (D1 < D05)
    assert dm.select_variant(_table({**base, "XS": 0.73}))[0] == "XS"
    assert dm.select_variant(_table({**base, "D1S": 0.70, "D05S": 0.70, "D05": 0.70, "D1": 0.70}))[0] == "D1"   # ties: listed order


def test_adoption_rule():
    assert dm.adoption({"1": 0.012, "2": 0.011}, {"1": 0.0, "2": -0.01}, -0.01)["adopted"]
    assert not dm.adoption({"1": 0.02, "2": -0.004}, {"1": 0.0}, 0.0)["adopted"]
    assert not dm.adoption({"1": 0.008, "2": 0.009}, {"1": 0.0}, 0.0)["adopted"]
    assert not dm.adoption({"1": 0.02, "2": 0.02}, {"1": -0.02}, 0.0)["adopted"]
    assert not dm.adoption({"1": 0.02, "2": 0.02}, {"1": 0.0}, -0.04)["adopted"]
