"""Issue #75: decoder re-tuning grid, selection and adoption rule (scoring itself needs external data, not tested)."""
# ruff: noqa: E501
from __future__ import annotations

import sys

import pytest
import yaml

from mtah2027.config import REPO_ROOT, route_config
from mtah2027.postprocessing.spell_optimizer import SpellParams

sys.path.insert(0, str(REPO_ROOT / "scripts"))
import decoder_retune_eval as dr  # noqa: E402


def test_grid_is_the_preregistered_36_cells():
    assert len(dr.GRID) == len(set(dr.GRID)) == 36
    assert {c[0] for c in dr.GRID} == {-11.5, -9.5, -7.5, -5.5}
    assert {c[1] for c in dr.GRID} == {-6.0, -4.5, -3.0}
    assert {c[2] for c in dr.GRID} == {1.2, 0.6, 0.0}
    assert dr.NOMINAL in dr.GRID
    assert set(dr.named_decoders("grid")) == {dr.cell_name(c) for c in dr.GRID} | {"ship"}


def test_grid_values_reach_the_learned_route():
    raw = yaml.safe_load((REPO_ROOT / "configs" / "default.yaml").read_text(encoding="utf-8"))
    cfg = dr.decoder_config(raw, (-9.5, -4.5, 0.6), None)
    learned = route_config("specialists", cfg)
    assert learned["transition_matrix"]["ha_closed"] == -9.5
    assert learned["transition_matrix"]["ha_occluded"] == -4.5
    assert SpellParams.from_config(learned).micro_poss_s == 0.6
    assert raw["transition_matrix"]["ha_closed"] == -11.5          # the input is not modified


def test_selection_takes_best_seed0_cell_with_grid_order_ties(monkeypatch):
    def table(best):
        res = {dr.cell_name(c): {"spell_f1": 0.7 + (0.01 if c == best else 0.0)} for c in dr.GRID}
        return {("b33", 0): {"m": res}}

    assert dr.select_cell(table(dr.GRID[17]))[0] == dr.cell_name(dr.GRID[17])
    assert dr.select_cell(table(None))[0] == dr.cell_name(dr.GRID[0])


def test_adoption_rule():
    ok = dr.adoption({"1": 0.012, "2": 0.011}, {"1": 0.0, "2": -0.01}, -0.01)
    assert ok["adopted"]
    assert not dr.adoption({"1": 0.02, "2": -0.004}, {"1": 0.0}, 0.0)["adopted"]       # one seed below -0.003
    assert not dr.adoption({"1": 0.008, "2": 0.009}, {"1": 0.0}, 0.0)["adopted"]       # mean below +0.010
    assert not dr.adoption({"1": 0.02, "2": 0.02}, {"1": -0.02}, 0.0)["adopted"]       # B'(1) guard
    assert not dr.adoption({"1": 0.02, "2": 0.02}, {"1": 0.0}, -0.04)["adopted"]       # A guard


@pytest.mark.parametrize("cell", [dr.SHIPPED, dr.NOMINAL])
def test_reference_cells_are_not_grid_edges_misnamed(cell):
    assert dr.cell_name(cell).startswith("hc")
