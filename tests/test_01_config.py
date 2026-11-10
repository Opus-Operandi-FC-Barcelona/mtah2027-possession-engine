import os
import random
import subprocess
import sys

import numpy as np
import pytest

from mtah2027 import config as cfg


def test_constants():
    assert cfg.N_VIDEO_FRAMES == 199319
    assert cfg.FPS == 25
    assert cfg.PITCH == (105.3, 68.0)
    assert cfg.HALF1 == (11185, 90651)
    assert cfg.HALF2_DEAD == (110808, 113741)
    assert cfg.HALF2_PLAY == (113742, 190197)


def test_graded_block_sizes_match_spec():
    sizes = [b - a + 1 for a, b in cfg.GRADED_BLOCKS]
    assert sizes == [79467, 2934, 76456]
    ungraded = [b - a + 1 for a, b in cfg.NOT_EVALUATED_BLOCKS]
    assert ungraded == [11185, 20156, 9121]
    assert sum(sizes) + sum(ungraded) == cfg.N_VIDEO_FRAMES


def test_games_cross_check_passes():
    assert cfg.cross_check_games() is True
    import games

    assert games.half_video_blocks("BAR-ATH") == [(1, 11185, 90651), (2, 110808, 190197)]


def test_games_cross_check_detects_mismatch():
    with pytest.raises(cfg.ConfigError):
        cfg.cross_check_games(cfg.MatchConfig(half1=(11186, 90651)))
    with pytest.raises(cfg.ConfigError):
        cfg.cross_check_games(cfg.MatchConfig(fps=30))


def test_yaml_load():
    conf = cfg.load_config()
    assert conf.team_name == "OpusOperandi"
    assert conf.seed == 0
    assert conf.match == cfg.MatchConfig()
    assert conf.match.graded_blocks == cfg.GRADED_BLOCKS
    assert conf.kpi_alarms.possessions == (150, 600)
    assert conf.kpi_alarms.mean_duration_s == (5.0, 30.0)
    assert conf.kpi_alarms.dead_pct_of_graded == (15.0, 55.0)
    assert conf.kpi_alarms.home_pct_of_alive == (35.0, 75.0)
    with pytest.raises(Exception):
        conf.seed = 1  # frozen


def test_yaml_mismatch_rejected(tmp_path):
    path = tmp_path / "bad.yaml"
    path.write_text("match:\n  half1: [11000, 90651]\n", encoding="utf-8")
    with pytest.raises(cfg.ConfigError):
        cfg.load_config(path)
    path.write_text("kpi_alarms:\n  possessions: [600, 150]\n", encoding="utf-8")
    with pytest.raises(cfg.ConfigError):
        cfg.load_config(path)


def test_set_global_seeds_deterministic():
    cfg.set_global_seeds(0)
    a = (random.random(), np.random.rand(3).tolist())
    cfg.set_global_seeds(0)
    b = (random.random(), np.random.rand(3).tolist())
    assert a == b
    assert os.environ["PYTHONHASHSEED"] == "0"
    cfg.set_global_seeds(7)
    assert os.environ["PYTHONHASHSEED"] == "7"
    cfg.set_global_seeds(0)


def test_import_finds_games_without_pythonpath_root():
    code = "import mtah2027.config as c; print(c.cross_check_games())"
    env = dict(os.environ, PYTHONPATH=os.path.join(str(cfg.REPO_ROOT), "src"))
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, env=env, cwd=str(cfg.REPO_ROOT.parent)
    )
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == "True"


def test_route_config_merges_overrides_without_touching_shared_sections(tmp_path):
    from mtah2027.config import load_section, route_config

    path = tmp_path / "cfg.yaml"
    path.write_text(
        "transition_matrix:\n  ha_closed: -11.5\n  ha_open: -1.4\n"
        "route_overrides:\n  heuristic:\n    transition_matrix:\n      ha_closed: -15.0\n",
        encoding="utf-8")
    merged = route_config("heuristic", path)
    assert load_section("transition_matrix", merged) == {"ha_closed": -15.0, "ha_open": -1.4}
    assert load_section("transition_matrix", path) == {"ha_closed": -11.5, "ha_open": -1.4}
    assert route_config("specialists", path) == path


def test_default_heuristic_overrides_are_valid_parameters():
    from mtah2027.config import route_config
    from mtah2027.decoding.transition_matrix import TransitionParams
    from mtah2027.models.heuristic_emissions import EmissionParams
    from mtah2027.postprocessing.spell_optimizer import SpellParams

    cfg = route_config("heuristic")
    assert isinstance(cfg, dict)
    assert TransitionParams.from_config(cfg) != TransitionParams.from_config()
    SpellParams.from_config(cfg)
    EmissionParams.from_config(cfg)
