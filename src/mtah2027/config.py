"""Typed configuration, immutable match constants and the bridge to the official ``games.py``.

The constants below are the published BAR-ATH values. When the vendored ``games.py`` is importable
(repo root), they are cross-checked against ``games.GAMES`` and ``games.half_video_blocks`` and a
mismatch raises :class:`ConfigError` -- the official module is the source of truth.
"""
from __future__ import annotations

import importlib
import os
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any, Mapping

import numpy as np
import yaml

GAME = "BAR-ATH"
N_VIDEO_FRAMES = 199319
FPS = 25
PITCH = (105.3, 68.0)

# Inclusive video-frame blocks.
HALF1 = (11185, 90651)
HALF2_DEAD = (110808, 113741)
HALF2_PLAY = (113742, 190197)
HALF2 = (HALF2_DEAD[0], HALF2_PLAY[1])

GRADED_BLOCKS = (HALF1, HALF2_DEAD, HALF2_PLAY)
NOT_EVALUATED_BLOCKS = (
    (0, HALF1[0] - 1),
    (HALF1[1] + 1, HALF2_DEAD[0] - 1),
    (HALF2_PLAY[1] + 1, N_VIDEO_FRAMES - 1),
)

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = REPO_ROOT / "configs" / "default.yaml"


class ConfigError(ValueError):
    """Raised when a configuration or the constants disagree with the official sources."""


@dataclass(frozen=True)
class MatchConfig:
    game: str = GAME
    n_video_frames: int = N_VIDEO_FRAMES
    fps: int = FPS
    pitch: tuple[float, float] = PITCH
    half1: tuple[int, int] = HALF1
    half2_dead: tuple[int, int] = HALF2_DEAD
    half2_play: tuple[int, int] = HALF2_PLAY

    @property
    def graded_blocks(self) -> tuple[tuple[int, int], ...]:
        return (self.half1, self.half2_dead, self.half2_play)

    @property
    def not_evaluated_blocks(self) -> tuple[tuple[int, int], ...]:
        return (
            (0, self.half1[0] - 1),
            (self.half1[1] + 1, self.half2_dead[0] - 1),
            (self.half2_play[1] + 1, self.n_video_frames - 1),
        )


@dataclass(frozen=True)
class KpiAlarms:
    """Inclusive (low, high) alarm ranges. Alarms, never optimisation targets."""

    possessions: tuple[float, float] = (150, 600)
    mean_duration_s: tuple[float, float] = (5.0, 30.0)
    dead_pct_of_graded: tuple[float, float] = (15.0, 55.0)
    home_pct_of_alive: tuple[float, float] = (35.0, 75.0)


@dataclass(frozen=True)
class Config:
    team_name: str = "OpusOperandi"
    seed: int = 0
    match: MatchConfig = MatchConfig()
    kpi_alarms: KpiAlarms = KpiAlarms()


def _pair(value: Any, name: str, cast: type, ordered: bool = True) -> tuple:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ConfigError(f"{name} must be a [low, high] pair, got {value!r}")
    low, high = cast(value[0]), cast(value[1])
    if ordered and low > high:
        raise ConfigError(f"{name} has low > high: {value!r}")
    return (low, high)


def _section(raw: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    section = raw.get(key, {})
    if not isinstance(section, Mapping):
        raise ConfigError(f"'{key}' must be a mapping")
    return section


def config_from_dict(raw: Mapping[str, Any]) -> Config:
    """Build a typed :class:`Config` from a parsed YAML mapping, validating and cross-checking."""
    defaults = Config()
    match_raw = _section(raw, "match")
    dm = defaults.match
    match = MatchConfig(
        game=str(match_raw.get("game", dm.game)),
        n_video_frames=int(match_raw.get("n_video_frames", dm.n_video_frames)),
        fps=int(match_raw.get("fps", dm.fps)),
        pitch=_pair(match_raw.get("pitch", dm.pitch), "match.pitch", float, ordered=False),
        half1=_pair(match_raw.get("half1", dm.half1), "match.half1", int),
        half2_dead=_pair(match_raw.get("half2_dead", dm.half2_dead), "match.half2_dead", int),
        half2_play=_pair(match_raw.get("half2_play", dm.half2_play), "match.half2_play", int),
    )
    alarms_raw = _section(raw, "kpi_alarms")
    da = defaults.kpi_alarms
    alarms = KpiAlarms(
        possessions=_pair(alarms_raw.get("possessions", da.possessions), "kpi_alarms.possessions", float),
        mean_duration_s=_pair(
            alarms_raw.get("mean_duration_s", da.mean_duration_s), "kpi_alarms.mean_duration_s", float
        ),
        dead_pct_of_graded=_pair(
            alarms_raw.get("dead_pct_of_graded", da.dead_pct_of_graded),
            "kpi_alarms.dead_pct_of_graded",
            float,
        ),
        home_pct_of_alive=_pair(
            alarms_raw.get("home_pct_of_alive", da.home_pct_of_alive), "kpi_alarms.home_pct_of_alive", float
        ),
    )
    config = Config(
        team_name=str(raw.get("team_name", defaults.team_name)),
        seed=int(raw.get("seed", defaults.seed)),
        match=match,
        kpi_alarms=alarms,
    )
    _check_internal_consistency(config.match)
    cross_check_games(config.match)
    return config


def load_config(path: str | os.PathLike | None = None) -> Config:
    """Load the YAML configuration (default: ``configs/default.yaml``) into a typed :class:`Config`."""
    target = Path(path) if path is not None else DEFAULT_CONFIG_PATH
    with open(target, encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    if not isinstance(raw, Mapping):
        raise ConfigError(f"{target}: top level must be a mapping")
    return config_from_dict(raw)


def _check_internal_consistency(match: MatchConfig) -> None:
    if match.half2_dead[1] + 1 != match.half2_play[0]:
        raise ConfigError("half2_dead and half2_play must be contiguous")
    if not (match.half1[1] < match.half2_dead[0] and match.half2_play[1] < match.n_video_frames):
        raise ConfigError("match blocks are out of order or outside the video")


def load_games_module() -> ModuleType | None:
    """Import the vendored official ``games.py`` (repo root); ``None`` if it is not available."""
    try:
        return importlib.import_module("games")
    except ImportError:
        pass
    if (REPO_ROOT / "games.py").is_file():
        root = str(REPO_ROOT)
        if root not in sys.path:
            sys.path.insert(0, root)
        return importlib.import_module("games")
    return None


def cross_check_games(match: MatchConfig | None = None, games: ModuleType | None = None) -> bool:
    """Compare ``match`` against ``games.GAMES`` / ``games.half_video_blocks``.

    Returns True when checked, False when ``games.py`` is not importable. Raises
    :class:`ConfigError` on any mismatch.
    """
    match = match or MatchConfig()
    games = games if games is not None else load_games_module()
    if games is None:
        return False
    entry = games.GAMES.get(match.game)
    if entry is None:
        raise ConfigError(f"games.py has no game {match.game!r}")
    problems = []
    if entry["n_video_frames"] != match.n_video_frames:
        problems.append(f"n_video_frames {match.n_video_frames} != {entry['n_video_frames']}")
    if entry["fps"] != match.fps:
        problems.append(f"fps {match.fps} != {entry['fps']}")
    if tuple(entry["pitch"]) != tuple(match.pitch):
        problems.append(f"pitch {match.pitch} != {tuple(entry['pitch'])}")
    blocks = {half: (first, last) for half, first, last in games.half_video_blocks(match.game)}
    if blocks.get(1) != tuple(match.half1):
        problems.append(f"half1 {match.half1} != {blocks.get(1)}")
    if blocks.get(2) != (match.half2_dead[0], match.half2_play[1]):
        problems.append(f"half2 {(match.half2_dead[0], match.half2_play[1])} != {blocks.get(2)}")
    if problems:
        raise ConfigError("constants disagree with games.py: " + "; ".join(problems))
    return True


def set_global_seeds(seed: int = 0) -> int:
    """Seed ``random``, ``numpy`` and ``PYTHONHASHSEED`` (effective for child processes)."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    return seed


# Fail loudly at import time if the constants drift from the official module.
cross_check_games()


# --------------------------------------------------------------------------------------------
# Per-module parameter sections (``configs/default.yaml``)
# --------------------------------------------------------------------------------------------
def _load_raw(path: str | os.PathLike | Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Parsed YAML mapping of ``path`` (default file when ``None``); a mapping is returned as is."""
    if isinstance(path, Mapping):
        return dict(path)
    target = Path(path) if path is not None else DEFAULT_CONFIG_PATH
    with open(target, encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    if not isinstance(raw, Mapping):
        raise ConfigError(f"{target}: top level must be a mapping")
    return dict(raw)


def load_section(name: str, path: str | os.PathLike | Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Raw mapping of one top-level YAML section (``{}`` when the section is absent).

    ``path`` may also be an already parsed configuration mapping (see :func:`route_config`)."""
    return dict(_section(_load_raw(path), name))


def route_config(route: str, path: str | os.PathLike | Mapping[str, Any] | None = None):
    """Configuration of one decoding route: ``route_overrides.<route>`` merged key by key into its sections.

    The heuristic and the learned routes share the decoder sections (``transition_matrix``,
    ``spell_optimizer``, ...) but were tuned separately on the external matches, so a route may override
    them.  Returns ``path`` unchanged when the route has no overrides, else the merged in-memory mapping
    (accepted by :func:`load_section` and every ``from_config``).
    """
    raw = _load_raw(path)
    overrides = (raw.get("route_overrides") or {}).get(route) or {}
    if not overrides:
        return path
    merged = dict(raw)
    for section, values in overrides.items():
        if not isinstance(values, Mapping):
            raise ConfigError(f"route_overrides.{route}.{section} must be a mapping")
        merged[section] = {**dict(raw.get(section) or {}), **dict(values)}
    return merged


def params_from_mapping(cls: type, mapping: Mapping[str, Any], section: str = "") -> Any:
    """Instantiate the frozen parameter dataclass ``cls`` from ``mapping``.

    Unknown keys raise :class:`ConfigError` (typos must not pass silently); YAML lists are turned
    into tuples; ints are accepted for float fields.  Missing keys keep the dataclass default.
    """
    import dataclasses

    fields = {f.name: f for f in dataclasses.fields(cls)}
    unknown = sorted(set(mapping) - set(fields))
    if unknown:
        raise ConfigError(f"unknown keys in section '{section or cls.__name__}': {unknown}")
    kwargs: dict[str, Any] = {}
    for key, value in mapping.items():
        default = fields[key].default
        if isinstance(value, list):
            value = tuple(value)
        if isinstance(default, float) and isinstance(value, int) and not isinstance(value, bool):
            value = float(value)
        kwargs[key] = value
    return cls(**kwargs)
