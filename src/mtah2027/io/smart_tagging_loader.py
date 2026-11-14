"""Smart Tagging (SportsCode XML) -> per-frame arrays and structured events (Issue #6).

Smart Tagging is Metrica's automatic annotation of the tactical video: *phases* of play (Build Up,
Progression, ...), ``SET PIECES`` and ``GOALS``, each an interval with a code and labels (Team, Half,
Type).  Seconds are from the first frame of the tactical video, so ``round(seconds * 25)`` is a video
frame.  Intervals are half-open on the right (``end_frame`` is the first frame after the interval),
exactly like ``starter/load_smart_tagging.py``.

The tags are inferred automatically (false positives/negatives exist) and their intervals are
*windows*, not instants -- e.g. ``Kick Off (Start)`` opens ~3.5 s before the kick.  They are a soft
prior, never ground truth; exact restart frames come from the tracking (Issue #14).

Everything is case-insensitive: codes, group names, tag values.  The parser is also tolerant to the usual
SportsCode export variants: tag names in any case or namespace, no ``<ALL_INSTANCES>`` wrapper, surrounding
whitespace/newlines in any text, decimal commas, ``Set Piece`` / ``set_pieces`` / ``Goal`` code spellings,
labels without a ``<group>`` (the team, half and type are then recognised from the text) and instances without
usable times (skipped and counted, never fatal).  Team names map to match sides
``Barcelona -> 0 (H)``, ``Athletic -> 1 (A)``, ``N/A`` (or missing) ``-> -1``.
"""
from __future__ import annotations

import dataclasses
import os
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field

import numpy as np

from mtah2027.config import FPS, HALF1, HALF2_PLAY, N_VIDEO_FRAMES, load_games_module

HOME, AWAY, NO_TEAM = 0, 1, -1
SET_PIECE_CODE = "set pieces"
GOAL_CODE = "goals"
# Phases whose ``Team`` is the team *performing the event* (OVERVIEW.md) but whose event is a defensive
# action: the possession of the phase belongs to the opponent, so their coverage counts for the other side.
DEFENDING_CODES: tuple[str, ...] = ("defending in attacking third", "defending in middle third",
                                    "defending in defensive third", "defensive transition")
# Issue #43: phases that mark a change of possession (the possession side is the team that gains the ball)
TRANSITION_CODES: tuple[str, ...] = ("attacking transition", "defensive transition")
DEFAULT_TEAM_ALIASES: dict[str, int] = {
    "barcelona": HOME, "fc barcelona": HOME, "barca": HOME, "barça": HOME,
    "atletic club": AWAY, "athletic club": AWAY, "atletic": AWAY, "athletic": AWAY,
    "athletic bilbao": AWAY,
}
_NA = {"", "n/a", "na", "none", "null", "-", "?"}


@dataclass(frozen=True)
class TagEvent:
    """One ``<instance>``; ``type`` and ``code`` are lower-cased, ``team`` is H=0, A=1, N/A=-1."""

    code: str
    start_frame: int
    end_frame: int          # exclusive
    start_s: float
    end_s: float
    team: int
    half: int               # 1, 2 or 0 when unknown
    type: str
    tags: dict[str, str]    # lower-cased group -> original text


@dataclass(frozen=True)
class SetPiece:
    type: str               # lower-cased, e.g. 'throw-in', 'kick off (start)'
    team: int               # 0 / 1 / -1 (Team = N/A)
    start_frame: int
    end_frame: int          # exclusive
    half: int


@dataclass(frozen=True)
class Goal:
    team: int
    start_frame: int
    end_frame: int          # exclusive
    half: int


@dataclass
class SmartTagging:
    """Parsed Smart Tagging; per-frame arrays have length ``n_frames``."""

    events: list[TagEvent]
    set_pieces: list[SetPiece]
    goals: list[Goal]
    phase_count_home: np.ndarray       # [N] int16: phases tagged H covering each frame
    phase_count_away: np.ndarray       # [N] int16
    phase_prior_home: np.ndarray       # [N] float32 in [-1, 1]: (h - a) / (h + a), 0 where none
    phase_team: np.ndarray             # [N] int8: 0 (H), 1 (A), -1 none or tie
    in_set_piece: np.ndarray           # [N] bool
    n_frames: int
    fps: int = FPS
    codes: dict[str, int] = field(default_factory=dict)
    n_flipped: int = 0                 # defensive phases whose team was flipped to the possessing side
    phase_cover: np.ndarray | None = None      # [N] bool: any phase code covers the frame (Issue #43)
    transitions: list = field(default_factory=list)   # (start_frame, possession_side) of Transition phases

    def set_pieces_of(self, kind: str) -> list[SetPiece]:
        """Set pieces whose type equals ``kind`` (case-insensitive)."""
        kind = kind.strip().lower()
        return [s for s in self.set_pieces if s.type == kind]


def _norm(text: str | None) -> str:
    return re.sub(r"\s+", " ", (text or "").strip().lower())


def _team_code(text: str | None, aliases: dict[str, int]) -> int:
    key = _norm(text)
    if key in _NA:
        return NO_TEAM
    if key in aliases:
        return aliases[key]
    for alias, code in aliases.items():      # tolerate e.g. "Atletic Club (A)" or "FC Barcelona B"
        if alias in key:
            return code
    return NO_TEAM


def _half_code(text: str | None) -> int:
    key = _norm(text)
    match = re.search(r"([12])", key)
    if match:
        return int(match.group(1))
    return 1 if "first" in key else 2 if "second" in key else 0


def smart_tagging_path(data_dir: str | os.PathLike | None = None) -> str:
    games = load_games_module()
    if games is None:
        raise ImportError("the vendored games.py is required to locate the Smart Tagging file")
    return games.require(["smart_tagging"], games.GAME, data_dir)[0]


_CODE_ALIASES = {"set piece": SET_PIECE_CODE, "set pieces": SET_PIECE_CODE, "setpiece": SET_PIECE_CODE,
                 "setpieces": SET_PIECE_CODE, "goal": GOAL_CODE, "goals": GOAL_CODE}
_HALF_TEXT = re.compile(r"(?:\b([12])\s*(?:st|nd)?\s*(?:half|time|part)\b)|(?:\bhalf\s*([12])\b)|"
                        r"\b(first|second)\s+half\b")


def _local(tag: str) -> str:
    return tag.rsplit("}", 1)[-1].lower()


def _child_text(el: ET.Element, name: str) -> str | None:
    """Text of the first child called ``name`` (any case / namespace), ``None`` when absent."""
    for child in el:
        if _local(child.tag) == name:
            return child.text or ""
    return None


def _seconds(text: str | None) -> float | None:
    try:
        value = float((text or "").strip().replace(",", "."))
    except ValueError:
        return None
    return value if np.isfinite(value) else None


def _code_key(text: str | None) -> str:
    key = _norm(re.sub(r"[_\-]+", " ", text or ""))
    return _CODE_ALIASES.get(key, key)


def _ungrouped(texts: list[str], aliases: dict[str, int]) -> dict[str, str]:
    """Team / half / type of labels that carry no ``<group>``, recognised from their text."""
    tags: dict[str, str] = {}
    rest = []
    for text in texts:
        key = _norm(text)
        if "team" not in tags and (key in _NA or key in aliases or _team_code(key, aliases) != NO_TEAM):
            tags["team"] = text
        elif "half" not in tags and _HALF_TEXT.search(key):
            tags["half"] = text
        else:
            rest.append(text)
    if rest:
        tags["type"] = rest[0]
    return tags


def parse_events(path: str | os.PathLike, fps: int = FPS,
                 team_aliases: dict[str, int] | None = None) -> list[TagEvent]:
    """Events in file order; frames are ``round(seconds * fps)`` with a half-open end."""
    aliases = {**DEFAULT_TEAM_ALIASES, **{_norm(k): v for k, v in (team_aliases or {}).items()}}
    events: list[TagEvent] = []
    for instance in ET.parse(path).getroot().iter():
        if _local(instance.tag) != "instance":
            continue
        start_s, end_s = _seconds(_child_text(instance, "start")), _seconds(_child_text(instance, "end"))
        if start_s is None or end_s is None:
            continue
        tags: dict[str, str] = {}
        loose: list[str] = []
        for label in instance:
            if _local(label.tag) != "label":
                continue
            group, text = _norm(_child_text(label, "group")), (_child_text(label, "text") or "").strip()
            if group:
                tags[group] = re.sub(r"\s+", " ", text)      # last one wins
            elif text:
                loose.append(re.sub(r"\s+", " ", text))
        for group, text in _ungrouped(loose, aliases).items():
            tags.setdefault(group, text)
        events.append(TagEvent(
            code=_code_key(_child_text(instance, "code")), start_s=start_s, end_s=end_s,
            start_frame=int(round(start_s * fps)), end_frame=int(round(end_s * fps)),
            team=_team_code(tags.get("team"), aliases), half=_half_code(tags.get("half")),
            type=_norm(tags.get("type")), tags=tags))
    return events


def _coverage(events: list[TagEvent], team: int, n: int) -> np.ndarray:
    """Number of events of ``team`` covering each frame (difference array, O(N + events))."""
    sel = [e for e in events if e.team == team and e.end_frame > e.start_frame]
    diff = np.zeros(n + 1, dtype=np.int32)
    if sel:
        lo = np.clip([e.start_frame for e in sel], 0, n)
        hi = np.clip([e.end_frame for e in sel], 0, n)
        np.add.at(diff, lo, 1)
        np.add.at(diff, hi, -1)
    return np.cumsum(diff[:n]).astype(np.int16)


def possession_side(event: TagEvent, defending_codes: tuple[str, ...] = DEFENDING_CODES) -> int:
    """Side in possession during a phase: the tagged team, or its opponent for the defensive phases."""
    if event.team >= 0 and event.code in defending_codes:
        return AWAY if event.team == HOME else HOME
    return event.team


def build_smart_tagging(events: list[TagEvent], n_frames: int = N_VIDEO_FRAMES,
                        fps: int = FPS, defending_codes: tuple[str, ...] = DEFENDING_CODES) -> SmartTagging:
    """Per-frame tensors plus structured set-piece and goal lists from parsed events.

    The phase prior counts the side *in possession*: for the ``defending in ... third`` and ``defensive
    transition`` phases (``defending_codes``; ``()`` disables) the tagged team is the defender, so the
    opponent is credited.  ``Team`` is "the team performing the event" (``OVERVIEW.md``)."""
    keys = tuple(_code_key(c) for c in defending_codes)
    phases = [e for e in events if e.code not in (SET_PIECE_CODE, GOAL_CODE)]
    n_flipped = sum(1 for e in phases if possession_side(e, keys) != e.team)
    phases = [dataclasses.replace(e, team=possession_side(e, keys)) for e in phases]
    home = _coverage(phases, HOME, n_frames)
    away = _coverage(phases, AWAY, n_frames)
    total = (home + away).astype(np.float32)
    prior = np.where(total > 0, (home - away) / np.maximum(total, 1.0), 0.0).astype(np.float32)
    team = np.where(home > away, HOME, np.where(away > home, AWAY, NO_TEAM)).astype(np.int8)

    cover = (home + away) > 0
    cover_nt = np.zeros(n_frames + 1, dtype=np.int32)       # phases of unknown team (N/A) also count as live
    for e in phases:
        if e.team < 0 and e.end_frame > e.start_frame:
            cover_nt[max(0, min(e.start_frame, n_frames))] += 1
            cover_nt[max(0, min(e.end_frame, n_frames))] -= 1
    cover = cover | (np.cumsum(cover_nt[:n_frames]) > 0)
    transitions = sorted((e.start_frame, e.team) for e in phases
                         if e.code in TRANSITION_CODES and e.team >= 0)

    sets = [SetPiece(e.type, e.team, e.start_frame, e.end_frame, e.half)
            for e in events if e.code == SET_PIECE_CODE]
    goals = [Goal(e.team, e.start_frame, e.end_frame, e.half) for e in events if e.code == GOAL_CODE]
    diff = np.zeros(n_frames + 1, dtype=np.int32)
    for s in sets:
        lo, hi = max(0, min(s.start_frame, n_frames)), max(0, min(s.end_frame, n_frames))
        if hi > lo:
            diff[lo] += 1
            diff[hi] -= 1
    codes: dict[str, int] = {}
    for e in events:
        codes[e.code] = codes.get(e.code, 0) + 1
    return SmartTagging(events=events, set_pieces=sets, goals=goals, phase_count_home=home,
                        phase_count_away=away, phase_prior_home=prior, phase_team=team,
                        in_set_piece=np.cumsum(diff[:n_frames]) > 0, n_frames=n_frames, fps=fps, codes=codes,
                        n_flipped=n_flipped, phase_cover=cover, transitions=transitions)


def load_smart_tagging(data_dir: str | os.PathLike | None = None, n_frames: int = N_VIDEO_FRAMES,
                       fps: int = FPS, team_aliases: dict[str, int] | None = None,
                       path: str | os.PathLike | None = None,
                       defending_codes: tuple[str, ...] = DEFENDING_CODES) -> SmartTagging:
    """Parse the tactical Smart Tagging XML (``path`` overrides the bundle lookup)."""
    path = path or smart_tagging_path(data_dir)
    return build_smart_tagging(parse_events(path, fps, team_aliases), n_frames, fps, defending_codes)


def check_time_base(st: SmartTagging, half1_start: int = HALF1[0], half2_start: int = HALF2_PLAY[0]) -> dict:
    """Does a ``Kick Off (Start)`` window of each half contain the published kick-off frame?"""
    rows = []
    for half, truth in ((1, half1_start), (2, half2_start)):
        kicks = st.set_pieces_of("kick off (start)")
        tagged = [(s.start_frame, s.end_frame) for s in kicks if s.half == half]
        rows.append({"half": half, "kickoff_frame": truth, "tagged": tagged,
                     "contains_kickoff": any(a <= truth <= b for a, b in tagged)})
    return {"halves": rows, "agrees": all(r["contains_kickoff"] for r in rows)}
