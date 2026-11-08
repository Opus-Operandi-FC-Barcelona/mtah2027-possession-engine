"""Streaming parser of the Automatic Tracking Data (Issue #5).

The ATD is a FIFA/EPTS pair: ``*_atd.xml`` (metadata + ``DataFormatSpecifications``) and
``*_atd.txt`` (one line per video frame, ``<frame_id>:<slot>;<slot>;...:<ball>``, groups separated by
commas, ``NaN`` for absent values).

Nothing about the channel layout is hard-coded: the token layout of a line is read from the
``DataFormatSpecification`` of the XML, every declared per-player and per-ball channel is parsed and
returned (the ball group may carry ``z``, the owning team or an alive/dead status -- algorithmic
signals that are first-class features).  Text channels are mapped to numeric codes (see
:attr:`TrackingData.ball_channel_codes`).  If the XML has no specification the layout assumed by the
official baseline (``x,y,speed`` per slot and for the ball) is used.

Alignment: ``video_frame = frame_id - 1``; row ``i`` of every array is video frame ``i`` and frames
missing from the file are NaN rows.  Positions are converted from normalised coordinates to metres
(``x * 105.3``, ``y * 68.0``).

Performance: the file is read in large binary blocks, separators are translated to blanks with a
``bytes`` table and the numbers are parsed in one C call per block (``np.fromstring``), writing
straight into preallocated float32 arrays.  Parsed data are cached as ``.npz`` keyed by the size and
mtime of the source files.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import warnings
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from mtah2027.config import FPS, PITCH, REPO_ROOT, load_games_module

CACHE_VERSION = 2
DEFAULT_CACHE_DIR = REPO_ROOT / "cache"
BLOCK_BYTES = 8 * 1024 * 1024
DEFAULT_CHANNELS = ("x", "y", "speed")
_METRE_UNITS = {"m", "meter", "meters", "metre", "metres"}
_KMH_UNITS = {"km/h", "kmh", "kph"}
_STATE_CODES = {"alive": 1, "dead": 0}
_NUMERIC_SEPARATORS = ":;,"
_NUMBER = re.compile(r"[-+]?(\d+\.?\d*|\.\d+)([eE][-+]?\d+)?")
_MISSING_TEXT = {"nan", "", "none", "null", "-"}


class AtdFormatError(ValueError):
    """Raised when the ATD files are inconsistent with the declared format."""


# --------------------------------------------------------------------------------------------
# Metadata
# --------------------------------------------------------------------------------------------
def _local(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


@dataclass
class AtdLayout:
    """Token layout of a raw ATD line, derived from the XML."""

    n_tokens: int
    frame_col: int
    player_channels: tuple[str, ...]
    player_cols: np.ndarray            # [S, P] token column of (slot, channel)
    ball_channels: tuple[str, ...]
    ball_cols: np.ndarray              # [B]
    team_of_slot: np.ndarray           # [S] int8, 0 = home (first <Team>), 1 = away
    player_ids: tuple[str, ...]
    team_ids: tuple[str, ...]
    team_names: tuple[str, ...]
    units: dict[str, str]
    separators: str
    from_spec: bool
    end_frame: int | None
    frame_rate: float | None

    @property
    def n_slots(self) -> int:
        return int(self.player_cols.shape[0])


def _channel_units(root: ET.Element) -> dict[str, str]:
    units: dict[str, str] = {}
    for el in root.iter():
        if _local(el.tag) == "Channel":
            cid = (el.get("id") or "").lower()
            unit = next((c.text or "" for c in el if _local(c.tag) == "Unit"), "")
            if cid and cid not in units:
                units[cid] = unit.strip().lower()
    return units


def parse_atd_meta(meta_path: str | os.PathLike) -> AtdLayout:
    """Read teams, slot -> team assignment and the declared channel layout from ``atd.xml``."""
    root = ET.parse(meta_path).getroot()
    teams = [(t.get("id"), next((c.text for c in t if _local(c.tag) == "Name"), None))
             for t in root.iter() if _local(t.tag) == "Team"]
    if len(teams) != 2:
        raise AtdFormatError(f"{meta_path}: expected 2 teams, found {len(teams)}")
    team_ids = tuple(str(t[0]) for t in teams)
    team_names = tuple(str(t[1]) for t in teams)
    players = [(p.get("id"), p.get("teamId")) for p in root.iter() if _local(p.tag) == "Player"]
    if not players:
        raise AtdFormatError(f"{meta_path}: no <Player> entries")
    team_of_player = {pid: (0 if tid == team_ids[0] else 1) for pid, tid in players}
    units = _channel_units(root)
    frame_rate = next((float(el.text) for el in root.iter()
                       if _local(el.tag) == "FrameRate" and el.text), None)

    pc_map = {el.get("id"): (el.get("playerId"), el.get("channelId"))
              for el in root.iter() if _local(el.tag) == "PlayerChannel"}

    spec = next((el for el in root.iter() if _local(el.tag) == "DataFormatSpecification"), None)
    if spec is None:
        return _default_layout(players, team_of_player, team_ids, team_names, units, frame_rate)

    seps = {el.get("separator") for el in spec.iter() if el.get("separator")}
    if any(len(s) != 1 for s in seps):
        raise AtdFormatError(f"{meta_path}: multi-character separators are not supported: {sorted(seps)}")
    separators = "".join(sorted(seps)) or _NUMERIC_SEPARATORS
    end_frame = int(spec.get("endFrame")) if spec.get("endFrame") else None

    leaves: list[tuple[str, Any]] = []
    for el in spec.iter():
        name = _local(el.tag)
        if name == "StringRegister":
            leaves.append(("frame" if el.get("name") == "frameCount" else "skip", None))
        elif name == "PlayerChannelRef":
            pcid = el.get("playerChannelId")
            if pcid in pc_map:
                pid, ch = pc_map[pcid]
            else:  # "<player>_<channel>" naming convention
                pid, _, ch = str(pcid).rpartition("_")
            leaves.append(("player", (pid, str(ch).lower())))
        elif name == "BallChannelRef":
            ch = el.get("channelId") or el.get("ballChannelId") or el.get("id")
            leaves.append(("ball", str(ch).lower()))
    frame_cols = [i for i, (kind, _) in enumerate(leaves) if kind == "frame"]
    if not frame_cols:
        raise AtdFormatError(f"{meta_path}: specification has no frameCount register")

    slot_ids: list[str] = []
    slot_cols: dict[str, dict[str, int]] = {}
    ball_pairs: list[tuple[str, int]] = []
    for col, (kind, val) in enumerate(leaves):
        if kind == "player":
            pid, ch = val
            if pid not in slot_cols:
                slot_ids.append(pid)
                slot_cols[pid] = {}
            slot_cols[pid][ch] = col
        elif kind == "ball":
            ball_pairs.append((val, col))
    if not slot_ids or not ball_pairs:
        raise AtdFormatError(f"{meta_path}: specification declares no player or ball channels")
    unknown = [p for p in slot_ids if p not in team_of_player]
    if unknown:
        raise AtdFormatError(f"{meta_path}: slots without a <Player>/team: {unknown[:3]}")
    p_channels = tuple(slot_cols[slot_ids[0]])
    for pid in slot_ids:
        if set(slot_cols[pid]) != set(p_channels):
            raise AtdFormatError(f"{meta_path}: slot {pid} declares channels different from the first slot")
    if "x" not in p_channels or "y" not in p_channels:
        raise AtdFormatError(f"{meta_path}: player channels must include x and y, got {p_channels}")
    b_channels = tuple(c for c, _ in ball_pairs)
    if "x" not in b_channels or "y" not in b_channels:
        raise AtdFormatError(f"{meta_path}: ball channels must include x and y, got {b_channels}")
    return AtdLayout(
        n_tokens=len(leaves), frame_col=frame_cols[0], player_channels=p_channels,
        player_cols=np.array([[slot_cols[p][c] for c in p_channels] for p in slot_ids], dtype=np.int64),
        ball_channels=b_channels, ball_cols=np.array([c for _, c in ball_pairs], dtype=np.int64),
        team_of_slot=np.array([team_of_player[p] for p in slot_ids], dtype=np.int8),
        player_ids=tuple(slot_ids), team_ids=team_ids, team_names=team_names, units=units,
        separators=separators, from_spec=True, end_frame=end_frame, frame_rate=frame_rate)


def _default_layout(players, team_of_player, team_ids, team_names, units, frame_rate) -> AtdLayout:
    """Format assumed by the official baseline: ``frame : (x,y,speed ;)* : x,y,speed``."""
    n_slots, n_ch = len(players), len(DEFAULT_CHANNELS)
    cols = 1 + np.arange(n_slots * n_ch, dtype=np.int64).reshape(n_slots, n_ch)
    return AtdLayout(
        n_tokens=1 + n_slots * n_ch + n_ch, frame_col=0, player_channels=DEFAULT_CHANNELS,
        player_cols=cols, ball_channels=DEFAULT_CHANNELS,
        ball_cols=1 + n_slots * n_ch + np.arange(n_ch, dtype=np.int64),
        team_of_slot=np.array([team_of_player[p[0]] for p in players], dtype=np.int8),
        player_ids=tuple(str(p[0]) for p in players), team_ids=team_ids, team_names=team_names,
        units=units, separators=_NUMERIC_SEPARATORS, from_spec=False, end_frame=None,
        frame_rate=frame_rate)


# --------------------------------------------------------------------------------------------
# Result container
# --------------------------------------------------------------------------------------------
@dataclass
class TrackingData:
    """ATD aligned to the video: row ``i`` of every array is video frame ``i``."""

    players_xy: np.ndarray                 # [N, S, 2] float32, metres
    players_speed: np.ndarray              # [N, S] float32 (m/s; NaN if undeclared)
    ball_xy: np.ndarray                    # [N, 2] float32, metres
    ball_channels: dict[str, np.ndarray]   # every declared ball channel [N] (x, y in metres)
    ball_channel_codes: dict[str, dict[str, int]]   # text channel -> {text: code}
    players_extra: dict[str, np.ndarray]   # other per-player channels [N, S]
    team_of_slot: np.ndarray               # [S] int8, 0 = home, 1 = away
    frame_index: np.ndarray                # [N] int64 = arange(N) = video frame
    present: np.ndarray                    # [N] bool, frame has a record in the file
    source: str = "tactical"
    fps: float = float(FPS)
    pitch: tuple[float, float] = PITCH
    team_ids: tuple[str, ...] = ()
    team_names: tuple[str, ...] = ()
    player_ids: tuple[str, ...] = ()
    report: dict[str, Any] = field(default_factory=dict)

    @property
    def n_frames(self) -> int:
        return int(self.players_xy.shape[0])

    @property
    def n_slots(self) -> int:
        return int(self.players_xy.shape[1])

    @property
    def ball_speed(self) -> np.ndarray:
        return self.ball_channels.get("speed", np.full(self.n_frames, np.nan, dtype=np.float32))

    @property
    def ball_visible(self) -> np.ndarray:
        return ~np.isnan(self.ball_xy).any(axis=1)


# --------------------------------------------------------------------------------------------
# Block parsing
# --------------------------------------------------------------------------------------------
class _Sink:
    """Preallocated, growable float32 destination arrays (normalised units, as in the file)."""

    def __init__(self, layout: AtdLayout, capacity: int, extra_p: list[str]):
        self.layout, self.extra_p = layout, extra_p
        self.ix = layout.player_channels.index("x")
        self.iy = layout.player_channels.index("y")
        self.isp = layout.player_channels.index("speed") if "speed" in layout.player_channels else None
        self.cap = 0
        self.xy = np.empty((0, layout.n_slots, 2), dtype=np.float32)
        self.speed = np.empty((0, layout.n_slots), dtype=np.float32)
        self.ball = np.empty((0, len(layout.ball_cols)), dtype=np.float32)
        self.extra = {c: np.empty((0, layout.n_slots), dtype=np.float32) for c in extra_p}
        self.present = np.zeros(0, dtype=bool)
        self.max_row = -1
        self.grow(max(capacity, 1))

    def grow(self, capacity: int) -> None:
        old = self.cap

        def ext(arr):
            new = np.full((capacity,) + arr.shape[1:], np.nan, dtype=np.float32)
            new[:old] = arr[:old]
            return new

        self.xy, self.speed, self.ball = ext(self.xy), ext(self.speed), ext(self.ball)
        self.extra = {c: ext(a) for c, a in self.extra.items()}
        present = np.zeros(capacity, dtype=bool)
        present[:old] = self.present[:old]
        self.present, self.cap = present, capacity

    def write(self, rows: np.ndarray, data: np.ndarray) -> None:
        top = int(rows.max()) + 1
        if top > self.cap:
            self.grow(max(top, self.cap * 2))
        lay, pc = self.layout, self.layout.player_cols
        self.xy[rows, :, 0] = data[:, pc[:, self.ix]]
        self.xy[rows, :, 1] = data[:, pc[:, self.iy]]
        if self.isp is not None:
            self.speed[rows] = data[:, pc[:, self.isp]]
        for name in self.extra_p:
            self.extra[name][rows] = data[:, pc[:, lay.player_channels.index(name)]]
        self.ball[rows] = data[:, lay.ball_cols]
        self.present[rows] = True
        self.max_row = max(self.max_row, top - 1)


class _TokenCoder:
    """Maps non-numeric tokens (team ids, alive/dead, ...) to numeric codes, per token column."""

    def __init__(self, layout: AtdLayout):
        seed = {tid: i for i, tid in enumerate(layout.team_ids)}
        seed.update({n: i for i, n in enumerate(layout.team_names)})
        self.seed = {**seed, **_STATE_CODES}
        self.codes: dict[int, dict[str, int]] = {}

    def code(self, col: int, text: str) -> float:
        table = self.codes.setdefault(col, {})
        if text not in table:
            if text.lower() in self.seed:
                table[text] = self.seed[text.lower()]
            elif text in self.seed:
                table[text] = self.seed[text]
            else:
                used = set(table.values()) | set(self.seed.values())
                table[text] = next(i for i in range(len(used) + 1) if i not in used)
        return float(table[text])

    def parse(self, tokens: np.ndarray) -> np.ndarray:
        """Slow path: ``tokens`` is a [L, T] str array; non-numeric columns are coded."""
        out = np.full(tokens.shape, np.nan, dtype=np.float64)
        for col in range(tokens.shape[1]):
            column = tokens[:, col]
            try:
                out[:, col] = column.astype(np.float64)
            except ValueError:
                for text in np.unique(column):
                    lowered = str(text).lower()
                    if lowered in _MISSING_TEXT:
                        continue
                    value = float(lowered) if _NUMBER.fullmatch(lowered) else self.code(col, str(text))
                    out[column == text, col] = value
        return out


def _parse_block(block: bytes, layout: AtdLayout, table: bytes, coder: _TokenCoder,
                 state: dict[str, bool]) -> tuple[np.ndarray, int]:
    """Parse whole lines into a float64 [L, n_tokens] array; returns (array, malformed lines)."""
    if state["fast"]:
        n_lines = block.count(b"\n") + (0 if block.endswith(b"\n") else 1)
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("error", DeprecationWarning)
                flat = np.fromstring(block.translate(table), dtype=np.float64, sep=" ")
        except (ValueError, DeprecationWarning):
            flat = None
        if flat is not None and flat.size == n_lines * layout.n_tokens:
            return flat.reshape(n_lines, layout.n_tokens), 0
    rows, bad = [], 0
    for line in block.splitlines():
        if not line.strip():
            continue
        toks = line.translate(table).split()
        if len(toks) != layout.n_tokens:
            bad += 1
            continue
        rows.append([t.decode("ascii", "replace") for t in toks])
    if not rows:
        return np.empty((0, layout.n_tokens)), bad
    if bad == 0:
        state["fast"] = False   # text channels: stay on the (vectorised per column) slow path
    return coder.parse(np.array(rows, dtype=str)), bad


def _iter_blocks(path: str | os.PathLike, block_bytes: int):
    """Yield byte blocks that end on a line boundary."""
    tail = b""
    with open(path, "rb") as handle:
        while True:
            chunk = handle.read(block_bytes)
            if not chunk:
                break
            chunk = tail + chunk
            cut = chunk.rfind(b"\n")
            if cut < 0:
                tail = chunk
                continue
            tail = chunk[cut + 1:]
            yield chunk[:cut + 1]
    if tail.strip():
        yield tail


def parse_atd_raw(raw_path: str | os.PathLike, layout: AtdLayout, capacity: int = 0,
                  block_bytes: int = BLOCK_BYTES) -> tuple[_Sink, _TokenCoder, dict[str, int]]:
    """Single streaming pass over the raw file into preallocated arrays."""
    extra_p = [c for c in layout.player_channels if c not in ("x", "y", "speed")]
    sink = _Sink(layout, capacity, extra_p)
    coder = _TokenCoder(layout)
    table = bytearray(range(256))
    for ch in layout.separators + "\r\n\t":
        table[ord(ch)] = ord(" ")
    table = bytes(table)
    state = {"fast": True}
    stats = {"n_records": 0, "malformed_lines": 0, "dropped_frames": 0, "duplicate_frames": 0}
    for block in _iter_blocks(raw_path, block_bytes):
        data, bad = _parse_block(block, layout, table, coder, state)
        stats["malformed_lines"] += bad
        if not len(data):
            continue
        ids = data[:, layout.frame_col]
        good = np.isfinite(ids) & (ids >= 1)
        stats["dropped_frames"] += int((~good).sum())
        data, rows = data[good], ids[good].astype(np.int64) - 1
        if not len(rows):
            continue
        inside = rows[rows < sink.cap]
        stats["duplicate_frames"] += int(sink.present[inside].sum())
        sink.write(rows, data)
        stats["n_records"] += len(rows)
    return sink, coder, stats


# --------------------------------------------------------------------------------------------
# Assembly, cache and public entry point
# --------------------------------------------------------------------------------------------
def _cache_key(paths: list[str], source: str) -> str:
    h = hashlib.sha256(f"v{CACHE_VERSION}|{source}".encode())
    for p in paths:
        st = os.stat(p)
        h.update(f"|{os.path.basename(p)}:{st.st_size}:{st.st_mtime_ns}".encode())
    return h.hexdigest()[:16]


def save_tracking(td: TrackingData, path: str | os.PathLike) -> Path:
    """Write ``td`` to an uncompressed ``.npz`` (atomic rename)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    meta = {
        "ball_channels": list(td.ball_channels), "players_extra": list(td.players_extra),
        "ball_channel_codes": td.ball_channel_codes, "source": td.source, "fps": td.fps,
        "pitch": list(td.pitch), "team_ids": list(td.team_ids), "team_names": list(td.team_names),
        "player_ids": list(td.player_ids), "report": td.report,
    }
    arrays: dict[str, np.ndarray] = {
        "players_xy": td.players_xy, "players_speed": td.players_speed, "ball_xy": td.ball_xy,
        "team_of_slot": td.team_of_slot, "present": td.present,
        "meta": np.frombuffer(json.dumps(meta).encode("utf-8"), dtype=np.uint8),
    }
    arrays.update({f"ball__{k}": v for k, v in td.ball_channels.items()})
    arrays.update({f"extra__{k}": v for k, v in td.players_extra.items()})
    tmp = path.with_name(path.name + ".tmp.npz")
    np.savez(tmp, **arrays)
    os.replace(tmp, path)
    return path


def load_tracking_cache(path: str | os.PathLike) -> TrackingData:
    """Read a file written by :func:`save_tracking`."""
    with np.load(path, allow_pickle=False) as z:
        meta = json.loads(bytes(z["meta"]).decode("utf-8"))
        n = z["players_xy"].shape[0]
        return TrackingData(
            players_xy=z["players_xy"], players_speed=z["players_speed"], ball_xy=z["ball_xy"],
            ball_channels={k: z[f"ball__{k}"] for k in meta["ball_channels"]},
            ball_channel_codes={k: {t: int(c) for t, c in v.items()}
                                for k, v in meta["ball_channel_codes"].items()},
            players_extra={k: z[f"extra__{k}"] for k in meta["players_extra"]},
            team_of_slot=z["team_of_slot"], frame_index=np.arange(n, dtype=np.int64),
            present=z["present"], source=meta["source"], fps=meta["fps"], pitch=tuple(meta["pitch"]),
            team_ids=tuple(meta["team_ids"]), team_names=tuple(meta["team_names"]),
            player_ids=tuple(meta["player_ids"]), report=meta["report"])


def load_atd_files(meta_path: str | os.PathLike, raw_path: str | os.PathLike, *, source: str = "tactical",
                   n_frames: int | None = None, pitch: tuple[float, float] = PITCH,
                   expected_frames: int | None = None, block_bytes: int = BLOCK_BYTES) -> TrackingData:
    """Parse an explicit (xml, txt) pair, without cache."""
    layout = parse_atd_meta(meta_path)
    capacity = n_frames or layout.end_frame or expected_frames or 1024
    sink, coder, stats = parse_atd_raw(raw_path, layout, capacity, block_bytes)
    n = n_frames if n_frames else max(sink.max_row + 1, layout.end_frame or 0)
    if sink.max_row + 1 > n:
        raise AtdFormatError(f"{raw_path}: frame id {sink.max_row + 1} exceeds n_frames={n}")
    if sink.cap < n:
        sink.grow(n)

    length, width = pitch
    metres = layout.units.get("x", "") in _METRE_UNITS
    kmh = layout.units.get("speed", "") in _KMH_UNITS
    players_xy = sink.xy[:n]
    if not metres:
        players_xy[..., 0] *= length
        players_xy[..., 1] *= width
    speed = sink.speed[:n]
    if kmh:
        speed /= 3.6
    ball_channels: dict[str, np.ndarray] = {}
    for j, ch in enumerate(layout.ball_channels):
        col = sink.ball[:n, j]
        if not metres and ch in ("x", "y"):
            col = col * (length if ch == "x" else width)
        elif ch == "speed" and kmh:
            col = col / 3.6
        ball_channels[ch] = np.ascontiguousarray(col, dtype=np.float32)
    col_name = {int(c): layout.ball_channels[j] for j, c in enumerate(layout.ball_cols)}
    codes = {col_name[c]: t for c, t in coder.codes.items() if c in col_name}
    report = dict(stats)
    report.update({
        "n_frames": int(n), "n_present": int(sink.present[:n].sum()), "from_spec": layout.from_spec,
        "player_channels": list(layout.player_channels), "ball_channels": list(layout.ball_channels),
        "ball_extra_channels": [c for c in layout.ball_channels if c not in ("x", "y", "speed")],
        "n_slots": layout.n_slots, "expected_frames": expected_frames,
        "matches_expected": expected_frames is None or n == expected_frames,
        "contiguous": bool(sink.present[:n].all()),
    })
    return TrackingData(
        players_xy=players_xy, players_speed=speed,
        ball_xy=np.stack([ball_channels["x"], ball_channels["y"]], axis=1),
        ball_channels=ball_channels, ball_channel_codes=codes,
        players_extra={k: v[:n] for k, v in sink.extra.items()}, team_of_slot=layout.team_of_slot,
        frame_index=np.arange(n, dtype=np.int64), present=sink.present[:n].copy(), source=source,
        fps=float(layout.frame_rate or FPS), pitch=(float(length), float(width)),
        team_ids=layout.team_ids, team_names=layout.team_names, player_ids=layout.player_ids,
        report=report)


def load_atd(data_dir: str | os.PathLike | None = None, source: str = "tactical",
             cache_dir: str | os.PathLike | None = DEFAULT_CACHE_DIR, use_cache: bool = True,
             n_frames: int | None = None) -> TrackingData:
    """Load the ATD of ``source`` ('tactical' or 'broadcast') aligned to video frames.

    ``data_dir`` defaults to the official bundle root (``games.DATA_DIR`` / ``MTAH_DATA_DIR``).  The
    parsed result is cached under ``cache_dir`` keyed by the size and mtime of the source files.
    """
    games = load_games_module()
    if games is None:
        raise ImportError("the vendored games.py is required to locate the ATD files")
    meta_path, raw_path = games.require(["atd_meta", "atd_raw"], games.GAME, data_dir, source)
    expected = games.source_frames(source, games.GAME)
    pitch = tuple(games.GAMES[games.GAME]["pitch"])
    cache_file = None
    if use_cache and cache_dir is not None:
        key = _cache_key([meta_path, raw_path], source)
        cache_file = Path(cache_dir) / f"atd_{source}_{key}.npz"
        if cache_file.is_file():
            try:
                return load_tracking_cache(cache_file)
            except (OSError, ValueError, KeyError):
                pass  # unreadable cache: re-parse and overwrite
    td = load_atd_files(meta_path, raw_path, source=source, n_frames=n_frames, pitch=pitch,
                        expected_frames=expected)
    if cache_file is not None:
        save_tracking(td, cache_file)
    return td
