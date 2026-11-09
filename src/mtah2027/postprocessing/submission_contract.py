"""Fixed submission contract and CSV writer (Issue #26).

The competition grades only three blocks of the 199,319-frame video (``MatchConfig``; the official
values by default, scaled ones for the short synthetic match):

==========================  =================================================================
``0 .. half1[0]-1``          not evaluated  -> ``X``
``half1``                    graded         -> no ``X`` left (forward-filled, back-filled at the head)
``half1[1]+1 .. half2_dead[0]-1``  half-time -> ``X``
``half2_dead``               graded, ball dead 100% -> ``D``
``half2_play``               graded         -> no ``X`` left
``half2_play[1]+1 ..``       post-match     -> ``X``
==========================  =================================================================

``spellF1`` forward-fills ``X`` inside a graded run and ``macroF1`` counts every remaining ``X`` as a miss, so
the exported file never contains an ``X`` inside a graded block.  Half 2 is one contiguous graded run
(``half2_dead`` + ``half2_play``) exactly like the scorer's ``valid_runs``; the fill therefore runs across the
forced ``D`` into the play block.  A graded run that is entirely ``X`` becomes ``D``.

``write_submission_csv`` writes ``frame,label`` with ``\\n`` line ends exactly like ``starter/baseline.py``'s
``write_submission`` (but independent of the platform's newline convention) and validates the file with the
official format check.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np

from mtah2027.config import MatchConfig
from mtah2027.validation import eval_bridge as eb

H, A, D, X = eb.H, eb.A, eb.D, eb.X


def graded_runs(blocks: MatchConfig) -> list[tuple[int, int]]:
    """Maximal contiguous graded runs (inclusive) of the match layout: half 1 and half 2 (dead + play)."""
    runs: list[list[int]] = []
    for a, b in sorted(blocks.graded_blocks):
        if runs and a <= runs[-1][1] + 1:
            runs[-1][1] = max(runs[-1][1], b)
        else:
            runs.append([a, b])
    return [(a, b) for a, b in runs]


def fill_run(codes: np.ndarray) -> np.ndarray:
    """Forward-fill ``X`` in a 1-D code array (leading ``X``: first non-``X`` value; all ``X``: ``D``)."""
    out = np.array(codes, dtype=np.int8, copy=True)
    if out.size == 0:
        return out
    known = out != X
    if not known.any():
        out[:] = D
        return out
    idx = np.where(known, np.arange(len(out)), -1)
    last = np.maximum.accumulate(idx)
    first = int(np.argmax(known))
    last[last < 0] = first
    return out[last]


def enforce_submission_contract(labels, blocks: MatchConfig | None = None) -> np.ndarray:
    """Return the label codes (int8: H=0, A=1, D=2, X=3) after applying the fixed contract.

    ``labels`` is a full-length array of letters or codes; ``blocks`` the match layout (default: the
    official BAR-ATH one).  The input is not modified.
    """
    blocks = blocks or MatchConfig()
    codes = np.array(eb.to_codes(labels), dtype=np.int8, copy=True)
    if len(codes) != blocks.n_video_frames:
        raise ValueError(f"expected {blocks.n_video_frames} labels, got {len(codes)}")
    for a, b in blocks.not_evaluated_blocks:
        codes[a:b + 1] = X
    a, b = blocks.half2_dead
    codes[a:b + 1] = D
    for a, b in graded_runs(blocks):
        codes[a:b + 1] = fill_run(codes[a:b + 1])
    return codes


def labels_to_text(labels) -> str:
    """The CSV text (header + one ``frame,label`` row per frame, ``\\n`` line ends)."""
    letters = eb.to_letters(labels)
    rows = ["frame,label"]
    rows.extend(f"{i},{c}" for i, c in enumerate(letters))
    return "\n".join(rows) + "\n"


def write_submission_csv(path: str | os.PathLike, labels, *, validate: bool = True,
                         n_expected: int | None = None) -> Path:
    """Write ``frame,label`` for ``labels`` (letters or codes) and, by default, validate the file.

    Raises ``ValueError`` with the official reason when the written file fails the format check.
    """
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with open(target, "w", encoding="ascii", newline="\n") as handle:
        handle.write(labels_to_text(labels))
    if validate:
        result = eb.validate_submission_file(target, len(labels) if n_expected is None else n_expected)
        if not result:
            raise ValueError(f"{target}: invalid submission: {result.reason}")
    return target
