"""Issue #26: segment consensus (MBR) and the fixed submission contract."""
from __future__ import annotations

import hashlib

import numpy as np
import pytest

from mtah2027.config import HALF1, HALF2_DEAD, HALF2_PLAY, N_VIDEO_FRAMES, MatchConfig
from mtah2027.decoding import mbr_consensus as mbr
from mtah2027.postprocessing import submission_contract as sc
from mtah2027.validation import eval_bridge as eb

H, A, D, X = 0, 1, 2, 3


def _spec(*parts):
    return np.concatenate([np.full(n, c, dtype=np.int8) for c, n in parts])


# ------------------------------------------------------------------------------ contract
def test_default_layout_is_the_official_one():
    codes = np.full(N_VIDEO_FRAMES, H, dtype=np.int8)
    out = sc.enforce_submission_contract(codes)
    assert out.shape == (N_VIDEO_FRAMES,) and out.dtype == np.int8
    assert (out[:HALF1[0]] == X).all()
    assert (out[HALF1[0]:HALF1[1] + 1] == H).all()
    assert (out[HALF1[1] + 1:HALF2_DEAD[0]] == X).all()
    assert (out[HALF2_DEAD[0]:HALF2_DEAD[1] + 1] == D).all()
    assert (out[HALF2_PLAY[0]:HALF2_PLAY[1] + 1] == H).all()
    assert (out[HALF2_PLAY[1] + 1:] == X).all()
    assert eb.validate_labels_array(out)


def test_internal_x_forward_filled_and_head_back_filled():
    codes = np.full(N_VIDEO_FRAMES, X, dtype=np.int8)
    codes[HALF1[0] + 10:HALF1[0] + 20] = A
    codes[HALF1[0] + 20:HALF1[0] + 30] = X
    codes[HALF1[0] + 30:HALF1[0] + 40] = H
    codes[HALF2_PLAY[0] + 5:HALF2_PLAY[0] + 9] = A
    out = sc.enforce_submission_contract(codes)
    assert (out[HALF1[0]:HALF1[0] + 10] == A).all()            # head back-filled
    assert (out[HALF1[0] + 20:HALF1[0] + 30] == A).all()       # internal forward-filled
    assert (out[HALF1[0] + 30:HALF1[0] + 40] == H).all()
    assert (out[HALF1[0] + 40:HALF1[1] + 1] == H).all()        # trailing X continue the last label
    assert not (out[HALF1[0]:HALF1[1] + 1] == X).any()
    # half 2 is one graded run with the dead block: X after the forced D continue D until the first label
    assert (out[HALF2_PLAY[0]:HALF2_PLAY[0] + 5] == D).all()
    assert (out[HALF2_PLAY[0] + 5:HALF2_PLAY[0] + 9] == A).all()


def test_all_x_graded_run_becomes_dead_and_input_untouched():
    codes = np.full(N_VIDEO_FRAMES, X, dtype=np.int8)
    before = codes.copy()
    out = sc.enforce_submission_contract(codes)
    assert (out[HALF1[0]:HALF1[1] + 1] == D).all()
    assert np.array_equal(codes, before)


def test_letters_input_and_length_check():
    letters = np.array(["H"] * N_VIDEO_FRAMES)
    assert sc.enforce_submission_contract(letters)[HALF1[0]] == H
    with pytest.raises(ValueError):
        sc.enforce_submission_contract(np.zeros(100, dtype=np.int8))


def test_custom_blocks_for_the_short_synthetic_match(synthetic_spec):
    spec = synthetic_spec
    blocks = MatchConfig(n_video_frames=spec.n_frames, half1=spec.half1, half2_dead=spec.half2_dead,
                         half2_play=spec.half2_play)
    out = sc.enforce_submission_contract(np.full(spec.n_frames, A, dtype=np.int8), blocks)
    assert (out[spec.half1[0]:spec.half1[1] + 1] == A).all()
    assert (out[spec.half2_dead[0]:spec.half2_dead[1] + 1] == D).all()
    assert out[0] == X and out[-1] == X and out[spec.half1[1] + 1] == X
    assert eb.validate_labels_array(out, spec.n_frames)
    assert sc.graded_runs(blocks) == [spec.half1, (spec.half2_dead[0], spec.half2_play[1])]


def test_csv_is_byte_exact_lf_and_valid(tmp_path):
    codes = sc.enforce_submission_contract(np.full(N_VIDEO_FRAMES, H, dtype=np.int8))
    path = sc.write_submission_csv(tmp_path / "sub" / "team.csv", codes)
    raw = path.read_bytes()
    assert raw.startswith(b"frame,label\n0,X\n") and b"\r" not in raw
    assert raw.count(b"\n") == N_VIDEO_FRAMES + 1 and raw.endswith(b"\n")
    assert eb.validate_submission_file(path)
    # identical to the starter's writer
    starter = "frame,label\n" + "".join(f"{i},{c}\n" for i, c in enumerate(eb.to_letters(codes)))
    assert raw == starter.encode("ascii")
    assert hashlib.sha256(raw).hexdigest() == hashlib.sha256(sc.labels_to_text(codes).encode()).hexdigest()


def test_write_rejects_invalid_file(tmp_path):
    with pytest.raises(ValueError):
        sc.write_submission_csv(tmp_path / "bad.csv", np.zeros(10, dtype=np.int8), n_expected=11)
    assert sc.write_submission_csv(tmp_path / "short.csv", np.zeros(10, dtype=np.int8)).is_file()


# ------------------------------------------------------------------------------ MBR
def test_segments_and_matching():
    codes = _spec((X, 5), (H, 20), (D, 5), (A, 30), (X, 5))
    segs = mbr.possession_segments(codes, [(5, 59)])
    assert segs.tolist() == [[0, 5, 24], [1, 30, 59]]
    other = mbr.possession_segments(_spec((X, 5), (H, 18), (D, 7), (A, 30), (X, 5)), [(5, 59)])
    m = mbr.match_segments(segs, other, 0.3)
    assert [(r, o) for r, o, _ in m] == [(0, 0), (1, 1)]
    wrong_team = mbr.possession_segments(_spec((X, 5), (A, 20), (D, 35)), [(5, 59)])
    assert mbr.match_segments(segs, wrong_team, 0.3) == []


def test_consensus_takes_median_boundaries_and_drops_unsupported_segments():
    n = 400
    blocks = [(0, n - 1)]

    def run(h_end, a_start, extra_blip=False):
        c = np.full(n, D, dtype=np.int8)
        c[:h_end] = H
        c[a_start:] = A
        if extra_blip:
            c[300:312] = H
        return c

    runs = [run(100, 110), run(104, 112), run(98, 109, extra_blip=True)]
    out = mbr.consensus_labels(runs, blocks)
    assert (out[:100] == H).all() and out[100] == D        # median end of H: 100 -> last H frame at index 99
    assert (out[:100] == H).all() and (out[101:110] == D).all()
    assert (out[110:300] == A).all()
    assert (out[300:312] == A).all()                        # the unsupported H blip is dropped (1 of 3)
    assert mbr.consensus_labels(runs, blocks).tolist() == out.tolist()
    # a single run is returned unchanged
    assert np.array_equal(mbr.consensus_labels([runs[0]], blocks), runs[0])


def test_consensus_reference_choice_is_order_independent_for_identical_runs():
    c = _spec((H, 50), (A, 50))
    out = mbr.consensus_labels([c, c.copy(), c.copy()], [(0, 99)])
    assert np.array_equal(out, c)
    with pytest.raises(ValueError):
        mbr.consensus_labels([c, c[:50]], [(0, 99)])
    with pytest.raises(ValueError):
        mbr.consensus_labels([], [(0, 99)])


def test_consensus_overlaps_split_at_midpoint():
    blocks = [(0, 99)]
    r1 = _spec((H, 50), (A, 50))
    r2 = _spec((H, 60), (A, 40))
    r3 = _spec((H, 40), (A, 60))
    out = mbr.consensus_labels([r1, r2, r3], blocks)
    assert (out[:50] == H).all() and (out[50:] == A).all()


def test_mbr_params_validation_and_default_off():
    assert mbr.MbrParams.from_config().enabled is False
    with pytest.raises(ValueError):
        mbr.MbrParams(scales=(1.0, 2.0), ha_occluded_shifts=(0.0,))
    with pytest.raises(ValueError):
        mbr.MbrParams(scales=(0.0,), ha_occluded_shifts=(0.0,))
