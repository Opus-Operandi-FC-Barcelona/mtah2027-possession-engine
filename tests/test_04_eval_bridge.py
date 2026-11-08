import time
from pathlib import Path

import numpy as np
import possession_metrics as pm
import pytest
import run_evaluation

from mtah2027.validation import eval_bridge as eb

ROOT = Path(__file__).resolve().parents[1]
BASELINE = ROOT / "evaluation" / "results" / "baseline_smooth50.csv"
N = eb.N_VIDEO_FRAMES


@pytest.fixture(scope="module")
def public_gt():
    return eb.public_gt_codes()


def test_uses_the_vendored_scorer_module():
    assert eb.pm is pm
    assert Path(pm.__file__).resolve() == ROOT / "evaluation" / "possession_metrics.py"


def test_to_codes_matches_official_label_codes():
    raw = np.array([" h", "A", "d ", "x", "H", "D"], dtype=object)
    assert np.array_equal(eb.to_codes(raw), pm.label_codes(raw))
    assert eb.to_codes(["H", "A", "D", "X"]).tolist() == [0, 1, 2, 3]
    assert eb.to_codes(np.array([0, 1, 2, 3])).tolist() == [0, 1, 2, 3]
    assert eb.to_letters([0, 1, 2, 3]).tolist() == ["H", "A", "D", "X"]
    with pytest.raises(ValueError):
        eb.to_codes(["H", "Z"])
    with pytest.raises(ValueError):
        eb.to_codes([0, 4])
    with pytest.raises(ValueError):
        eb.to_codes([[0, 1]])


def test_perfect_prediction_scores_one(public_gt, synthetic_gt):
    result = eb.evaluate_arrays(public_gt, public_gt.copy())
    assert result["spell_f1_mean"] == pytest.approx(1.0)
    assert result["macro_f1"] == pytest.approx(1.0)
    assert eb.spell_f1(synthetic_gt, synthetic_gt) == pytest.approx(1.0)
    assert eb.macro_f1(synthetic_gt, synthetic_gt) == pytest.approx(1.0)
    letters = eb.to_letters(public_gt)
    assert eb.spell_f1(letters, letters) == pytest.approx(1.0)


def test_strings_and_codes_agree(public_gt):
    pred = np.roll(public_gt, 7)
    by_codes = eb.evaluate_arrays(public_gt, pred)
    by_letters = eb.evaluate_arrays(eb.to_letters(public_gt), eb.to_letters(pred))
    assert by_codes["spell_f1_mean"] == by_letters["spell_f1_mean"]
    assert by_codes["macro_f1"] == by_letters["macro_f1"]


def test_constant_prediction_scores_zero(public_gt):
    for label in ("H", "A"):
        pred = np.full(N, label)
        assert eb.spell_f1(public_gt, pred) == pytest.approx(0.0)
    assert eb.macro_f1(public_gt, np.full(N, "X")) == pytest.approx(0.0)


def test_one_frame_lag_is_nearly_free_and_one_second_is_not(synthetic_gt):
    one = eb.spell_f1(synthetic_gt, np.roll(synthetic_gt, 1))
    second = eb.spell_f1(synthetic_gt, np.roll(synthetic_gt, 25))
    assert one > 0.95
    assert second < one


def test_length_mismatch_raises(public_gt):
    with pytest.raises(ValueError):
        eb.evaluate_arrays(public_gt, public_gt[:-1])


def test_evaluate_matches_official_cli_scorer(public_gt):
    pred = eb.read_submission_codes(BASELINE)
    ours = eb.score_against_public_gt(pred)
    official, reason = run_evaluation.score_submission(str(BASELINE), public_gt, pm.TAUS, 0)
    assert reason is None
    assert ours["spell_f1_mean"] == official["spell_f1_mean"]
    assert ours["macro_f1"] == official["macro_f1"]
    assert ours["n_pred_spells"] == official["n_pred_spells"]


def test_public_baseline_smooth50_known_scores():
    pred = eb.read_submission_codes(BASELINE)
    result = eb.score_against_public_gt(pred)
    assert result["spell_f1_mean"] == pytest.approx(0.3256, abs=1e-4)
    assert result["macro_f1"] == pytest.approx(0.6207, abs=1e-4)
    assert result["n_gt_spells"] == 17
    assert eb.spell_f1(eb.public_gt_codes(), pred) == pytest.approx(0.3256, abs=1e-4)


def test_score_is_fast(public_gt):
    pred = eb.read_submission_codes(BASELINE)
    t0 = time.perf_counter()
    eb.score_against_public_gt(pred)
    assert time.perf_counter() - t0 < 2.0


def test_score_against_public_gt_requires_full_length():
    with pytest.raises(ValueError):
        eb.score_against_public_gt(np.zeros(100, dtype=int))


# ---------------------------------------------------------------- validation
def write_csv(path, header="frame,label", rows=None, n=N, label="X"):
    rows = rows if rows is not None else [f"{i},{label}" for i in range(n)]
    path.write_text("\n".join([header] + list(rows)) + "\n", encoding="utf-8")
    return path


def test_valid_file_accepted(tmp_path):
    result = eb.validate_submission_file(write_csv(tmp_path / "ok.csv"))
    assert result.ok and bool(result) and result.reason is None
    assert eb.validate_submission_file(BASELINE)


def test_invalid_files_rejected(tmp_path):
    cases = {
        "short": write_csv(tmp_path / "short.csv", n=N - 1),
        "long": write_csv(tmp_path / "long.csv", n=N + 1),
        "bad_label": write_csv(tmp_path / "bad.csv", label="Z"),
        "blank": write_csv(tmp_path / "blank.csv", rows=[f"{i}," for i in range(N)]),
        "one_based": write_csv(tmp_path / "one.csv", rows=[f"{i + 1},H" for i in range(N)]),
        "no_label_col": write_csv(tmp_path / "col.csv", header="frame,pred"),
    }
    for name, path in cases.items():
        result = eb.validate_submission_file(path)
        assert not result.ok, name
        assert result.reason, name
    assert "row count" in eb.validate_submission_file(cases["short"]).reason
    assert "outside" in eb.validate_submission_file(cases["bad_label"]).reason
    assert "empty label" in eb.validate_submission_file(cases["blank"]).reason
    assert "0.." in eb.validate_submission_file(cases["one_based"]).reason
    assert "could not read" in eb.validate_submission_file(cases["no_label_col"]).reason


def test_missing_and_unparseable_files_rejected(tmp_path):
    assert "no such file" in eb.validate_submission_file(tmp_path / "nope.csv").reason
    garbage = tmp_path / "garbage.csv"
    garbage.write_bytes(b"\x00\xff\xfe not a csv \x00")
    assert not eb.validate_submission_file(garbage)
    with pytest.raises(ValueError):
        eb.read_submission_codes(garbage)


def test_custom_expected_rows(tmp_path):
    path = write_csv(tmp_path / "small.csv", n=50)
    assert eb.validate_submission_file(path, n_expected=50)
    assert not eb.validate_submission_file(path)


def test_validate_labels_array():
    assert eb.validate_labels_array(np.full(N, "X"))
    assert eb.validate_labels_array(np.zeros(N, dtype=int))
    assert eb.validate_labels_array(["H", "a", "D"], n_expected=3)
    assert "row count" in eb.validate_labels_array(np.full(N - 5, "H")).reason
    assert "outside" in eb.validate_labels_array(["H", "Q"], n_expected=2).reason
    assert "empty label" in eb.validate_labels_array(["H", None], n_expected=2).reason
    assert "empty label" in eb.validate_labels_array([1.0, float("nan")], n_expected=2).reason
    assert not eb.validate_labels_array([[1, 2]], n_expected=1)
    assert not eb.validate_labels_array([0, 9], n_expected=2)


def test_validation_agrees_with_official_function(tmp_path):
    path = write_csv(tmp_path / "x.csv", label="D")
    table = pm.read_label_csv(path)
    assert pm.validate_submission(table, N) is None
    assert eb.validate_submission_file(path).ok
