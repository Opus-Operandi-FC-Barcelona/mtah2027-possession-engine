import re
import subprocess
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]

IGNORED = [
    "data/tactical/BAR-ATH_tactical_atd.txt",
    "data/anything/else.bin",
    "data/foo.csv",
    "cache/atd_players.npy",
    "cache/sub/dir/file.txt",
    "BAR-ATH_tactical.mp4",
    "tactical/video.mp4",
    "BAR-ATH_tactical_atd.txt",
    "x/BAR-ATH_tactical_atd.xml",
    "x/BAR-ATH_tactical_homography.csv",
    "x/BAR-ATH_tactical_smart_tagging.xml",
    "arrays.npy",
    "arrays.npz",
    "OpusOperandi.csv",
    "outputs/OpusOperandi.csv",
    "evaluation/results/OpusOperandi.csv",
    "evaluation/reports/report.json",
    "evaluation/scoreboard.csv",
]

NOT_IGNORED = [
    "data/README.md",
    "evaluation/results/baseline_smooth50.csv",
    "evaluation/results/baseline_raw.csv",
    "evaluation/gt/BAR-ATH_possession_gt_public.csv",
]


def check_ignore(path: str) -> bool:
    proc = subprocess.run(
        ["git", "check-ignore", "-q", "--no-index", path], cwd=ROOT, capture_output=True, text=True
    )
    assert proc.returncode in (0, 1), proc.stderr
    return proc.returncode == 0


@pytest.mark.parametrize("path", IGNORED)
def test_private_paths_are_ignored(path):
    assert check_ignore(path), f"{path} must be gitignored"


@pytest.mark.parametrize("path", NOT_IGNORED)
def test_public_paths_are_not_ignored(path):
    assert not check_ignore(path), f"{path} must not be gitignored"


def test_no_private_files_tracked():
    tracked = subprocess.run(
        ["git", "ls-files"], cwd=ROOT, capture_output=True, text=True, check=True
    ).stdout.splitlines()
    pattern = re.compile(r"^data/.*\.(mp4|txt|xml|csv)$|^cache/|\.npy$|\.npz$|\.mp4$")
    offenders = [f for f in tracked if pattern.search(f)]
    assert offenders == []


def load_ci():
    return yaml.safe_load((ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8"))


def test_ci_parses_and_has_security_gate():
    ci = load_ci()
    assert ci["env"]["PYTHONHASHSEED"] == "0"
    steps = ci["jobs"]["verify-pipeline"]["steps"]
    names = [s.get("name", "") for s in steps]
    gate = next(s for s in steps if "Security Gate" in s.get("name", ""))
    assert "git ls-files" in gate["run"] and "exit 1" in gate["run"]
    assert any("Ruff" in n for n in names)
    assert any("pytest" in s.get("run", "") for s in steps)
    py = next(s for s in steps if s.get("uses", "").startswith("actions/setup-python"))
    assert py["with"]["python-version"] == "3.11"


def test_ci_determinism_step_is_conditional():
    steps = load_ci()["jobs"]["verify-pipeline"]["steps"]
    det = next(s for s in steps if "SHA256" in s.get("name", ""))
    assert "if [ -f scripts/generate_submission.py ]" in det["run"]
    assert "sha256sum" in det["run"]


def test_ci_triggers():
    ci = load_ci()
    triggers = ci.get("on") or ci.get(True)  # PyYAML parses a bare `on` key as boolean True
    assert "develop" in triggers["push"]["branches"]
    assert "feat/**" in triggers["push"]["branches"]


def test_issue_template_and_tracker():
    template = (ROOT / ".github" / "ISSUE_TEMPLATE" / "task.md").read_text(encoding="utf-8")
    assert template.startswith("---")
    tracker = (ROOT / "docs" / "ISSUES_TRACKER.md").read_text(encoding="utf-8")
    pattern = r"^- \[( |x|~)\] (?:COMPLETADA|PENDIENTE|OLEADA [12]) · #(\d\d) · (wave-\d)"
    rows = re.findall(pattern, tracker, re.M)
    assert [int(n) for _, n, _ in rows] == list(range(1, 29))
    waves = {int(n): w for _, n, w in rows}
    assert {n for n, w in waves.items() if w == "wave-3"} == {10, 11, 12}
    wave1 = {1, 2, 3, 4, 5, 6, 7, 8, 9, 13, 14, 16, 23, 24, 25, 26, 28}
    assert {n for n, w in waves.items() if w == "wave-1"} == wave1
    state = {int(n): d for d, n, _ in rows}
    assert state[1] == "x" and state[2] == "x"
