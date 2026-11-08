---
name: Atomic task (1 module + 1 test)
about: One atomic Issue of the MTAH2027 possession engine
title: "#NN <title>"
labels: ""
assignees: ""
---

## Branch
`feat/issue-NN-<slug>` (created from `develop`)

## Scope
<!-- One production file + one test file. No stubs, no TODOs. -->

## Files
- `src/mtah2027/...` / `scripts/...`
- `tests/test_NN_....py`

## Acceptance criteria
- [ ] `python -m pytest tests/test_NN_*.py -q` is green
- [ ] `python -m ruff check src tests scripts` is clean
- [ ] `docs/ISSUES_TRACKER.md` marks the issue `[x] COMPLETADA`
- [ ] Commit `feat(issue-NN): <title> (closes #NN)`, merged `--no-ff` into `develop`

## Data safety checklist
- [ ] No file from `data/`, `cache/`, videos or real-match prediction CSVs is read, printed or committed
- [ ] Nothing is tuned on the public 5-minute window (11185..18684)
