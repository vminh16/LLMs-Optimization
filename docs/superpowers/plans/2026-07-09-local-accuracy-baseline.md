# Local Accuracy Baseline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make local GPQA reports use the measured 43/120 baseline while preserving the official 0.40 competition scoring defaults.

**Architecture:** Keep GPQA answer scoring independent in `eval/gpqa.py`, and let the thin evaluator CLI combine that result with the existing benchmark accuracy multiplier. Record the calibrated local gate and three-run baseline in documentation.

**Tech Stack:** Python 3.11+, argparse, unittest/pytest, JSON, Markdown.

## Global Constraints

- Do not change the official `baseline_accuracy=0.40` defaults.
- Do not change TTFT or TPOT scoring thresholds.
- Keep scripts thin and reusable scoring logic in `src`.
- Preserve the organizer-required vLLM entrypoint.

---

### Task 1: Local GPQA Gate Report

**Files:**
- Modify: `src/inference_opt/eval/gpqa.py`
- Modify: `scripts/evaluate_gpqa_results.py`
- Create: `tests/test_evaluate_gpqa_results_script.py`

**Interfaces:**
- Consumes: `score_gpqa_results` and `accuracy_multiplier`.
- Produces: `LOCAL_BASELINE_ACCURACY` and `build_report(answer_key, rows, baseline_accuracy)`.

- [x] **Step 1: Write the failing test**

```python
from scripts.evaluate_gpqa_results import build_report
from inference_opt.eval.gpqa import LOCAL_BASELINE_ACCURACY


def test_local_report_uses_measured_baseline():
    report = build_report({0: "A", 1: "B"}, [
        {"request_id": 0, "text": "A"},
        {"request_id": 1, "text": "B"},
    ], baseline_accuracy=1.0)
    assert report["baseline_accuracy"] == 1.0
    assert report["accuracy_delta"] == 0.0
    assert report["accuracy_multiplier"] == 1.0
    assert LOCAL_BASELINE_ACCURACY == 43 / 120
```

- [x] **Step 2: Run the focused test and verify it fails**

Run: `.\.venv\Scripts\python.exe -m pytest tests/test_evaluate_gpqa_results_script.py -q`

Expected: import failure for the missing constant or function.

- [x] **Step 3: Implement the minimum report glue**

Add `LOCAL_BASELINE_ACCURACY = 43 / 120` to `eval/gpqa.py`. Add a
`build_report` function and `--baseline-accuracy` argument to the evaluator,
then include `baseline_accuracy`, `accuracy_delta`, and
`accuracy_multiplier` in its JSON output.

- [x] **Step 4: Run focused and scoring tests**

Run: `.\.venv\Scripts\python.exe -m pytest tests/test_evaluate_gpqa_results_script.py tests/test_benchmark_scoring.py -q`

Expected: all tests pass, and official 0.40 behavior remains unchanged.

### Task 2: Lock Baseline Context

**Files:**
- Create: `docs/baseline/results.md`
- Modify: `docs/baseline/gpqa-benchmark.md`
- Modify: `docs/baseline/trace-benchmark.md`
- Modify: `AGENTS.md` Team Decisions Log only

**Interfaces:**
- Consumes: the three local trace summaries and three GPQA reports.
- Produces: the fixed L4 comparison baseline and phased optimization order.

- [x] **Step 1: Record measured medians**

Document median TTFT `21329.47 ms`, P95 TTFT `51667.18 ms`, median TPOT
`270.32 ms`, P95 TPOT `339.41 ms`, throughput `244.77 output chunks/s`,
zero errors, and GPQA `43/120`.

- [x] **Step 2: Document the local gate**

State that `43/120` is local-only, full credit starts at `31/120`, the linear
penalty spans `24-30/120`, and `23/120` or fewer gives zero.

- [x] **Step 3: Document optimization priority**

Order work as measurement/configuration free wins, simple serving
configuration, precision/memory changes, speculative decoding, and custom
kernels. Require one variable per experiment and GPQA checks after every
candidate that can affect correctness.

- [x] **Step 4: Update the Team Decisions Log**

Add one dated entry that locks the L4 and local GPQA references while keeping
the official competition constants unchanged.

### Task 3: Verification And Commit

**Files:**
- Verify all modified files.

**Interfaces:**
- Consumes: Tasks 1 and 2.
- Produces: a tested commit ready for optimization experiments.

- [x] **Step 1: Re-evaluate an existing baseline**

Run:

```powershell
.\.venv\Scripts\python.exe scripts/evaluate_gpqa_results.py `
  --requests results/gpqa-baseline/baseline-01/requests.jsonl `
  --output results/gpqa-baseline/baseline-01/local_gate.json
```

Expected: accuracy `43/120`, delta `0`, multiplier `1`.

- [x] **Step 2: Run the complete test suite**

Run: `.\.venv\Scripts\python.exe -m pytest -q`

Expected: all tests pass.

- [x] **Step 3: Check the diff**

Run: `git diff --check` and `git status --short`.

Expected: no whitespace errors; unrelated `requirements.txt`,
`results.tar.gz`, and user edits remain unstaged unless directly incorporated.

- [x] **Step 4: Commit**

```powershell
git add AGENTS.md docs scripts src tests
git commit -m "feat: calibrate local accuracy gate"
```
