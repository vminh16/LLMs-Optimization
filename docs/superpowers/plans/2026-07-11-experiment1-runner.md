# Experiment 1 Runner Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a safe, repeatable runner for the five independent BF16 vLLM Experiment 1 candidates and compare them with the locked H0.1 baseline.

**Architecture:** `serving/sweep.py` owns exact final vLLM commands and validates flag conflicts. `serving/experiment.py` owns preflight, immutable run identity, Docker lifecycle, diagnostics, and resume behavior; scripts remain thin CLI wrappers. Benchmark reporting reuses H0.1 summaries and applies explicit screening thresholds.

**Tech Stack:** Python 3.11+, standard library, Docker Compose, vLLM `v0.22.1`, unittest/pytest.

## Global Constraints

- Preserve `python3 -m vllm.entrypoints.openai.api_server` exactly.
- Keep the model `Qwen/Qwen3.5-2B` in BF16 for every Experiment 1 candidate.
- Use `data/trace-round1-diverse-content.jsonl` and require 120 successful requests.
- Do not combine scheduler or FP8 KV changes with Experiment 1.
- Do not silently overwrite or resume a run with a different trace or command fingerprint.

---

### Task 1: Exact Candidate Commands

**Files:**
- Modify: `src/inference_opt/serving/sweep.py`
- Modify: `tests/test_serving_sweep.py`

**Interfaces:**
- Produces: `build_experiment1_candidates()`, `select_candidates()`, and `validate_command_args()`.

- [ ] Write tests for all five candidate command deltas, prefix on/off exclusivity, duplicate flag rejection, and unknown candidate rejection.
- [ ] Run `python -m pytest tests/test_serving_sweep.py -q` and confirm the new tests fail because the interfaces do not exist.
- [ ] Replace additive candidate construction with exact final command tuples and remove FP8-coupled scheduler builders.
- [ ] Run the focused tests and confirm they pass.

### Task 2: Safe Experiment Lifecycle

**Files:**
- Create: `src/inference_opt/serving/experiment.py`
- Create: `tests/test_serving_experiment.py`

**Interfaces:**
- Produces: `ExperimentRun`, `build_preflight_commands()`, `prepare_run()`, and `run_experiment()`.

- [ ] Write tests for deterministic fingerprints, overwrite refusal, exact-match resume, preflight commands, failure diagnostics, and guaranteed cleanup.
- [ ] Run the focused test and confirm failure because the module does not exist.
- [ ] Implement file hashes, manifests, preflight command construction, lifecycle execution, log capture, and cleanup.
- [ ] Run the focused tests and confirm they pass.

### Task 3: CLI and Screening Report

**Files:**
- Modify: `scripts/run_serving_sweep.py`
- Create: `scripts/summarize_experiment.py`
- Modify: `src/inference_opt/benchmark/report.py`
- Modify: `tests/test_serving_sweep.py`
- Modify: `tests/test_benchmark_report.py`

**Interfaces:**
- Produces: `compare_candidate_to_baseline()` and CLIs for `--suite experiment1`, candidate filtering, preflight-only, dry-run, resume, and force.

- [ ] Write failing CLI and threshold-classification tests.
- [ ] Implement the minimal orchestration and report output.
- [ ] Verify focused tests pass and dry-run emits no Docker execution.

### Task 4: Baseline Alignment and Documentation

**Files:**
- Modify: `tests/test_local_compose.py`
- Modify: `docs/optimization/free-wins.md`

**Interfaces:**
- Consumes: current BF16 submission compose and Experiment 1 CLI.

- [ ] Align stale assertions with the current BF16 submission and static prefix override without changing serving configuration.
- [ ] Replace obsolete FP8-coupled sweep instructions with Experiment 1 commands and acceptance rules.
- [ ] Run focused tests, then the full test suite.
- [ ] Run Docker Compose config validation and CLI dry-run; record Docker-engine limitations explicitly.
