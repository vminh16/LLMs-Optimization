# Repository Structure Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Create the approved production-lite repository skeleton and personal navigation skill.

**Architecture:** Keep root files minimal, move reusable Python code under `src/inference_opt`, keep command wrappers in `scripts`, and keep local run outputs in `results`. The personal skill remains outside the repo and points agents back to `AGENTS.md` plus the repo map.

**Tech Stack:** Python 3.11+, setuptools package layout, httpx, pytest, vLLM OpenAI-compatible API server.

## Global Constraints

- Preserve `python3 -m vllm.entrypoints.openai.api_server`.
- Keep directory depth shallow; source code should stay at `src/inference_opt/<domain>/<file>.py`.
- Keep `scripts` thin; reusable logic belongs in `src`.
- Do not add optimization code before a measurable benchmark or eval target exists.

---

### Task 1: Repository Skeleton

**Files:**
- Create: `pyproject.toml`
- Create: `docker-compose.yml`
- Create: `src/inference_opt/__init__.py`
- Create: `src/inference_opt/serving/__init__.py`
- Create: `src/inference_opt/trace/__init__.py`
- Create: `src/inference_opt/benchmark/__init__.py`
- Create: `src/inference_opt/eval/__init__.py`
- Create: `src/inference_opt/clients/__init__.py`
- Move: `trace-round1.jsonl` to `data/trace-round1.jsonl`
- Create: `.gitkeep` placeholders for empty tracked folders.

**Interfaces:**
- Produces: importable Python package `inference_opt`.
- Produces: main compose file matching the organizer-compatible baseline entrypoint.

- [x] **Step 1: Create top-level folders**

Run: `New-Item -ItemType Directory -Force -Path configs,docker,src\inference_opt,scripts,evals,data,tests,docs,results`

- [x] **Step 2: Move trace into data**

Run: `Move-Item -LiteralPath trace-round1.jsonl -Destination data\trace-round1.jsonl`

- [x] **Step 3: Add package and metadata files**

Add `pyproject.toml`, `docker-compose.yml`, and package `__init__.py` files.

- [x] **Step 4: Verify structure**

Run: `rg --files -uu`

Expected: approved folders appear and trace is under `data/`.

### Task 2: Architecture Docs

**Files:**
- Create: `docs/architecture/repo-structure.md`
- Create: `docs/superpowers/specs/2026-07-08-repo-structure-design.md`
- Create: `docs/superpowers/plans/2026-07-08-repo-structure.md`

**Interfaces:**
- Produces: written source boundary and dependency direction for future agents.

- [x] **Step 1: Document folder ownership**

Add the approved top-level folder map and `src/inference_opt` module map.

- [x] **Step 2: Document dependency direction**

Record allowed imports between `serving`, `trace`, `clients`, `benchmark`, and `eval`.

- [x] **Step 3: Verify docs have no unfinished markers**

Run a docs search for unfinished marker words before handoff.

Expected: no matches.

### Task 3: Personal Repo Navigation Skill

**Files:**
- Create outside repo: `C:\Users\USER\.codex\skills\repo-navigation\SKILL.md`

**Interfaces:**
- Produces: personal skill named `repo-navigation`.

- [x] **Step 1: Create skill folder**

Run with approval because it writes outside the repo workspace.

- [x] **Step 2: Add minimal SKILL.md**

Include the trigger, AGENTS.md first-read rule, folder map, source dependency direction, and `rg` search patterns.

- [x] **Step 3: Validate skill frontmatter**

Check `name` and `description` are present and the name matches the folder.
