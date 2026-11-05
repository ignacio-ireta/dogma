# Maintenance Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Complete I-20's local hooks and full gate, including the previously deferred content type check.

**Architecture:** Keep app typing strict. Check each content problem independently with fresh plain non-strict mypy configuration, staging `solution.py` as `submission.py` so the unchanged tests resolve the actual reference implementation. Local pre-commit hooks reuse frozen Python dependencies and the existing pnpm tools without installing hooks automatically.

**Tech Stack:** Python 3.12, frozen uv environment, mypy, pytest, pre-commit, pnpm/Biome.

**Spec:** `docs/superpowers/specs/2026-10-24-learning-sequence-design.md`; `docs/ENGINEERING_STANDARDS.md` §§5.2, 13; `docs/ROADMAP.md` I-20 and I-03 handoff.

## Global Constraints

- Use `uv run --frozen`; no dependency or lockfile changes.
- Never weaken tests, widen tolerances, or alter runtime content behavior to hide typing errors.
- No live models or learner-state access. Root coordinates Makefile, roadmap, verification and commits.
- No automatic installation into `.git`. Hooks are configuration available for explicit installation.

## Review Focus

- Reused `submission` module names must not collide across problem packages.
- SQL problems' Python tests must receive the same type check.
- Typed function bodies and imports must report errors despite the app's separate strict configuration.
- Missing content or required tests must fail clearly instead of silently checking nothing.
- Whitespace errors and oversized binary files must fail; fixture exceptions require a documented reason.

### Task 1: Faithful content type check

**Files:** Create `scripts/check_content_types.py`, `tests/unit/scripts/test_check_content_types.py`.

**Interfaces:** `main(argv: list[str] | None = None) -> int`; CLI `--root PATH` selects a scratch/repository root. Exit 0 means all discovered problems pass, 1 means type errors, 2 means invalid/incomplete input or checker infrastructure failure.

- [ ] Write CLI behavior tests for independent conflicting module names, typed solution/test errors, SQL test discovery, starter exclusion, non-strict configuration, missing content and incomplete packages.
- [ ] Run the new tests and observe failure because the checker is missing.
- [ ] Implement deterministic discovery under `content/*/*/problems/*`, staging only solution/tests in a temporary directory; invoke mypy separately per problem with a fresh plain non-strict config and `ignore_missing_imports = True` for existing unstubbed dependencies.
- [ ] Run focused tests, scoped lint/format and the real content checker. Report exact legacy annotation fixes to root before editing content.

### Task 2: Local hooks and commit-file checks

**Files:** Create `.pre-commit-config.yaml`, `scripts/check_commit_files.py`, `tests/unit/scripts/test_check_commit_files.py`.

**Interfaces:** Commit-file CLI accepts filenames; exits 1 with actionable diagnostics for trailing whitespace, missing final newlines or files larger than 1,000,000 bytes. Explicit large-fixture exceptions are restricted to `tests/fixtures/` and include a reason.

- [ ] Write real scratch-file tests for whitespace, binary size checks, valid fixture exceptions and rejected undocumented/outside-fixture exceptions; observe failures before implementation.
- [ ] Implement file checks without modifying input files.
- [ ] Configure local system hooks for frozen ruff check/format, frozen privacy audit, commit-file checks and existing pnpm/Biome lint. Pass no filenames to whole-repository privacy or web checks; serialize checks that share their environment.
- [ ] Validate pre-commit configuration and run scoped behavior tests, lint and format. Do not install hooks.

### Task 3: Coordinator gate integration

**Files:** Root owns `Makefile`, `docs/ROADMAP.md`, any approved legacy content annotation fixes.

- [ ] Add the content checker to `make types`; preserve strict app mypy and import contracts.
- [ ] Ensure `make test-all` runs the full check, e2e/slow Python tests excluding live models, and browser smoke tests.
- [ ] Run the aggregate gates on the exact final tree, document results and commit separately from content additions.

## Feasibility audit

The 135 authored problem packages take about 12 seconds using in-process, isolated mypy invocations with a shared temporary cache. Plain non-strict checking finds eight existing annotation errors in four files; expanding to `check_untyped_defs` introduces 20 failing packages, largely intentional invalid-input tests. Follow the documented plain non-strict policy while retaining normal checks of annotated functions; do not disable error categories or replace submission imports with `Any`.
