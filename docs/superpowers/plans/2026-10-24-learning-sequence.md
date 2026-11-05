# Learning Sequence Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Complete the approved learning, content, retrieval, tutor and local browser sequence.

**Architecture:** Keep content as versioned packages and all learner state in SQLite.
Add pure learning services with thin CLI/API adapters. Preserve the isolated grader.
Independent content work owns only its assigned topic directories.

**Tech Stack:** Python 3.12, uv, Pydantic, Typer/Rich, SQLite/SQLAlchemy/Alembic,
FSRS; later FTS5, httpx, FastAPI and React/TypeScript.

**Spec:** `docs/superpowers/specs/2026-10-24-learning-sequence-design.md`,
`docs/ARCHITECTURE.md`, `docs/ENGINEERING_STANDARDS.md`, `docs/CONTENT_SPEC.md`.

## Global Constraints

- Use `uv run --frozen`; only the coordinator changes dependencies and shared docs.
- Keep content ids stable. Never weaken tests or widen tolerances to pass a solution.
- Never read `private/` or use the real learner database for verification.
- Preserve existing working-tree changes. Content and platform changes remain separate.
- No external services in core paths; model calls go through the loopback gateway.
- Only verified/reviewed content enters sessions; learner state is SQLite truth.
- Schema changes are forward-only Alembic migrations and preserve existing data.
- Tests use explicit timestamps and scratch settings; production failures are structured.

## Review Focus

- Edited/retired cards must preserve history while disappearing from due queues as appropriate.
- Duplicate ratings or closes must not double-count or overwrite completed work.
- Unseen topics and unsolved prerequisites must not leak into eligible review/study queues.
- Existing databases and existing no-record author commands must survive new migrations/hooks.
- EOF, empty queues, unavailable models and incomplete corpus sources need useful outcomes.

## Execution record

The user explicitly requested implementation after reviewing the proposed sequence.
No additional design/plan approval is required. Work is on `codex/learning-loop` in
the existing checkout, preserving uncommitted draft content and FSRS dependency.
The branch is the isolation boundary; no second checkout with missing drafts is used.
Detailed task evidence is kept in the plan-scoped SDD ledger. No automatic commits
to main, pushes, merges, or publishing. Scope decisions favor the existing design.

### Task 1: Persistent FSRS reviews (I-09)

**Files:** create `src/dogma/learning/{__init__,scheduler,cards,review}.py`,
`src/dogma/db/review_models.py`, an Alembic revision, `src/dogma/cli/review.py`,
`tests/unit/learning/`, `tests/unit/cli/test_cli_review.py`; modify DB model
registration, migration tests and backup expectations as needed. Coordinator
registers the CLI command in `src/dogma/cli/main.py`.

**Interfaces:** `sync_cards(settings: Settings, *, now: datetime) -> int`;
`due_cards(settings: Settings, *, now: datetime, limit: int = 20) -> tuple[ReviewCard, ...]`;
`rate_card(settings: Settings, card_id: str, rating: int, *, now: datetime,
expected_due: datetime) -> ReviewResult`. Dataclasses/Pydantic values carry card id,
topic id, question, answer and aware due timestamp; no live ORM objects cross the boundary.
`review_command(ctx, limit=20)` is the CLI adapter. Caller-supplied `now` is UTC-aware.

- [ ] Add behavior tests for solved-only enrollment, deterministic due dates, all four
  ratings, QA/cloze rendering, edited/retired content, stale ratings, lapse handling,
  naive timestamps, transaction rollback and old-database migration preservation.
- [ ] Run new tests and record the expected missing-behavior failures.
- [ ] Implement FSRS target retention 0.90 with fuzzing disabled, migrations,
  idempotent synchronization, bounded due queue, atomic ratings and CLI reveal/quit.
- [ ] Run `uv run --frozen pytest tests/unit/learning tests/unit/db tests/unit/cli -q`,
  plus ruff/mypy/import checks. Expected: all pass. Record default suite outcome.
- [ ] Independent task review; fix material findings; update roadmap and handoff.

### Task 2: Daily planner and close (I-10)

**Files:** `src/dogma/learning/{planner,sessions}.py`,
`src/dogma/db/session_models.py`, migration, `src/dogma/cli/{today,close}.py`,
tests under `tests/unit/learning/`, `tests/unit/cli/` and `tests/unit/db/`.
**Interfaces:** typed `SessionPlan` and `TransitionNote`; `today(settings, *,
now, energy)` returns an existing or newly persisted plan; `close_session(settings,
session_id, note, *, answer, now)` persists once and invokes Task 1 enrollment.

- [ ] Write failing tests for prerequisite gating, unavailable/draft content,
  exact block budgets, low-energy no-new-material, repeated today, close idempotency,
  stored transition notes, empty history and migration preservation.
- [ ] Implement pure planning and transactional persistence, then thin CLI commands.
- [ ] Verify targeted suites and full default suite; independently review.

### Task 3: Complete the active content wave

**Files:** only week-4/week-5 topic directories from `curriculum/paths/core.yaml`;
coordinator updates `docs/ROADMAP.md`.
**Interfaces:** existing CONTENT_SPEC topic/problem/card formats remain unchanged.

- [ ] Independently review week-4 math, contracts, oracle tests, failure messages,
  cards, citations and time estimates. Add regression tests before fixes.
- [ ] Finish all thirteen week-5 drafts with original lessons, contracts, solutions,
  independent tests, hints, cards and interview notes; keep stable ids.
- [ ] Verify each package with `dogma content verify`, exercise meaningful mutants,
  and get independent review. Record findings per topic and accurate statuses.
- [ ] Run `make check` once the wave is complete. Expected: green full gate.

### Task 4: Corpus retrieval (I-11)

**Files:** `src/dogma/corpus/`, `src/dogma/cli/corpus.py`, corpus tests.
**Interfaces:** `sync(settings)` incrementally converts and indexes allowed sources;
`search(settings, query, *, topics=None, k=5)` returns typed source/page/heading chunks.

- [ ] Write a focused plan against ARCHITECTURE §4.6 before implementation.
- [ ] Test incremental sync, heading/page provenance, malformed inputs, deletion,
  safe FTS queries and deterministic ranking with fixture documents.
- [ ] Implement docmd adapter, chunker, SQLite FTS5 index and CLI sync/search.
- [ ] Verify fixture integration plus a bounded real-source smoke test; review.

### Task 5: Local tutor (I-12)

**Files:** `src/dogma/tutor/`, versioned prompts, CLI hint/debug adapters, tests.
**Interfaces:** `LLMProvider` protocol and typed completion objects per ARCHITECTURE §4.5;
hint/debug services consume GradeReport and Task 4 citations.

- [ ] Write focused plan; add gateway, context, prompt and failure-path tests first.
- [ ] Implement local/fake providers, loopback enforcement, timeouts, bounded context,
  prompt-hash logs, hint ladder and graceful unavailable results.
- [ ] Verify with fake provider, no live calls by default; review.

### Task 6: Local API and browser (I-14 through I-17)

**Files:** `src/dogma/api/`, `web/`, API/frontend tests, dependency and build config.
**Interfaces:** versioned OpenAPI endpoints from ARCHITECTURE §4.7; generated TS types.

- [ ] Write focused API/browser plans; reuse Tasks 1, 2, 4 and 5 services.
- [ ] Test and implement API catalog, attempts, today/close, review and progress.
- [ ] Build offline web shell, Problem, Today, Review, Map and Progress screens,
  accessible keyboard controls, bundled rendering dependencies, generated API types.
- [ ] Verify service/route tests, browser smoke flows and production build; review.

### Task 7: Bounded maintenance and runtime readiness

**Files:** `Makefile`, hook config, runner/harness modules/tests, runtime group config.

- [ ] Align I-20 test-all with the full gate plus e2e/slow excluding live LLM calls;
  add pre-commit hooks and close documented type-check gaps without weakening tests.
- [ ] Plan and implement I-21 network/write denial, read-only hidden tests and safe
  run retention preserving attempts; verify platform-specific failure paths.
- [ ] Add I-19 optional torch group/runtime support before week-8 authoring;
  test supported/unavailable runtime behavior.
- [ ] Final full gate, independent whole-branch review and concise user handoff.
