# Timed Re-solves, Retained Mastery and Readiness (I-13)

**Goal:** turn "familiar practice" into scheduled, timed, hint-free re-solves of core problems at
about 7, 21 and 60 days, earn `retained` mastery from them, plan them in Consolidate sessions and
show honest readiness counts. `interview-ready` needs rubric-scored mocks and stays with I-18.

**Spec:** `docs/ARCHITECTURE.md` §4.4 (mastery chain, scheduling, session table), §4.9, §5;
ADR-0013 (no streaks or debt counters); `docs/LEARNER_PROFILE.md`.

**Split:** I-13a = domain, storage, planner and CLI (this plan's Tasks 1–4). I-13b = API routes,
generated client, Today/Progress page wiring (Task 5), started after I-13a is merged.

## Decisions

- **Timed window, not a new attempt mode.** A `resolve_window` row records one timed re-solve:
  `id` (32 hex), `problem_id`, `problem_version`, `topic_id`, `started_at`, `budget_s`,
  `hints_used` (int, default 0), `status` (`open|passed|partial|failed|abandoned`), `attempt_id`
  (nullable, the deciding attempt), `closed_at` (nullable). A partial unique index allows at most
  one `open` row. The `attempt` table and its CHECK constraint are unchanged; the window links to
  the deciding submit. One forward-only Alembic migration; `dogma backup` includes the table.
- **Rules of a window.** `run` (public tests) is unlimited. The **first recorded submit** of that
  problem while the window is open decides it: all tests pass, within `budget_s`, with
  `hints_used == 0` → `passed`; all pass but over budget or hinted → `partial`; any test fails,
  or the run is rejected/timeout/oom/error → `failed`. Budget = the problem's `meta.minutes` × 60.
  A window open longer than 12 hours is closed as `abandoned` the next time anything reads
  windows. `abandoned` carries no penalty.
- **Starting resets the workspace** to the starter, keeping the old file in
  `var/workspace-history/` exactly like `dogma open --fresh` (reuse that helper). Only a
  `verified`/`reviewed` core problem (p01) of a topic with a passing submit can be started. Starting
  while another window is open is an error naming the open one and how to abandon it.
- **Hints during a window** are recorded: the CLI `hint`/`debug` commands increment `hints_used`
  for an open window on that problem after an explicit confirmation ("this makes the re-solve count
  as practice"; `--yes` skips the prompt). The API tutor routes increment without a prompt (I-13b
  shows the warning in the browser).
- **Schedule is derived, not stored.** A pure function computes each core problem's re-solve stage
  and due time from (first passing submit time, closed windows in order). Stage = number of
  `passed` windows, capped at 3. Intervals by stage: 7, 21, 60 days, then 120 days maintenance.
  Anchor = the last `passed` window's close time, or the first passing submit for stage 0. After a
  `partial` or `failed` window, the problem is due again 3 days after that window closed, same
  stage. `abandoned` windows are ignored. Every result carries a one-line reason.
- **Mastery.** After each window closes and after each card rating, re-evaluate the topic:
  `consolidating` → `retained` when a `passed` window on its core problem closed ≥ 7 days after
  `solved_at` and no `failed` window on that problem came after it (set `retained_at`). A `failed`
  window drops `retained`/`interview-ready` to `consolidating` (never lower; matches the existing
  card-lapse rule in `learning/review.py`). A `passed` window while the topic is still `solved`
  advances the schedule; the topic becomes `retained` once a card review moves it to
  `consolidating` (re-evaluation catches up). `retained_at` updates on re-promotion.
- **Planner.** Consolidate's 25-minute re-solve block takes the most important due re-solves:
  lowest stage first, then earliest due, never today's Learn topic, at most two problems whose
  budgets fit the block. The block detail and first action name `dogma resolve <problem>`, the
  budget and "no hints". If nothing is due, keep today's familiar untimed practice but label it
  untimed. Minimum-viable sessions keep untimed familiar practice (no timed pressure on low days).
  Never show a backlog/overdue count (ADR-0013); show only the chosen next items.
- **Readiness** (`dogma progress`): per path week, topic counts by mastery; a "Re-solves" section
  with the next three due re-solves (problem, stage, due date) and the last five timed results
  (problem, verdict, time used vs budget). No retention estimates, streaks or comparisons.

## Ownership (I-13a)

`src/dogma/db/models.py` (new model or a `resolve_models.py`), one new migration under
`src/dogma/db/migrations/versions/`, `src/dogma/db/backup.py` (include the table),
`src/dogma/db/progress.py` (snapshot reads), new `src/dogma/learning/resolves.py` (pure schedule
+ mastery re-evaluation) and a small storage module for windows, `src/dogma/learning/planner.py`,
`src/dogma/learning/sessions.py` (gather due re-solves), `src/dogma/learning/review.py` (call the
re-evaluation after a rating), `src/dogma/cli/practice.py` (submit closes a window),
new `src/dogma/cli/resolve_cmd.py` + registration in `cli/main.py`, `src/dogma/cli/tutor.py`
(hint recording), `src/dogma/cli/progress.py`, matching tests under `tests/unit/**` and
`tests/e2e/cli/**`, `docs/ARCHITECTURE.md` §4.4/§4.9/§6 wording. Not: API, web, content,
pyproject/uv.lock, Makefile, ROADMAP.

## Tasks

### Task 1: Storage
- [ ] Failing tests: migration upgrades an existing DB (with attempts, cards, sessions) and keeps
  every row; at most one open window (second insert fails); backup exports the new table;
  `dogma db` commands still work.
- [ ] Model + migration + backup; storage helpers `start_window`, `open_window`,
  `close_window_for_attempt`, `abandon_window`, `windows_for(problem_id)` with an injected `now`.

### Task 2: Pure schedule and mastery
- [ ] Table-driven tests for every rule above: stage/interval progression, 3-day retry after
  partial/failed, abandoned ignored, 120-day maintenance, reasons; mastery promotion only from
  `consolidating`, ≥ 7-day rule, failed drop never below `consolidating`, catch-up after a later
  card review, retained_at on re-promotion.
- [ ] Implement in `learning/resolves.py` with no I/O; storage-backed wrappers call it inside
  the same transaction as the write that triggered it.

### Task 3: CLI
- [ ] e2e tests (`CliRunner`, scratch repo): `dogma resolve P` resets the workspace with a
  history copy and prints budget/deadline/rules; refuses non-core, unsolved or unready problems and
  a second open window; `--status` and `--abandon`; `dogma submit P` closes the window with the
  right verdict for pass / over budget (injected clock) / hinted / failed, prints the verdict and
  the next due date; `dogma run` never closes it; `--submission` (authoring) grading never
  touches windows; `DOGMA_RECORD_ATTEMPTS=false` never creates or closes windows.
- [ ] `dogma hint`/`debug` during a window: confirmation, `--yes`, increments `hints_used`.
- [ ] `dogma progress` readiness sections as specified; empty states are explicit.

### Task 4: Planner and sessions
- [ ] Planner tests: Consolidate picks due re-solves by stage then due date, excludes today's
  Learn topic, fits ≤ 2 budgets into 25 minutes, falls back to labelled untimed practice; Minimum
  stays untimed; no backlog counts in any text.
- [ ] Wire `sessions._gather` to supply due re-solves; keep plan persistence unchanged.

### Task 5 (I-13b): API and browser
- [x] API tests first: start (needs a first pass, core, recording on, one window at a time),
  current (null or the window), abandon (404 when none), submit closes the window (passed,
  partial when late or hinted, failed with a 3-day retry), run never closes, hint and SSE hint
  and debug count against an open window, readiness on `GET /progress`, every resolve code
  mapped in `api/errors.py`, the Today timed-block parse, Origin guard on the new routes.
- [x] API: `POST /api/v1/resolves` (201), `GET /api/v1/resolves/current` (200, `null` when none),
  `POST /api/v1/resolves/current/abandon`; `POST /attempts` returns `timed_resolve`; tutor routes
  count hints; `GET /progress` returns `readiness`; `SessionOut.timed_resolve` for Today.
  `web/src/api/schema.d.ts` regenerated and `gen:api --check` passes.
- [x] Browser tests first: the Today start action (start, refusal, continue), the Problem page
  window panel (budget, countdown, rules, Abandon with confirmation), the hint confirmation, the
  verdict note, the Readiness section (no streak, debt or overdue wording), the planned-action
  wording. Playwright: one start, submit, verdict, readiness flow.
- [x] Docs: ARCHITECTURE §4.4, §4.7 and §4.8 updated.
- Deviations: (1) Today reads the timed problem back from the planner's stored action text
  (`timed_problem_of`), because a plan block stores no structured problem id; a structured
  column would need a migration, which is I-13a's ownership. (2) The Playwright flow opens the
  window through `POST /api/v1/resolves` and does the rest in the browser: the browser smoke
  clock is fixed on the day of the first pass, so Today never plans a due re-solve there. The
  Today start action is covered by Vitest. (3) `GET /resolves/current` answers 200 with `null`
  when no window is open, rather than 404. (4) The CLI's readiness moved into
  `resolve_service.readiness`, so the CLI and the API share one implementation.

## Verification (I-13a)
`uv run --frozen pytest -q`, the new e2e tests with `-m ""`, `ruff check`, `ruff format --check`,
`mypy`, `lint-imports`. Never use the real learner DB: scratch `Settings(repo_root=tmp_path)`.
