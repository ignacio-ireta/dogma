# Roadmap: incremental build plan

*The single source of truth for **what to build next**. Every increment is sized for one coding
session (one agent), owns a disjoint set of paths, and has acceptance criteria you can check.
Work stops and resumes across sessions through the status table and the handoff log.*

---

## How to run a session

1. Read `CLAUDE.md`, then this file. Read ARCHITECTURE / ENGINEERING_STANDARDS / CONTENT_SPEC as
   your increment requires.
2. Pick `todo` increments whose dependencies are `done`. Prefer, in order: anything blocking the
   learner's current week → the content runway → platform features.
3. Increments can run **in parallel only if their "Owns" paths don't overlap.** Shared files
   (`pyproject.toml`, `uv.lock`, `Makefile`, this file) are edited only by the increment that
   lists them, or by the coordinating session.
4. Use `uv run --frozen …`. Run `make check` before declaring done.
5. Set the status (`todo` → `in-progress` → `done` / `blocked`) and append a **handoff entry**
   (§4): what changed, what you verified (commands and results), what's left.

**Status legend:** `done` · `in-progress` · `todo` · `blocked`

## 1. Status

| ID | Increment | Milestone | Depends on | Status |
|---|---|---|---|---|
| I-00 | Planning, syllabus + tiers + core path, specs, example topic, `scripts/grade.py`, core skeleton, gate | M0 | — | done |
| I-01 | Import, scrub & vendor (sources, corpus/raw, tools/docmd, privacy audit) | M0 | I-00 | done |
| I-02 | Content model: schemas, loader, validation, catalog, coverage | M1 | I-00 | done |
| I-03 | Harness + grading + sandboxed runner | M1 | I-00 | done |
| I-04 | Week-1 content (11 topics, 3 parallel authors: a/b/c) | M0 | I-00 | done |
| I-05 | CLI v0: `dogma open/run/submit`, `content verify/coverage/new`, templates | M1 | I-02, I-03 | done |
| I-06 | Review week-1 content (second agent, 2 reviewers) | M0 | I-04 | done |
| I-07 | Week-2 content (13 topics, 4 authors: calculus / optimization+stats / probability / toolbox) | M0 | I-00 | done |
| I-08 | Persistence: SQLite + SQLAlchemy + Alembic, attempts, `dogma progress` | M2 | I-05 | done |
| I-09 | Cards + FSRS: `dogma review` | M2 | I-08 | done |
| I-10 | Planner & sessions: `dogma today`, `dogma close`, transition notes, low-energy mode | M2 | I-09 | done |
| I-11 | Corpus: docmd conversion, chunker, FTS5 index, `dogma corpus search` | M3 | I-01, I-05 | done |
| I-12 | Tutor v0: gateway (local + fake), hint/debug modes, versioned prompts, `dogma hint` | M3 | I-11 | done |
| I-13 | Timed re-solves, mastery states, readiness (plan `2026-10-27-timed-resolves.md`; I-13a backend+CLI, I-13b API+web) | M5 | I-10 | done |
| I-14 | HTTP API v1 (FastAPI) | M4 | I-10 | done |
| I-15 | Web shell + Problem page (CodeMirror, run/submit, results) | M4 | I-14 | done |
| I-16 | Web: Today + Review pages | M4 | I-15 | done |
| I-17 | Web: Map + Progress pages | M4 | I-15 | done |
| I-18 | Mock interviews, rubric scoring, `sys` track scaffolding (plan `2026-10-27-mocks-and-cases.md`; I-18a–e) | M5 | I-12, I-13 | done |
| I-19 | `py-torch` runtime (dependency group + runner support) | M1 | I-03 | done |
| I-20 | pre-commit hooks, `make test-all`, content type check in `make types` | — | I-05 | done |
| I-21 | Runner hardening: `sandbox-exec` network/write deny; move hidden tests out of the learner-writable run dir; run-dir retention | — | I-03 | done |
| I-23 | API layering + tutor SSE (prereq for I-15): move `cli.resolve`/`cli.path_view`/grade seam into a shared module, add an import-linter contract (api must not import cli), add SSE variants of /tutor/* (ADR-0017), coverage endpoint for the Map page | M4 | I-14 | done |
| I-22 | Tutor grounding: show the authored `hints.md` level first (instant, reviewed, no model), then an LLM nudge on the learner's code; dedupe corpus hits per source; small hand-graded prompt eval | M3 | I-12 | done |
| I-24 | Light visual refresh, bundled OpenDyslexic, responsive reading layouts | M4 | I-15 | done |
| I-25a | Additive Library refresh: selected 72 PDFs + 36 notes, provenance and retrieval | — | I-01, I-11 | done |
| I-25b | Strengthen numerical foundations, autodiff and optimization from primary sources | — | I-25a | done |
| I-25c | Strengthen probability, inference and uncertainty assumptions | — | I-25a | done |
| I-25d | Precision, LoRA and verified primary-paper citations | — | I-25a | done |
| I-25e | Selected existing tier-B topic packages: constraints, EM, packing and SARSA | — | I-25b, I-25c, I-25d | done |
| I-25f | Complete the four remaining optional system-design cases | — | I-25e | done |
| I-25g | Import the 72 refresh PDFs, convert, index and re-check `corpus:` citations on a machine that has the Library PDFs | — | I-25a | done |
| I-26 | Close implementation handoff: safe backup, structured re-solve target, model/phone checks, feedback and disk cleanup | — | I-18 | done |
| I-27 | Context-agnostic requirements, source/code cleanup and publication privacy checks | — | I-01, I-20 | done |
| I-28 | Exercise audit: mutation strength, blind solves, grader probes and message quality across all problems | — | I-03, I-05 | done |
| I-29 | Introduction: a Start here page and a guided first problem | — | I-15, I-16 | done |
| I-30a | Time-warp backend + CLI: personal calendar offset, sync to the furthest exercise, `dogma warp`, `dogma course reset` (plan `2026-11-03-time-warp.md`) | — | I-10, I-13 | done |
| I-30b | Time-warp API + web: `/calendar`, `/calendar/warp`, `/course/reset`, Progress page controls, Today offer | — | I-30a, I-17 | done |
| I-31a | Data realism platform: `dogma_data` (real offline datasets, frozen realistic generators, seeds), `hf` dependency group + HF snapshot sync, submission ban, test-import allowlist, `exercise_audit realism` (plan `2026-11-09-data-realism.md`) | — | I-03, I-28 | done |
| I-31b | Data realism content sweep: real and realistic data in the tests, statements and lessons of the affected problems (slices S1–S7) | — | I-31a | done |
| I-31c | Data realism review: second-agent review per slice, fresh exercise audit, statuses back to `reviewed` | — | I-31b | done |
| I-32a | Test-strength platform: hidden-test lint, `variants/` manifest, `scripts/exercise_gate.py`, docs and variants for the example problem (plan `2026-11-09-test-strength.md`; a1–a4) | — | I-31a | done |
| I-32b | Hygiene sweep: messages, ids, Hypothesis flags and `pytest.raises` in hidden tests, outcomes unchanged | — | I-32a, I-31c | done |
| I-32c | One blind solution per problem, committed as `variants/accept_blind` | — | I-32a, I-31c | done |
| I-32d | Reject sweep and mutant triage: one wrong solution per Contract bullet, gaps closed by gated tests | — | I-32b, I-32c | done |
| I-32e | Test-strength close-out: enforce the rules in the gate, re-audit, report | — | I-32d | done |
| I-33 | Real datasets everywhere: seeded real draws (`sample_rows`, `sample_window`), the Online Retail snapshot, and every problem moved off random numbers onto real values, sizes and rates (251 of 251), each under the refactor gate | — | I-31c, I-32e | done |
| I-34 | Test strength, second pass: machine probes (`scripts/exercise_probe.py`) for arguments changed in place, results returned in a laxer form and input forms no test gives; every gap closed by a gated hidden test (plan `2026-11-11-test-strength-2.md`) | — | I-32e, I-33 | in-progress |

### Content runway (one wave per path week; ARCHITECTURE §7)

Week N+1 must be `verified` before week N starts, and `reviewed` before its study week starts.
**C-Wnn** = author wave for week nn (3–4 author agents split by sub-theme, briefs modelled on
I-04/I-07). **R-Wnn** = review wave (2 reviewer agents, brief modelled on I-06). A review wave can
run alongside the next author wave, because their paths don't overlap.

| Path week | Study starts | Theme | Content due | Authored | Reviewed |
|---|---|---|---|---|---|
| W1 | 26 Oct | Linear algebra & numpy fluency | 25 Oct | ✅ I-04 | ✅ I-06 |
| W2 | 2 Nov | Calculus, optimization, probability I | 26 Oct | ✅ I-07 | ✅ R-W02a/b |
| W3 | 9 Nov | Probability & statistics II, experimentation | 2 Nov | ✅ C-W03 a/b/c/d | ✅ R-W03a/b |
| W4 | 16 Nov | Regression & classification | 9 Nov | ✅ C-W04 a/b/c/d | ✅ R-W04a/b |
| W5 | 23 Nov | ML workflow, validation, time series | 16 Nov | ✅ C-W05 a/b/c/d | ✅ R-W05a/b (12 reviewed; stacks-and-queues verified) |
| W6 | 30 Nov | Trees, ensembles, unsupervised, retrieval | 23 Nov | ✅ C-W06 a/b/c/d | ✅ R-W06 (all 13 topics) |
| W7 | 7 Dec | Neural networks from scratch | 30 Nov | ✅ C-W07 (`1271c3b`) | ✅ all 12 reviewed |
| W8 | 14 Dec | PyTorch & DL optimization (I-19 done) | 7 Dec | ✅ C-W08 a/b/c/d (13 topics) | ✅ R-W08 a/b/c/d (all 13) |
| W9 | 21 Dec | Attention & transformers (holiday, lighter) | 14 Dec | ✅ C-W09 a/b/c/d (12 topics) | ✅ R-W09ab/c/d (all 12) |
| W10 | 28 Dec | LLM fundamentals I (holiday, lighter) | 21 Dec | ✅ C-W10 a/b/c (10 topics) | ✅ R-W10a/b/c (all 10) |
| W11 | 4 Jan | LLM fundamentals II & inference basics | 28 Dec | ✅ C-W11 a/b/c/d (13 topics) | ✅ R-W11ab/cd (all 13) |
| W12 | 11 Jan | Serving at scale & fine-tuning | 4 Jan | ✅ C-W12 a/b/c/d (14 topics) | ✅ R-W12a/b/c/d (all 14) |
| W13 | 18 Jan | RL basics, alignment, RAG & evaluation | 11 Jan | ✅ C-W13 a/b/c/d (13 topics) | ✅ R-W13a/b/c/d (all 13) |
| W14 | 25 Jan | Interview week: mocks + re-solves, no new topics | — | — | — |

Topics per week: `curriculum/paths/core.yaml`.

## 2. Increment cards

### I-01 · Import, scrub & vendor
**Owns:** `scripts/import_sources.py`, `scripts/audit_privacy.py`, `sources/**` (not
`manifest.yaml` unless a rule is wrong), `corpus/raw/**`, `tools/docmd/**`, `Makefile` (add an
`audit` target and add it to `check`), `tests/unit/scripts/**`.
**Do:** implement IMPORT_PLAN §4. Read `sources/manifest.yaml` + `private/import_roots.yaml`, copy,
rename, scrub per IMPORT_PLAN §2, and write `sources/manifest.lock.json`. Vendor docmd with
`VENDORED.md`. The audit script scans the listed dirs for `private/denylist.txt` regexes plus
email/phone patterns.
**Accept:** `make audit` exits 0 with zero hits. `make check` is green. The lock lists every file
with origin sha256, final sha256 and `scrubbed`. `corpus/raw/library-papers` has 124 PDFs. `uv run --project tools/docmd docmd --help` works and docmd's
own test suite passes. Re-running the import is a no-op.

### I-02 · Content model
**Owns:** `src/dogma/content/**`, `tests/unit/content/**`, `tests/fixtures/content/**`.
**Do:** Pydantic v2 schemas (`extra="forbid"`) for the syllabus, core path, `topic.yaml`, lesson
front-matter, cards and `meta.yaml` (CONTENT_SPEC §2–6). Loader for topic packages. Validation
returning `Issue(code, path, message)` for every CONTENT_SPEC rule: ids, titles and tiers match the
syllabus; `requires` exist and are acyclic; the 7 lesson H2 sections are in order; card ids;
`pNN-slug` dirs; `meta.id` matches the path; exactly one core problem (p01); required files per
language. `build_catalog()` → `var/catalog.json` (atomic). `coverage()` → per-topic rows +
per-track/tier summary (pure data, no printing).
**Accept:** the real `content/` validates with zero issues. The syllabus loads 538 topics. The tier
A set equals the core-path topics. Every rule has a failing test. mypy strict + ruff clean.

### I-03 · Harness + grading + runner
**Owns:** `src/dogma_harness/**`, `src/dogma/grading/**`, `src/dogma/runner/**`,
`tests/unit/grading/**`, `tests/unit/runner/**`, `tests/integration/runner/**`,
`tests/fixtures/problems/**`.
**Do:** ARCHITECTURE §4.2–4.3 and CONTENT_SPEC §6.4.
- `python -m dogma_harness --run-dir D --tests … --report report.json` runs pytest in-process
  with a collector plugin.
- `dogma.grading`: `GradeReport` / `TestResult` / `StaticFinding` models + static checks
  (syntax, entry present, banned imports; port from `scripts/grade.py`).
- `dogma.runner.run(problem_dir, submission, include_hidden=…)` → `GradeReport`:
  `var/runs/<attempt_id>/`, `python -I -m dogma_harness`, new session/process group, timeout →
  `timeout`, psutil RSS watchdog → `oom`, scrubbed env, missing report → `error`.
**Accept:** integration tests cover passed, failed, rejected (syntax / missing entry / banned
import), timeout, oom, a huge-stdout truncation case and a SQL (DuckDB) problem. The example
topic's solution passes and its starter fails through `runner.run`. Import-linter contracts hold.

### I-04 · Week-1 content (authors a / b / c)
**Owns:** only the listed `content/<track>/<topic>/` directories.
- **a:** `cml/matrix-shapes-and-transpose`, `cml/matrix-multiplication`,
  `dl/building-tensors-by-broadcasting`, `cml/vector-and-matrix-norms`
- **b:** `cml/cosine-similarity`, `cml/matrix-inverse-and-rank`, `math/determinants-and-trace`,
  `cml/eigenvalues-and-eigenvectors`
- **c:** `sql/sql-select-filter-and-sort`, `sql/sql-group-by-and-having`, `algo/hash-maps-and-sets`

**Accept:** each package follows CONTENT_SPEC and matches the example topic's shape. Every
problem passes `scripts/grade.py verify`. `ruff check` on the package is clean. `status: verified`.
No names, employers or personal context.

### I-05 · CLI v0
**Owns:** `src/dogma/cli/**`, `templates/**`, `tests/e2e/cli/**`, `scripts/grade.py`, the Makefile
`verify-content` target, CONTENT_SPEC §7, README quickstart.
**Do:** a Typer app with `open`, `run`, `submit` (via `runner`), `content verify [--all|PATH…]`
(runner + validation + ruff + determinism), `content coverage`, `content new <topic-id>`
(scaffold from `templates/`, generated from the example topic), `catalog build`. Rich output that
names error types. Reduce `scripts/grade.py` to a deprecated wrapper.
**Accept:** e2e tests via `CliRunner`. `make check` uses `dogma content verify --all`. Docs updated.

### I-06 · Review week-1 content (second agent)
**Owns:** the week-1 topic directories (review edits only).
**Do:** the ENGINEERING_STANDARDS §5.6 checklist on each topic. Fix clear defects (wrong math,
unclear contract, weak or misleading failure messages, a missing edge case), **never weakening
tests**, then re-verify. Set `status: reviewed` on topic and problems. Record findings per topic
in the handoff log.
**Accept:** every week-1 topic is `reviewed` or has a listed blocker. Verify passes everywhere.

### I-07 · Week-2 content
Same as I-04 for the path-week-2 topics (`curriculum/paths/core.yaml`), split into 3 authors by
sub-theme (calculus & optimization / probability / toolbox).

### I-08 … I-21
Write the full card when an increment is picked up, following ARCHITECTURE §4. Each one that adds
a dependency owns `pyproject.toml` + `uv.lock` for that change: I-08 sqlalchemy + alembic,
I-09 fsrs, I-12 httpx, I-14 fastapi + uvicorn, I-19 the torch group.

### I-24 · Light visual refresh
**Owns:** shared browser styles, shell/navigation, page presentation, font assets and related
documentation. **Requested:** a prettier light interface with OpenDyslexic. **Accept:** locally
bundled fonts, consistent page design, responsive layouts, keyboard access and existing study
flows preserved. Plan and validation: `docs/superpowers/plans/2026-10-27-light-visual-refresh.md`.

### I-25a–f · Library refresh and optional content

**Plan:** [Library refresh](superpowers/plans/2026-10-28-library-refresh.md).
**Evidence/design:** [source audit and topic map](superpowers/specs/2026-10-28-library-refresh-design.md),
with an explicit [checksum inventory](source-refresh-2026-10-28.json).
**Owns:** importer/manifest/lock and refresh sources in I-25a; exact existing topic packages in
the design's batch map for I-25b–d; the four existing tier-B topic paths in I-25e; the four
remaining `sys` case paths in I-25f. Shared roadmap edits stay with the coordinator.
**Accept:** additive import preserves old citations and scrub history; primary-source locators
support revised claims; no core-path/tier-A changes; second content review and `make check`.

### I-26 · Implementation handoff follow-ups

**Owns:** `src/dogma/{db/backup.py,db/session_models.py,db/migrations/versions/b8c9d0e1f2a3_session_timed_problem.py,cli/backup.py,learning/planner.py,learning/sessions.py,api/services/learning.py}`,
their tests, optional live tutor smoke tests, mock phone checks and `web/src/styles/interview.css`;
public feedback tests and metadata for SQL first/last activity, grid components p02 and GRPO p02;
README/architecture/development/roadmap documentation and ignored worktree dependency cleanup.
**Accept:** pre-upgrade SQLite snapshot includes committed WAL data and fails before migration
if it cannot be written; timed target survives resume/migration without runtime prose parsing;
real model outcomes reported honestly; mock scoring/debrief works at 320/390px; four faulty
submission probes name the mistake; hidden tests untouched; full gate and browser/e2e pass.

### I-27 · Context-agnostic product and publication checks

**Owns:** `scripts/audit_privacy.py`, `tests/unit/scripts/test_{audit_privacy,privacy_gate,import_sources}.py`,
`Makefile`, `.pre-commit-config.yaml`, `.gitignore`, `CLAUDE.md`, README and privacy/design docs;
`sources/{manifest.yaml,manifest.lock.json,withdrawn.yaml}`; vendored docmd package metadata,
two IR test inputs, one golden Markdown file and `VENDORED.md`. Local private rules remain ignored.
**Accept:** published requirements describe general behavior rather than an individual; imports
use reviewed reference allowlists; code, metadata, filenames, staged blobs and newly built assets
are checked without echoing private terms; histories and binary-coverage limits are explicit;
required upstream attribution is preserved; full gate and relevant vendor tests pass.

### I-28 · Exercise audit

**Owns:** `scripts/exercise_audit.py`, `tests/unit/scripts/test_exercise_audit.py`,
`tests/integration/runner/test_grader_probes.py`, `docs/audits/`, additions to problem packages,
and `src/dogma/grading/static.py` or `src/dogma_harness/collector.py` only for confirmed verdict bugs.
**Do:** measure all 251 problems the same way: mutation strength, branch coverage, effective
precision, blind solves, seed robustness, grader probes and failure-message quality.
**Accept:** a report with a per-problem scorecard. The unmodified references pass through the tool.
A gap is fixed only with evidence (an executable witness, an untested contract clause, or a
requirement missing from the statement). `test_hidden.py` changes are additions only, each kills
its mutant and bumps the version. A hidden test that contradicts its statement is reported, not
changed. `make test-all` passes.

### I-29 · Introduction

**Owns:** `web/src/pages/WelcomePage.tsx`, `web/src/pages/welcome/`, `web/src/components/{RootRedirect,TourStrip}.tsx`,
`web/src/lib/welcome*.ts`, `web/src/styles/welcome.css`, their tests and `web/tests/e2e/welcome.spec.mjs`, plus small
edits to `App.tsx`, `NavRail.tsx`, `Icon.tsx` and `ProblemPage.tsx`.
**Do:** show a new learner what dogma is, how a session runs, where things live and what to do first, on `/welcome`,
with a guide strip on the first problem. Frontend only.
**Accept:** `/` opens `/welcome` once, then `/today`. Deep links are never redirected. Blocked browser storage falls
back to `/today`. Every link on the page works. No horizontal page scroll at 320px or 390px. `make test-all` passes.

### I-30a, I-30b · Time-warp

Plan: `docs/superpowers/plans/2026-11-03-time-warp.md`. ADR-0019.
**Owns:** see the plan, per sub-increment. I-30b starts after I-30a is merged.
**Do:** let the learner move the course calendar by whole weeks (sync to the furthest exercise run, pick a week,
move ahead or back, restore the authored dates) and reset the course after a backup, in the CLI and the web app.
Card and re-solve due dates never move. No wording about being behind.
**Accept:** as listed in the plan. `make check` for I-30a, `make test-all` for I-30b.

### I-31a, I-31b, I-31c · Data realism

Plan: `docs/superpowers/plans/2026-11-09-data-realism.md`. ADR-0020.
**Owns (I-31a):** `pyproject.toml` and `uv.lock` (the `hf` group and the `dogma_data` packaging and
import contracts), `src/dogma_data/`, `tests/unit/dogma_data/`, `scripts/sync_hf_datasets.py`,
`scripts/exercise_audit.py`, `src/dogma/grading/static.py`, `src/dogma/content/validate.py` (test
import allowlist only), their unit tests, `tests/fixtures/problems/py-realism/`,
`tests/integration/runner/test_runner_dogma_data.py`, ADR-0020, `docs/DATASETS.md`, and the
CONTENT_SPEC §6.4/§6.6, ENGINEERING_STANDARDS §5.2/§12, ARCHITECTURE §4.3/§8 and template edits
the plan lists. Sub-slices run in parallel worktrees on disjoint paths: a1 package, a2 guards and
audit, a3 HF sync, then a4 Seatbelt test and docs.
**Owns (I-31b):** the topic directories of slices S1–S7 in the plan; nothing under `src/`, `docs/`
or this file. **Owns (I-31c):** the same topic directories, for review fixes only.
**Do:** give tests real offline datasets, pinned HF snapshots and realistic generators where
synthetic data hides failures; name the data in `problem.md`; show a real-data failure in the
lesson. Keep every hand-built edge case.
**Accept:** `dogma_data` loads only numpy on import and is rejected in submissions; every dataset's
shape, hash and quirks are pinned by unit tests; each new realism test maps to a Contract sentence,
fails a hand-written realistic wrong solution that passes the old tests (`exercise_audit
realism`), passes the blind solution in under 10 s and survives `exercise_audit seeds` with the
data salt; versions are bumped. `make test-all` passes.

### I-32a–e · Test strength

Plan: `docs/superpowers/plans/2026-11-09-test-strength.md`. Design:
`docs/superpowers/specs/2026-11-09-test-strategy-design.md`. One runbook per step in
`docs/superpowers/plans/2026-11-09-test-strength/`; the coordinator starts at `01-coordinator.md`.
Done on 2026-11-11: the report is `docs/audits/2026-11-11-test-strength.md`, the decisions and
the open points for the owner are in `docs/audits/test-strength-decisions.md`.
**Owns:** per runbook, listed in the plan's increment table. The sweeps (I-32b–d) own topic
directories only: `test_hidden.py`, `variants/` and the version in `meta.yaml`.
**Do:** give every problem committed proof for its tests (`variants/`: solutions that must pass
and wrong solutions that must fail), a lint for the hidden-test rules, and one gate command for
any test edit. Then sweep every problem: fix the lint findings, add one blind solution, write one
wrong solution per Contract bullet and close the gaps they expose with gated tests.
**Accept:** as listed per runbook. Each sweep's `worklist` is empty or every problem left has a
note in `docs/audits/test-strength-decisions.md`; the report is in `docs/audits/`;
`make test-all` passes. Hidden tests are only added to, apart from message-only edits the gate
proves equal.

### I-33 · Real datasets everywhere

**Owns:** `src/dogma_data/` (`draws.py`, the `online_retail` loader and catalog entry,
`hf_manifest.yaml`, the snapshot and its lock and NOTICE), `tests/unit/dogma_data/`,
`docs/DATASETS.md`, CONTENT_SPEC §6.6 (one rule), and in the topic directories only
`test_hidden.py` and one sentence under "Done looks like" in `problem.md`. No `solution`,
`starter`, `test_public.py`, `meta.yaml` or `variants/` file changes.
**Do:** wherever a hidden test builds an input that can hold values from random numbers (a
feature matrix, labels, scores, a sample, a series, text, tokens, an image, an order table, and
also a tensor inside a network, a distribution, rewards, a transition model, request sizes, a
rate), draw it from a real dataset instead. Keep hand-built cases, Hypothesis properties over the
stated domain, scalar parameters and generators that set a property exactly. A problem whose
arguments are only sizes, counts or rates reads those off real data (rows, features, updates per
epoch, words in a passage, a rate counted from a table).
The rules are in DATASETS.md, "Real draws instead of random numbers".
**Accept:** every edited problem passes `exercise_gate.py gate <id> --mode refactor` (no test
function removed, reference and every accept pass, every recorded reject still fails its test, no
mutant weakened, seeds 1 to 5 with the data salt, time). `make test-all` passes.

### I-34 · Test strength, second pass

Plan: `docs/superpowers/plans/2026-11-11-test-strength-2.md`. Report:
`docs/audits/2026-11-11-test-strength-2.md`.
**Owns:** `scripts/exercise_probe.py`, `scripts/exercise_gate.py` (`START_TAGS`,
`prune-uncovered`), `tests/unit/scripts/test_exercise_probe.py` and the gate's unit tests,
CONTENT_SPEC §6.7 and §6.8, two lines of ARCHITECTURE, `docs/audits/` (the report and the
decisions file), and in the topic directories: `test_hidden.py` (additions through `add-test`
only), `variants/`, the version in `meta.yaml`, and in `problem.md` one new last Contract bullet
plus one sentence under "Done looks like", where a test needs a sentence the statement lacks. No
`solution`, `starter`, `hints.md` or `test_public.py` change.
**Do:** find by machine what one hand-written wrong solution per Contract bullet cannot: record
what the tests give each argument; grade the reference inside wrappers that change an argument in
place or return the right value in a laxer form; close every gap the Contract promises with one
test, proven by the wrapper as the wrong solution.
**Accept:** every new test entered through `exercise_gate.py add-test` (reference twice, every
accept, every recorded reject, the wrong solution on the old and the new tests, five other seeds,
run time). `exercise_gate.py variants` has no mismatch, `lint` has no finding, `exercise_audit.py
seeds` and `mutate` are no worse than at the start. `make test-all` passes.

## 3. Milestones


| Milestone | The learner can… | Target |
|---|---|---|
| **M0** | Study week 1 from verified content with `scripts/grade.py` | 25 Oct |
| **M1** | Use `dogma open/run/submit` with a sandboxed runner | early Nov |
| **M2** | Get a daily plan, FSRS reviews and transition notes | mid Nov |
| **M3** | Ask the local-model tutor for hints, with citations | late Nov |
| **M4** | Do everything in the browser | December |
| **M5** | Run timed mocks and see readiness | before W14 (25 Jan) |

## 4. Handoff log

*Newest first. One entry per session or agent: what changed · what was verified (exact commands
and results) · what's left or blocked.*

### 2026-11-11 · Real datasets in every problem that takes data · I-33 done

- **Why:** the owner asked that every problem that can use a real-world dataset (scikit-learn,
  HuggingFace, Kaggle-style tables) use one instead of dummy data, and that the answers be
  checked. Before this, 28 of 251 problems tested on real data; the bulk inputs of the rest were
  `rng.normal`, random labels, white-noise series, toy vocabularies, hand-typed numbers and
  generated order logs.
- **Result:** all 251 problems now have hidden tests that read real data. In 221 the problem
  takes data, and the arrays its hidden tests draw are real values or are derived from real
  values by a stated rule. A scan of every hidden test finds 10 problems that still call a
  random distribution for an array, and the audit lists each. In seven the call computes the
  expected output: it replays the function's own seeded draws (dropout, initialisers,
  bootstrap, inverse-transform sampling). In two it is the test's own Monte Carlo check and
  is never passed to the function. In one it is passed to the function: the standard normal
  of a closed-form check, E[exp(Z)] = e^(1/2), that the Monte Carlo test is named after.
  Hand-built edge
  cases with exact answers and Hypothesis properties over the stated domain are unchanged.
  In 30 the function has no argument that could hold dataset values (the audit prints the 30
  signatures), and the sizes, rates and prices its tests feed it are read off real datasets
  where they were drawn at random before. 246 problems changed: `test_hidden.py`, and in 236
  of them one sentence in `problem.md` naming the data. In 243 of them no starter, public
  test, `meta.yaml` or variant changed and no version moved: every test edit is a refactor
  under the gate, with no test function added or removed. Three problems changed beyond
  their tests (Bradley-Terry, k-NN classify and ACF, below).
- **The rule:** every input that can hold values is real. A problem whose arguments are only
  sizes, counts or rates reads those off real data (rows and features of a dataset, updates
  per epoch over it, words in a passage, a rate counted from a table). No dataset is read at
  import: a parametrized case names the dataset and the test resolves it.
- **Platform:** `dogma_data.sample_rows(name, gen, n, d, standardize=)` and
  `sample_window(name, gen, n)` give seeded draws of real rows, columns and series runs, so a
  test keeps its shapes and its seeds and gets real values. `california_housing` is drawable.
  Two new pinned HuggingFace snapshots. `online_retail` (Online Retail II, UCI, CC BY 4.0,
  876 KB): every invoice line of the 60 customers whose id ends in 77, two years, with real
  cancellations, duplicate lines and product descriptions as typed; `retail_events()` gives it
  as an activity log and `retail_invoices()` as one row per invoice. `mt_bench_judgments`
  (MT-Bench human judgments, LMSYS, CC BY 4.0, 140 KB): 3,355 expert verdicts on pairs of chat
  models, 582 items judged in both presentation orders; the conversation text and the
  annotator id are left out. 13 unit tests for the draws, 10 for the two tables. DATASETS.md
  has the table of which real data stands in for which kind of input, and the rules.
- **What the data is now:** feature matrices, labels and scores from wine, breast_cancer,
  diabetes, digits, iris and california_housing, with the errors of simple real models (a
  stump, nearest centroid, least squares) where a test needs predictions; categorical tables
  and counts from the adult census; samples and series from engel, nile, sunspots, elnino and
  co2; images, grids and whole numbers from digits; text, tokens, word counts and a bigram
  model from three novels and GSM8K; orders, activity logs, postings, sessions, shipments and
  daily revenue from online_retail, for all 22 SQL and pandas problems. In the second pass:
  tensors inside a network (29 torch problems) draw standardised breast_cancer measurements
  through one helper, `_real_values`, as a stand-in that the helper names as one; a greedy
  decoder runs on a bigram model counted from forty lines of a novel; a Bayes model is a
  class-by-band table counted from iris, wine, breast_cancer or California districts, as
  exact fractions; target and draft distributions are word shares in two novels; DPO
  log-probabilities come from two unigram models; rewards are invoice amounts and series
  changes, Q tables are wine and digit values, and an MDP is counted from transitions between
  bands of El Niño temperature; request sizes are GSM8K sentence and step counts and invoice
  line counts, arrivals are invoice times, and overflow flags are runs of cancelled invoices; tool-call arguments are GSM8K questions as written;
  trees, coin amounts, cache keys and dependency graphs come from wine and the retail lines.
- **What is still placed by a test, and said so in its docstring:** a known truth over real
  inputs (lasso, ridge, logistic, normal equation, permutation importance); a property that
  must be exact (the spectrum over real principal axes in power iteration and gradient
  descent, the rank over real columns, gaps of exactly thirty minutes on real clocks, dates on
  a year boundary); dirt and gaps the source lacks (missing measurements, a missing region, a
  status typed with a capital, labels in another letter case, a missing timestamp); and a
  fixed shift of real dates where a statement names a calendar window (monthly active users,
  top five orders of 2024, both streak problems). In the SQL problems an invoice line stands
  for a session, a posting, a view or a shipment, as each fixture says.
- **What real data broke, and the fix each time:** all-positive columns let a sign mistake and
  an in-place clip pass (standardised draws); values stored to four decimals let an in-place
  rounding pass (a ratio column); whole-number hours let a truncation pass (iris
  measurements); a strong predictor left no mixed ties for the F1, ROC and AP tie rules (a weak
  predictor, rounded); a good ranker never put the first relevant item beyond k (a poor ranker
  on odd seeds); a real stump is sometimes perfect, a small draw sometimes has a constant
  column or a rank-deficient design, a river series has equal neighbours (the draw is repeated,
  and the docstring says why); real queries needed a repeated word and an unknown word first
  (BM25); real labels differ in spaces but never in letter case alone (one in seven is
  lower-cased); the longest real streak has to cross the year end (twenty busy days moved onto
  it); a pivot mutant in matrix rank was caught only by luck before (the dependent row is now
  inserted at every position, with both signs). No tolerance was widened and no assertion
  removed.
  In the second pass: a greedy model on real text almost never emits the pad id as its last
  token (the word the longest rows end on trades ids with the pad id); a row of standardised
  measurements can lie wholly inside one, which hid an in-place rescale (the draw is repeated
  until an entry exceeds one); real distributions over a dataset smaller than the sequence
  (California districts for 1,500 observations).
- **Third pass (sizes, rates and the last stand-ins):** index generators take the row counts
  of real datasets; schedules last one epoch over a real dataset; networks are sized by real
  features and classes; sequence, prompt and context lengths are word counts of real
  passages; acceptance rates come from the next-word distributions of two novels; verdict
  pairs are MT-Bench items judged in both orders; token timings in the latency problems are
  real waits between invoices; prices are real prices in pounds. What broke: a line amount
  is a whole multiple of its own unit price, so floor and ceiling agreed (the two prices
  come from different lines); no passage of whole lines is one word long (single lines
  count as passages); a prompt never filled the budget exactly (the budget is drawn from the
  longest prompt upward).
- **Fourth pass (no generated draw left in a data problem):** the floating-point problems
  lost their random bit patterns and magnitudes: real numbers now span forty orders of
  magnitude (measurements, census weights, the probabilities of real phrases down to 1e-45,
  their differences, logarithms and reciprocals), so float32 subnormals, zeros and infinities
  come from real values, NaN from the missing weeks of the CO2 record, and only the special
  words are typed by hand. Vector projection draws every pair from real tables, one of them
  digit pixels as a shortfall from full ink, which gives the exact zeros a scaling mutant
  needs. The debiased win rate draws only real verdict pairs (all nine combinations occur).
  Monte Carlo resamples three real populations, and its coverage test uses the exact
  population mean as the truth. Reservoir sampling streams real words in every test, and the
  memory test is three times longer because a stored list of shared words is small.
- **Fifth pass (secondary draws):** a scan of every hidden test found about ninety data
  problems whose main draw was real but whose other tests still drew arrays at random. Two
  tools rewrote those calls inside test functions (never at module level): normal draws and
  `torch.randn` became standardised `breast_cancer` measurements of the same shape
  (`_real_normal`, `_real_values`); uniform draws, whole numbers in a range, Dirichlet
  vectors and random masks became the same measurements passed through the normal
  distribution function, cut into bands, normalised or thresholded (`_real_unit`,
  `_real_ints`, `_real_shares`). 84 problems were rewritten; 80 passed the gate as they were,
  three of the other four were then done by hand, and so was k-NN classify (below).
  Logistic regression fits the real diagnosis and
  the spector grades. The prior-shift simulation resamples real tumours
  at the deployment rate under Gaussians fitted to a real measurement. The coverage test
  samples a real population whose mean is known exactly. ROC AUC times real diagnoses.
  Only the two-layer MLP went back unchanged: its draws are the seeded initialisation the
  function itself performs. Coin systems, arrival steps and bandit arms are real as well
  (order quantities and prices, invoice minutes, the weekday of each invoice line), and the
  microsecond sessions fixture is built from real customers and real waits between invoices
  (read in seconds), with only the gaps at the thirty-minute boundary placed.
- **k-NN classify: a tie the Contract did not define, now defined.** The integer-grid test
  expected rows at the same distance on paper to tie exactly. That holds for rows whose
  coordinate differences from the query are the same up to sign, whatever the arithmetic.
  It does not hold for rows such as (1, 3, 3) and (3, 3, 1): on a real-derived grid the
  blind accept, which rescales before squaring, was separated from the exact oracle by one
  rounding step on one query in three hundred. The Contract now says which ties are exact
  and that the other kind may go either way; the test compares only queries whose ties are
  exact (586 of 1,200 grid queries, 436 of them with an equal-distance pair, and it asserts
  that at least 300 are compared) on real-derived grids. Reference, blind accept and every
  reject behave as recorded; no mutant is weakened; the version is unchanged because the
  gate tool derives it from the tests. Gated against a snapshot of the amended statement and
  again with base `HEAD` in a scratch worktree.
- **ACF: an accepted variant broke the Contract on some series, and is replaced.** After
  the conversions I ran every accepted variant of the 246 changed problems under three other
  data salts (1,365 runs; the gate runs accepts on the default data only). Two failed.
  `accept_loop_sums.py` of `p02-acf` added the lag sums in a Python loop and the denominator
  with a dot product, so lag 0 was not exactly 1 on some real windows, which clause C3
  requires; a normal draw had never shown it. The variant now adds the denominator the same
  way, the old form is recorded as `reject_lag_zero_sums_differ.py`, and the lag-0 test
  checks twelve real windows where it checked one. Gate passed; both accepts pass eight
  salts.
- **Logistic regression: the engineered separable case is now a real rule, and the blind
  accept passes every salt.** The other failure of that run was the blind accept of logistic
  regression, which did not converge on one separable case under data salt 1 (so at
  `6ce4070` too). The case drew a random direction over the real features, so each salt set
  a different bar. It is now a real rule with no draw in it: the tumours whose worst area is
  above the median, and those whose worst perimeter is in the top hundredth. One real
  measurement separates them perfectly, the raw scale still drives linear scores into the
  hundreds, the reference and both accepts converge, and the reject without a line search
  still fails. Gate passed; both accepts pass twelve salts. The run over every accepted
  variant was repeated on the final tree, for all 251 problems and five salts: 461 accepted
  variants, 2,305 runs, no failure.
- **A reference solution was wrong at one point; the problem is corrected and re-versioned:**
  `llm/bradley-terry-reward-models/p01-reward-pair-loss-and-accuracy` returned a gradient of 0
  for a pair whose two rewards are equal, where the loss has -1/(2B): it wrote max(x, 0) as
  `clamp(x, min=0)`, which has a kink at 0. Real rewards tie; random normals never did. A
  reward head initialised at zero ties every pair, so that solution could not start
  training. Now: the Contract states the gradient at every margin, a tie included (clause
  C5), and the Constraints, hint 2 and the lesson name the pitfall; the reference writes
  max(x, 0) as (x + |x|) / 2 (the same mean loss, to the last bit, on 14 batches of 64 pairs
  at scales from 1e-30 to 1e300, 1e30 in float32, and the analytic gradient everywhere); a
  new hidden test, `test_tied_pairs_receive_the_gradient_of_the_smooth_loss`, entered through
  `add-test` with the old form recorded as `reject_tied_pair_gets_no_gradient.py`; the two
  older gradient tests take tied draws again; the three accepts were replaced through the
  tool (sum over count, `torch.maximum` for the positive part, and a new blind solution
  written by the local model server from the amended statement, the starter and the public
  tests only); the eight older rejects carry the corrected line, so each still has one
  mistake; version 4 to 5. Both gate runs passed (`add`, then `refactor`). The gate freezes
  the statement and the reference against its base, so the base was a snapshot of the
  corrected statement and reference: once through `add-test --base` with snapshots made by
  `git stash create` (a commit object, no branch moved), and once more with the default
  base, `HEAD`, in two scratch worktrees checked out at those snapshots and removed after.
- **Verified (answers are right):** each of the 246 problems passed
  `exercise_gate.py gate <id> --mode refactor` against `HEAD` after its last edit: the
  reference passes twice, every accept variant passes (blind solutions included), every
  recorded reject still fails the test named in its manifest, no sampled mutant that the old
  tests killed survives the new ones, seeds 1 to 5 pass with the data salt, and the run time
  holds. Gates that print FAIL do so only on the `names` check for numbered ids such as
  `case-0`, which are unchanged and are the open A2 point of I-32. Expected values come from
  library oracles on the same data (NumPy, SciPy, scikit-learn, statsmodels, pandas, DuckDB,
  torch) or from the brute-force references already in the tests. `exercise_audit.py seeds`:
  2,000 runs on the first 100 problems with one failure (permutation importance p01, salt 17:
  two permutation seeds gave the same mean drops on correlated real columns; the test now
  compares six other seeds, and the 20 runs of that problem pass), and 980 runs on the other
  49 with none, and 1,360 runs on the 68 problems of the second pass with none, 100 runs
  on the five test files edited after that audit with none, 660 runs on the 33 problems of the
  third pass with none, 160 runs on the eight problems of the fourth pass and Bradley-Terry
  with none, 1,600 runs on the 80 problems of the fifth pass with none, and 220 runs on the
  problems finished by hand after it with none.
- **Verified (whole repository):** `make test-all` exit 0 on the final tree of tests, solutions,
  statements, variants and data (`var/i33/test-all-10.log`): Ruff and format, strict mypy
  (181 files), 6 import contracts, content types (251 problems), 2,247 Python tests, `All 175
  topic(s) verified`, 187 browser unit tests and the build, audit clean (6,261 files),
  `variants: 2310 ok, 0 mismatches in 251 problems`, 11 e2e/slow tests, 22 Playwright tests.
  Earlier attempts of the run failed three times and each cause was fixed: mypy on a helper
  that reused one name for a list and a tensor (greedy generation), mypy on a case list that
  mixes names and numbers (A/B sample size), and the example topic `cml/vector-operations`,
  which is also linted as a copy outside the repository where `dogma_data` sorts as
  third-party (its import now sits inside the helper). This entry, the card, DATASETS.md and
  the audit file were written after that run; Ruff and the privacy audit were run again on
  them. `sync_hf_datasets.py --check`: clean.
- **Open:** the retail sample has 60 customers and no Saturdays, which is thin for
  per-customer and streak tests (a product family stands for the user there); `linnerud`,
  `macrodata` and the three generated interview-prep CSVs are in the catalog but unused.
  Merged into `main` at `90ffc70` on the owner's instruction, on top of the I-32 branch
  (`fbce158`, which carries the five commits of `origin/main`: CI, licences, run-timeout
  scale). `make test-all` on the merge: exit 0, 2,252 Python tests, `All 175 topic(s)
  verified`, `variants: 2310 ok, 0 mismatches in 251 problems`, 187 browser tests, 22
  Playwright tests. Pushed to `origin/main`; the merged branches `i33-real-datasets` and
  `i32-test-strength` and their worktrees were removed. `i34-test-strength-2` belongs to
  another session and was left alone.
- **Decided by the owner (2026-11-11):** the 30 problems whose functions take only sizes,
  rates or prices stay as they are, and so do the ten problems whose tests still call a
  random distribution (seven seeded replays, two Monte Carlo checks, one closed-form normal).
  They are not to be redesigned or have tests retired for this.

### 2026-11-11 · Test strength finished: last 81 problems swept, rules enforced, report written · I-32d and I-32e done

- **Reject sweep and mutant triage for the 81 problems in the 48 I-31 topics (I-32d):** 618 wrong
  solutions that the tests reject, 140 accepted ones and 89 Contract bullets marked uncovered
  with a reason; 43 hidden tests added. The mutant triage looked at 220 surviving mutants: 135
  cannot change a result, 69 change something no Contract sentence covers, 1 has no usable
  witness, and 11 hidden tests were added. All four worklists (`blind`, `hygiene`, `reject`,
  `mutants`) print `total: 0` for all 251 problems.
- **Corrections during the sweep:** one commit (`0c75771`, ridge p01) went in with a failing gate
  after a data swap and was fixed by `f8f9c0d`. A wrong solution in `cml/k-means/p02` that the
  tests rejected only on some runs was replaced (`df854cd`). Two executor tests did not reach a
  commit (an unreadable-date test for pandas-groupby p01, a subnormal-input test for
  feature-scaling); the reasons and the saved patches are in
  `docs/audits/test-strength-decisions.md`.
- **Close-out, Part A (`05559f0`):** the hidden-test lint now runs in `dogma content validate`
  (A1 on) and `VARIANTS_REQUIRED` is `True` (A3 on). **A2 is off:** the id-leak check in
  `content verify` refuses any digit in a parametrized id, and 35 problems in 25 topics have
  neutral but numbered ids such as `case-0` and `grid0`. With the check, `content verify --all`
  ended with `25 of 175 topic(s) failed`, so it was undone as the runbook says. The 35 problems
  are listed in the report.
- **Close-out, Part B:** `docs/audits/2026-11-11-test-strength.md`, all numbers pasted from the
  tools. `baseline: 251/251 references pass`; `variants: 2308 ok, 0 mismatches in 251 problems`;
  `lint: 0 findings in 0 problems`; 10,551 mutants, `killed 93.7%`; scorecard: 251 problems, 251
  with a blind accept, 461 accepts, 1,847 rejects, every bullet covered 251, both sweeps 251.
- **Seeds finding, not changed:** `seeds: 3 failing runs on 1 problems`.
  `cml/lagrange-multipliers/p01-constrained-quadratic-min` fails
  `test_solution_is_feasible_and_satisfies_stationarity` under Hypothesis seeds 1, 13 and 15.
  The grader pins seed 0, which passes, so a Submit does not see it. I-32 did not write or
  tighten that test.
- **Verified:** `make test-all` exit 0 on `8b49d37` (the report commit) in a clean worktree: Ruff and
  format (4,401 files), strict mypy (180 files), 6 import contracts, 2,198 Python tests, 175
  topics verified, 187 browser tests and the build, audit clean (6,240 files),
  `exercise_gate.py variants` 2,308 ok and 0 mismatches in 251 problems, 11 e2e/slow tests, 22
  Playwright tests. `make check` exit 0 on `de302dd` (the end of the sweeps). The Part A accept
  commands passed on `05559f0`. This entry is a docs-only commit after the gate; the privacy
  audit was run again on it.
- **Open decisions, by kind (ledger rows):** blocked 4, question 13, unpromised 217, no-witness
  10, overreach 9, message-defect 102, inspect 13 (327 rows are done). The tolerance table in
  the report lists 7 problems whose tests are stricter than the declared `rtol` and 1 whose
  tests are looser; nothing was changed.
- **For the owner:** the "Waiting for the owner" sections of
  `docs/audits/test-strength-decisions.md`. The main ones: the A2 rule against the numbered ids;
  the lagrange seeds; Contract sentences that are missing (KV-cache minimums, non-tensor
  arguments, defaults that live only in a signature); the kl-divergence blind accept that
  disagrees with the reference on an underflowed probability; the sampling problem whose
  statement does not fix how the value is drawn; `uncovered` entries that later tests made
  stale (adamw C1 and C3, prefill C2, k-means p02 C5).
- **State of the branches:** `main` was fast-forwarded to the end of `i32-test-strength`, and
  both were pushed at the owner's request. Before the push, the five commits that `origin/main`
  had gained (CI workflow, licenses, CLI test fixture, run-timeout scale; the entry below) were
  read one by one and merged. One of them put the demo's address in the README; it holds the
  account name, which the local privacy audit refuses (4 hits, one line). The README now points
  to the website link in the repository page's About box instead, and the audit is clean.
- **Left:** nothing in the status table. Every increment is `done`.

### 2026-11-10 · Public-repository hardening (agent): CI, license, run-timeout scale

- **Added:** `.github/workflows/ci.yml`, two macOS jobs on every push: lint, types, tests and the web checks;
  and `dogma content verify --all`. The privacy audit stays local because its rules are never committed.
  `LICENSE` (MIT, code) and `LICENSE-CONTENT.md` (curriculum all rights reserved), with a License section in the
  README. The README's Requirements now state that Run and Submit need macOS. New setting
  `DOGMA_RUN_TIMEOUT_SCALE` (default 1, cannot go below 1) multiplies every run's time limit, in
  `runner.run.effective_timeout_s`; documented in `.env.example` and DEVELOPMENT.
- **Fixed:** two CLI help tests failed only on GitHub Actions, where Typer forces Rich's terminal mode and the
  help text carries ANSI styling. The shared `cli` fixture now uses a dumb terminal; the duplicate fixture in
  `test_cli_runs.py` is gone. Reproduce the old failure with `GITHUB_ACTIONS=true uv run --frozen pytest`.
- **Found on CI:** content verification failed twice on the shared runner, each time on one starter run ending
  with status `timeout` at 30 s, and on a different problem each time (`cml/shuffling-splitting-and-batching-data`
  p01, then `cml/ranking-metrics` p01). Both are property-heavy and pass locally in a few seconds. The content
  job sets the scale to 3. No problem's tests, tolerances, contract or `timeout_s` changed.
- **Changed:** `tools/docmd/LICENSE` names the dogma contributors, so the audit's attribution exemption table
  is empty.
- **Verified:** locally `make lint`, mypy strict (173 files), 4 import contracts, 1700 Python tests (also under
  `GITHUB_ACTIONS=true`), `gen:api --check`; `dogma content verify --all` on all 175 topics before the scale
  change, and on the two affected topics with the scale set to 3. `make audit` was not run: this checkout has no
  `private/denylist.txt`. CI result for this commit is on GitHub.
- **Left:** content verification takes about 47 minutes on the shared runner. Sharding it by track would cut
  that, at the cost of running the repository-level checks separately.

### 2026-11-10 · First 170 problems fully swept; hygiene and blind accepts done for all 251 · I-32b and I-32c done, I-32d in progress

- **Done for the 170 problems outside the 48 I-31 topics:** hygiene (I-32b), the reject sweep
  (I-32d) and the mutant triage (I-32d2). Their manifests now record 1,218 wrong solutions that the
  tests reject, 321 accepted ones and 101 Contract bullets marked uncovered with a reason. The
  mutant triage looked at about 418 surviving one-token mutants once: 268 cannot change a result,
  124 change something no Contract sentence covers, 2 have no usable witness, and 19 hidden tests
  were added in 13 problems.
- **Done for the 81 problems in the I-31 topics, after `main` was merged in (`8d744a0`):** a blind
  accept each (I-32c: 62 from the solutions written blind during I-31, submitted again against
  the current tests, and 19 from fresh blind solvers; all 81 passed on the first submission) and
  the hygiene fixes for the 15 problems that had lint findings. `worklist --kind blind` and
  `worklist --kind hygiene` are empty for all 251 problems.
- **Tag:** `i32-start` now sits on the merge `8d744a0`, so the version check of the I-31 problems
  starts from their I-31c versions. The old position (`dc33fc6`) is kept as `i32-start-phase1`
  for the close-out statistics.
- **Review by the coordinator:** every topic was gated per problem, verified and Ruff-checked
  before its commit, and every new hidden test was read against its Contract. In the mutant
  triage five executor tests did not reach a commit as written (a 4% kill margin; two tests built
  on an object that imitates a `str`; a non-tensor argument the Raises bullet does not name; a
  test that depended on the learner's free choice of start vector, replaced by one that does
  not). The reasons and the saved patches are in `docs/audits/test-strength-decisions.md`.
- **Verified:** `make check` exit 0 on `fba52e0` (the end of the first 170). `make test-all` exit 0
  on `8d744a0` in a clean worktree: Ruff and format, strict mypy (180 files), 6 import contracts,
  content types (251 problems), 2,192 Python tests, 175 topics verified, 187 browser tests and
  the build, audit clean (5,388 files), `exercise_gate.py variants` 1,539 ok and 0 mismatches in
  170 problems, 11 e2e/slow tests, 22 Playwright tests. `main` was fast-forwarded to `8d744a0`.
  Commits after `8d744a0` (blind accepts, hygiene and reject work in the I-31 topics) are gated
  per problem and not yet under a full gate.
- **Left:** the reject sweep and the mutant triage for the 81 problems in the I-31 topics (one
  topic is done), then I-32e (close-out), then the last merge to `main`.
- **For the owner:** `docs/audits/test-strength-decisions.md` has three entries from this session
  with a "Waiting for the owner" section each. The main ones: Contract sentences that are missing
  (minimums in the KV-cache problems, non-tensor arguments, defaults that live only in a
  signature), the kl-divergence blind accept that disagrees with the reference on an underflowed
  probability, and the sampling problem whose statement does not fix how the value is drawn.
  The branch `i32-test-strength` was pushed to `origin` at `dfc48f2` at the owner's request;
  `main` was not pushed.

### 2026-11-10 · I-32 merged to main as far as it is swept · I-32b, I-32c, I-32d in progress

- **Merged:** `i32-test-strength` at `b3f9739` plus `main` at `d11beb0` (the nine I-31c review
  commits), as merge `3145444`, then fast-forwarded onto `main`. One conflict, in this file: the
  status table keeps I-31c as `done` beside the I-32 rows, and the log keeps both sides' entries.
- **What that brings to `main`:** the I-32 platform (lint, `variants/`, `scripts/exercise_gate.py`,
  CONTENT_SPEC §6.7 and §6.8, ADR-0022) and, for the 170 problems outside the 48 I-31 topics: a
  blind accept each, the hygiene fixes, the reject sweep (1,199 wrong solutions and 321 accepted
  ones recorded; 97 problems gained hidden tests and one version bump each) and the first 13
  problems of mutant triage.
- **Two sessions:** the planning session built the platform, ran the first blind accepts and the
  pilots, and did this merge. The data-realism session picked the sweeps up from the roadmap and
  coordinates all of them on `i32-test-strength`; it writes its own entry.
- **Verified:** `make test-all` exit 0 on `3145444` in a clean worktree: Ruff and format (3,762
  files), strict mypy (180 files), 6 import contracts, content typing (251 packages), 2,192
  Python tests, all 175 topics verified, 187 browser unit tests, production build, audit clean
  (5,369 files), `exercise_gate.py variants` 1,520 ok and 0 mismatches in 170 problems, 11
  e2e/slow tests, 22 Playwright tests.
- **Left (from `worklist` on the merge):** hygiene 15 problems, blind 81, reject 81, all in the
  I-31 topics; mutant triage 238. The tag `i32-start` stays on `dc33fc6` until the first 170
  problems are fully swept; the sweeping session moves it when the I-31 topics join.
- **For the owner:** `docs/audits/test-strength-decisions.md` has a "Waiting for the owner"
  section. `main` is ahead of `origin/main` and nothing was pushed.

### 2026-11-10 · Test-strength pilots · I-32b and I-32d in progress

- **Pilots, on problems I-31 does not touch, Haiku executors:** hygiene on five problems (seven
  lint sites fixed, four in mode `text`, one `pytest.raises` replaced in mode `refactor` with 38
  mutants and none weakened); reject sweep on five topics, six problems.
- **Reject sweep result:** 41 wrong solutions recorded and four alternative accepts, none
  refused. Six wrong solutions passed every existing test, in four of the six problems, and each
  got one new hidden test through `add-test`: `cml/vector-operations/p01` (rescales `b` in
  place), `algo/lru-cache/p01` (returns `None` for a stored falsy value; refuses a list value),
  `dl/layer-normalization/p01` (squeezes a one-feature input; `view` on a non-contiguous input),
  `inf/kv-cache-size-arithmetic/p02` (accepts upper-case dtype names). Those four problems went
  from version 2 to 3. The coordinator read each new test against its Contract.
- **Runbook changes from the pilots:** executors may read their problem's harvest folder; a
  wrong SQL query that fails every test on the shared type check is recorded as `uncovered`;
  `K` bullets need no entry; named cases keep their meaning as ids.
- **Verified:** `exercise_gate.py gate` passed for all eleven pilot problems in the mode of
  their edit; `dogma content verify` on the ten topics; `exercise_gate.py variants`: 219 ok, 0
  mismatches. `make check` was not re-run after the pilots.
- **Left, from `worklist`:** hygiene 70 problems (209 sites), blind 81, reject 245, mutants 251.

### 2026-11-10 · Test-strength platform and first blind accepts · I-32a done, I-32c in progress

- **Where:** branch `i32-test-strength` (worktree `.claude/worktrees/i32`), 18 commits on top of
  `5b6e734`. Not merged to `main`. One coordinating session, Haiku agents as executors.
- **I-32a (platform):** `dogma.content.testlint` (four rules for hidden tests and an id-leak
  check); the `variants/manifest.yaml` schema, loader fields and `VARIANT_*` validation;
  `scripts/exercise_gate.py` (`lint`, `clauses`, `probe`, `note`, `ledger`, `add-accept`,
  `add-reject`, `add-uncovered`, `remove-variant`, `adopt-blind`, `variants`, `gate`,
  `bump-version`, `add-test`, `mark-swept`, `worklist`, `scorecard`); CONTENT_SPEC §6.4, §6.7 and
  §6.8; ADR-0022; `make verify-variants` inside `test-all`; variants for the example problem
  (three rejects, one alternative accept).
- **I-32c (blind accepts):** 170 of 251 problems have `variants/accept_blind`, written by agents
  that saw only the statement, the starter and the public tests. 168 passed on the first
  submission, 2 on the second, none failed. The other 81 problems are in the 48 topics I-31
  touched and wait for I-31c.
- **Departure from the plan:** blind solving and adoption for problems outside the I-31 topics
  started before I-31c was done. I-31c owns only its own topic directories, so it cannot change
  these problems.
- **Runbook and tool defects found by running them, all fixed on the branch:** the protocol asked
  early runbooks for a ledger command that did not exist yet; two test files with one basename
  collided at collection; `ruff format` also checks Python blocks in Markdown; a test literal
  read as an e-mail address to the privacy audit; installing a variant refused 37 working blind
  solutions on style rules (style findings are now marked with `noqa`, real errors still
  refuse); the blind-similarity check counted lines the starter already has.
- **Trial of `add-test`:** on `cml/vector-operations/p01-vector-projection`, with a solution
  that rescales `b` in place (it passes all 17 current tests). The gate passed, the version went
  2 to 3 and the reject was recorded against "Must not mutate its inputs." The trial was then
  undone: the gap is left for I-32d so that the problem's version is bumped once.
- **Verified:** `make check` exit 0 on `dc33fc6`: Ruff and format (2,435 files), strict mypy (180
  files), 6 import contracts, content typing (251 packages), 2,184 Python tests, all 175 topics
  verified, 187 browser unit tests, production build, audit clean (4,022 files).
  `exercise_gate.py variants`: 174 ok, 0 mismatches in 170 problems. `exercise_gate.py gate
  --mode none`: 170 of 170 adopted problems pass (run before the example's four extra variants;
  the example was gated again after them). `make test-all` was not run.
- **Tag:** `i32-start` is on `dc33fc6`. Version checks count from it. When the I-31 topics join
  the sweeps (after I-31c is merged into this branch), move the tag to that merge, and only
  after the sweeps for the first 170 problems are complete.
- **Left:** I-32b (216 lint sites in 75 problems), I-32c for the 81 held problems, I-32d, I-32e,
  and the merge to `main`. Notes so far are in `docs/audits/test-strength-decisions.md`.
### 2026-11-10 · Data realism review and follow-ups · I-31c done

- **Follow-ups (owner decisions):** rolling-statistics p01's property check now uses the
  running-update bound (1e-6 times scale; the reference failed 7 of 20 seeds, flagged in I-28);
  the subquery contracts state the exact 0.25-multiple amount domain, with a lesson note on cents;
  every runtime now sets what a submission may import (ADR-0021, rule `runtime-import`; 178 of 231
  Python problems could import libraries their statements rule out); ridge p01 returned to its
  pre-I-31 tests (its longley killer was only 2x its tolerance and rejected a correct eigh solve).
- **Review:** seven second agents (Sonnet for linear models and PCA) reviewed all 47 I-31 topics.
  They corrected lesson numbers that did not reproduce (Nile window mean, conformal coverage 90.0%
  and 80.0%, the cents claim, permutation-importance drops, a dirty-label example, cohort and lag
  shares), statement errors (a quoted line not in the book, an example exposing hidden inputs, the
  ulp limit of the tie rule, `?` handling in adult), and two test issues: PCA p01's data span
  (a correct eigh solution failed 7 of 12 draws; now 0.03 to 30) and pandas-groupby p01's HEAD
  test, which had lost its DuckDB oracle. All 47 topics are `reviewed`.
- **Audit (fresh, 44 problems, against I-28):** 44/44 references pass; raw kill rate higher on 21,
  equal on 9, lower on 0 (the 14 SQL problems have no I-28 row); `seeds` 880 runs, 0 failing. A
  per-mutant comparison at 9ab85f7 confirmed no mutant the old tests killed survives the new ones.
- **Verified:** `make test-all` stages on 21fa147: Ruff and format, strict mypy (179 files), 6 import contracts, content types (251), 2015 unit and integration tests, 175 topics verified, web check (187 tests) and build, privacy audit clean, 11 e2e/slow tests, 22 Playwright tests.
- **Open (for I-32):** logistic p01's rare-separable test kills full Newton without line search on
  about 25% of data draws (fair under the Contract; the grader's draw is one of them); the normal
  equation's longley test rejects an eigh-of-Gram route (Gram cond 2.4e19); PCA p01's eigh route
  still fails about 1 in 60 salted draws (the grader's draw passes); the word tokenizer and
  majority vote miss non-ASCII whitespace and casefold variants (pre-I-31); `synth.event_log` has
  whole-second timestamps; `_random_*(rng: ...)` helpers shadow the imported `rng`; the static
  rules still miss `sys.modules` and dynamic imports.

### 2026-11-09 · Data realism content sweep · I-31b done

- **Result:** 42 problems in 47 topics changed, each with one version bump (re-solve material),
  plus 43 lesson paragraphs with re-measured real-data numbers. No reference solution changed.
  Every new test kills a hand-written realistic wrong solution that passed the old suite
  (`exercise_audit realism`), sits next to the existing tests (none dropped: checked against
  `9ab85f7`), survives `exercise_audit seeds` with the data salt, and runs under 10 s with a
  blind solution.
- **What real data caught:** overflow on separable real features and ill-conditioned designs
  (logistic, normal equation, ridge); tie grouping by tolerance (ROC, AP, F1, NDCG); zero-IQR
  pixels that vary (robust scaling); rare and unseen categories (pipelines, target encoding);
  integer and float32 storage (ACF, rolling, bootstrap, Welch, ANOVA); curly quotes, case and
  punctuation in real prose (TF-IDF, BPE, tokenizers, chunking); thousands separators (gsm8k
  votes); narrow integer counts; dirty SQL keys, duplicate log rows, milliseconds, multi-year
  and leap-year windows, `FLOAT` cent totals.
- **Unchanged with evidence:** SVD, covariance, GMM, k-means, kNN, BM25, latency, quantization,
  drift, the DL loops, class imbalance, tree splits, permutation importance, subqueries, top-N,
  running totals, day-over-day change: every realistic mistake built for them was already caught.
  Several of those topics got lesson paragraphs only.
- **Process:** a session restart lost the coordinator's own uncommitted drafts for about 35
  problems; six reviewers held each to the kill gate (14 reverted, with the learner's approval;
  the drafts are kept in `var/audit-i31/uncommitted-2026-11-09-2300.patch`). Reviewers restored
  randomized HEAD tests that drafts had replaced.
- **Verified:** `make test-all` on `a91f293` in a clean worktree: Ruff and format, strict mypy
  (179 files), 6 import contracts, content types (251), 1996 unit tests, 175 topics verified,
  web check (187 tests) and build, privacy audit clean, 11 e2e/slow tests, 22 Playwright tests.
- **Open (for I-31c or I-32):** rolling p01's reference fails some Hypothesis seeds at HEAD on
  constant windows (held fix in `var/audit-i31/S4b/held/`); in the subquery problems, DuckDB's
  average of identical cent amounts differs from the value in about 9% of cases, so a strict
  `amount > AVG` is float-sensitive and the contract is silent; S3's inverse/determinant GMM
  density and SVD p02's normal-equation route can't be separated from the reference at a
  tolerance the conditioning allows; PSI and absmax Constraints say stdlib and NumPy, but their
  `banned_imports` don't block sklearn or statsmodels; `synth.event_log` has whole-second
  timestamps only (lag p02 builds millisecond clocks inline); confusion-matrix p01's uint8
  killer is a weak realism case. I-31c review is next.

### 2026-11-09 · Data realism · I-31a done, I-31b in progress

- **Why:** bulk test data was iid same-scale normals, labels independent of scores, white-noise
  series and toy text, so some wrong solutions passed and some lessons taught the wrong intuition.
  Plan `docs/superpowers/plans/2026-11-09-data-realism.md`, ADR-0020.
- **Added (I-31a):** test-only `src/dogma_data` (25 offline datasets with pinned shapes, hashes
  and quirks; frozen pure-numpy generators; `rng(key)` with the `DOGMA_DATA_SALT` audit salt);
  `datasets` in the opt-in `hf` group and `scripts/sync_hf_datasets.py` (gsm8k, adult,
  california_housing, three Gutenberg slices, pinned revisions, licence policy, `--check`);
  platform-wide submission ban on `dogma_data`/`datasets`/`huggingface_hub`; test-import allowlist
  (`TEST_IMPORT_NOT_ALLOWED`); `exercise_audit realism` and the per-seed data salt; a Seatbelt
  integration test; CONTENT_SPEC §6.6, `docs/DATASETS.md`.
- **Spike:** pyarrow switches pandas 3.1 string storage; with it the content verify pass set is
  unchanged but slower (upper bound 1348 s vs 774 s under load), so `datasets` stays opt-in.
- **Coordinator fixes:** `peaked_probs` was degenerate (top probability 0.9995 on every row):
  replaced by a margin model; feature names are plain `str`; HF snapshots load through
  `load_frame`/`load_text`; Gutenberg slices start at each book's first sentence.
- **Content (I-31b) so far:** 19 topics committed; each new test kills a hand-written realistic
  wrong solution that passes the old tests (`exercise_audit realism`), survives `seeds` with the
  salt, and runs under 10 s with a blind solution. Many slices found their old suites already
  catch every realistic mistake tried (SVD, covariance, GMM, k-means, kNN, BM25, BPE, the
  serving and DL problems), and those stay unchanged. Lasso, elastic net and regression metrics
  got lesson paragraphs only (their new tests killed nothing).
- **Verified:** `make test-all` on `9ab85f7` in a clean worktree: Ruff and format, strict mypy,
  6 import contracts, content types (251), 1996 unit tests, 175 topics verified, web check and
  build, privacy audit clean, e2e/slow pytest, 22 Playwright tests (the web stage re-run after a
  real offline `pnpm install`; a symlinked `node_modules` is refused by pnpm). Each content
  commit: `dogma content verify` on its topics, `check_commit_files`, staged privacy audit.
- **Open:** adoption review of uncommitted realism edits made before a session restart (about 35
  problems across serving/DL, geometry, text, counts, trees and SQL), each held to the kill gate;
  the rest of the SQL slice (S7); the rolling-statistics p01 reference fails some Hypothesis
  seeds at HEAD on constant windows (held fix in `var/audit-i31/S4b/held/`, a test-strictness
  question for I-32); I-31c review.

### 2026-11-03 · Time-warp API and web · I-30b done

- **Added:** `GET /calendar`, `POST /calendar/warp` (dry run or apply) and `POST /course/reset` ({confirm:
  "reset"}), as thin services over `dogma.learning.calendar`. `GET /progress` reads the shifted calendar. The
  Progress page has a Calendar section (this week, the furthest exercise's week, match / go to week / original
  dates, each previewed with the before → after lines) and a Start over section with a typed `reset` dialog that
  shows the backup and snapshot paths. Today shows one neutral line linking there when the calendar is at least
  two weeks past the furthest exercise.
- **Coordinator fixes:** with no exercise run, the Today line said "Your furthest exercise is in week 1"; it now
  says "You haven't run an exercise yet", with a test for both cases. The preview's before → after values did
  not line up at desktop width (letter spacing outgrew the label's `min-width`); it is now a two-column grid that
  stacks below 30rem. Checked by eye at 1280px and 390px with no horizontal scroll.
- **Verified:** `make test-all` exit 0 on `21dfb35`: Ruff, strict mypy (173 files), 4 import contracts, 1695
  Python tests, all 175 topics, 176 browser unit tests, production build, audit clean (3583 files), 11 e2e/slow
  tests, 16 Playwright tests (including the new calendar spec: preview and move, week shown, no scroll at 390px).
- **Learner data:** still at `b8c9d0e1f2a3`. The first `dogma` command after this branch is merged upgrades
  `var/dogma.db` to `a9b0c1d2e3f4`; branches without the migration cannot open it afterwards.

### 2026-11-03 · Time-warp backend and CLI · I-30a done

- **Why:** the learner had not started week 1 when the calendar reached week 2, and asked for a way to pick
  the course up after a break without the calendar running ahead. Plan `2026-11-03-time-warp.md`, ADR-0019.
- **Added:** `calendar_warp` (append-only log; the newest row's `offset_weeks` is in force) and migration
  `a9b0c1d2e3f4` on top of `b8c9d0e1f2a3` (the plan named the wrong parent; fixed there). `dogma.learning.calendar`
  shifts the authored path by whole weeks. The planner, `dogma path` and `dogma progress` read the shifted path.
  `dogma warp` syncs this week to the furthest core-path week with a recorded attempt (run or submit; week 1 with
  none), or takes `--to-week`, `--ahead`, `--back`, `--original`, with `--dry-run` and `--yes`. `dogma course
  reset` backs up (JSON and DB snapshot), clears answer, mock_session, session_block, study_session,
  resolve_window, card_review, card, topic_state and attempt in one transaction, keeps the warp log, sets
  workspace files aside under `var/workspace-history/reset-<stamp>/`, and makes this week week 1. Card and
  re-solve due dates never move.
- **Coordinator fixes:** `--ahead`/`--back` help said the opposite of what they do; `db/backup.py` now imports
  `calendar_models`; the two new error codes have API statuses (409, 422).
- **Test fix (separate commit):** `test_default_week_is_the_current_one` assumed the course had not started and
  failed from Mon 2 Nov. It now pins the date and adds a week-2 case.
- **Verified:** `make check` exit 0 on `b135a35`: Ruff, strict mypy (170 files), 4 import contracts, 1680 Python
  tests, all 175 topics, 166 browser unit tests, production build, audit clean (3570 files). CLI smoke test in a
  scratch `var/`: sync preview and apply, `dogma path` note, `--ahead 2` preview, conflicting flags (exit 2),
  reset cancelled by a wrong word and completed by `reset`.
- **Learner data:** not migrated yet. The first command on this branch upgrades `var/dogma.db` to `a9b0c1d2e3f4`,
  which older branches cannot open.
- **Open:** I-30b (API and web), closed in the entry above.

### 2026-11-01 · Introduction · I-29 done

- **Added:** `/welcome` (what dogma is, the session shape, where things live, a tutorial on the first
  problem, a glossary) and a guide strip on that problem, shown only by `?tour=1`. Frontend only.
- **Behaviour:** `/` opens `/welcome` once per browser, then `/today`. Deep links are never redirected.
  The seen flag is in browser storage, and blocked storage falls back to `/today`. Start here stays in
  the nav. Existing learners see the page once.
- **Also fixed:** the Mock nav item never had an icon, hidden by an `as IconName` cast. A test now checks
  that every nav item draws one.
- **Reviewed by eye:** screenshots at 1280px and 390px showed orphaned cards, misaligned headings and a
  guide strip that pushed the editor off screen. All tests had passed. The strip is now one row of step
  titles, and the layouts use container queries.
- **Verified:** `make test-all` exit 0 on `cc41371`: Ruff and format, strict mypy (166 files), 4 import
  contracts, 1630 Python tests, all 175 topics, 166 browser unit tests, production build, audit clean
  (3561 files), 11 e2e/slow tests, 14 Playwright tests (8 new, including 320px and 390px with no
  horizontal scroll, every card link, a deep link and keyboard use).
- **Dropped from this sprint by decision:** a streak and activity calendar. The no-streaks rule in
  ARCHITECTURE §5 stands. A "how it works" page on the teaching science and curriculum, and everyday
  exercises, were deferred.
- **Open:** the problem page has two `h1` headings, one in its header and one in the statement. This
  predates I-29 and was left alone.

### 2026-10-31 · Exercise audit · I-28 done

- **Report:** [`docs/audits/2026-10-31-exercise-audit.md`](audits/2026-10-31-exercise-audit.md), with
  a generated scorecard for all 251 problems. Tool: `scripts/exercise_audit.py`.
- **Measured:** 10,551 mutants, 88.1% killed raw. Blind solves: 251/251, 235 first try, no forbidden
  reads across 1,115 tool calls. Branch coverage median 100%, lowest 82.1%. 5,020 seed runs, 2 problems
  fail on some seed, neither at the pinned seed. 96% of hidden-only kills name a mistake.
- **Fixed:** 15 statements now state numeric limits their tests already enforced (no version bump).
  544 of 549 real gaps closed by 425 new hidden tests in 144 problems, each version bumped. Two
  misdirecting messages, checked as text-only by an AST comparison. One grader crash on a source that
  overflows the parser (`MemoryError` escaped `run()`), with 35 probes pinning the verdict behaviour.
- **Gate on new tests:** each had to pass the reference, pass an independent blind solution and fail
  its mutant, all through the real grader. Final content: 251/251 references, 251/251 blind solutions,
  544/549 gap mutants dead. Five exact-float-tie tests were dropped on purpose. `test_hidden.py` diff
  across 144 files: +4,302, -0. Mutation score excluding equivalents went from 93.2% to 98.6%.
- **Verified:** `make test-all` exit 0 on `db16852`: Ruff and format, strict mypy (166 files), 4 import
  contracts, 1630 Python tests, all 175 topics through the Seatbelt runner, content typing for 251
  packages, 148 browser unit tests, production build, audit clean (3544 files), 11 e2e/slow tests, 6
  Playwright tests.
- **Corrected:** the previous entry's matrix-rank flake note was wrong (see the fix in that entry).
- **Open, all in the report:** `rolling-mean-std` has a tolerance at the rounding floor (10 of 20 other
  seeds fail; changing it means loosening a hidden tolerance, so it needs a decision); two
  `lagrange-multipliers` instances of 5,001 fail, neither at seed 0; `meta.yaml` tolerances are never
  read by grading; `kv-cache-size-arithmetic` and `training-an-mlp-from-scratch` statements leave
  minimums or bounds unstated. New problems should state extreme-value requirements in the contract.

### 2026-10-31 · macOS gate, numerical review and PDF import · I-25a–g done

- **Environment:** macOS with Accelerate, the first full gate after the Linux session below. Local
  `main` was 26 commits behind and was fast-forwarded to `dc906aa`.
- **Two regressions the Linux session could not see, both from I-25f adding cases:**
  `test_a_case_covering_the_weakest_recent_part_is_preferred` assumed three cases (`635ec7d`; its
  assertion is unchanged, and a planner that ignores coverage now fails it), and the Playwright
  scratch repo lacked the `library:` notes two new cases cite, so all six browser tests failed on
  a catalog 500 (`f4ac3de`). Product code was correct in both.
- **I-25b numerical review finished (`47cb6d3`):** all four topics are `reviewed`. Beyond the
  grader and `content verify`, fresh solutions written from the statements passed p02 of both
  changed problems, and five mutants (`min(m, n)` margin, `max|A|` scale, strict `<`, coarse default
  cutoff, Frobenius cutoff) were rejected. Every lesson number was recomputed. Several were one
  BLAS's rounding, so they are now ranges or per platform: covariance error 3% to 6% (not 6.1%),
  ill-conditioned `solve` differs by platform, rank sweep about 1.5% (84 or 89).
- **I-25g (`ab3f9aa`, `7a56d25`):** all 72 PDFs matched the inventory. The lock gained 72 entries
  and changed none, and a second run copied nothing. `dogma corpus sync` converted 68; 4 scans
  failed and 5 more converted only their text cover pages, about 100 bytes per page, reported as
  success. Those 9 were re-extracted with `--ocr auto` (Apple Vision, on device). The 5 cover-only
  files were moved aside for one sync so that they counted as new; docmd skips an unchanged hash.
  The same run also recovered the 6 older `library-papers` scans. Index: 124 `library-papers` and
  72 `library-refresh` sources. Retrieval confirmed for LoRA, DPO, conformal, normal equations and
  linear programming.
- **Citations read in the PDFs:** stale "not in the corpus", "from memory" and "unchecked" notes
  are gone for seven papers and two PDFs wrongly called absent. Four claims changed: the `f·P` loss
  is Eq. 4 of Switch Transformer, not Shazeer 2017 (`moe-top-k-routing`); the stability boundary is
  the Library note's synthesis, not Goldberg's (`matrix-inverse-and-rank`); strict convexity's
  "at most one point" is Boyd and Vandenberghe §4.2.1, p.137, not §3.1 (`ridge`); ST-MoE's fp32 and
  bf16 passage is p.8, not p.9 (`mixed-precision`). Confirmed unchanged: the other ST-MoE and Sparse
  Transformer pages, XGBoost Eqs. 5–7, LoRA §4.1 and §7, DPO Eqs. 3–5 and 7, Friedman §§2–3,
  Penrose Theorem 1, Golub and Reinsch §1.1, Halko Theorem 10.5, Bottou §4.2 and Neyman and
  Pearson. Corpus `p.` values are PDF pages, not printed pages. Papers still absent keep their
  notes (Freund and Schapire, Bradley and Terry, Micikevicius, GRPO, CPC, speculative decoding).
- **Verified:** `make test-all` exit 0 on `f4ac3de`: Ruff and format, strict mypy (166 files), 4
  import contracts, 1543 Python tests, all 175 topics through the Seatbelt runner, 148 browser unit
  tests, production build, audit clean (3541 files), 11 e2e/slow tests, 6 Playwright tests.
- **Open:**
  - `p02-matrix-rank`: the default tolerance's rare miss can reach three Hypothesis properties. The
    reference failed them in 5 of 900,000 random draws, but this is not a Submit risk: the harness
    pins `--hypothesis-seed=0`, so Submit runs the same examples every time, and they pass.
  - The citation edits were checked against the PDFs by this session but had no separate second
    reviewer. The topics stay `reviewed`.
  - "(my reproduction)" numbers outside these four topics were not re-run on a second platform.
  - Seven older `library-papers` conversions are still cover-only, predate this work and are
    unused by content: `fisher-theoretical-statistics-1922` (and its two parts),
    `euler-calculus-of-variations-1744`, `grassmann-ausdehnungslehre-1844`,
    `jacobi-eigenvalue-algorithm-1846`, `riemann-trigonometric-series-integration-1854`.
  - `corpus sync` reports any docmd note as a failure, so the one-time "source path changed"
    notice after the rebrand printed as `CORPUS_CONVERSION_FAILED`. It cleared itself.
  - Four topics remain `verified` and were not in this work: `algo/stacks-and-queues` and the
    `sys` churn, fraud and RASS cases.

### 2026-10-31 · Library refresh executed · I-25a–f done, I-25g blocked (cloud session, Linux)

- **Environment:** a Linux container. The runner's macOS Seatbelt sandbox is unavailable, so
  `dogma content verify` and `make verify-content` refuse every problem with `RUNNER_CRASHED`
  (expected). Problems were verified with `scripts/grade.py verify` (same pytest harness, no OS
  sandbox) and mutants ran through `scripts/grade.py submit --submission`. The Library git clone
  has no PDFs (gitignored upstream), so only the 36 notes were imported. **Run the full
  `make check` on macOS before trusting the gate**; it was not green here for environmental
  reasons only (see below).
- **I-25a (commits `0bdcdaa`, `fb52f2e`):** importer gains `--source-id`, `--kind`, corpus
  `include:` lists and `--inventory` checksum validation (15 tests). `library-refresh` snapshot
  and corpus entries added; 36 notes imported, scrubbed (vault links, unimported relative links,
  acquisition narration) and locked (+36 entries, 549 old entries unchanged). `dogma corpus sync
  --no-convert` indexed them as `sources:library-refresh/<path>`. IMPORT_PLAN §6 documents it.
- **I-25f (`a21926e`, `da3924e`, `f2018da`):** four original synthetic cases authored and
  independently reviewed (numbers recomputed with scipy LP/MILP; precision column, pre-revealed
  twist, degeneracy wording fixed). All four `reviewed`.
- **I-25e (`91245c0`, `15af386`, `5430221`):** Lagrange multipliers, GMM/EM, document packing
  masks, Sarsa/expected Sarsa authored with one core problem each (20–23 mutants per problem),
  then independently reviewed (reviewer solved each from the statement; sign-of-multiplier claim,
  unreproducible EM figures and an ambiguous contract fixed). All four `reviewed`; problems v2.
- **I-25b/c/d revisions (one commit per batch, then one review commit per batch):** 29 existing
  topics revised against the Library notes (cited as `source: library-refresh/<path>`, locators
  relayed "per the Library note"; no PDF was inspected, and every paper citation nobody inspected
  stays qualified). Hidden tests were only ever added (lasso p01 v3, Bayes p01 v4, permutation
  p02 v3, Monte Carlo p01 v3, gauss-jordan p01 v3, matrix-rank p02 v3, normal-equation p01 v3,
  pseudo-inverse p02 v2); reviewers confirmed additions only. Notable corrections: reward-model
  scale vs offset identification, MAP boundary rule, AdaBoost weight growth, the linear link's
  derivative is X not X^T, loss-scaler diagnostics, conformal recency claim, bootstrap on AR(1).
  25 of 29 topics are `reviewed`.
- **Stopped mid-way:** the second reviewer of the I-25b numerical batch (`cml/matrix-inverse-and-rank`,
  `cml/normal-equation`, `cml/singular-value-decomposition`, `cml/principal-component-analysis`)
  reported "everything passes so far" and was stopped before its final verification pass and
  status change. Its edits are committed (`19dcc5e`); the four topics stay `verified`.
  **To continue:** re-run `grade.py verify` on their 7 problems, `dogma content verify` on the
  four ids, re-derive the lesson numbers if desired, then set `status: reviewed`.
- **Web:** flaky `ProblemPage` client-route test now awaits the editor (`aa1ae31`); the gen-api
  artifact test has a 30 s timeout (`2832988`). `pnpm build` passed; `pnpm check` passed except
  the gen-api timeout before that fix (not re-run after it).
- **Verified here (Linux):** `make lint types` exit 0 (166 files, 4 import contracts, 247 then
  251 typed packages); `make test` baseline 1487 passed / 33 failed, all 33 environmental
  (Seatbelt absent, root can read chmod-0 dirs); `make audit` clean after every commit (with a
  local placeholder `private/denylist.txt`); `grade.py verify` on every changed problem; `ruff`
  on every changed dir; `dogma content verify` validation issues 0 on every touched topic. The
  full `make check` and `make test-all` were **not** run end-to-end after the final commit.
- **Next steps:** (1) on macOS run `make check` and `make test-all`; (2) finish the numerical
  batch review above; (3) I-25g: run
  `uv run --frozen python scripts/import_sources.py --source-id library-refresh --kind corpus --inventory docs/source-refresh-2026-10-28.json`
  on a machine with the Library PDFs, `dogma corpus sync`, then revisit the `corpus:` citations
  marked unchecked; (4) `web/scripts/gen-api.test.mjs` and the Linux-only pytest failures are
  worth a Linux runner story if the app should run outside macOS.

### 2026-10-31 · Rebrand from deepml to dogma

- Moved the checkout, its history, ignored data and agent worktrees to `dogma/` and added the
  GitHub remote. A one-shot script rewrote tracked text (2065 replacements in 332 files) and
  renamed the `dogma` and `dogma_harness` packages, so the CLI, `DogmaError` and the `DOGMA_`
  environment prefix all changed. `uv.lock` keeps the same 104 packages and versions.
- Local data moved with it: `var/dogma.db` and 2977 `.dogma-completed.json` run markers. Learner
  rows were not changed, so the session planned on 30 Oct still shows the old command name in
  its stored text. It clears when that session closes.
- **Kept on purpose:** `deep-ml` and `origin: deepml` (provenance of the third-party outline),
  two historical gate log paths below, and `sources/` (imported files pinned by SHA-256).
  Existing `var/backups/` files keep their old names and format label; nothing reads them back.
- **Verification:** `make check` exit 0: Ruff/format, strict mypy (166 files), four import
  contracts, 1528 Python tests, all 167 topics verified, 148 browser unit tests, production
  build and privacy audit (3421 files). `pytest -m "(e2e or slow) and not llm" -q` 11 passed;
  `pnpm test:e2e` 6 passed. `dogma runs prune --older-than-days 1` (dry run) lists migrated runs.
  `dogma today` ran on a copy of the database; the real file was byte-identical afterwards.
- **Open:** `ProblemPage.test.tsx` "loads a client route with Run disabled…" is flaky and fails
  `make check` intermittently. It asserts the editor synchronously after waiting for the Run
  button. It also failed 1 of 6 full suite runs on the pre-rebrand `4267faf`, so the rebrand did
  not cause it. The test was left unchanged. A fresh clone lacks the gitignored
  `private/denylist.txt`, which `make audit` requires.

### 2026-10-29 · Follow-up review, regression fixes and commits · I-26/I-27

- Reviewed the pending implementation and publication changes from `7bee9ae`. Committed privacy
  and planning documentation as `f41e2e5`, structured timed sessions as `a46f622`, SQLite backup
  protection as `6b4bf2c`, phone mock scoring as `fe4b297`, opt-in model smoke tests as `b6e045f`,
  and exercise feedback as the separate content-only commit `ba387d8`.
- Independent code review found that ignored build symlinks could bypass the privacy audit.
  Reproduced four failing fixtures covering the build root, nested directory, file and dangling
  link. Explicit build discovery now collects and rejects those links without reading their
  targets. All four fixtures pass; no material review findings remain.
- Verification exposed a hint unit test that could contact the live model. A live-call guard
  reproduced the problem; the test now uses the fake provider and checks a successful hint
  without creating a resolve window. All 21 resolve-route tests pass.
- **Verification:** `UV_CACHE_DIR=/private/tmp/dogma-uv-cache UV_NO_SYNC=1 make check`
  exit 0 on the corrected tree: Ruff/format, strict mypy (166 files), four import contracts,
  content typing (247 packages), 1528 Python tests, all 167 topics verified under Seatbelt,
  148 browser unit tests, generated API freshness, production build and post-build privacy
  audit (3421 files). Full log: `/private/tmp/deepml-check-20261029-final.log`.
  Additional checks passed: 80 script tests, 5 affected vendored IR tests, 69 database/planner/
  session tests, `pytest -m '(e2e or slow) and not llm' -q` (11 passed), and
  `pnpm --dir web test:e2e` (6 passed, including 320/390px mock scoring).
  Targeted content verification passed for all three changed exercises, including solution
  repeatability and starter rejection. Four faulty-submission probes were rejected with
  specific feedback. Hidden grading tests are unchanged.
- **Scope:** I-26 and I-27 are committed; I-25a–f remain planned. The committed inventory has
  108 unique relative paths, including 72 PDFs and 36 notes with SHA-256 values. This session
  did not import external sources, alter real learner data or rerun the opt-in live-model suite;
  its four tests were collected successfully. Git history and existing publication archives
  were not rewritten or regenerated.

### 2026-10-29 · Context-independent product and publication privacy · I-27 done

- Replaced the individual learner profile with general learning-design requirements, retaining
  its path for existing links. Generalized architecture, README, import notes and source-withdrawal
  reasons; removed individual background, behavioral descriptions and learner-state incident
  details from public handoffs. Course settings remain product configuration.
- Reference-lesson import now allows only the four reviewed technical notes. Excluded narrative
  filenames are not recorded in the manifest or import docs. Checked authored source code,
  tests, configuration and public text for the requested context references: none remain.
- Extended local-only private filters; their values are not committed or copied to exports.
  The audit now covers every tracked/unignored publication file, metadata, filenames and browser
  assets. A history-free export is scanned across all public folders, including vendored code.
  Common escaped text and UTF-16 are supported; arbitrary binary analysis is not claimed.
- Closed privacy-tool gaps with failing regression fixtures: metadata/filename omissions,
  private values in diagnostic labels, missing rules, oversized files, symlink publication,
  staged blobs differing from disk, force-staged private files later removed from disk, nested
  public folders and escaped account paths. `--staged` checks actual index blobs; the commit hook
  uses it. It rejects private index paths without reading their contents. `make check` audits
  the rebuilt product tree after the browser build, including under parallel make.
- Neutralized vendored converter package author metadata and illustrative authors in two IR
  tests plus one golden output. The extractor behavior is unchanged. Updated the four scrubbed
  lock entries and VENDORED notes, preserving origin hashes. Required upstream MIT/OFL
  attribution is retained with checksum-pinned exceptions. Two exact opaque parser tables have
  reviewed checksum exemptions for coincidental short letter sequences; changed data and
  surrounding code remain checked.
- Independent review found the staged/index, symlink, export-discovery and escaping cases above;
  all findings were corrected and reproduced in regression tests. No source lessons, hidden
  grading tests, learner data, external source originals or Git history were modified by I-27.
- **Verification:** `UV_CACHE_DIR=/private/tmp/dogma-uv-cache UV_NO_SYNC=1 make check` exit 0:
  strict typing/import contracts, 247 typed problem packages, **all 167 topics verified**,
  **148 browser unit tests**, generated API freshness, production build and the final post-build
  privacy audit (**3421 text files**, including canonically equivalent filesystem-path aliases).
  Final standalone default Python suite after review fixes: **1524 passed**, 15 deselected;
  script suites: **76 passed**; affected vendored IR tests: **5 passed**. Ruff and diff checks pass.
- **Publication artifact:** `dist/dogma-public-source.zip`, generated from the cleaned current
  files and browser build, independently audited without Git, with a per-file checksum manifest.
  The archive excludes Git history, local identity rules, learner state, dependencies and agent
  artifacts; timestamps and permissions are normalized and filesystem ownership is omitted.
  Existing Git history retains earlier versions and has not been rewritten: use the clean
  snapshot for history-free sharing. All changes remain local and uncommitted.

### 2026-10-28 · Library audit and implementation follow-ups · I-26 done, I-25a–f planned

- **Source audit:** read the updated Library without changing it. Independently verified all
  206 cataloged PDF checksums; all 124 originally imported PDF hashes remain present. The two
  acquisition batches add 71 PDFs and 35 notes. The refresh selects those plus the earlier
  Goldberg PDF/note: **72 PDFs + 36 notes**, frozen by checksum in
  `docs/source-refresh-2026-10-28.json`. Ten other unmatched historical/alternate PDFs remain
  outside this selection. Upstream working-tree changes mean a Git commit alone is insufficient
  provenance. Existing manifest globs miss the new directory layout.
- **Plan:** source-to-topic map and reviewed design in
  `docs/superpowers/specs/2026-10-28-library-refresh-design.md`; six increments in
  `docs/superpowers/plans/2026-10-28-library-refresh.md`. Prioritize improvements to existing
  numerical, statistics, autodiff, optimization, precision and LoRA/alignment lessons; then
  selected existing tier-B packages and the four remaining `sys` cases. LP/evolution stays
  optional. Core path and tier-A set unchanged. **No refresh sources imported or new content
  authored this session.** Sutton–Barto, PagedAttention and Bradley–Terry primary sources are
  still absent; LoRA/DPO can be verified after import and direct section review.
- **Backup safety fixed:** `dogma backup` now atomically snapshots an existing SQLite database
  (including committed WAL rows) before `open_session` can migrate it, then exports current
  tables as JSON. Failure to save the snapshot stops before migration. A live-WAL regression
  and a failed-write regression pin both behaviors. Backup artifacts remain local under
  `var/backups/`; individual learner-state and backup details are excluded from this handoff.
- **Structured target:** migration `b8c9d0e1f2a3` adds nullable `study_session.timed_problem_id`
  and backfills the two known legacy action templates. New plans persist the id; Today reads
  the actual open-window state, without parsing prose. Tests cover migration/data preservation,
  altered instructions, resume and low-energy replanning clearing the target.
- **Phone and feedback fixes:** the mock question input caused horizontal overflow at 320px;
  its width now shrinks, and score labels provide at least 44px tap targets. New 320/390px
  browser flows select all 30 scores and reach the debrief. Public tests now name grid bounds,
  GRPO input mutation and SQL event selection mistakes; the SQL alphabetical-minimum mutant
  previously passed the public tests and now fails a concrete counterexample. Four incorrect
  submission probes fail with useful feedback. Versions: components p02 → 3, GRPO p02 → 2,
  SQL first/last p01 → 2. **Hidden tests unchanged.** A second reviewer checked these changes.
- **Real model:** opt-in smoke run against the existing `qwen3.8-27b-q6` server: **4 passed**.
  Score mode accepted 3 dimensions with exact evidence and counted 1 dropped item; full-rubric
  model compliance is therefore not established. Partial results are permitted advisory output,
  never self-scores. Interviewer matching/unrelated-question checks both passed. Tests use
  synthetic answers and temporary gateway logs. Counts saved in `/tmp/dogma-live-model.xml`.
- **Disk:** removed 16 ignored `.venv`/`web/node_modules` directories across the nine agent
  worktrees, reclaiming **10,190,368,768 allocated bytes** (about 10.2 GB). Worktree sources and
  branch refs retained; the directories now total about 784 MB. Seven branches are entirely
  patch-equivalent to main; `agent-a52505bcea47eb21d` and `agent-a933d24f551a20e9b` contain
  non-equivalent patches, so the claim that all nine can simply be deleted was not assumed.
- **Verification on the working tree based on `7bee9ae`:**
  `UV_CACHE_DIR=/private/tmp/dogma-uv-cache UV_NO_SYNC=1 make check` exit 0 using the installed
  locked dependencies: **1490 Python tests**, strict mypy **166 files**, **4 import contracts**,
  content typing **247 packages**, **all 167 topics verified under Seatbelt**, privacy audit
  clean (**3286 files**), **148 Vitest tests**, generated API freshness and production build.
  The added session lifecycle test also passed separately with style checks. Python
  `pytest -m '(e2e or slow) and not llm' -q`: **11 passed**; all Playwright flows: **6 passed**;
  live model suite: **4 passed**. Independent review found no remaining material defect.
  Full gate log: `/tmp/deepml-check-20261028.log`. Changes remain uncommitted and unpushed.

### 2026-10-28 · Coordinator session (continued) · I-18 done, whole core path authored and reviewed
- **Every increment in §1 is done**, and every core-path week W1–W13 is authored and reviewed
  (167 topics incl. the 3 `sys` seed cases; W14 has no new topics). M4 (everything in the
  browser) and M5 (timed re-solves, mocks, readiness) are reached.
- **I-18d merged** (`516097c`..`a442cc7`): tutor `score` (strict JSON, verbatim-quote check,
  advisory scores in `answer.model_score`, never changes self-scores or mastery) and
  `interviewer` (`dogma mock ask`: sees only the brief and the numbered `when_asked` list,
  releases one earned reveal, falls back to keyword matching when the model is down); versioned
  prompts with golden tests; migration `a7b8c9d0e1f2`. **I-18e merged** (`005a877`..`7be4b1a`):
  interview-answer, rubric and mock API routes; the Answer tab, `/mock` page and interview-ready
  counts in the browser; a Playwright mock smoke flow.
- **W13 fully reviewed** (R-W13a/b/c/d): BCE-with-logits shortcuts closed, DPO beta claim and GRPO
  std factor corrected, speculative-decoding and conformal edge cases pinned, four RL survivors
  caught. Coordinator: GRPO requires rl/reinforce; MDP p02 accepts in-place sweeps.
- **Final gate on `b59c7e8`** (clean worktree): `make check` exit 0: 1485 Python tests, strict
  mypy 165 files, 4 import contracts, content typing 247 packages, **all 167 topics verified
  under the Seatbelt sandbox**, audit clean (3271 files), 148 Vitest tests, build; e2e/slow 11
  passed; Playwright 4 passed.
- **Follow-ups worth a future increment** (none block the learner):
  1. Today finds a planned timed re-solve by parsing the planner's text (pinned by a test); store
     the problem id in a structured field (small migration).
  2. Live-model smoke tests (`-m llm`) for `score`/`interviewer`; real-model JSON and
     verbatim-quote compliance is unverified, and dropped items are counted, not hidden.
  3. Check the 30-item mock form (120 radios) on a phone-width screen.
  4. A few learner messages could name their mistake better: two SQL first/last-activity
     mutants, a raw IndexError in the components public test, an in-place mutation in GRPO p02.
  5. Content beyond the core path: the remaining `sys` cases (predictive maintenance,
     recommendations, logistics, RAG system design) and tier-B topics linked from lessons
     (e.g. llm/document-packing-and-cross-document-masks, rl/sarsa-and-expected-sarsa).
  6. Citations flagged "from memory / not in corpus" in topic.yaml (e.g. Sutton & Barto chapters,
     LoRA, PagedAttention) can be verified if those sources are added to the corpus.
  7. Learner note: the next `dogma` command migrates `var/dogma.db` through four new revisions;
     `dogma backup` first is cheap insurance.

### 2026-10-28 · Coordinator session (continued) · I-13 done, I-18 started, W10/W11 authoring
- **I-13b merged** (`a249266`..`d0cfb89`): `POST /resolves`, `GET /resolves/current`, `POST
  /resolves/current/abandon`, submit verdicts and readiness over HTTP, tutor routes count hints;
  Today's "Start timed re-solve", a calm countdown, hint confirmation and verdict on the Problem
  page, Readiness on Progress. **Gate on `d0cfb89`** (clean worktree): `make check` exit 0
  (1199 Python tests, 114 topics, 131 Vitest), e2e/slow 11 passed, Playwright 3 passed →
  **I-13 done.** Follow-up: Today detects the timed problem by parsing the planner's block text
  (pinned by a test); a structured field needs a small migration.
- **Content gate gap found:** `dogma content verify` accepted scaffold `TODO` text; the reviewed
  SDPA problem had shipped template hints (fixed in `ecbd18d`). Coordinator now greps for TODO
  before each content commit; a validation rule follows once I-18b (which edits the validator)
  merges. Briefs updated.
- **I-18 planned** (`docs/superpowers/plans/2026-10-27-mocks-and-cases.md`, split a–e); I-18a
  (rubrics, `answer` table, interview-ready) and I-18b (`sys` track, case content kind) running
  in worktrees. W10b/c and W11b committed; reviewers R-W10b/c running. A Haiku session limit
  paused every agent once; all were resumed with their context.
- **I-18a merged**: versioned rubrics (`curriculum/rubrics/mock-v1.yaml`, 30 paraphrased items;
  `topic-answer-v1.yaml`, 4 dimensions), `answer` table + migration `e5f6a7b8c9d0`, `dogma answer
  <topic>` (written or `--spoken`, self-scored), `interview-ready` = retained + every dimension
  >= 3 (answers never lower mastery; re-solve failures and card lapses still drop it). On main:
  `pytest -q` 1268 passed, e2e 10, mypy 148 files, 4 contracts, API types current.
- **I-18b merged** (`a868dcc`..`82ada56`): `sys` track (7 stable tier-B ids; syllabus now 545
  topics in 15 tracks), the `case` content kind (10 validation rules, catalog/coverage/API/Map
  support, `dogma content new` case template, CONTENT_SPEC §9) and three verified seeds
  (RASS framework, churn, fraud). On main: `pytest -q` 1311 passed, Vitest 131, API types
  current, audit clean **with** the private denylist (2991 files).
- **Validator hardened** (`2b8809a`): `TOPIC_TODO` (leftover scaffold text or a pNN-todo dir) and
  `SOURCE_NOT_FOUND` (`library:`/new `source:` citations must exist). All 140 committed topics
  pass; the sys seeds now cite `source: interview-prep/...`.
- **Gate on `bebf867`** (W12 reviewed, W13b/c/d authored, I-18c merged): `make check` exit 0
  (1372 Python tests, 162 topics verified, content typing 241 packages, 131 Vitest), e2e/slow 11,
  Playwright 3, audit clean. A Haiku session limit paused all agents twice more; all resumed.
- **I-18c merged** (`977d442`..`5a0018b` + `a9e6f14`): `mock_session` table and migration
  `f6a7b8c9d0e1` (rebuilds study_session/session_block/answer to widen CHECKs; rows and FKs kept,
  reviewed by the coordinator: child tables rebuilt, old tables dropped child-first, indexes
  recreated), `dogma mock start/reveal/score/debrief/abandon/status`, Interview sessions weekly
  from path week 3 and every session in week 14 (never on low energy), the weakest rubric part
  in plan notes. Coordinator regenerated the API types (session kind `interview`) and added the
  Today label. On main: `pytest -q` 1372 passed, e2e 10, Vitest 131, build ok. **Note for the
  learner:** the next `dogma` command migrates `var/dogma.db`; `dogma backup` first is cheap.
- **W12 fully reviewed** (R-W12a/b/c/d). Notable fixes: int8 scale underflow and the needed
  zero-point clip, a hidden message that leaked an expected value, a shared-schema test
  isolation bug, a privacy-audit false positive (a Decimal method name matched by substring) renamed away.
- **Gate on `6517072`** (W10/W11 reviewed + I-18a): `make check` exit 0 (1268 tests, 137
  topics, content typing 205 packages, 131 Vitest), e2e/slow 11, Playwright 3.
- **W10 fully reviewed** (R-W10a/b/c). Coordinator restored two hidden topological-sort tests a
  reviewer had rebuilt weaker (original generator, seeds and messages from git) and replaced a
  BPE hint that gave nearly the full loop. Fixed `dogma run/submit` rendering: long test names
  were pushed below an empty "✗ public" line at 80 columns (`dfd0d64`).

### 2026-10-27 · Coordinator session (resumed after the usage limit) · I-21 done, I-13a merged, W8 reviewed
- **Resumed** the stopped agents with their context (SendMessage) instead of restarting them.
- **I-21 done.** I-21a (Seatbelt: no network, writes only under the attempt dir and data writes
  to /dev/null; TMPDIR/HOME inside it; hidden tests in a sealed sibling bundle; fail closed)
  reviewed and merged as `96171c4`, combined with I-21b's completion marker in `run.py`.
  **Acceptance:** `make check` on `96171c4` in a clean worktree exit 0: 1083 Python tests,
  strict mypy 133 files, 4 contracts, content typing 174 packages, **all 114 topics verified
  under the sandbox** (incl. 25 py-torch), audit clean, 100 Vitest tests, build.
- **I-13a merged** (`e0bd592`..`2222bac`): `resolve_window` table + forward-only migration,
  pure verdict/schedule/mastery rules, `dogma resolve [--status|--abandon]`, submit verdicts,
  hint/debug confirmation, readiness in `dogma progress`, Consolidate plans due re-solves.
  Verified on main: `pytest -q` 1168 passed, e2e 10 passed, mypy 139 files, 4 contracts.
  Decisions: re-solve intervals are gaps between passes (7, 21, 60, then 120 days; ARCHITECTURE
  wording clarified in `4a66388`); card ratings re-evaluate mastery only on solved→consolidating
  so a lapse on a retained topic stands; a partial verdict never changes mastery.
  **I-13b** (API routes, Today/Problem/Progress UI) is running in a worktree.
- **W8 fully reviewed** (R-W08a/b/c/d: 4 commits, ~220 reviewer mutants; backdoors such as
  torch.group_norm/instance_norm and `.data` closed; generic oracle messages now name causes);
  **W9 fully reviewed** (R-W09ab/c/d; last `487f62c`). C-W10 (3 authors) and C-W11 (4 authors)
  started with the author brief updated by the W8/W9 review lessons (named generic-oracle
  messages, dtype guards, backdoor bans, results-only SQL, non-flaky time limits, verified-only
  citations, sandbox: no downloads).

### 2026-10-27 · Coordinator session · backlog committed, W8/W9 authoring, I-21 and I-13a started
- **Committed the pending tree** in coherent pieces after `make check` passed on it (exit 0: 993
  Python tests, strict mypy 128 files, 4 import contracts, 89 topics, audit clean, 94 Vitest
  tests, build): `1271c3b` W7 content (12 reviewed topics) · `baa79ff` disabled /docs and /redoc
  stay 404 with a build · `acdd196` **I-19 done** (torch 2.14.1 group, runtime guard, API 503
  mapping now tested; `pytest tests/unit/runner tests/integration/runner/test_runner_torch.py
  -m ""` passed) · `1fa9941` **I-20 done** (content type check in `make types`; verify-content
  and the type check use `--group torch`; 8 legacy annotation-only content fixes, no behaviour
  change; `pre-commit validate-config` ok; 42 script tests passed) · `0a301e9` I-16/I-17/I-24
  web pages · `4ead02c` docs · `eb2ac69` I-13 plan.
- **Agents (Haiku), running:** C-W08a (tensors, framework autograd, one step, training loop),
  C-W08b (momentum, Adam, AdamW), C-W08c (LR decay, warmup, SQL LAG/LEAD), C-W08d (batch/layer/
  RMS norm), C-W09a (residual, SDPA, causal masking), C-W09b (positional encodings, MHA, FFN),
  C-W09c (transformer block, KL, conv2d), C-W09d (SQL frames, LRU cache, DP). Shared brief
  requires torch oracles with banned imports, mutants and per-topic verify; authors don't commit.
  I-21a (Seatbelt policy + protected staging) and I-21b (retention) and I-13a (backend + CLI) run
  in separate git worktrees and commit on their own branches for the coordinator to merge.
  A read-only reviewer checks I-16/I-17 against the plan's review focus.
- **Progress (same session):** gate on `4ead02c` passed in a clean worktree (`make check` exit 0
  incl. content typing over 135 packages; e2e/slow 9 passed; Playwright 2 passed). I-16/I-17
  review: 5 confirmed findings (404 stale-card recovery, focus loss on reveal/rate, false "no core
  problem" on catalog failure, shrinking batch count, >200-line files) fixed test-first in
  `80074fe`; Vitest 100 passed, Playwright 2 passed, pytest 1048 passed on `ed6504a` →
  **I-16/I-17 done**. I-21b retention (`dogma runs prune`, dry-run default) merged as `cc827c8`
  after review. W8 all 13 and W9 9/12 topics verified and committed by author batch, with
  cross-batch `requires` added by the coordinator (MHA→SDPA, residual→layer norm, transformer
  block→its parts, warmup→Adam). One reviewer per batch is running.
- **Stopped at the usage limit (resume here):** all 25 W8/W9 topics authored, verified and
  committed (last `d6d8a32`); 6 W9 topics reviewed. Reviewers R-W08a/b/c/d and R-W09c were stopped
  mid-review: their **uncommitted edits are in the working tree** under their topic dirs
  (`git status content/`). Next session: for each of those topics, re-run `uv run --frozen --group
  torch dogma content verify <topic>`, read the diff, and either finish the review or keep the
  verified state; never commit a topic marked `reviewed` without a passing verify. R-W09d
  (sql/sql-running-totals-and-frames, algo/lru-cache, algo/dynamic-programming) not started;
  coordinator decision for it: drop the SQL p01 text-level `ROWS` keyword check (grade results
  only; RANGE + unique tiebreaker is correct) and teach explicit ROWS in the lesson; tighten the
  ~950-word lessons. Briefs to reuse: author/reviewer briefs were in the session scratchpad
  (gone); recreate from ENGINEERING_STANDARDS §5.2/§5.6.
- **Platform agents stopped:** I-21a (Seatbelt policy + protected staging) and I-13a (timed
  re-solves backend + CLI) ran in git worktrees under `.claude/worktrees/agent-*`
  (branches `worktree-agent-a52505bcea47eb21d`, `worktree-agent-a933d24f551a20e9b`). Inspect
  their commits/uncommitted changes there; review before cherry-picking onto main. I-21a must
  show `dogma content verify --all` passing under the policy, including torch topics.
- **Next (original plan):** gate the exact HEAD in a clean worktree (`make check` + e2e + Playwright); integrate
  and spot-check author output, add cross-batch `requires`, commit W8/W9; launch R-W08/R-W09
  review waves; merge I-21a/b and I-13a after review; then I-13b (API + web).

### 2026-10-27 · I-24 done · Light theme and OpenDyslexic visual refresh
- **Changed:** warm cream/sage/slate palette, shared page introductions, illustrated accents,
  SVG navigation, softer cards/controls, improved code presentation and a prominent first
  action on Today. All five main pages use the same visual system. Mobile session tables
  become readable steps. Code retains JetBrains Mono; four OpenDyslexic faces are bundled
  locally with the upstream OFL license, pinned revision and checksums. No new dependencies.
- **Review fixes:** enlarged-text header/filter/workbench overflow; long title wrapping;
  mobile shortcut popover placement; stacked session title/time cells at large text sizes;
  sufficient text contrast on tinted surfaces; inset navigation focus rings.
- **Existing I-16 blocker resolved:** `browserInstruction` now accepts a sentence-ending
  period after a lesson path, and its helper no longer shadows global `escape`. The early
  test assertion expected a raw internal path, contradicting the later browser-instruction
  regression; it now checks the explicit "Lesson tab" wording. Test coverage was preserved.
- **Verified:** `make check` exit 0: 993 Python tests, strict mypy on 128 files, four import
  contracts, all 89 topics, clean privacy audit, frontend lint/types, 94 Vitest tests, current
  generated API and production build. Both Playwright learning-workflow tests passed. Browser
  checks confirmed all four font faces loaded locally, light under dark OS preference, and
  no external requests/browser errors or page overflow across six screens at 320/390/768 px.
  Independent review checked 320 px with 200% text, long titles, popovers, skip-link focus and
  retained mobile table semantics. The final frontend gate/build and screen captures passed
  after the last focus-ring adjustment.
- **Artifacts:** desktop/mobile previews under `.superpowers/sdd/2026-10-27-visual-refresh/`.
  Verification used disposable data; existing code/content/dependency work was preserved.
- **Limits / next:** Vite still emits its nonfatal bundle-size advisory. Visual checks used
  Chromium; Safari/Firefox and platform screen readers were not exercised. I-16/I-17 feature
  acceptance and I-19/I-20 acceptance remain separate work; their previous frontend test/lint
  blockers recorded below are now resolved.

### 2026-10-27 · Documentation maintenance
- **Changed:** README setup and practice/browser instructions; new `docs/DEVELOPMENT.md` and
  `.env.example` covering actual settings, optional torch commands, API generation, checks and
  troubleshooting; corpus rebuild/search instructions. Corrected architecture and standards
  references to CLI commands, hidden feedback, content verification, implemented tutor modes,
  browser routes and pending runner protections. CONTENT_SPEC now describes optional torch
  verification and test imports.
- **Status reconciliation:** I-19/I-20 have implementation artifacts in the working tree and
  are now `in-progress`, with acceptance pending. W7's 12 authored packages are marked `reviewed`
  in YAML and passed the aggregate content gate; their content commit remains pending.
  I-16/I-17 remain in progress. Existing code, content and dependency changes were preserved.
- **Validation:** `make check` passed Python lint/format, strict mypy (128 files), all four
  import contracts, 993 Python tests, all 89 authored topics and the privacy audit (2015 files).
  It then exited 2 at frontend lint: `TodayPage.tsx` shadows global `escape`; that file and
  `learning.test.tsx` also have formatting errors. These are existing I-16 working-tree files.
  Independently, `pnpm --dir web build` and `pnpm --dir web gen:api --check` passed.
  `pnpm --dir web test` reported 93 passed, 1 failed: the Today-page instruction test expects
  the first action's lesson path to become "the Lesson tab", but the path remains visible.
  An ad hoc consistency check passed for 31 local links/anchors, all seven environment keys
  and defaults, 21 CLI help commands, toolchain versions and all 12 W7 review metadata entries.
  `git diff --check` and `scripts/check_commit_files.py` passed for the documentation.
- **Next:** resolve the I-16 lint/format issues and instruction regression, then complete
  acceptance/review for I-16/I-17/I-19/I-20 and the W7 content commit. I-21 network/write denial,
  hidden-test isolation and run retention remain open. Extended e2e, optional torch and live
  model suites were not run for this documentation-only change.

### 2026-10-25 · I-15 done · week 6 reviewed · review-and-commit continuation
- **API preparation committed** as `2eb08a4`: shared learner-facing grading summaries, safe attempt response
  models, offline static hosting and same-origin loopback requests. Review closed two defects with regressions:
  index symlinks escaping the build root, and wrapped assertion values bypassing the hidden-message filter.
- **I-15 browser:** React shell, Problem index/page, CodeMirror Python/SQL, statement/lesson/hint tabs,
  sanitized Markdown/KaTeX, streamed hints, run/submit and learner-safe results. Exact package pins and lockfile,
  generated OpenAPI/client types, bundled fonts and assets. `make check` now checks/builds the browser too.
- **Independent review fixes:** oversized buffers cannot grade an old disk file; empty starters create a workspace
  file; saves serialize edits made during a write; failed departure saves preserve recoverable text/error in session
  memory; pending/failed buffers get unload prompts. Aborted or malformed hint streams cannot replace newer/final
  content. Failed API loads have explicit states. Keyboard actions honor busy/reveal/confirmation guards.
- **Verified exact staged tree:** `make check` exit 0: 973 Python tests, strict mypy 127 files, four import contracts,
  all 77 published topics verified, privacy audit, frontend lint/types, 58 Vitest tests, generated API current,
  production build. The original repo's configured denylist audit also passes; unrelated W7 drafts were preserved.
  A real loopback HTTP smoke used a temporary learner repo: root/client-route reload 200, unknown API route 404,
  same-origin workspace PUT, starter run failed, reference submit passed/recorded, progress solved count 1.
- **R-W06 committed** as `f857df6`: all 13 topics/25 problems reviewed. Numerical fixes cover overflowing medians,
  tree thresholds, AdaBoost weights/tiny errors, KNN norms/label coercion and k-means++ rescaling. Corrected teaching
  claims, citations and hidden feedback. Nine new regressions, 25 rejected mutants, ten valid alternatives accepted;
  a separate clean full staged-tree gate passed before commit.
- **Next work:** all 12 W7 packages are independently reviewed and await their aggregate gate/content commit.
  I-16 Today/Review and I-17 Map/Progress are authored in separate unstaged files, with independent review and real
  browser smoke tests in progress. No real learner progress was used for verification.
- **Limits:** unloading a browser is best-effort protection; session-memory buffers do not survive a process crash.
  The production bundle currently emits Vite's nonfatal chunk-size advisory. Full runner/collection-error filtering
  remains I-21, as recorded in the I-15 plan.

### 2026-10-25 · I-23 done · week 6 mostly authored (agents, verified by coordinator)
- **I-23** (addendum in `docs/superpowers/plans/2026-10-25-api.md`): helpers moved out of the CLI (`content/resolve.py`,
  `content/coverage_scope.py`, `learning/path.py`, `runner/attempt.py`; `cli/resolve.py` deleted, callers updated).
  **Two new import contracts**: the API never imports the CLI, and the domain packages never import the CLI or API
  (a planted violation was caught). SSE tutor routes `/tutor/hint/stream` and `/tutor/debug/stream` relay the provider
  stream through the unmodified gateway (call log, loopback check and guards preserved). Event order: reviewed → deltas → one
  terminal event. `GET /coverage`. Coordinator made the tutor's L4 lock message neutral (no CLI-only `--yes` wording).
  **Verified:** `pytest -q` 929 passed, e2e 8 passed, mypy strict (125 files), 4 import contracts kept, audit clean.
  Open: a client disconnect doesn't cancel the model call (worker runs to the gateway timeout). (Done since:
  `cli/coverage_view.py` uses `content/coverage_scope.py`; the stale `cli/serve.py` lazy-import comment is fixed.)
- **Week 6 (C-W06b/c/d)**: 9 topics verified and committed (b003843, 3504cda, 7f3fabd): PCA/SVD, XGBoost objective,
  k-NN, k-means, ranking metrics, TF-IDF/BM25, SQL ranking windows, heaps. C-W06a (entropy/MI, tree split, forests &
  permutation importance, boosting) verified and committed (61dada1; 88 author mutants caught). All 13 week-6 topics
  are verified; R-W06a/b/c review them. Citations written from memory are flagged for R-W06.
- **`dogma open --fresh`** (an I-13 prerequisite, done early): keeps the old workspace file under
  `var/workspace-history/<track>/<topic>/pNN-slug.<UTC stamp>.py` (copied before the starter replaces it), then
  restores the starter. `open` on an existing file now says how to start over.
- **Finding (corpus):** SQLite FTS5's bm25 uses Robertson IDF clamped to ~1e-6 for terms in more than half the rows, so
  very common terms barely affect ranking. Acceptable because queries AND all terms. Revisit if search quality complaints
  appear.

### 2026-10-25 · I-14 done · week 5 reviewed (agents, verified by coordinator)
- **I-14 API v1** (plan `docs/superpowers/plans/2026-10-25-api.md`): `src/dogma/api/` (app factory, routers, schemas,
  services reusing content/runner/db/learning/corpus/tutor), `dogma serve`, 10 `API_*` codes, deps fastapi + uvicorn,
  140 API tests + 1 e2e learner-day. Security: loopback-only bind, Host/Origin guard, explicit CORS origins, 1 MiB body /
  256 KiB workspace limits, ids validated before path resolution, no solution/test exposure, `/docs` disabled (offline).
  **Decision ADR-0017:** synchronous v1 calls bounded by runner/gateway timeouts, with SSE for the tutor before the browser
  relies on it. **Follow-up I-23:** the API imports a few `dogma.cli` helpers, so move them to a shared module and add an
  import contract.
- **R-W05a/b** (13 topics): 12 `reviewed`. `algo/stacks-and-queues` stays `verified` (a list-as-deque solution can't be
  separated without a flaky timing test). Highlights: a real KS p-value bug for tiny λ (fixed with the theta series),
  overflow-safe quantile/median contracts, float64-pinned split protocol, ADF attribution corrected (Dickey–Fuller 1979 is
  the unit-root test; augmented is later), Breck → Polyzotis et al. 2019, non-reproducible simulation figures replaced with
  reproduced ones and stated setups. ~180 mutants across both batches.
- **Verified (coordinator):** `pytest -q` 880 passed, mypy strict clean (118 files), import contracts kept, audit clean.
  **64/164 core topics built (63 reviewed, 1 verified); weeks 1–5 complete.**

### 2026-10-25 · I-22 done (agent, verified by coordinator)
- **Tutor grounding:** for L1–L3, `dogma hint` prints the reviewed `hints.md` level first (instant, no model, works offline),
  then a model "personalised nudge" whose prompt carries the authored levels ≤ N as an explicit ceiling (`hint.md` v3).
  Server down → reviewed hint + one-line note, exit 0. Non-loopback URL → LLM_ENDPOINT_REFUSED, exit 1 (config error,
  deliberately loud). Missing/malformed hints.md → I-12 model-only behaviour with a note. Corpus context keeps one
  passage per source (dedupe). L1/L2 leak guard (`tutor/ceiling.py`): strips nudge sentences containing a ≥3-token
  formula that only a later hint level reveals. 17-row scripted eval + every repo hints.md parses to 3 levels.
- **Verified (coordinator):** tutor + CLI tests 170 passed. mypy strict clean (118 files). Server-down output checked.
  Live run (agent): 32 s, reviewed hint first, 8 distinct cited sources.
- **Open:** 2-token formula fragments (e.g. `b · b`) still pass the guard. The nudge didn't point at the learner's exact
  buggy line, so a v4 prompt could require naming the LEARNER FILE expression (needs golden updates and a live check).

### 2026-10-25 · I-12 done (agent, verified by coordinator) · week 5 authored · corpus PDFs converted · **M3 reached**
- **I-12 tutor** (plan: `docs/superpowers/plans/2026-10-25-tutor.md`): `src/dogma/tutor/` (gateway = the only model
  caller, providers `local` (httpx, loopback-validated before any client exists, connect 3 s / first token 60 s / total
  180 s via streaming) and `fake`, bounded context ~36K chars with S# citation tags, evidence reader that never opens
  private/, solution.py or test files, output guards that strip correct/incorrect verdicts and long code in debug, and
  an "Unsourced:" notice when nothing is cited). Prompts `hint.md` v2 and `debug.md` v2 with golden snapshots. `dogma hint
  PROBLEM [--level 1-4] [--yes]` (L4 locked until 2 failed recorded submits, else explicit confirmation) and `dogma debug
  PROBLEM` (latest failing attempt). `var/llm_calls.jsonl` holds metadata only. New dep: httpx.
- **Verified (coordinator):** mypy strict (85 files), import contracts, `pytest -q` 679 passed, degradation checked live
  (unreachable server → LLM_UNAVAILABLE; LAN URL → LLM_ENDPOINT_REFUSED; L4 gate → TUTOR_LOCKED). The agent's live smoke
  test against qwen3.8-27b-q6: L1 hint in 33–42 s, cited sources. **Known weakness:** L1 still partly reveals the formula,
  so I-22 grounds L1–L3 in the reviewed `hints.md`.
- **Corpus:** full conversion ran once (15 min). 118/124 PDFs converted, and 6 scanned historical PDFs need `--ocr auto`
  (optional; only LeCun 1989 matters much). Index 112 MB, md 16 MB, both gitignored.
- **Week 5 authored** (13 topics, 26 problems; commits 6feaf8a, 1157128, 9dd0549). 64/164 core topics built.
- **Milestone M3 reached** (corpus search + local tutor with citations).

### 2026-10-24 · Learner session 2 (on main) · I-11 done + learning-loop hardening, reviewed by coordinator
- **Learner-led, while agents were rate-limited:**
  - **Learning-loop hardening** (closes the I-10 follow-ups): close + topic introduction + card enrolment in ONE
    transaction (`sync_cards_in_session`). Open sessions resume across local midnight and keep their study date.
    Ratings are bound to the displayed card's due time AND content hash (`REVIEW_STALE` otherwise). The Practice
    step tells the learner to `submit` to record a solve. `today` no longer double-reports the resumed session as stale.
  - **I-11 corpus** per `docs/superpowers/plans/2026-10-24-corpus.md`: `src/dogma/corpus/{models,chunker,index,sync,
    convert}.py`, `dogma corpus sync [--convert/--no-convert] [--ocr off|auto|force]`, `dogma corpus search QUERY
    [--topic] [--limit]`. Separate `corpus/index.db` (FTS5, porter, external content + triggers, schema v1). Queries
    are reduced to quoted literal terms with AND, parameterised. Read-only search connections. Symlink/escape-safe paths.
    Fence-aware `### Page N` provenance, never merging pages. Per-source atomic replace. sha256 + chunker version
    incremental sync. docmd invoked via `uv run --project tools/docmd` (no shell, timeout). CORPUS_* error codes. 53 tests.
- **Coordinator review:** no changes needed. Verified: ruff + format clean, mypy strict clean (74 files), import
  contracts kept, `pytest -q` 570 passed, FTS5 available (SQLite 3.53.1). Real Markdown sync: 326 sources, second sync
  all unchanged. Searches return relevant cited hits, and a hostile operator/punctuation query returns no hits without
  error. PDFs not yet converted (`corpus/md` absent): run `uv run dogma corpus sync` (with conversion) once.
- **Week 5 (partial):** C-W05a (splitting/batching, scaling, pipelines, missing values & outliers) and C-W05c (rolling
  stats/ACF/seasonal-naive MASE, PSI/KS drift) done and verified (6 topics, 12 problems). C-W05b/d resumed.

### 2026-10-24 · I-10 done (agent, verified by coordinator) · R-W04a done · **M2 reached**
- **I-10** (built to the learner's spec, `docs/superpowers/specs/…learning-sequence-design.md` §Sessions): pure planner
  `dogma.learning.planner` + `sessions` (persistence), revision `c3d4e5f6a7b8` (`study_session`, `session_block`),
  `dogma today [--energy normal|low]` and `dogma close [--stopped-where --next-step --blocker --answer]`. Rules:
  ready = verified/reviewed topic with a verified core problem. New topics need every `requires` ≥ solved. In-progress
  topic first, then path order. Learn 5/3/12/20/5, Consolidate 15/25/5, low energy 10/10 with no new material. An
  empty Learn slot falls back to Consolidate with a reason. Repeat `today` resumes, and a different energy re-plans the
  open slot in place. `close` is idempotent, marks the unseen Learn topic `introduced`, enrols its cards, and never
  solves. Simplifications: no Learn pairing, one re-solve per Consolidate, re-solve = oldest passing submit (spacing
  schedule is I-13), closing an earlier day's session isn't supported (reported as stale).
- **Coordinator:** fixed the one stale assertion (hard-coded schema head → `head_revision()`). Verified: `pytest -q`
  517 passed, mypy strict clean (67 files), import contracts kept, scratch-repo smoke (today → resume → close →
  low-energy today shows the transition note first). Real `var/` untouched.
- **R-W04a done**: 7 week-4 topics `reviewed` (55 mutants. Gaps closed: tol-ignoring lasso/elastic-net solvers.
  Oracles computed before learner calls. Unreproducible figures replaced with closed-form examples). Coordinator
  made the BCE relative-precision requirement explicit in its contract (it was test-only), version 2 → 3.
- **Milestone M2 reached** (memory: progress DB, FSRS reviews, daily planner, transition notes).
- **Open follow-ups:** `dogma open` never overwrites, so a re-solve reuses the old file (done since: `open
  --fresh`). Card enrolment in `close` runs after the close commits (it retries on the next today/review).

### 2026-10-24 · Learner session (codex/learning-loop) · I-09 done, reviewed by coordinator
- **Learner-led work** while agents were rate-limited: design spec + plans in `docs/superpowers/` (learning
  sequence: I-09 → I-10 → content wave → I-11 → I-12 → I-14..I-17 → I-19/I-20/I-21; corpus plan for I-11), and
  **I-09 implemented**: `src/dogma/learning/{scheduler,cards,review}.py` (FSRS, retention 0.90, fuzzing off,
  UTC-only timestamps), `src/dogma/db/review_models.py` + forward-only revision `b2c3d4e5f6a7` (`card`,
  `card_review` with content snapshots), `dogma review [--limit N]`, REVIEW_* error codes, 42 tests.
  Behaviour: solved topics' cards enrol on sync. Edits keep schedules. Removed cards are deactivated, never
  deleted. Ratings are atomic and reject stale/early/invalid input. A good review ≥3 days after solving earns
  consolidating, and a lapse from retained+ returns to consolidating.
- **Coordinator review:** design matches ARCHITECTURE §4.4 and ADR-0010/0013 (bounded queue, no backlog
  counter). Fixes were style only (ruff format on 6 files, long lines, function-local imports hoisted, fixed-name
  SQL annotated). Verified: mypy strict clean (61 files), import contracts kept, `pytest -q` 426 passed, end-to-end
  smoke in a scratch repo (solve → enrol → review QA + cloze → reschedule → progress). Real `var/` untouched.
  Merged into `main` (fast-forward).
- **Follow-up:** daily new-card cap (the queue is already bounded per session) and a `--topic` filter are optional.
  Next per the plan: I-10 (planner + `today`/`close`).

### 2026-10-24 · Session 1 · week 4 authored, week 3 reviewed
- **C-W04 done** (4 authors, resumed once after a usage limit): 13 topics, 25 problems. Regression metrics + Huber,
  ridge, lasso + elastic net, sigmoid/logit, L2 logistic regression (Newton + Armijo), softmax/log-softmax,
  cross-entropy (multiclass + binary from logits), confusion matrix + count scores, precision/recall/F1 averaging +
  best-F1 threshold, ROC-AUC rank statistic + step-wise AP, balanced class weights + prior-shift correction,
  SQL conditional aggregation, pandas groupby/transform. Each author ran 50–80 mutants.
- **R-W03a/b done**: all 13 week-3 topics `reviewed`. About 240 mutants. Gaps closed: a statistic called on the
  whole resample matrix, scaled 1−CDF p-values, orphan foreign keys in SQL, accidental passes. False or
  non-reproducible claims fixed (det ≈ 1e-380 not 1e-420, Anscombe IV description, "0.05-pt lift at 1M users is
  significant", which is false). Setups are now stated for every simulated figure.
- **Coordinator:** timeout/OOM headlines in the CLI now name the usual causes (endless loop, too-slow algorithm,
  oversized arrays). An empty 0-byte `var/dogma.db` (no tables) was removed. The progress DB is untouched by
  agents since the recording fix.
- **Verified (coordinator):** `make check` exit 0 (384 tests, mypy 54 files, 51 topics verified, audit clean).
  Statuses: 38 reviewed (W1–W3), 13 verified (W4). 51/164 core topics built.
- **Next:** R-W04 (2 reviewers) ∥ C-W05 (4 authors, due 16 Nov). Platform: I-09 (cards + FSRS `dogma review`).

### 2026-10-24 · Session 1 · I-08 done + attempt-recording fix
- **I-08 done** (agent): `src/dogma/db/` (SQLAlchemy 2 models `attempt` + `topic_state`, WAL, Alembic in code,
  forward-only revision `a1b2c3d4e5f6`), `dogma progress [--week N]`, `dogma backup`, `dogma db upgrade`, attempt
  hook in run/submit (a passing core submit → topic `solved`, never downgraded; DB failure = warning only).
  57 new tests. Added deps sqlalchemy + alembic.
- **Recording defect found by I-08:** alternative-file mutation tests (`dogma submit --submission …`)
  could be recorded as learner progress.
  **Coordinator fix:** only the learner's workspace file is recorded by default. `--submission` runs are recorded
  only with `--record`, `--no-record` opts out, and `DOGMA_RECORD_ATTEMPTS=false` is a global kill switch. 4
  regression tests cover the recording boundary. Individual database contents and local cleanup
  details are excluded from the public handoff.
- **Verified (coordinator):** `pytest -q` passes (see the commit). mypy strict clean (54 files). lint-imports kept.
  Progress reporting was checked against a scratch database.
- **Follow-ups:** move `StorageErrorCode` into `core.ErrorCode` (StorageError → DogmaError). Backups are written
  0600 (fine for private data). Run-dir retention is still I-21.

### 2026-10-24 · Session 1 · week 3 authored, week 2 reviewed, grader fix
- **C-W03 done** (4 authors): 13 topics, 20 problems. Expectation/variance, covariance/correlation, MVN log-pdf,
  MLE/MAP, Monte Carlo, inverse-transform, reservoir sampling, t-intervals, bootstrap/permutation, Welch t/ANOVA,
  A/B sample size + z-test, SQL subqueries/CTEs, binary search. Each author ran 40–60 mutants.
- **R-W02a/b done**: all 13 week-2 topics `reviewed`. Reviewers re-measured lesson numbers and replaced several
  that couldn't be reproduced (a −262 cancellation figure, a "10^15 coefficients" claim that turned out false, and
  "170! overflows float64", which is really 171!). Mutants that slipped through were closed. Hypothesis tests now
  route through a single failure site. A per-call SIGALRM time box turns hangs into named failures.
- **Coordinator fix (grading):** a bare banned name (e.g. `random`) also matched attributes (`rng.random`),
  rejecting valid code. Fixed in `dogma.grading.static` and `scripts/grade.py`, with a regression test.
  CONTENT_SPEC §6.4 now records the recurring test-authoring lessons (report_multiple_bugs=False, neutral
  param ids, raise helpers, wrapping learner crashes) and the banned-name semantics.
- **Verified (coordinator):** `make check` exit 0 (ruff, mypy 37 files, import contracts, 323 tests,
  `dogma content verify --all` → 38 topics, audit clean). Statuses: 25 reviewed (W1–W2), 13 verified (W3).
- **Next:** R-W03 (2 reviewers) ∥ C-W04 (4 authors, due 9 Nov). Then I-08 persistence. Housekeeping: `var/runs`
  has 1000+ agent attempt dirs (safe to delete; retention belongs to I-21).

### 2026-10-24 · Session 1 (resumed again) · baseline commits
- **Committed** (on request): `fb4cc56` docs + syllabus/path · `72112b1` platform code + tests · `a8ebb1c`
  vendored docmd + scrubbed sources · `b266756` weeks 1–2 content (25 topics, all verified right before
  committing; week 1 reviewed). The repo uses a neutral local git identity (`dogma <dogma@localhost>`),
  so no PII is in commit metadata. Change it with `git config user.name/user.email` if you want.
- **Left uncommitted on purpose:** week-3 topic dirs (C-W03 in progress) and R-W02 review edits. Commit them
  as `content(w03)` / `content(w02): review` once they verify.
- A second usage limit cut off C-W03 a/b/c/d and R-W02a/b. All were resumed with SendMessage.

### 2026-10-23 · Session 1 (resumed after a usage-limit pause) · I-05, I-06 done; I-07 a/b/c done
- **Usage limit:** all running agents were cut off mid-task at the same moment. Each was resumed with
  SendMessage and continued from its saved state. Expect this again; nothing was lost.
- **I-05 done** (agent). `dogma` CLI: `open/run/submit/path`, `content verify/coverage/new`, `catalog build`,
  templates in `templates/topic/*.tmpl`, 77 unit + 7 e2e tests. **Verified (coordinator):** `pytest -q` 322 passed,
  mypy strict clean (37 files), `dogma path --week 2` and `dogma submit` behave as specified. Coordinator
  stripped pytest's "Failed:" prefix in hidden summaries (CLI and grade.py) and switched `make verify-content`
  to `dogma content verify --all`. `scripts/grade.py` stays until the content waves stop using it, then
  becomes a deprecated wrapper (small follow-up, any session).
- **I-06 done** (2 reviewers). All 12 week-1 topics are `reviewed`, versions bumped where tests changed.
  Headlines: the example solution under/overflowed for tiny or huge nonzero `b` (fixed by scaling); about
  100 mutants run; gaps closed (partial pivoting, absolute vs. relative tolerances, exact SQL status
  match, clipping, loop detection for broadcasting); hidden messages no longer leak values; unverifiable
  citations softened to chapter titles. Determinant tests use a SIGALRM guard so cofactor expansion fails
  with a named "too slow" message (works inside the real runner too, verified).
- **I-07 done** (4 authors): 13 topics, 17 problems, all pass `dogma content verify` (coordinator-run).
  Authors each ran 20–40 mutants. The I-07d author flagged that the week-1 group-by test helper maps `pd.NA`
  to the string "<NA>" (probably latent) and that its hypothesis test has several assert sites. Check in a
  later review pass.
- **Launched:** C-W03 a/b/c/d (week-3 authors), R-W02a (calculus/optimization review), R-W02b
  (probability/toolbox review). Next platform increment: I-08 (persistence). Start it in a session with
  usage headroom, because it changes `pyproject.toml`/`uv.lock`.
- **Follow-ups:** citations/dates written from memory (week-2) get checked in R-W02. Some lessons run
  slightly over 900 words. Some cards have two-fact backs. Whole-run timeouts in the runner aren't
  attributed to a test (the determinant guard is the per-topic workaround; consider a per-test timeout in
  the harness).

### 2026-10-23 · Session 1 · I-01 done (agent, verified by coordinator)
- **Changed:** `scripts/import_sources.py` (manifest-driven, idempotent, scrub-preserving, `--relock`),
  `scripts/audit_privacy.py` + `make audit` (in `check`), `sources/**` imported + scrubbed (183 files edited,
  4 withdrawn; see `sources/withdrawn.yaml`), `sources/manifest.lock.json`, `tools/docmd/` vendored with
  `VENDORED.md`, `corpus/raw/library-papers` (124 PDFs, gitignored), 29 tests.
- **Coordinator decisions:** unapproved non-ML PDFs excluded from the committed lock; lock now
  549 entries. Required third-party license attribution is retained. Local privacy filters were
  extended, with false positives corrected for ordinary technical text and scholarly attribution.
- **Verified (coordinator):** `make audit` clean (556 files). `make check` green (see below).
- **Note:** the importer once ran `git rev-parse HEAD` in the vendored tool's origin, which resolved the
  enclosing workbench repo (commit hash only, no content read). It now runs git only in roots with their own `.git`.

### 2026-10-23 · Session 1 · I-03 done (agent, verified by coordinator)
- **Changed:** `src/dogma_harness/` (pytest collector, bounded stdout capture, report), `src/dogma/grading/`
  (`GradeReport`, `RunStatus`, `check_submission`), `src/dogma/runner/` (`run`, `load_meta`,
  `new_attempt_id`; process-group kill, RSS + disk watchdogs, scrubbed env), 116 tests + fixture problems.
- **Verified (coordinator):** `pytest -q` 245 passed. mypy strict clean. lint-imports 2 kept.
  `runner.run` on the example problem: solution passed 4/4 + 11/11, starter failed.
- **Decisions:** child uses `-P -s`, not `-I` (`-I` ignores `PYTHONHASHSEED`, confirmed). Default memory
  limit lowered to 2048 MB. Hidden-test exposure and missing network/write denial deferred to I-21.

### 2026-10-23 · Session 1, wave 1 results (coordinator)
- **I-02 done** (agent). `src/dogma/content/{schemas,markdown,pyfacts,loader,validate,catalog,coverage}.py`,
  98 tests. API: `validate_repo(settings) -> list[Issue]`, `load_topic`, `build_catalog`/`write_catalog`
  (refuses to build when any issue exists), `coverage(syllabus, packages)`. **Coordinator verified:**
  `pytest tests/unit/content` 98 passed. ruff clean. lint-imports 2 kept. `validate_repo` on the real
  tree → 0 issues (after fixing one `PROBLEM_EXAMPLES` issue in the group-by problem).
- **I-04 done** (3 agents): 11 topics, 13 problems (matrix-inverse-and-rank has p02 rank).
  **Coordinator verified:** every problem passes `scripts/grade.py verify`. Authors ran mutation tests.
- **Coordinator fixes:** the example lesson wrongly claimed `np.linalg.norm` rescales (it overflows
  to inf, verified). The group-by problem.md got a second example block. `scripts/grade.py` now shows
  the first *meaningful* line of hidden failures (numpy `err_msg`) without leaking values.
- **Follow-ups (not yet assigned):** (1) content `.py` files aren't type-checked by the gate. Add a
  non-strict mypy pass for content in I-05 or I-20 (needs a pandas override or pandas-stubs).
  (2) Runtime `py-stdlib` is used by the hash-map problem. Add it to the CONTENT_SPEC runtime list.
  (3) The CLI (I-05) must print all validation issues, not just the first.
  (4) Authors cited chapter numbers from memory. The I-06 reviewers check or soften them.
- **Launched:** I-06 (2 reviewers), I-07 (4 authors). I-01 and I-03 still running.

### 2026-10-23 · Session 1 (coordinator, Opus): planning + I-00
- **Changed:** git init. uv project (Python 3.12, dev group). Docs: ARCHITECTURE,
  ENGINEERING_STANDARDS, IMPORT_PLAN, CONTENT_SPEC, LEARNER_PROFILE, ROADMAP. Syllabus with tiers
  (A 164 / B 221 / C 153) + `curriculum/paths/core.yaml`. Example topic
  `content/cml/vector-operations`. `scripts/grade.py`. `src/dogma/core/{config,errors,fsio}.py`.
  Makefile. `private/denylist.txt` and `private/import_roots.yaml` (gitignored).
- **Verified:** `make check` green: ruff, ruff format, mypy strict, import-linter, pytest
  (2 passed), example problem verify ✓ (public 4, hidden 11, deterministic, starter fails).
- **Next:** wave 1 launched in parallel: I-01, I-02, I-03, I-04a/b/c. Then I-05, I-06, I-07.
