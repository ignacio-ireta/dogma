# Engineering Standards

*How code and content are built in this repo. **MUST** = enforced by the gate or review.
**SHOULD** = the default; deviations need a one-line reason in the commit or PR. These standards
extend the conventions already proven in docmd and MVS (uv, ruff, strict mypy, failures-as-data,
self-checking exercises).*

---

## 1. Principles

1. **Content is the product; the platform is thin.** Build only what makes studying better this
   week. When unsure, author content instead of building features.
2. **Verifiable or it doesn't ship.** Every problem passes the content gate. Every module has
   tests. "Seems right" is not a state.
3. **Local-first, private by default.** No network in core paths. External services are opt-in
   per call. Personal and licensed material never enters git.
4. **Failures are data.** Bad learner code, a corrupt PDF or a down model each produce a
   structured result with an error code, never a traceback in the UI.
5. **Small, boring technology.** One language per layer, one database file, one resident model.
   Before adding a dependency, ask whether stdlib or an existing dependency covers it.
6. **Predictable structure.** Same layouts, same section orders, same command shapes everywhere,
   for the learner and for developers.
7. **Self-contained.** Everything the project needs lives in this repo: imported material in
   `sources/`, vendored tools in `tools/`, third-party documents in gitignored `corpus/raw/`.
   Nothing reads or depends on other projects on this machine. **The only exception is the AI
   runtime** (model weights, llama.cpp / Ollama servers), reached over loopback through the
   gateway (ADR-0014).

## 2. Language

Docs, code, identifiers and content are in **English**. Use the US spelling of identifiers
(`color`, `normalize`). Prose spelling is free.

## 3. Toolchain

| Concern | Tool | Notes |
|---|---|---|
| Python | **3.12** via `uv` (`.python-version`) | Not the system 3.15. Torch/JAX wheels |
| Env & deps | `uv` + `uv.lock` (committed) | `dev` is installed by default; `torch` is optional. Other runtime groups are planned |
| Lint / format | `ruff check`, `ruff format` | Config in §4.1 |
| Types | `mypy --strict` on `src/` (`dogma` + `dogma_harness`) | Content solutions: checked non-strictly (§5.2) |
| Tests | `pytest`, `hypothesis`, `pytest-cov` | Markers in §6 |
| Architecture | `import-linter` | Contracts from ARCHITECTURE §8 |
| CLI | `typer` + `rich` | |
| Config / schemas | `pydantic` v2, `pydantic-settings` | Env prefix `DOGMA_` |
| API | `fastapi`, `uvicorn` | OpenAPI is the contract |
| DB | SQLite (WAL), `sqlalchemy` 2.0, `alembic` | |
| HTTP client | `httpx` | Timeouts always explicit |
| Web | Node ≥22, pnpm 12.8.1, React 19, Vite, TypeScript (strict) | Requirements and package-manager pin in `web/package.json` |
| Web lint / format | `biome` | One tool for both |
| Web tests | `vitest` (+ Testing Library), `playwright` (few smoke flows) | |
| Task runner | `make` | Preinstalled on macOS. Targets in §13 |
| Hooks | `pre-commit` | Fast subset of the gate |

Pin by lockfile, not by `==` in `pyproject.toml`. Use lower bounds (`>=`) unless an upper cap is
*known* to be needed, with a comment saying why (as MVS does for scikit-learn < 1.8).

## 4. Python code

### 4.1 Style

The checked-in [pyproject.toml](../pyproject.toml) is the source of truth for ruff rules,
exclusions and per-file exceptions. Target Python 3.12, use a 100-character line limit and
run both `ruff check` and `ruff format --check`. Content allows conventional mathematical
names; application code keeps the stricter naming and output rules.

- `from __future__ import annotations` at the top of every module.
- Names: modules `snake_case`, classes `PascalCase`, constants `UPPER_SNAKE`. Math code may use
  conventional symbols (`X`, `W`, `mu`, `sigma`) when that mirrors the formula, with the formula
  cited in the docstring.
- Functions do one thing. Aim for ≤ 40 lines. Prefer pure functions in `content/`, `grading/`
  and `learning/`; I/O lives at the edges (`cli`, `api`, `fsio`).
- Comments explain **why**, not what. Match the density of the surrounding code.

### 4.2 Types

- `mypy --strict` passes on `src/`. No bare `Any`. If one is unavoidable, add `# Any: <reason>`.
- Arrays: `numpy.typing.NDArray[np.float64]` in signatures. Write the expected shape in the
  docstring (`(n_samples, n_features)`).
- Domain data crosses module boundaries as Pydantic models or frozen dataclasses, never raw
  dicts.

### 4.3 Docstrings

Public functions and classes get a one-line summary plus whatever a reader can't infer: shapes,
units, invariants, the formula and its source. NumPy style for anything with parameters:

```python
def bm25_scores(
    query: list[str], index: InvertedIndex, k1: float = 1.2, b: float = 0.75
) -> NDArray[np.float64]:
    """Score every document against ``query`` with Okapi BM25.

    Uses the Robertson–Spärck Jones IDF with +1 smoothing so scores stay non-negative.

    Returns
    -------
    scores : (n_docs,) array, higher is more relevant.
    """
```

### 4.4 Errors

- Domain exceptions live in `dogma.core.errors`. Each carries a stable `code`
  (`CONTENT_SCHEMA_INVALID`, `RUNNER_TIMEOUT`, `LLM_UNAVAILABLE`, …) and a learner-safe message.
- Never use a bare `except:`. Never `except Exception` without re-raising or converting to a
  domain error *with* logging.
- **Learner code failing is not an exception.** It's a `GradeReport` with a status. Only
  infrastructure failures raise.
- New failure modes get a new code and a test.

### 4.5 Logging & output

- stdlib `logging` with a `rich` handler. Log at module level:
  `log = logging.getLogger(__name__)`.
- **Never log** the contents of `private/`, secrets, or full prompts. Log the prompt *hash*,
  model id and latency.
- `print` is banned in `src/` (ruff `T20`). CLI output goes through `rich.console`.

### 4.6 Configuration

All settings live in one `Settings(BaseSettings)` in `dogma.core.config` (env prefix `DOGMA_`,
optional gitignored `.env` in the working directory). Environment variables take precedence over
`.env`. `.env.example` is committed with every key and a comment; keep it aligned with `Settings`.
The [developer guide](DEVELOPMENT.md#configuration) lists defaults and scope. Code never reads
`os.environ` directly.

### 4.7 Filesystem & ids

- `pathlib` only (ruff `PTH`). All paths are derived from `Settings` roots. No hardcoded `~/…`
  in `src/`.
- **Every write is atomic**: write a temp file in the same directory, then `os.replace`. Helper:
  `dogma.core.fsio.atomic_write`.
- Commands that sync, build or import are **idempotent**. Re-running them with no changes does
  nothing and says so.
- **Ids are stable and never renumbered or reused:** topics `track/slug`, problems
  `track/slug/pNN-slug`, cards `<topic>#<n>`. Retire with `status: retired` instead of deleting.
- Store times as UTC ISO-8601. Convert to local time only for display.
- Randomness: always explicit seeds via `np.random.default_rng(seed)`. Never the global state.

## 5. Content standards (the curriculum is code)

### 5.1 Topic packages

- The layout in ARCHITECTURE §4.1 is mandatory. `dogma content new <id>` creates it from
  `templates/`. Don't hand-roll packages.
- Every topic id must exist in `curriculum/syllabus.yaml` (or a local track defined there).
- `requires:` lists only topics that already exist. The prerequisite graph must stay acyclic
  (checked by the gate).
- One topic → 1–3 problems. The first is the **core problem** (the session's must-do). Others go
  deeper or vary the context.

### 5.2 Problem quality bar

A problem is shippable only if **all** of the following hold:

- [ ] `problem.md` states the task, a precise I/O contract (types, shapes, dtypes, edge cases),
      ≥2 worked examples, constraints, and what "done" looks like.
- [ ] `starter.py` contains the exact signature, a docstring and `raise NotImplementedError`.
      It runs, and it fails ≥1 hidden test.
- [ ] `solution.py` is **readable reference code**, not golfed. Comments tie each step to the
      formula or system in the lesson. It uses only the imports the learner is allowed.
- [ ] ≥3 public tests, including ≥1 edge case. ≥5 hidden tests covering empty or degenerate
      input, shapes, numerical stability (large or small magnitudes), and dtype.
- [ ] **Oracle comparison** on randomized inputs whenever a library implementation exists.
      If none exists, add a property test or an analytic case.
- [ ] Bulk random data is realistic where realism matters: real or semi-synthetic data via
      `dogma_data`, or a `dogma_data.synth` generator; see CONTENT_SPEC §6.6.
- [ ] Hidden tests pass the lint (CONTENT_SPEC §6.4), and `variants/` holds one blind accept
      and, for every Contract bullet, a reject or an `uncovered` reason (CONTENT_SPEC §6.8).
- [ ] Gradient problems include a finite-difference check.
- [ ] Tolerances are declared in `meta.yaml` (`atol`, `rtol`) and justified when looser than
      `1e-6`.
- [ ] Banned imports are declared for from-scratch problems (e.g. `sklearn`, `torch.nn.functional`).
- [ ] Runs under the time budget (default 10 s per test file).
- [ ] `hints.md` has exactly three levels: **concept → formula/approach → code shape**. None of
      them contains the solution.
- [ ] Problem text, tests and solution are **original**. Nothing is copied from deep-ml or other
      problem sites.

Solutions and tests must pass plain (non-strict) mypy and ruff. `make types` (part of
`make check`) runs `uv run --frozen --group torch python scripts/check_content_types.py`, which
checks each problem package on its own.

### 5.3 Lesson template (fixed section order)

Every `lesson.md` uses these H2 sections in this order, so the learner always knows where things
are:

1. **The system**: the rule set, subgroups or decision tree, stated explicitly. A diagram if one
   helps.
2. **Worked example**: small numbers, every step shown, using the semantic colours.
3. **Why it works**: the intuition and the one derivation that matters, with a citation.
4. **Signal vs. noise**: what interviewers and practice actually need, and what's safe to ignore
   for now.
5. **Where it breaks**: failure modes and how to spot them in real data.
6. **Interview angle**: the 60-second answer shape plus the two most likely follow-ups (full
   versions in `interview.md`).
7. **Connections**: prerequisites, what this unlocks, the historical lineage (link the Library
   note).

Front-matter carries `minutes` and a **Minimum-viable path** line for low-energy days.

### 5.4 Writing style

- Explicit beats implicit. Name the rule, the exception and the subgroup. Never write "you'll
  get a feel for it."
- Short sentences, concrete numbers, one idea per paragraph.
- Math is KaTeX (`$…$`, `$$…$$`). Define every symbol at first use, with its shape.
- **Name the kind of difficulty** wherever there is one ("this part is bookkeeping, not
  insight"; "this is the one step people get wrong").
- Interview answers sound like a calm senior: trade-offs first, caveats explicit, no hype.

### 5.5 Sources, citations, originality

- Every lesson cites ≥1 source in `topic.yaml: sources` and inline where a claim depends on it:
  `[library:AI-ML/…/Kingma Ba Adam 2015.md]`, `[library-papers/1412.6980 p.3]`.
- Quote third-party text only in short excerpts, attributed. Paraphrase everything else.
- No PII, employer, client, colleague or project names anywhere in the repo (ADR-0016). `make audit`
  enforces this.

### 5.6 Review checklist (to mark `reviewed`)

- [ ] I could solve the problem from the statement alone, without guessing hidden requirements.
- [ ] Every hidden-test failure message tells me what kind of mistake I made.
- [ ] The lesson's "system" section would let me rebuild the solution from memory.
- [ ] Signal vs. noise matches what senior interviews actually ask.
- [ ] Cards follow minimum information: one card, one question, no ambiguity.
- [ ] Citations resolve and support the claims they're attached to.
- [ ] Time estimates are realistic (adjust them after the first real attempt).

### 5.7 Status & versioning

- Status lifecycle: `draft` → `verified` (automated content gate passes) → `reviewed` (§5.6) →
  `retired`. Only `verified`+ appears in sessions, and `verified`-only items show a badge.
- Bump `meta.yaml: version` whenever tests, tolerances or the I/O contract change. Attempts are
  keyed by version. Typo fixes don't bump it.

### 5.8 Visual language

Semantic colours, used everywhere (lessons, widgets, plots), ported from MVS: **blue** = inputs /
original variables, **orange** = latent / components / learned representations, **green** =
fitted values / model output, **red** = residuals / error / loss, qualitative palette =
groups / clusters. Colour is never the only signal. Pair it with a label or line style.

## 6. Testing

| Marker | Scope | Runs in default `make test` |
|---|---|---|
| *(none)* | Unit: pure functions, schemas, planner logic, grading comparisons | yes |
| `integration` | Multiple components: runner + harness on fixture problems, DB + migrations, corpus indexing on fixture docs | yes |
| `content` | Marker reserved for content-focused pytest suites; the authored-content gate runs separately via `make verify-content` | `make check` verifies all topics |
| `e2e` | CLI via `CliRunner`, API via `TestClient`, 2–3 Playwright flows | `make test-all` |
| `llm` | Live calls to the local model (smoke only) | **never by default** |
| `slow` | Anything over 5 s | `make test-all` |

- **No live model calls in the default suite.** Use the `fake` provider and recorded responses.
- Prompts get **snapshot tests**: the rendered prompt for each mode is compared against a golden
  file. Read golden diffs before committing, never commit them blind (same rule as docmd).
- The **planner and FSRS adapter** get table-driven tests: given state X and Y minutes, expect
  plan Z. Hypothesis checks invariants (blocks sum to available time; nothing new in low-energy
  mode; prerequisites are respected).
- **Runner** tests cover timeout, the OOM and disk-growth watchdogs, syntax errors, banned
  imports and infinite output, plus real Seatbelt enforcement on macOS: denied network,
  denied writes outside the attempt directory (including through symlinks and child
  processes) and tests that learner code cannot change.
- Coverage: ≥ 85% branch coverage on `grading/`, `runner/`, `learning/`, `content/`. No numeric
  target for `api/`, `cli/`, `web/`. Behaviour tests matter more there.
- A bug fix starts with a failing test that reproduces it.

## 7. Frontend

- TypeScript `strict: true`, `noUncheckedIndexedAccess: true`. No `any` (use `unknown` and
  narrow).
- **API types are generated** from the FastAPI OpenAPI schema (`pnpm gen:api`). They're never
  written by hand. CI fails if the generated client is stale.
- Server state uses TanStack Query. Local UI state uses React state. No global store until it
  clearly pays for itself.
- Components: one per file, ≤ 200 lines, props typed, no business logic. Logic lives in hooks or
  the API layer.
- **Keyboard first:** Run ⌘↵, Submit ⌘⇧↵, next hint ⌘H, review ratings 1–4. Visible focus states.
  Respect `prefers-reduced-motion`. The learner's explicit light-theme preference takes
  precedence over the OS color scheme.
- OpenDyslexic Bold is the interface font. Rendered Markdown uses Inter for readable prose
  alongside KaTeX formulas. Bundle the real font files and their licenses locally; keep
  JetBrains Mono for code alignment.
- Works fully offline. No runtime CDNs. Fonts and KaTeX are bundled.
- Design tokens (colours from §5.8, spacing, type scale) live in one CSS file. No inline colour
  literals.
- Memphis coral/violet decoration uses separate tokens from the semantic colours in §5.8.
  Keep reading and code surfaces quiet; place illustration on introductory surfaces and compact
  topic headers. Draw figures and subject motifs as editable SVG; keep them decorative and static.
  Normal text must retain 4.5:1 contrast and focus/control boundaries 3:1 where required.

## 8. API

- REST + JSON under `/api/v1`. Breaking changes need `/api/v2`. Additive changes are fine.
- Request and response bodies are Pydantic models. No untyped dicts in signatures.
- Errors use RFC 9457 `application/problem+json`, with the domain error `code`.
- API v1 grades attempts synchronously, bounded by the runner timeout, and supports both JSON
  and SSE tutor endpoints. Tutor operations use the gateway's explicit timeouts. This is the
  accepted exception to the target of immediate ids or streams for long work; see
  [ADR-0017](decisions/0017-synchronous-api-v1.md).
- Mutations are idempotent where possible: workspace `PUT` replaces the draft, session close
  preserves the first accepted note, and review ratings carry stale-view guards.

## 9. Data & migrations

- A single SQLite file at `var/dogma.db`, WAL mode, foreign keys on.
- Schema changes go **only** through Alembic. Migrations are forward-only, and each is tested
  by upgrading a fixture DB copy in CI.
- **Never** write learner progress anywhere else (no JSON side files, no localStorage as truth).
- `dogma backup` → timestamped JSON export in `var/backups/`. Copy it to the External SSD
  weekly.
- Never delete learner data in a migration. Deprecate columns instead.

## 10. LLM usage

- All model calls go through `dogma.tutor.gateway`. **No direct SDK or HTTP calls to models
  anywhere else.**
- The `local` provider validates that its base URL is loopback, and refuses anything else.
- External providers are **deny by default**: they need `DOGMA_ALLOW_EXTERNAL=1` **and** a
  per-request opt-in. Never in background jobs. `private/` context is always stripped first.
- Prompts are versioned files in `src/dogma/tutor/prompts/<mode>.md` with front-matter
  `version:`. Changing a prompt bumps the version and updates its snapshot.
- Every call appends a line to `var/llm_calls.jsonl`: ts, mode, model, prompt version, prompt
  hash, tokens in/out, latency, outcome. No prompt bodies.
- Explicit timeouts (connect 3 s, first token 60 s, total 180 s). On failure, return a
  `LLM_UNAVAILABLE` result. The UI keeps working.
- The tutor **never** states that code is correct or incorrect. It refers to the GradeReport.
- Tutor answers that assert facts include ≥1 citation from the lesson or corpus, or say clearly
  that they're unsourced.

## 11. Security & privacy

- **Threat model:** one trusted user on their own machine. Guard against runaway code, accidental
  data leaks (git, external LLMs, logs) and supply-chain surprises. We are not defending against
  a malicious user.
- **Runner:** timeout, process-group kill, RSS and disk-growth watchdogs, scrubbed env, isolated
  cwd and output truncation, plus (I-21) macOS Seatbelt network and write denial outside the
  attempt directory, tests in a sealed bundle the learner process cannot modify, and explicit
  dry-run retention. Execution fails closed where Seatbelt is unavailable. Reads are not
  restricted, so this is integrity protection, not secrecy. Untrusted *files* (PDFs,
  downloads) are parsed by docmd and never executed. Scripts that read downloaded data run with
  `python -I`.
- **Privacy boundaries** (IMPORT_PLAN §2) are enforced by `.gitignore` plus `make audit`
  (`scripts/audit_privacy.py`). It checks all tracked/unignored working-tree files, filenames,
  metadata and built browser assets against local private rules plus contact/account-path rules.
  Missing/empty rules and oversized files block certification. Diagnostics redact private terms.
  The commit hook uses `--staged` to check actual index blobs as well, including paths removed
  only from disk; private blobs are rejected by path without being read. Publication symlinks
  are rejected. No individual biography, medical profile or private work narrative belongs in
  product requirements: write general design principles and synthetic examples instead.
  Checksum-pinned third-party license attribution and opaque parser tables have narrow, tested
  exemptions. Neither this audit nor `.gitignore` removes prior Git history.
- Secrets live only in `.env` or the macOS keychain. Never in code, logs, prompts or fixtures.
- Dependency changes show up as `uv.lock` / `pnpm-lock.yaml` diffs in their own commit with a
  one-line reason.

## 12. Git workflow & decisions

- Trunk-based: `main` always passes the gate. Use short-lived branches for anything over one
  commit.
- **Conventional commits** with scopes: `content`, `runner`, `grading`, `learning`, `tutor`,
  `corpus`, `sources`, `data`, `db`, `api`, `cli`, `web`, `docs`, `build`. Examples:
  `feat(runner): add RSS watchdog`, `content(cml): add gradient-descent p01`.
- Keep **content commits separate from code commits**. A content commit touches only `content/`
  and `curriculum/`.
- Never commit `private/`, `corpus/*`, `var/`, `workspace/` or `.env`. `sources/` **is**
  committed, but only after its scrub passes `make audit`.
- **ADRs:** any decision that changes ARCHITECTURE, adds a top-level dependency, or reverses an
  earlier ADR gets `docs/decisions/NNNN-kebab-title.md`:

```markdown
# NNNN. Title
Date: YYYY-MM-DD · Status: proposed | accepted | superseded by NNNN
## Context        — the forces at play, in 3–6 lines
## Decision       — what we do, in 1–3 lines
## Consequences   — what gets easier, what gets harder, what we'll watch
## Alternatives   — each rejected option + one-line reason
```

## 13. Quality gate

```text
make check    = lint (ruff check + ruff format --check)
              + types (mypy strict on src/ + lint-imports + content typing per package)
              + test (pytest: unit + integration)
              + verify-content (dogma content verify --all, torch group)
              + web-check (biome, tsc --noEmit, vitest, gen:api --check, production build)
              + audit (working tree and newly built assets; runs after all checks above)
make test-all = check + verify-variants (every problem's variants replayed through the grader)
              + Python e2e/slow suites (excluding llm) + Playwright browser smoke
```

`pre-commit` runs ruff, ruff-format, the privacy audit, a whitespace/large-file check and web lint.
Install it explicitly with `uv run --frozen pre-commit install`. Files
over 1 MB are blocked unless they're under `tests/fixtures/` with a reason. The full `make check`
must pass before every commit to `main`.

## 14. Documentation

- `README.md`: what it is, plus how to start CLI practice in ≤ 3 commands.
- `docs/DEVELOPMENT.md`: setup, configuration, development checks and troubleshooting.
- `docs/ARCHITECTURE.md` stays true. A PR that changes architecture updates it in the same commit.
- Every CLI command has `--help` text with one example.
- Templates in `templates/` are the documentation for content authors. Keep them annotated.

## 15. Definition of Done

**Feature:** gate green; tests for new behaviour (including the failure paths); docs updated
(ARCHITECTURE / ADR / `--help`); works offline; no new privacy surface (or the audit has been
extended to cover it); demonstrated end-to-end via CLI or browser.

**Content item:** checklist §5.2 complete; lesson follows §5.3; `content verify` green;
`status: verified`; cards and interview notes present; coverage report updated.

## 16. Rules for agents working in this repo

1. Read `CLAUDE.md`, then this file and ARCHITECTURE before your first edit in a session.
2. Run `git status` first. Preserve unrelated work. Never reset, stash or delete it.
3. **Never weaken a test, widen a tolerance, or edit `test_hidden.py` just to make a solution
   pass.** If a test is wrong, say so and fix it in a separate, explained commit.
   Check every test edit with `scripts/exercise_gate.py gate` (CONTENT_SPEC §6.8).
4. Never read or write `private/` unless the task explicitly requires it. Never send its
   contents to an external model.
5. Don't read outside the repo. Imported material lives in `sources/` and `corpus/`. If you need
   more, propose a manifest change.
6. Don't touch `var/` (learner state) except through the app's own commands.
7. Keep content ids stable. Never renumber. Retire instead of deleting.
8. Report outcomes truthfully: say what you ran, what passed, what failed, and what you skipped.
9. Respect the runway rule (ARCHITECTURE §7). When content is behind, author content before
   adding features.
10. **Stay inside the repo.** Never add a dependency on a path outside it (no `path =` deps on
    other projects, no reading `~/…` at runtime). The AI runtime is the only exception, through
    the gateway.
11. **Use `uv run --frozen`.** Never edit `pyproject.toml` or `uv.lock` unless your increment
    explicitly owns dependency changes. Parallel agents share one environment.
12. **Work from [ROADMAP.md](ROADMAP.md).** Touch only the paths your increment owns. Before
    finishing, update the increment's status and add a handoff entry: what you did, what you
    verified (commands and results), and what's left.
