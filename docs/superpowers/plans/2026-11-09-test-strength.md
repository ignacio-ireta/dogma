# Plan: I-32 Test strength: a design method, committed proof and a gate for every problem's tests

*Master plan, 2026-11-09. Nothing here is implemented yet. The design is
[2026-11-09-test-strategy-design.md](../specs/2026-11-09-test-strategy-design.md). Each step has
its own runbook in [2026-11-09-test-strength/](2026-11-09-test-strength/). An executor reads the
protocol and its own runbook, and nothing else from that folder.*

## Context

The tests are sound on the basics (I-28: 98.6% of non-equivalent mutants killed, 251/251 blind
solves) and I-31 is adding real data. Four things are missing:

- **The proof is not kept.** Blind solutions, wrong solutions and triage verdicts live in
  git-ignored `var/`. The I-28 set is gone. A later test edit can only be re-checked by a full
  audit.
- **Only syntactic mistakes are measured.** Mutants are one-token edits of the reference.
  Conceptual mistakes are covered only where I-31 hand-wrote them. One check made while writing
  this plan shows the cost: a solution to the example problem
  (`cml/vector-operations/p01-vector-projection`) that divides `b` in place passes all 17 tests,
  although the Contract says inputs must not change. The test for it uses a `b` whose largest
  entry is 1, so the division changes nothing. I-28 scored that problem as having no gap.
- **Known rule breaks have no gate.** 214 sites in 73 problems break CONTENT_SPEC §6.4 in hidden
  tests, and nothing stops new ones.
- **The method is scattered** over several documents and session memory.

I-32 adds one design method, one gate command, and proof committed inside each problem package.

## Decisions

Taken by the owner on 2026-11-09:

1. Proof lives in each problem package, under `variants/`.
2. Sonnet-class models run everything, the coordinator included. Judgement calls go to a
   decisions file for a person.
3. Every problem is swept, not a sample.

Defaults chosen by the planner. Change them here before execution starts:

4. At most one `meta.yaml` version bump per problem for the whole program. An edit that changes
   only messages, ids or Hypothesis reporting flags does not bump.
5. A problem with no passing blind solution gets its gaps recorded and no new hidden test.
6. A hidden test that is stricter than its statement is never edited by an executor. It goes to
   the decisions file.
7. Executors never edit `problem.md`, `solution.*`, `starter.*`, `hints.md` or `test_public.py`.
8. `meta.yaml` tolerances that differ from what the tests enforce (I-28 open finding) are
   reported at close-out, not changed.
9. Sweeps start after I-31c is `done`. I-31b is rewriting 51 hidden test files now.
10. Sweep order follows the authored calendar in `curriculum/paths/core.yaml`: the weeks after
    today's week first, then today's week, then earlier weeks, newest first. The tool computes
    it from the file and the date, and never opens the learner database.
11. `status` is not changed by this program. A new hidden test is proven by the gate (reference,
    blind solution, wrong solution), as in I-28.
12. New decision record: ADR-0022, or the next free number (written by I-32a4). ADR-0021 was
    taken on 2026-11-10 by the runtime import rule.

## Roles

| Role | Runs | Reads |
|---|---|---|
| **Coordinator** | One session. Sets up the worktree, dispatches work orders, re-runs gates, commits, writes the handoff | [00-protocol](2026-11-09-test-strength/00-protocol.md), [01-coordinator](2026-11-09-test-strength/01-coordinator.md), this file |
| **Platform executor** | One runbook `a*`, or Part A of `e` | Protocol, its runbook |
| **Hygiene executor** | Up to 5 problems | Protocol, [b-hygiene](2026-11-09-test-strength/b-hygiene.md) |
| **Blind solver** | Up to 5 problems, from bundles only | [c1-blind-solver](2026-11-09-test-strength/c1-blind-solver.md) and nothing else |
| **Variant author** | One topic | Protocol, [d-reject-sweep](2026-11-09-test-strength/d-reject-sweep.md) or [d2-mutant-triage](2026-11-09-test-strength/d2-mutant-triage.md), one section of the [catalogue](2026-11-09-test-strength/catalogue.md) |

## Increments

| ID | Runbook | Depends on | Owns | Done when |
|---|---|---|---|---|
| I-32a1 | [a1-testlint](2026-11-09-test-strength/a1-testlint.md) | I-31a `done` | `src/dogma/content/testlint.py`, `tests/unit/content/test_testlint.py` | Four rules, each with a failing and a passing test; counts on real content match the baseline |
| I-32a2 | [a2-variants-schema](2026-11-09-test-strength/a2-variants-schema.md) | I-31a `done` | `src/dogma/content/{schemas,loader,validate}.py`, `tests/unit/content/test_variants.py`, `tests/fixtures/content/packages/` | Manifest schema, loader fields and `VARIANT_*` issues, each with a failing test; real content still validates |
| I-32a3 | [a3a](2026-11-09-test-strength/a3a-gate-readonly.md), [a3b](2026-11-09-test-strength/a3b-gate-writers.md), [a3c](2026-11-09-test-strength/a3c-gate-checks.md), [a3d](2026-11-09-test-strength/a3d-gate-progress.md), in that order | a1, a2 | `scripts/exercise_gate.py`, `tests/unit/scripts/test_exercise_gate.py`, `tests/fixtures/problems/py-variants/`, `tests/integration/runner/test_exercise_gate.py` | Every command in the table below works on the fixture problem through the real grader |
| I-32a4 | [a4-docs-and-example](2026-11-09-test-strength/a4-docs-and-example.md) | a3 | `docs/CONTENT_SPEC.md`, `docs/ENGINEERING_STANDARDS.md`, `docs/ARCHITECTURE.md`, `docs/decisions/0022-*.md`, `templates/topic/problems/p01-todo/`, `Makefile`, `content/cml/vector-operations/` | Docs state the rules; the example problem carries variants and passes `gate` |
| I-32b | [b-hygiene](2026-11-09-test-strength/b-hygiene.md) | a, I-31c `done` | Topic directories, `test_hidden.py` only | `lint` reports nothing, or every remaining site has a `question` note; every edit passed `gate` in mode `text` or `refactor` |
| I-32c | [c-blind-accepts](2026-11-09-test-strength/c-blind-accepts.md) | a, I-31c `done` | `variants/` of every problem | `worklist --kind blind` lists only problems that have a note |
| I-32d | [d-reject-sweep](2026-11-09-test-strength/d-reject-sweep.md), then [d2-mutant-triage](2026-11-09-test-strength/d2-mutant-triage.md) | b, c | Topic directories: `variants/`, `test_hidden.py` (additions), `meta.yaml` (version) | `worklist --kind reject` and `--kind mutants` list only problems that have a note |
| I-32e | [e-closeout](2026-11-09-test-strength/e-closeout.md) | all | `src/dogma/content/validate.py`, `src/dogma/cli/verify.py`, their tests and fixtures, `docs/CONTENT_SPEC.md`, `docs/audits/`, `docs/ROADMAP.md` | Rules enforced in `make check` where their condition holds; report committed; `make test-all` passes |

a1 and a2 may run at the same time. Everything else in I-32a is sequential. Each content sweep
starts with a pilot on five problems, one per kind, before the fan-out:

```text
cml/vector-operations/p01-vector-projection                              py-numpy, function
sql/sql-group-by-and-having/p01-customers-with-three-completed-orders    sql-duckdb
algo/lru-cache/p01-lru-cache                                             py-stdlib, class entry
dl/layer-normalization/p01-layer-norm-forward                            py-torch
sql/pandas-groupby-aggregate-and-transform/p01-customer-order-features   py-data
```

If a pilot executor stops on something the runbook does not cover, the coordinator records it in
the decisions file and stops the sweep. A person fixes the runbook. Executors and the
coordinator never edit a runbook.

## The tool: `scripts/exercise_gate.py` (created by I-32a3)

Every command is run as
`uv run --frozen --group torch python scripts/exercise_gate.py <command> --out <OUT> …`.

| Command | Does | From |
|---|---|---|
| `lint [--only P]` | Lists lint findings in hidden tests; exit 1 if any | a3a |
| `clauses ID` | Prints the Contract and Constraints bullets with labels (`C1`, `K2`). Other commands take a label, never typed clause text | a3a |
| `probe ID FILE` | Grades one file against the working-tree tests and prints each failing test with the message the learner would see | a3a |
| `note ID --kind K --text T` | Appends a row to `<OUT>/ledger.jsonl` | a3a |
| `ledger [--all] [--kind K]` | Prints ledger rows as Markdown: of `<OUT>`, or with `--all` of every `var/audit-i32/*/` | a3a |
| `add-accept`, `add-reject`, `add-uncovered`, `remove-variant` | The only way `variants/manifest.yaml` is written. Each grades first and records what it saw, or changes nothing | a3b |
| `adopt-blind [--only P]` | Copies passing blind solutions into `variants/accept_blind.*` | a3b |
| `variants [--only P]` | Replays every manifest through the grader; exit 1 on any mismatch | a3b |
| `gate ID --mode M [--wrong FILE]` | Checks one problem after an edit. Read-only | a3c |
| `bump-version ID` | Sets the version to its value at tag `i32-start`, plus one if a test was added since | a3c |
| `add-test ID TESTFILE --wrong FILE …` | Appends one new hidden test, bumps the version, runs `gate` and records the wrong solution, as one transaction that either completes or restores every file | a3d |
| `mark-swept ID --kind S --mode M` | Runs `gate`; on PASS records the sweep in the manifest | a3d |
| `worklist --kind K` | Prints what is left for a sweep, in order, computed from the tree | a3d |
| `scorecard` | Prints the per-problem table for the close-out report | a3d |

`gate` prints one line per check and ends with `GATE: PASS` or `GATE: FAIL`:

| Check | Passes when |
|---|---|
| `frozen` | `problem.md`, `solution.*`, `starter.*`, `hints.md` and `test_public.py` equal the base; `meta.yaml` differs at most in `version` |
| `manifest` | The manifest has no static issue |
| `reference` | The reference passes the working-tree tests twice with identical outcomes |
| `accepts` | Every accept variant passes |
| `rejects` | Every reject variant fails, and its `killed_by` test is among the failures |
| `change` | The edit to `test_hidden.py` obeys `--mode`: `none` (no change), `add` (no deleted line, no top-level name defined twice, every old test still collected), `text` (same AST after dropping messages, `ids=` and the two Hypothesis flags), `refactor` (every sampled mutant the old tests killed is still killed) |
| `wrong` | Each `--wrong FILE` passed the old tests and fails the new ones, with a message that names a mistake and holds no array or float text |
| `version` | Version equals its value at `i32-start`, plus one if a test function was added since |
| `lint` | Modes `text` and `refactor`: no finding. Modes `none` and `add`: no more findings than the base |
| `names` | No new digit inside `[...]` in a hidden test name, dtype names excepted |
| `seeds` | The reference passes under five other seeds, or fails only where the old tests also fail |
| `time` | The run takes under 10 s, or no more than the old tests took plus a quarter |

## Working conventions

- **One branch, one worktree:** `i32-test-strength` at `.claude/worktrees/i32`, created from
  `main` by the coordinator. Executors work in it on disjoint paths and never commit. The
  coordinator commits one topic (or one platform runbook) at a time.
- **Results directory:** `var/audit-i32/<work order>/` inside the worktree. Git-ignored. Anything
  that must outlive the session goes into `variants/`, the handoff log or the decisions file.
- **Decisions file:** `docs/audits/test-strength-decisions.md`, tracked, written by the
  coordinator from the ledgers at the end of each wave.
- **Tag `i32-start`:** set by the coordinator on the commit where I-32a4 is committed. Version
  checks compare against it.
- **Commits:** content commits touch `content/` only (`content(<track>): … (I-32d)`). Code and
  docs commits are separate.
- **No state files.** What is done is in each manifest (`sweeps`) and in git. What is left is
  whatever `worklist` prints.

## Verification

```bash
# after each platform runbook: the runbook's own Accept block, then
uv run --frozen pytest -q
uv run --frozen ruff check . && uv run --frozen ruff format --check .
uv run --frozen mypy && uv run --frozen lint-imports
# after each work order (coordinator, per problem)
uv run --frozen --group torch python scripts/exercise_gate.py gate --out var/audit-i32/coord <id> --mode <mode>
uv run --frozen --group torch dogma content verify <topic id>
# after each wave, with nothing uncommitted in the worktree
make check
# before each merge to main, and at close-out
make test-all
```

Report the commands that were actually run, with exit codes, in the ROADMAP handoff entry.

## Size

Estimates, to be replaced by the pilot's measurements: I-32a about 7 agent runs. I-32b about 15
(73 problems, 5 per run). I-32c about 50 (251 problems, 5 per run). I-32d about 170 (one per
topic) and I-32d2 about 170 shorter ones. One grader run takes about a second, so the full replay
of roughly 2,000 variants takes a few minutes on ten workers.

## Not in scope

- Changing the grading pipeline, the sandbox or what the learner is shown.
- Lowering `max_examples`, widening a tolerance, or deleting a test. If a test is wrong, it is
  reported with evidence and a person decides (ENGINEERING_STANDARDS §16.3).
- Using variants in the tutor. They would fit ("your failure matches this known mistake"), but
  that is a feature for another increment.
- Public tests. The lint and the sweeps cover hidden tests only.
