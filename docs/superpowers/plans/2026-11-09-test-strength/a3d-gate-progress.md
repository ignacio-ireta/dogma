# I-32a3d: the gate tool, part 4: `add-test`, `mark-swept`, `worklist`, `scorecard`

**Owns:** `scripts/exercise_gate.py`, `tests/unit/scripts/test_exercise_gate.py`,
`tests/integration/runner/test_gate_tool.py`.
**Needs:** I-32a3c committed.
**Commit message (coordinator):** `feat(scripts): add tests as one transaction and track sweeps from the tree (I-32a3d)`

`add-test` turns "append a test, format, bump the version, run the gate, record the wrong
solution" into one command that either does all of it or changes nothing. The other three
commands let a sweep stop and resume with no state file: what is done is written in each
problem's manifest, and what is left is computed from the tree.

## Read first

- `scripts/exercise_gate.py` as part 3 left it (whole file).
- `src/dogma/content/schemas.py` lines 98 to 121 (`PathWeek`, `CorePath`).
- `src/dogma/content/loader.py`: `load_core_path`.
- `curriculum/paths/core.yaml` lines 1 to 30 (its shape).

## Build

### 1. `add-test ID TESTFILE --wrong FILE --mistake TEXT --clause LABEL [--base REV]`

TESTFILE holds one new hidden test, written by the caller outside `content/`. FILE is the wrong
solution it must catch, named `reject_<slug>.py` (or `.sql`).

Refuse before changing anything (`GateError`) when:

- the manifest has no accept with `origin: blind`
  (`no blind accept: record the gap with a no-witness note, add no test`);
- TESTFILE does not parse, or does not define exactly one top-level `def test_*`;
- TESTFILE defines a top-level name that `test_hidden.py` already defines;
- TESTFILE has more than 40 lines, or contains `pytest.raises`, `skip` or `xfail`;
- `lint_hidden_tests(TESTFILE text)` reports a finding (print the findings);
- `--mistake` fails `plain_words`, or LABEL is not a clause.

Then, as one transaction:

1. Keep the current bytes of `test_hidden.py`, `meta.yaml` and `variants/manifest.yaml`.
2. Append TESTFILE's text to `test_hidden.py` (after a blank line). Run `ruff format`, then
   `ruff check`, on `test_hidden.py`, as `install_variant` does for one file.
3. Do what `bump-version` does.
4. `run_gate(problem, mode="add", base=base, wrongs=[FILE], …)` and print its output.
5. On `GATE: PASS`: do what `add-reject` does for FILE with the given mistake and label. The
   new test must be among the failing functions; record it as `killed_by`. Print
   `ADD-TEST: added <test name> and <reject file name>`.
6. On `GATE: FAIL`, or on any error in steps 2 to 5: write the three kept files back exactly
   (delete the manifest if there was none), delete any variant file this call installed, print
   `ADD-TEST: FAILED, nothing changed`, exit 1.

### 2. `mark-swept ID --kind reject|mutants --mode none|add [--base REV]`

Run `run_gate` with the given mode and base, and print its output. On `GATE: PASS`, add the kind
to the manifest's `sweeps` (create the manifest if the problem has none) with `write_manifest`,
and print `marked <kind>`. On `GATE: FAIL`, write nothing and exit 1. A kind that is already
there is not added twice.

### 3. `worklist --kind hygiene|blind|reject|mutants [--today YYYY-MM-DD]`

A problem is **left** for a kind when:

| Kind | Left when |
|---|---|
| `hygiene` | `lint_hidden_tests` reports a finding for its `test_hidden.py` |
| `blind` | Its manifest has no accept with `origin: blind` (or it has no manifest) |
| `reject` | Its manifest's `sweeps` lacks `reject` (or it has no manifest) |
| `mutants` | Its manifest's `sweeps` lacks `mutants` (or it has no manifest) |

Order. Load the path with `load_core_path(Settings().curriculum_dir / "paths" / "core.yaml")`.
`today` is `--today` or the system date. The **current week** is the last week whose `starts` is
on or before `today` (week 1 if `today` is earlier than every start). Weeks are listed in this
order: current + 1 up to the last week, then the current week, then current − 1 down to 1.
Inside a week: topics in path order, problems by directory name. Problems whose topic is not on
the path come last under `not on the path` (four topics on 2026-11-09).

Annotate a problem with the count of its `blocked` and `question` notes in any
`var/audit-i32/*/ledger.jsonl`. Output:

```text
WORKLIST reject  today=2026-11-20  current week=4
week 5
  cml/k-fold-cross-validation/p01-kfold-indices
  cml/k-fold-cross-validation/p02-cross-validate  (blocked 1)
week 6
  ...
not on the path
  cml/lagrange-multipliers/p01-constrained-quadratic-min
total: 212
```

An empty list prints the header and `total: 0`. Exit 0 always.

Public helper: `week_order(weeks: Sequence[PathWeek], today: date) -> list[int]`.

### 4. `scorecard`

One Markdown table row per problem, in id order:

```text
| Problem | Version | Lint | Blind | Accepts | Rejects | Bullets | Sweeps | Notes |
```

- `Lint`: number of lint findings.
- `Blind`: `yes` or `no`.
- `Accepts`, `Rejects`: counts from the manifest.
- `Bullets`: `<covered>/<total>` over the Contract clauses `C1…` from `contract_clauses`. A
  clause is covered when `squash` of some reject or uncovered `clause` is a substring of it.
- `Sweeps`: the manifest's `sweeps`, joined by a comma, or `-`.
- `Notes`: counts by kind from every `var/audit-i32/*/ledger.jsonl`, such as `blocked 1`, or `-`.

Then a summary block: problems, problems with a blind accept, total accepts, total rejects,
problems with every bullet covered, problems with both sweeps, total lint findings.

Public helper: `bullet_coverage(problem_md: str, manifest: VariantsManifest | None) -> tuple[int, int]`.

## Tests to write

Unit, with the fake grader of part 3:

1. `add-test` success: the test is appended, the version is one higher, the manifest gains the
   reject with `killed_by` equal to the new test.
2. `add-test` refusals that change no file (compare bytes before and after): no blind accept;
   two tests in TESTFILE; a name that already exists; 41 lines; a `pytest.raises`; an assert
   without a message; a bad label.
3. `add-test` rollback: the gate fails because the wrong file still passes (no `CATCH` in the
   new test); because the reference fails (`BREAKREF` in the new test); because ruff rejects the
   appended text. In each case `test_hidden.py`, `meta.yaml` and the manifest are byte-identical
   to before, and no new file is in `variants/`.
4. Two successful `add-test` calls in a row bump the version once in total.
5. `week_order`: `today` before the path, inside week 4 of 13, and after the last week.
6. `worklist` on a small repo with three problems and a two-week path: each kind lists the right
   problems, in order; a problem off the path comes last; a `blocked` note shows as
   `(blocked 1)`; an empty list prints `total: 0`.
7. `mark-swept`: marks on PASS; marks nothing on FAIL; creates a manifest when there is none;
   does not duplicate a kind.
8. `bullet_coverage`: a reject clause equal to bullet 2 covers bullet 2 only; an `uncovered`
   entry counts as covered; `C0` text is not counted.
9. `scorecard` prints one row per problem and the summary block.

Integration (real grader), on the fixture copy: adopt a blind accept (a correct rewrite, added
with `add-accept --origin blind`), then `add-test` with the `bool` test and the candidate from
part 3 succeeds; a second `add-test` whose test asserts `add(1, 1) == 3` fails and leaves the
three files byte-identical.

Patch `load_core_path` or pass a small `core.yaml` in the temporary repo. Do not depend on the
real path file or on today's date.

## Accept

```bash
uv run --frozen pytest tests/unit/scripts/test_exercise_gate.py -q
uv run --frozen pytest -m integration tests/integration/runner/test_gate_tool.py -q
uv run --frozen ruff check scripts/exercise_gate.py tests/unit/scripts tests/integration/runner
uv run --frozen ruff format --check scripts/exercise_gate.py tests/unit/scripts tests/integration/runner
uv run --frozen pytest -q
uv run --frozen --group torch python scripts/exercise_gate.py worklist --out var/audit-i32/a3d --kind hygiene
uv run --frozen --group torch python scripts/exercise_gate.py worklist --out var/audit-i32/a3d --kind blind
uv run --frozen --group torch python scripts/exercise_gate.py scorecard --out var/audit-i32/a3d
git status --short
```

Expected: tests pass; the `hygiene` total equals the number of problems in the `lint` summary
from part 1; the `blind` total equals the number of problems (251 on 2026-11-09, when no problem
has variants yet); `scorecard` prints one row per problem; no path under `content/` changed.

## Traps

- The rollback must restore bytes, not re-derive them. Read the three files as bytes first and
  write those bytes back.
- Put steps 2 to 5 of `add-test` in one `try` block, with the restore in `except BaseException`
  (then re-raise anything that is not a `GateError`).
- Do not store progress anywhere except the manifest. No `done.json`, no cached worklist.
- Do not read `var/dogma.db` or call any `dogma` command that does. The week comes from the
  path file and the date.
