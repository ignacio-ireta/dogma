# I-32a3c: the gate tool, part 3: `gate` and `bump-version`

**Owns:** `scripts/exercise_gate.py`, `tests/unit/scripts/test_exercise_gate.py`,
`tests/integration/runner/test_exercise_gate.py`.
**Needs:** I-32a3b committed.
**Commit message (coordinator):** `feat(scripts): add the test-edit gate (I-32a3c)`

`gate` answers one question about one problem: after this edit, do the tests still accept correct
work and reject wrong work, and was nothing weakened? Executors are told to get `GATE: PASS` and
nothing else, so every rule below must be exact.

## Read first

- `scripts/exercise_gate.py` as part 2 left it (whole file).
- `scripts/exercise_audit.py` lines 507 to 512 (`mutants_for`), 564 to 587 (`verdict`, `record`),
  768 to 775 (`_plain_bundle`), 823 to 879 (`cmd_seeds`: copy its pytest command and
  environment), 1100 to 1147 (`working_suite`, `base_suite`, `_git`, `_git_show`, `grade_suite`).
- `src/dogma/content/testlint.py`.

## Build

### 1. Command line

```text
gate ID --out DIR [--mode none|add|text|refactor] [--base REV] [--wrong FILE]... [--workers N]
bump-version ID --out DIR [--base REV]
```

Defaults: `--mode none`, `--base HEAD`, `--workers 10`. `--wrong` with a mode other than `add`
is exit 2. An unknown REV is exit 2.

Terms: the **new** suite is `audit.working_suite(problem)`. The **base** suite is
`audit.base_suite(problem, base)`. `rel` is the problem directory relative to `audit.REPO`.
`suffix` is `.py` or `.sql`. Base file text comes from `audit._git_show(base, f"{rel}/<name>")`.

### 2. Helpers (public, unit-tested)

```python
@dataclass(frozen=True)
class Check:
    name: str
    status: str        # "PASS" | "FAIL" | "SKIP"
    detail: str = ""

def normalise_tests(source: str) -> str
def duplicate_top_level_names(source: str) -> list[str]
def test_function_names(source: str) -> set[str]
def expected_version(problem, base: str) -> int
def run_gate(problem, *, mode: str, base: str, wrongs: Sequence[Path], out: Path, workers: int) -> list[Check]
```

**`normalise_tests`** returns `ast.dump` of the tree after these edits, and nothing else:

1. Drop every statement that is only a string (docstrings).
2. `assert`: remove the message.
3. Calls with last name `assert_allclose`, `assert_array_equal`, `assert_array_almost_equal`,
   `assert_almost_equal`, `assert_equal`, `assert_close`: drop keywords `err_msg` and `msg`.
4. Calls to `pytest.fail`: drop all arguments.
5. `raise AssertionError(...)`: drop the arguments of the `AssertionError` call.
6. Calls with last name `parametrize`: drop keyword `ids`.
7. Calls with last name `settings`: drop keywords `deadline` and `report_multiple_bugs`. A
   decorator that is then `settings()` with nothing in it is removed.
8. `from hypothesis import …`: drop the name `settings`; drop the statement if it is then empty.

**`duplicate_top_level_names`**: names bound more than once at module level by `def`, `class`
or a plain assignment, sorted.

**`test_function_names`**: names of top-level `def test_*`.

**`expected_version`**: let `start` be the tag `i32-start` when it exists and has this
problem's `meta.yaml`; otherwise `base`. Result: the `version` in `meta.yaml` at `start`, plus 1
if `test_function_names(new hidden)` has a name that `test_function_names(hidden at start)`
lacks.

### 3. The checks, in this order

If `frozen`, `manifest` or `reference` fails, every later check is printed as `SKIP`.

| Check | PASS when |
|---|---|
| `frozen` | `problem.md`, `solution<suffix>`, `starter<suffix>`, `hints.md` and `test_public.py` equal their base text. `meta.yaml`, parsed as YAML, equals the base after removing the `version` key from both. A problem that does not exist at `base` passes with detail `new problem` |
| `manifest` | No `variants/` directory, or `validate_variants(load_problem(problem.path))` is empty |
| `reference` | The reference graded on the new suite twice: both `passed`, same `(file, name, outcome)` list |
| `accepts` | Every accept row from `replay_problem` is OK. No manifest: PASS, detail `none` |
| `rejects` | Every reject row from `replay_problem` is OK. No manifest: PASS, detail `none` |
| `change` | See the mode table below |
| `wrong` | Mode `add` only, else `SKIP`. For each `--wrong` file: `audit.verdict` on the base suite is `survived`; on the new suite the status is `failed` with at least one failing hidden test; the learner message of the first failing hidden test has kind `named` and no leak. With no `--wrong`: PASS, detail `none given` |
| `version` | `meta.yaml` version equals `expected_version` |
| `lint` | Modes `text` and `refactor`: `lint_hidden_tests(new hidden)` is empty. Modes `none` and `add`: for each rule, the new count is not above the base count |
| `names` | Let leaks be `id_leaks` of the hidden test names in the reference's report. Modes `text` and `refactor`: none. Mode `add`: no leak that the base-suite report does not also have. Mode `none`: PASS |
| `seeds` | For seeds 1 to 5 the reference passes the new suite under plain pytest (the command and environment of `cmd_seeds`). A failing seed is forgiven if the base suite fails the same seed. Mode `none`: PASS, with failing seeds in the detail |
| `time` | The reference's `duration_ms` on the new suite is at most 10,000; or at most 1.25 times its duration on the base suite plus 1,000. Mode `none`: PASS, with the time in the detail |

A **leak** is a match of `array\(|tensor\(|\d+\.\d{2,}|\d[eE][+-]?\d` in the message.

The base-suite run of the reference (`audit.grade_suite(problem, solution text, base suite,
settings)`) is needed in modes `add`, `text` and `refactor`. Do it once and reuse it.

`change`, by mode, on `test_hidden.py`:

| Mode | PASS when |
|---|---|
| `none` | The new text equals the base text |
| `add` | `git diff --numstat <base> -- <rel>/test_hidden.py` shows 0 deleted lines; `duplicate_top_level_names(new)` is empty; every hidden test name in the base-suite reference report is in the new-suite reference report |
| `text` | `normalise_tests(base) == normalise_tests(new)` |
| `refactor` | `test_function_names(base)` is a subset of `test_function_names(new)`; and no mutant is weakened (below) |

**Weakened mutants** (mode `refactor`): for each mutant of
`audit.mutants_for(problem, audit.MAX_MUTANTS)`, take `audit.verdict` on the base suite and on
the new suite (run the pairs through `audit._run_all`). A mutant is weakened when the base
verdict is not `survived` and the new one is. If the base verdict was `timeout`, `oom` or
`error`, grade that mutant again on both suites, alone, before deciding. PASS when none is
weakened. Detail: `<k> mutants, <w> weakened`, then up to three `change` texts.

### 4. Output and record

```text
GATE <id>  mode=<mode>  base=<rev>
frozen     PASS
manifest   PASS
reference  PASS  public 3/3 hidden 6/6, twice
accepts    PASS  1 of 1
rejects    PASS  2 of 2
change     PASS  +9 -0 lines, 1 new test
wrong      PASS  reject_accepts_bools.py: test_bool_raises_type_error
version    PASS  2
lint       PASS  0 findings
names      PASS
seeds      PASS  5 of 5
time       PASS  1.2 s (base 1.1 s)
GATE: PASS
```

Append one row to `<OUT>/gate.jsonl`: time, problem, mode, the commit id of `base`
(`git rev-parse`), `PASS` or `FAIL`, the checks, and `audit._text_sha` of `test_hidden.py`,
`test_public.py`, the solution and the manifest. Exit 0 on PASS, 1 on FAIL.

### 5. `bump-version ID`

Compute `expected_version`. If `meta.yaml` differs, replace the number on its `version:` line and
keep everything else on that line (some lines carry a comment): one substitution of
`^(version:\s*)\d+` in multiline mode, written with `atomic_write`. If a manifest exists, set its
`problem_version` to the same value with `write_manifest`. Print `version: <old> -> <new>` or
`version: unchanged (<n>)`.

## Tests to write

**Unit.** Extend the fake grader so that results depend on the tests it is given. A fake that
works: read the test function names of both files with `ast`; every test passes, except: if the
hidden text contains `BREAKREF`, the first hidden test fails for every source; if the source
contains `BUG` and the hidden text contains `CATCH`, the hidden test named `test_catches_bug`
fails with message `the sum is wrong for bools`. Patch the seed runner with a stub.

1. `normalise_tests` gives the **same** result before and after each of: adding an assert
   message; adding or rewording `err_msg`; adding `ids=`; adding `deadline=None` and
   `report_multiple_bugs=False` to an existing `settings`; adding a whole
   `@settings(deadline=None, report_multiple_bugs=False)` with its import; rewording
   `pytest.fail("…")`; rewording `raise AssertionError("…")`; editing a docstring or a comment.
2. `normalise_tests` gives a **different** result after each of: changing `rtol`; changing
   `max_examples`; removing an `assert`; changing a compared value; renaming a test; changing a
   `parametrize` value; adding `@pytest.mark.skip`; replacing `pytest.raises` with a helper call.
3. `duplicate_top_level_names` and `test_function_names` on small sources.
4. `expected_version`: with and without the tag; with and without an added test.
5. `bump-version` keeps a trailing comment on the line and updates the manifest.
6. `gate` on the fixture, one test per outcome: PASS in mode `none`; then one FAIL for each
   check, caused by one small change: `solution.py` edited (`frozen`); stale manifest version
   (`manifest`); `BREAKREF` (`reference`, and later checks print `SKIP`); an accept with `BUG`
   and `CATCH` present (`accepts`); a reject with no marker (`rejects`); mode `add` after
   deleting a line, and after appending a second `def` with an existing name (`change`); mode
   `text` after changing a compared value (`change`); `--wrong` file without `BUG` (`wrong`); a
   test added without `bump-version` (`version`); mode `add` with a new message-less assert
   (`lint`); mode `add` with a new `parametrize` whose ids hold digits (`names`).
7. `--wrong` with `--mode none` is exit 2. `gate.jsonl` gains one row per run.

**Integration** (real grader, same file as part 2):

1. Fixture as written: `gate --mode none` is `GATE: PASS`.
2. Append this test to the fixture copy's `test_hidden.py`, write a candidate that keeps the
   guard but drops the `bool` check, run `bump-version`, then
   `gate --mode add --wrong <candidate>` is `GATE: PASS`:

```python
def test_bool_raises_type_error():
    _raises_type_error(lambda: add(True, 1), "a bool is not a number here: raise TypeError")
```

3. From there, delete `test_mixed_int_and_float`: `gate --mode add` fails at `change`.
4. On a fresh copy, reword one assert message: `gate --mode text` passes. Change `-10` to `-11`
   in a test: it fails at `reference`.

## Accept

```bash
uv run --frozen pytest tests/unit/scripts/test_exercise_gate.py -q
uv run --frozen pytest -m integration tests/integration/runner/test_exercise_gate.py -q
uv run --frozen ruff check scripts/exercise_gate.py tests/unit/scripts tests/integration/runner
uv run --frozen ruff format --check scripts/exercise_gate.py tests/unit/scripts tests/integration/runner
uv run --frozen pytest -q
uv run --frozen --group torch python scripts/exercise_gate.py gate --out var/audit-i32/a3c cml/vector-operations/p01-vector-projection --mode none
uv run --frozen --group torch python scripts/exercise_gate.py gate --out var/audit-i32/a3c sql/sql-group-by-and-having/p01-customers-with-three-completed-orders --mode none
uv run --frozen --group torch python scripts/exercise_gate.py gate --out var/audit-i32/a3c dl/layer-normalization/p01-layer-norm-forward --mode none
git status --short
```

Expected: tests pass; the three real problems end `GATE: PASS`; `git status --short` shows no
path under `content/`. If a real problem fails in mode `none`, do not change the problem. Put
the output in your report: either the check is wrong or the finding is real, and the coordinator
decides.

## Traps

- Mode `none` must never fail a problem for something that was already there (old lint
  findings, a slow run, a flaky seed). Only the modes that describe an edit are strict.
- Compare `meta.yaml` as parsed YAML, not as text. Comments and spacing differ.
- `git diff --numstat` prints `added<TAB>deleted<TAB>path`. An unchanged file prints nothing.
- Do not let `gate` write to `content/`. It reads, grades and reports.
- Keep each check in its own small function that returns a `Check`. One long function will
  fail the `PL` lint rules and will be hard to test.
