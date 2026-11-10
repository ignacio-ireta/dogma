# I-32a3a: the gate tool, part 1: skeleton and read-only commands

**Owns:** `scripts/exercise_gate.py` (new), `tests/unit/scripts/test_exercise_gate.py` (new),
`tests/fixtures/problems/py-variants/` (new).
**Needs:** I-32a1 committed (`dogma.content.testlint`).
**Commit message (coordinator):** `feat(scripts): add exercise_gate with lint, clauses, probe and notes (I-32a3a)`

## Read first

- `scripts/exercise_audit.py` lines 1 to 132 (docstring, `Problem`, `discover`), 528 to 610
  (`grade_source`, `classify_message`, `verdict`, `record`, `Log`, `read_jsonl`) and 1408 to 1459
  (`main`). You reuse these. Do not edit that file.
- `src/dogma/grading/display.py` (102 lines): `learner_message`.
- `src/dogma/content/markdown.py` (92 lines): `parse_outline`.
- `tests/unit/scripts/test_exercise_audit.py` lines 1 to 32 (how a script is loaded in a test) and
  260 to 345 (`fake_grader`, `demo`, `_git`: the pattern for your tests).

## Build

### 1. Skeleton

`scripts/exercise_gate.py` starts like this, so that it can reuse the audit tool:

```python
sys.path.insert(0, str(Path(__file__).resolve().parent))
import exercise_audit as audit  # noqa: E402
```

- Always call the audit tool through the module (`audit.grade_source(...)`,
  `audit.discover(...)`, `audit.REPO`). Tests replace these attributes. A `from exercise_audit
  import grade_source` would escape them.
- Use `argparse` with sub-commands. Every sub-command takes `--out DIR` (required; create it).
- `main(argv: Sequence[str] | None = None) -> int`. Exit codes: 0 success, 1 a check failed,
  2 wrong usage.
- `find_problem(problem_id)`: look the id up in `audit.discover(audit.REPO, [])`. If it is
  unknown, print `no problem <id>` and up to three close ids from `difflib.get_close_matches` to
  stderr, and return exit code 2.

### 2. `lint [--only PREFIX]...`

For every problem (filtered by `--only` as the audit tool does), run
`dogma.content.testlint.lint_hidden_tests` on its `test_hidden.py`. Print one line per finding,
tab-separated: `problem id`, `rule`, `line`, `message`. Last line:
`lint: <n> findings in <m> problems`. Write the findings to `<OUT>/lint.jsonl`, replacing the
file. Exit 1 if `n > 0`.

### 3. `clauses ID`

Public helper, reused by later parts:

```python
def squash(text: str) -> str            # " ".join(text.split())
def contract_clauses(problem_md: str) -> list[tuple[str, str]]   # (label, text)
```

Take the `Contract` and `Constraints` sections from `parse_outline`. In each section a new
clause starts at every line that begins with `- ` or `* ` at column 0. Lines up to the next such
line belong to it (this keeps nested bullets with their parent). Non-empty text before the first
bullet is clause 0. Strip the leading bullet mark and `squash` the rest. Labels: `C1`, `C2`, … for
Contract (`C0` for text before the first bullet), `K1`, … for Constraints. Skip empty clauses.

Print `CLAUSES <id>`, then one line per clause, `<label>  <text>`, then
`Give a label (C2, K1) when a command asks for --clause.`

Later commands take a clause as a **label**, never as typed text, because statements are full of
backticks and `$`, which a shell mangles. Add the helper they will use:

```python
def clause_text(problem_md: str, label: str) -> str    # GateError "no clause <label>: run clauses" if absent
```

Also add one check used by every free-text option in this tool (`--text` here; `--mistake`,
`--reason`, `--freedom` and `--message-has` later):

```python
def plain_words(value: str) -> str    # returns value, or raises GateError
```

It accepts only text that fully matches `[A-Za-z0-9 ,.;:()/+=<>_'-]+`. Anything else is refused
with `use plain words: no backticks, quotes, dollar signs or backslashes`.

### 4. `probe ID FILE`

Grade the text of FILE with `audit.grade_source(problem, source, Settings())` and print:

```text
PROBE <id>  file=<FILE>
status: <status>   public <passed>/<total>   hidden <passed>/<total>
FAILED <public|hidden> <test name>
    kind=<audit.classify_message(...)>  message: <first line of learner_message(test)>
VERDICT: <verdict>
```

One `FAILED` block per test whose outcome is `fail` or `error`. Public helper:

```python
def probe_verdict(report: GradeReport) -> str
```

| Report | Verdict text |
|---|---|
| status `rejected` | `broken (rejected: <each finding message, joined by "; ">)` |
| status `timeout`, `oom` or `error` | `infra (<status>)` |
| status `passed` | `survived` |
| status `failed` and no test passed | `broken (no test passed)` |
| status `failed` and at least one test passed | `killed` |

After `VERDICT: killed`, print
`killed_by: <function name> (<public|hidden>)`: the first failing public test if any public test
failed, else the first failing hidden test. The function name is the test name up to the first
`[`. Exit 0 for every verdict; 2 if FILE does not exist.

### 5. `note ID --kind KIND --text TEXT`

Kinds: `blocked`, `question`, `unpromised`, `no-witness`, `overreach`, `reference-doubt`,
`message-defect`, `inspect`, `done`. Append one JSON line to `<OUT>/ledger.jsonl`:
`{"ts": <UTC ISO-8601>, "problem": ID, "kind": KIND, "text": TEXT}`. Print `noted: <kind> <id>`.
TEXT goes through `plain_words`. An unknown kind or problem is exit 2. Expose
`append_note(out, problem_id, kind, text)` for later parts (it does not apply `plain_words`, so
the tool's own notes may hold any text).

`GateError` is a plain `Exception` subclass. `main` prints its message to stderr and returns 1.

### 6. `ledger [--all] [--kind KIND]...`

Read `<OUT>/ledger.jsonl`, or with `--all` every `var/audit-i32/*/ledger.jsonl` under
`audit.REPO`. Filter by kinds if given. Print Markdown: a `### <kind>` heading per kind, then
`` - `<problem>`: <text> `` per row in file order. Last line: `ledger: <n> rows`.

### 7. Fixture problem

Create `tests/fixtures/problems/py-variants/` with exactly these files. Later parts use it too.
Its tests leave one gap on purpose: nothing checks that a bool is refused.

`meta.yaml`

```yaml
id: fixtures/py-variants/p01-add
version: 1
title: Add two numbers
language: python
runtime: py-stdlib
entry: add
difficulty: easy
minutes: 5
core: true
tolerance: {atol: 0.0, rtol: 0.0}
banned_imports: []
timeout_s: 30
status: draft
```

`problem.md`. Write it exactly as below, with one change: in the real file, put the four lines
under `## Examples` inside a fenced block (a line of three backticks followed by `python`, the
four lines without their indent, a line of three backticks).

```text
# Add two numbers

Implement `add(a, b)`, which returns the sum of two numbers.

## Contract

- `a`, `b`: ints or floats. A bool is not a number here.
- Returns: `a + b`. Sums of ints are exact, whatever their size.
- Raises `TypeError` when an argument is not an int or a float.

## Examples

    >>> add(2, 3)
    5
    >>> add(0.5, 0.25)
    0.75

## Constraints

- Allowed: the standard library.
- Not allowed: none.

## Done looks like

All public tests pass, then the hidden tests pass. They check other values, large ints and
arguments of the wrong type.
```

`solution.py`

```python
from __future__ import annotations


def add(a: int | float, b: int | float) -> int | float:
    """Return a + b, refusing anything that is not an int or a float."""
    for value in (a, b):
        if isinstance(value, bool) or not isinstance(value, int | float):
            raise TypeError("add takes ints or floats")
    return a + b
```

`starter.py`: the same signature and docstring, body `raise NotImplementedError`.

`test_public.py`

```python
from submission import add


def test_example_adds_two_ints():
    assert add(2, 3) == 5, "add(2, 3) should be 5"


def test_example_adds_two_floats():
    assert add(0.5, 0.25) == 0.75, "add(0.5, 0.25) should be 0.75"


def test_adding_zero_changes_nothing():
    assert add(7, 0) == 7, "adding zero should leave the value unchanged"
```

`test_hidden.py`

```python
import pytest
from submission import add


def _raises_type_error(call, mistake):
    try:
        call()
    except TypeError:
        return
    except Exception as exc:
        pytest.fail(f"{mistake}; raised {type(exc).__name__} instead")
    pytest.fail(mistake)


def test_sum_of_negative_ints():
    assert add(-4, -6) == -10, "the sum is wrong for negative ints: check the sign"


def test_large_ints_stay_exact():
    assert add(10**30, 1) == 10**30 + 1, "large ints lost precision: do not convert to float"


def test_mixed_int_and_float():
    assert add(1, 0.5) == 1.5, "an int plus a float should be their sum"


def test_none_raises_type_error():
    _raises_type_error(lambda: add(None, 1), "None is not a number: raise TypeError")


def test_text_raises_type_error():
    _raises_type_error(lambda: add("a", "b"), "text is not a number: raise TypeError")
```

## Tests to write (`tests/unit/scripts/test_exercise_gate.py`)

Load the script as `test_exercise_audit.py` loads its script (a module-scoped `gate` fixture).
Replace the grader with a fake as `fake_grader` does, by patching `gate.audit.grade_source`.
Build a small git repo in `tmp_path` as `demo` does.

1. `squash` and `contract_clauses`: bullets that wrap over two lines; a nested bullet stays with
   its parent; text before the first bullet becomes `C0`; `Constraints` gives `K1`, `K2`; a
   statement with no `Contract` section gives only `K` clauses. `clause_text` returns the text
   for `C2` and raises for `C9`. `squash(clause_text(...))` is a substring of
   `squash(problem_md)` for every label of the fixture. `plain_words` accepts
   `divides by n, not n - 1` and refuses text holding a backtick, `$`, `"` or `\`.
2. `probe_verdict`: one case per row of the table, built from `GradeReport` objects.
3. `probe` prints the `killed_by` line with the public test when a public test fails, and with
   the hidden one otherwise.
4. `note` then `ledger` round trip; `--kind` filter; unknown kind is exit 2.
5. `lint` on a repo with one clean and one unclean `test_hidden.py`: output lines, summary line,
   `lint.jsonl`, exit code 1; exit 0 when both are clean.
6. An unknown problem id is exit 2 and names a close match.

## Accept

```bash
uv run --frozen pytest tests/unit/scripts/test_exercise_gate.py -q
uv run --frozen ruff check scripts/exercise_gate.py tests/unit/scripts/test_exercise_gate.py tests/fixtures/problems/py-variants
uv run --frozen ruff format --check scripts/exercise_gate.py tests/unit/scripts/test_exercise_gate.py tests/fixtures/problems/py-variants
uv run --frozen pytest -q
uv run --frozen --group torch python scripts/exercise_gate.py clauses --out var/audit-i32/a3a cml/vector-operations/p01-vector-projection
uv run --frozen --group torch python scripts/exercise_gate.py probe --out var/audit-i32/a3a cml/vector-operations/p01-vector-projection content/cml/vector-operations/problems/p01-vector-projection/solution.py
uv run --frozen --group torch python scripts/exercise_gate.py probe --out var/audit-i32/a3a cml/vector-operations/p01-vector-projection content/cml/vector-operations/problems/p01-vector-projection/starter.py
uv run --frozen --group torch python scripts/exercise_gate.py lint --out var/audit-i32/a3a
```

Expected: the tests pass; `clauses` prints `C1` to `C4` and `K1`, `K2`; the first `probe` ends
`VERDICT: survived`; the second ends `VERDICT: broken (no test passed)`; the `lint` summary
matches the counts I-32a1 reported (its exit code 1 is expected here).

## Traps

- `ruff` checks `scripts/` with the repo rules. `print` is allowed there. Subprocess calls are
  allowed. Keep functions short enough for the `PL` rules, or split them.
- Do not write to `content/` in this part. Every command here is read-only apart from files
  under `<OUT>`.
- Do not copy code out of `exercise_audit.py`. Call it.
