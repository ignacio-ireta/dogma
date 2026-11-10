# I-32a1: a lint for hidden tests

**Owns:** `src/dogma/content/testlint.py` (new), `tests/unit/content/test_testlint.py` (new).
**Do not touch:** `src/dogma/content/validate.py`. Wiring the lint into the gate is I-32e.
**Commit message (coordinator):** `feat(content): add a lint for hidden tests (I-32a1)`

## Read first

- `src/dogma/content/pyfacts.py` (104 lines): the style to copy. Pure functions over `ast`,
  frozen dataclass, nothing imported or run.
- `docs/CONTENT_SPEC.md` lines 196 to 220: the rules this lint checks.

## Build

One module, `src/dogma/content/testlint.py`, with this public surface and nothing else public:

```python
RULES: tuple[str, ...] = (
    "HIDDEN_PARAM_IDS",
    "HIDDEN_BARE_RAISES",
    "HIDDEN_GIVEN_SETTINGS",
    "HIDDEN_ASSERT_MESSAGE",
)

@dataclass(frozen=True)
class LintFinding:
    rule: str      # one of RULES
    line: int      # 1-based line in the source
    message: str   # the fixed sentence for the rule, below

def lint_hidden_tests(source: str) -> list[LintFinding]: ...
def id_leaks(names: Iterable[str]) -> list[str]: ...
```

`lint_hidden_tests` parses `source` with `ast.parse`. On `SyntaxError` it returns `[]` (the
validator already reports syntax errors). Findings are sorted by `(line, rule)`.

Call "last name" of a call the text after the final dot of `ast.unparse(call.func)`:
`pytest.mark.parametrize` → `parametrize`, `np.testing.assert_allclose` → `assert_allclose`.

| Rule | A finding is reported for | `line` | `message` |
|---|---|---|---|
| `HIDDEN_PARAM_IDS` | A decorator (on a function or a class) that is a call with last name `parametrize`, unless it has an `ids` keyword, or its second positional argument is a list or tuple whose every element is a call with last name `param` that has an `id` keyword | The decorator's line | `parametrize needs ids=: default ids print the hidden inputs in the test name` |
| `HIDDEN_BARE_RAISES` | Any call whose function unparses to `pytest.raises` or to the bare name `raises` | The call's line | `use a helper that names the mistake instead of pytest.raises` |
| `HIDDEN_GIVEN_SETTINGS` | A function whose name starts with `test_` and that has a decorator call with last name `given`, unless it also has a decorator call with last name `settings` carrying both `deadline=None` and `report_multiple_bugs=False` (both as constants) | The `def` line | `@given needs @settings(deadline=None, report_multiple_bugs=False)` |
| `HIDDEN_ASSERT_MESSAGE` | (a) An `assert` statement with no message. (b) A call with last name in `assert_allclose`, `assert_array_equal`, `assert_array_almost_equal`, `assert_almost_equal`, `assert_equal`, `assert_close` that has neither an `err_msg` nor a `msg` keyword | The statement's or call's line | (a) `assert needs a message that names the kind of mistake` (b) `<last name> needs err_msg= (msg= for torch)` |

Walk the whole tree (`ast.walk`), so asserts inside helper functions count.

`id_leaks(names)` returns, in input order, each name that has a `[`…`]` part containing a digit
after dtype names are removed. Take the text between the first `[` and the final `]`. Remove
every match of `(?i)\b(?:u?int|float|complex|bool)(?:8|16|32|64|128)\b`. If a digit remains, the
name leaks. A name with no `[` never leaks.

## Tests to write (`tests/unit/content/test_testlint.py`)

Use small source strings. One `parametrize` table per rule is fine.

1. For each of the four rules: a source that triggers it once, and the nearest source that does
   not. Check `rule`, `line` and `message`.
2. `HIDDEN_PARAM_IDS`: `ids=[...]` passes; every case as `pytest.param(..., id="x")` passes; one
   `pytest.param` without `id` fails; a `parametrize` on a class fails without ids.
3. `HIDDEN_GIVEN_SETTINGS`: `settings(max_examples=50, deadline=None)` alone fails;
   `settings(deadline=None, report_multiple_bugs=False, max_examples=50)` passes; `@given` on a
   helper that is not named `test_*` is ignored.
4. `HIDDEN_ASSERT_MESSAGE`: an `assert` inside a helper function is reported;
   `np.testing.assert_allclose(a, b, err_msg="x")` passes; `torch.testing.assert_close(a, b, msg="x")`
   passes; `np.testing.assert_allclose(a, b)` fails.
5. A source with a syntax error returns `[]`.
6. Two findings on different lines come back sorted by line.
7. `id_leaks`: `test_a[180.0]` leaks; `test_a[empty]`, `test_a[float32]`, `test_a[int8-uint16]`,
   `test_a[case-a]` and `test_b` do not; `test_a[float32-n2]` and `test_a[seed0]` leak.

## Accept

Run each and copy the last line into your report.

```bash
uv run --frozen pytest tests/unit/content/test_testlint.py -q
uv run --frozen ruff check src/dogma/content/testlint.py tests/unit/content/test_testlint.py
uv run --frozen ruff format --check src/dogma/content/testlint.py tests/unit/content/test_testlint.py
uv run --frozen mypy
uv run --frozen lint-imports
```

Then count findings on the real content:

```bash
uv run --frozen python - <<'EOF'
from collections import Counter
from pathlib import Path

from dogma.content.testlint import lint_hidden_tests

sites, problems = Counter(), Counter()
for path in sorted(Path("content").glob("*/*/problems/*/test_hidden.py")):
    found = lint_hidden_tests(path.read_text(encoding="utf-8"))
    sites.update(finding.rule for finding in found)
    problems.update({finding.rule for finding in found})
for rule in sorted(sites):
    print(rule, sites[rule], problems[rule])
EOF
```

An independent scan on 2026-11-09 gave the numbers below. Yours should be close. The content has
changed since, so a difference of a few is expected. A rule that reports zero, or more than
double, means the rule is implemented wrongly: fix it before reporting.

```text
HIDDEN_ASSERT_MESSAGE 62 23
HIDDEN_BARE_RAISES 10 5
HIDDEN_GIVEN_SETTINGS 73 31
HIDDEN_PARAM_IDS 69 42
```

Put your four lines in the report.

## Traps

- `mypy` is strict here. Annotate everything. `ast.unparse` returns `str`.
- Do not import anything from `dogma.cli`, `dogma.api` or `dogma_data` (import contracts).
- Do not add configuration, suppression comments or a rule that is not in the table.
- Do not "fix" content files that the count reveals. That is I-32b.
