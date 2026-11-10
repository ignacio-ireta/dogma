# I-32a3b: the gate tool, part 2: manifest writers and the replay

**Owns:** `scripts/exercise_gate.py`, `tests/unit/scripts/test_exercise_gate.py`,
`tests/integration/runner/test_gate_tool.py` (new).
**Needs:** I-32a2 and I-32a3a committed.
**Commit message (coordinator):** `feat(scripts): write and replay problem variants (I-32a3b)`

The rule this part enforces: `variants/manifest.yaml` is written only by these commands, from
what the grader showed. Each command either finishes completely or leaves no trace.

## Read first

- `scripts/exercise_gate.py` as part 1 left it (whole file).
- `src/dogma/content/schemas.py`: the `VariantsManifest` models from I-32a2.
- `src/dogma/content/validate.py`: `validate_variants`.
- `src/dogma/cli/verify.py` lines 163 to 178 (`run_ruff`): how to call ruff with the repo
  configuration, or `--isolated` when there is no `pyproject.toml`.
- `src/dogma/core/fsio.py`: `atomic_write`.
- `scripts/exercise_audit.py` lines 923 to 945 (`cmd_submit`: the rows of `blind.jsonl`) and 1124
  to 1147 (`_suite_sha`, `_text_sha`, `grade_suite`).

## Build

### 1. Shared helpers

`GateError`, `clause_text`, `plain_words`, `squash`, `probe_verdict` and `append_note` exist
from part 1. Add:

```python
def read_manifest(problem) -> VariantsManifest | None
def write_manifest(problem, manifest: VariantsManifest) -> None
def install_variant(problem, source: Path, name: str) -> Path
def failing_functions(report, file: str) -> list[str]
```

- `write_manifest`: `yaml.safe_dump(manifest.model_dump(mode="json", exclude_none=True),
  sort_keys=False, allow_unicode=True, width=100)`, written with `atomic_write` to
  `variants/manifest.yaml`. A manifest never holds comments.
- `install_variant`: refuse if `source` already lies inside the problem directory
  (`write candidates under <OUT>/candidates, not in variants/`). Copy `source` to
  `variants/<name>`. For a `.py` file run `ruff format`, then `ruff check --fix`, then
  `ruff check` on that one file, with the configuration choice of `run_ruff`. If the last call
  fails, delete the copy and raise `GateError` with ruff's output.
- `failing_functions`: names of tests in `file` (`public` or `hidden`) with outcome `fail` or
  `error`, cut at the first `[`, without repeats, in report order.
- A new manifest starts with `problem_version` taken from `meta.yaml`.
- The suffix of every variant must fit the language: `.py` for Python, `.sql` for SQL.

### 2. `add-reject ID FILE --mistake TEXT --clause LABEL [--killed-by NAME] [--message-has TEXT]`

In this order. On any failure delete what this call created, raise `GateError`, write nothing.
`--mistake` and `--message-has` go through `plain_words`.

1. FILE's base name must match `^reject_[a-z0-9_]+\.(py|sql)$` and must not be in the manifest.
2. Resolve LABEL with `clause_text(problem.md text, LABEL)`. The manifest stores that text, not
   the label. A caller can never type a clause.
3. `install_variant`.
4. Grade the installed file (`audit.grade_source`). `probe_verdict` must be `killed`. Else print
   the same block `probe` prints and fail. For `survived`, the message is
   `SURVIVED: no test fails for this wrong solution (see your runbook: candidate gap)`.
5. Choose the killing test. With `--killed-by NAME`: NAME must be in `failing_functions` of the
   public or the hidden file. Without it: the first failing public function, else the first
   failing hidden one. `visible` is the file it came from.
6. With `--message-has TEXT`: the learner message of some failing case of that function must
   contain TEXT, ignoring case.
7. Append the entry and `write_manifest`. Print
   `added reject <name>: killed_by <function> (<visible>), message kind <kind>`.
8. If the kind (from `audit.classify_message` on that message) is not `named`, also call
   `append_note(out, id, "message-defect", "<name>: <function> shows kind=<kind>")`.

### 3. `add-accept ID FILE --origin blind|freedom [--freedom TEXT] [--clause LABEL]`

Same shape. The name must match `^accept_[a-z0-9_]+\.(py|sql)$`; for `--origin blind` it must
be `accept_blind.py` or `accept_blind.sql`. `--freedom` (through `plain_words`) is required for
`freedom` and refused for `blind`. A given label is resolved as in step 2 above. The verdict
must be `survived`. If it is not, print the probe block and fail with
`NOT ACCEPTED: a test fails for this solution (see your runbook: do not edit the test)`.
Print `added accept <name> (<origin>)`.

### 4. `add-uncovered ID --clause LABEL --reason TEXT`

Resolve the label, check `--reason` with `plain_words`, append, write. Print
`added uncovered <label>`.

### 5. `remove-variant ID NAME`

Delete `variants/NAME` and its manifest entry. If the manifest then has no accept, reject,
uncovered or sweeps, delete `manifest.yaml` and the empty `variants/` directory. Print
`removed <name>`. An unknown NAME is exit 2.

### 6. `adopt-blind [--only PREFIX]...`

For each problem that has no blind accept yet:

1. Look for `<OUT>/blind/<id>/solution.py` (or `.sql`). None: print `<id>  skipped: no file`.
2. Require a row in `<OUT>/blind.jsonl` with this `problem`, `verdict == "survived"` and
   `source_sha` equal to `audit._text_sha` of the file's text. None:
   `<id>  skipped: no passing submit for this file`.
3. Run the `add-accept` logic with origin `blind` and name `accept_blind.<suffix>`. A failure
   prints `<id>  skipped: <reason>` and continues.
4. Compare with the reference (`blind_similarity`). Take the non-blank lines that are not
   comments, stripped, from both files, and drop every line the starter also has: the signature
   and docstring are shared by every solution and say nothing about independence.
   `ratio = difflib.SequenceMatcher(None, a, b).ratio()`. If the reference keeps at least 15
   lines of its own and `ratio >= 0.9`, call
   `append_note(out, id, "question", "accept_blind is <ratio:.2f> similar to the reference")`.
5. Print `<id>  adopted  similarity=<ratio:.2f>`.

Last line: `adopt-blind: <a> adopted, <s> skipped`.

### 7. `variants [--only PREFIX]... [--workers N]`

Public helper for part 3: `replay_problem(problem, out) -> list[dict]`. For each problem with a
manifest:

1. `validate_variants(load_problem(problem.path))`: each issue is a mismatch row
   (`static <code>`).
2. Each accept: `probe_verdict` must be `survived`.
3. Each reject: verdict `killed`; `killed_by` in `failing_functions(report, visible)`;
   `message_has`, when set, found as in `add-reject` step 6.

Keep rows in `<OUT>/variants.jsonl` and reuse a row when its key is already there. Key:
`<problem>:<file name>:<_text_sha of the file>:<_suite_sha of audit.working_suite(problem)>:<_text_sha of the entry as JSON>`.
Run the grading jobs through `audit._run_all` (default 10 workers). Print one line per variant:

```text
<id>  <file>  expect=<pass|fail>  got=<verdict>  OK
<id>  <file>  expect=fail  got=survived  MISMATCH: no test fails
```

Last line: `variants: <ok> ok, <bad> mismatches in <n> problems`. Exit 1 if `bad > 0`.

## Tests to write

**Unit** (`tests/unit/scripts/test_exercise_gate.py`), with a fake grader patched onto
`gate.audit.grade_source`. Copy `tests/fixtures/problems/py-variants` into
`tmp_path/repo/content/fixtures/py-variants/problems/p01-add`, make it a git repo as `demo`
does, and point `gate.audit.REPO` at it. The fake reads markers in the submitted source and
returns a `GradeReport` whose test names exist in the fixture:

| Marker in source | Report |
|---|---|
| `BUG` | `failed`; `test_example_adds_two_ints` passes (public); `test_large_ints_stay_exact` fails (hidden) with message `large ints lost precision` |
| `PUBLICBUG` | `failed`; a hidden test passes; `test_example_adds_two_ints` fails (public) |
| `BROKEN` | `failed`; no test passes |
| `SLOW` | `timeout` |
| none | `passed`, one public and one hidden test |

1. `add-reject` with `BUG`: file installed, manifest entry has `killed_by:
   test_large_ints_stay_exact`, `visible: hidden`. With `PUBLICBUG`: `visible: public`.
2. `add-reject` refusals, each leaving no file and no manifest change: no marker (`survived`),
   `BROKEN`, `SLOW`, a label that does not exist (`C9`), a `--mistake` with a backtick, a bad
   file name, a `--killed-by` that did not fail, a `--message-has` that is not in the message, a
   FILE inside `variants/`. The stored `clause` of a successful call equals
   `clause_text(problem.md, label)`.
3. `add-accept`: passes with no marker; refused with `BUG`; `--origin blind` with another name
   refused; `--origin freedom` without `--freedom` refused.
4. `add-uncovered` and `remove-variant` (the last removal deletes the directory).
5. `adopt-blind`: adopted when `blind.jsonl` has the matching sha; skipped when the sha
   differs, when the verdict is not `survived`, and when the file is missing.
6. `variants`: all OK gives exit 0; a manifest whose reject now survives gives exit 1 and a
   `MISMATCH` line; a second run with unchanged files grades nothing (count calls to the fake).
7. `message-defect` note: make the fake return message `ValueError: boom` for the failing test;
   the ledger gains one row.

**Integration** (`tests/integration/runner/test_gate_tool.py`, `pytestmark =
pytest.mark.integration`): the same repo layout, the real grader, no fake.

1. `add-accept` of a correct rewrite (the guard, then `return b + a`) succeeds.
2. `add-reject` of a variant with no guard (`return a + b` alone), `--clause C3`, succeeds with
   `killed_by: test_text_raises_type_error`.
3. `add-reject` of a variant that keeps the guard but drops the `bool` check is refused with
   `SURVIVED`, and `variants/` does not contain it.
4. `variants` exits 0. After changing that reject's `killed_by` in the manifest to
   `test_sum_of_negative_ints`, `variants` exits 1.

## Accept

```bash
uv run --frozen pytest tests/unit/scripts/test_exercise_gate.py -q
uv run --frozen pytest -m integration tests/integration/runner/test_gate_tool.py -q
uv run --frozen ruff check scripts/exercise_gate.py tests/unit/scripts tests/integration/runner
uv run --frozen ruff format --check scripts/exercise_gate.py tests/unit/scripts tests/integration/runner
uv run --frozen pytest -q
uv run --frozen --group torch python scripts/exercise_gate.py variants --out var/audit-i32/a3b
git status --short
```

Expected: tests pass; `variants` prints `variants: 0 ok, 0 mismatches in 0 problems`;
`git status --short` shows no path under `content/`.

## Traps

- Roll back in a `try`/`except` around the whole command body, so that a crash half-way also
  leaves no file behind. Test this with the refusal cases above.
- Never accept a test name typed by the caller without checking it against the report.
- Do not format or lint anything except the one installed file.
- Do not run these commands against real `content/` in this part. The example topic gets its
  variants in I-32a4.
