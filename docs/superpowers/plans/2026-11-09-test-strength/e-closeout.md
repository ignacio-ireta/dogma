# I-32e: close-out (enforce, measure, report)

Two parts. **Part A** is code, done by one platform executor. **Part B** is measurement and
writing, done by the coordinator after Part A is committed.

## Part A: turn the rules on

**Owns:** `src/dogma/content/validate.py`, `src/dogma/cli/verify.py`,
`tests/unit/content/test_validate_lint.py` (new), `tests/unit/cli/test_cli_verify.py`,
`tests/fixtures/content/packages/`, `docs/CONTENT_SPEC.md`.
**Commit message (coordinator):** `feat(content): enforce the hidden-test lint and variants in the gate (I-32e)`

Read [00-protocol](00-protocol.md) first. Each step has a condition. If the condition does not
hold, **skip the step**, do not fix content, and say so in your report.

### A1. Enforce the lint in validation

Condition: this prints `lint: 0 findings in 0 problems`.

```bash
uv run --frozen --group torch python scripts/exercise_gate.py lint --out var/audit-i32/e
```

Read `src/dogma/content/validate.py` lines 537 to 556 (`_check_python`). In the branch for test
files, for `test_hidden.py` without a syntax error, read the file's text and add one issue per
finding of `lint_hidden_tests`:

```python
Issue(finding.rule, path, f"line {finding.line}: {finding.message}")
```

Then make the fixture packages pass. Run:

```bash
uv run --frozen pytest tests/unit/content -q
```

Tests that expect the fixtures to validate cleanly will now fail, because
`tests/fixtures/content/packages/tst/*/problems/*/test_hidden.py` have asserts without messages.
Add a short message to each such `assert` in those fixture files (they are test data, not
content) until those tests pass. Change nothing else in them.

New tests, in `tests/unit/content/test_validate_lint.py`: for each of the four rules, mutate the
`alpha_problem` copy's `test_hidden.py` so that it breaks the rule once, and check that the
rule's name is among the issue codes. Also check that the unmodified fixture gives none of them.

### A2. Refuse leaking test ids in `content verify`

Condition: A1 was done.

Read `src/dogma/cli/verify.py` lines 196 to 222 (`_check_problem`). After `first` is graded, add:

```python
leaks = id_leaks(test.name for test in first.tests if test.file == "hidden")
```

and, when `leaks` is not empty, the reason
`hidden test names show values (<up to three names>): give the parametrized cases neutral ids`.
Add one test to `tests/unit/cli/test_cli_verify.py`, following the fake-grader tests already
there: a report with a hidden test named `test_rate[180.0]` gives that reason; one named
`test_rate[large-rate]` does not.

Then check the real content (about 15 minutes):

```bash
uv run --frozen --group torch dogma content verify --all
```

If any problem fails with the new reason, undo A2 (`git checkout -- src/dogma/cli/verify.py
tests/unit/cli/test_cli_verify.py`) and list those problems in your report.

### A3. Require variants

Condition: both of these print `total: 0`.

```bash
uv run --frozen --group torch python scripts/exercise_gate.py worklist --out var/audit-i32/e --kind blind
uv run --frozen --group torch python scripts/exercise_gate.py worklist --out var/audit-i32/e --kind reject
```

Set `VARIANTS_REQUIRED = True` in `src/dogma/content/validate.py`. Run the content unit tests.
The fixture problems are `status: draft`, so they are not affected. If a test fails, read it:
fix the test's setup only if it builds a `verified` problem without variants on purpose.

### A4. Documents

Only for the steps you did.

- A1 done: in `docs/CONTENT_SPEC.md` §6.4, at the end of the bullet that starts `- **Lint.**`,
  add the sentence: `Validation reports each finding under the rule's name.`
- A3 done: in the §6.8 table of codes, add the row
  `| `VARIANTS_REQUIRED` | A `verified` or `reviewed` problem has a blind accept and, for its Contract, at least one reject or `uncovered` entry |`

### Accept (Part A)

```bash
uv run --frozen pytest tests/unit/content tests/unit/cli -q
uv run --frozen ruff check src tests && uv run --frozen ruff format --check src tests
uv run --frozen mypy && uv run --frozen lint-imports
uv run --frozen pytest tests/unit/cli/test_cli_new.py -q
uv run --frozen --group torch dogma content verify --all
git status --short
```

Expected: all pass; `content verify` ends with `All <n> topic(s) verified.`; only owned paths
changed. Report which of A1, A2, A3 you did and which you skipped, with the condition's output.

## Part B: measure and report (coordinator)

Run from the worktree, with nothing uncommitted. Each command can take many minutes. Run them
one at a time and let each finish.

### B1. Re-measure

```bash
uv run --frozen --group torch python scripts/exercise_audit.py baseline  --out var/audit-i32/final
uv run --frozen --group torch python scripts/exercise_audit.py mutate    --out var/audit-i32/final
uv run --frozen --group torch python scripts/exercise_audit.py seeds     --out var/audit-i32/final
uv run --frozen --group torch python scripts/exercise_audit.py precision --out var/audit-i32/final
uv run --frozen --group torch python scripts/exercise_audit.py summary   --out var/audit-i32/final
uv run --frozen --group torch python scripts/exercise_gate.py variants   --out var/audit-i32/final
uv run --frozen --group torch python scripts/exercise_gate.py scorecard  --out var/audit-i32/final
uv run --frozen --group torch python scripts/exercise_gate.py ledger     --out var/audit-i32/final --all
git diff --name-only i32-start..HEAD -- 'content/*/*/problems/*/meta.yaml' | wc -l
git diff --shortstat i32-start..HEAD -- 'content/*/*/problems/*/test_hidden.py'
```

`baseline` must report every reference passing, and `variants` must report zero mismatches. If
either does not, stop ([01-coordinator](01-coordinator.md) §8).

### B2. Declared against enforced tolerance

```bash
uv run --frozen python - <<'EOF'
import json
from collections import defaultdict
from pathlib import Path

import yaml

caught = defaultdict(list)
for line in Path("var/audit-i32/final/precision.jsonl").read_text(encoding="utf-8").splitlines():
    if line.strip():
        row = json.loads(line)
        if row["verdict"] != "survived":
            caught[row["problem"]].append(row["epsilon"])
print("| Problem | Declared rtol | Smallest error the tests catch | Reading |")
print("|---|---:|---:|---|")
for meta_path in sorted(Path("content").glob("*/*/problems/*/meta.yaml")):
    meta = yaml.safe_load(meta_path.read_text(encoding="utf-8"))
    rtol = float(meta["tolerance"]["rtol"])
    seen = caught.get(meta["id"])
    if not seen:
        continue
    tightest = min(seen)
    if tightest < rtol:
        print(f"| `{meta['id']}` | {rtol:.0e} | {tightest:.0e} | tests are stricter than declared |")
    elif rtol <= 1e-6 and tightest >= 1e-3:
        print(f"| `{meta['id']}` | {rtol:.0e} | {tightest:.0e} | tests are looser than declared |")
EOF
```

This only reads. It changes no tolerance (master plan, decision 8).

### B3. Write the report

Create `docs/audits/<today's date>-test-strength.md` with these sections, in this order. Paste
tool output. Do not retype or round a number, and do not add a claim no command printed.

1. `# Test strength` and one line: increment, date, the commands of B1.
2. `## Before and after`: a table with the "before" column from the design document §3 (214
   lint findings in 73 problems; 0 problems with committed variants) and the "after" column from
   the `lint`, `scorecard` summary and the two `git diff` lines.
3. `## Mutation and seeds`: the output of `summary`, and the last line of `seeds`.
4. `## What stays open`: the output of `ledger --all`, grouped as printed.
5. `## Declared against enforced tolerance`: the table from B2, with the sentence "Reported,
   not changed. A person decides."
6. `## Enforcement`: which of A1, A2 and A3 are on, from the Part A report.
7. `## Scorecard`: the full output of `scorecard`.

### B4. Finish

```bash
uv run --frozen python scripts/audit_privacy.py
git add docs/audits
git commit -m "docs: add the I-32 test-strength report"
```

Then [01-coordinator](01-coordinator.md) §7: merge, set I-32a to I-32e to `done` in
`docs/ROADMAP.md` (or `blocked` with the reason, for a sweep that left problems), and write the
handoff entry. In the entry, list the open decisions by kind and count, and say plainly which
enforcement steps are off and why.
