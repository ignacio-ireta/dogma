# I-32a4: documents, template, Makefile and the example topic

**Owns:** `docs/CONTENT_SPEC.md`, `docs/ENGINEERING_STANDARDS.md`, `docs/ARCHITECTURE.md`,
`docs/decisions/0022-problem-variants-and-test-gate.md` (new),
`templates/topic/problems/p01-todo/test_hidden.py.tmpl`, `Makefile`,
`content/cml/vector-operations/problems/p01-vector-projection/variants/` (new, through the tool).
**Needs:** I-32a3d committed.
**Commits (coordinator), two:** `docs: document test design, variants and the test gate (I-32a4)`
for docs, template and Makefile; `content(cml): add variants to the example problem (I-32a4)` for
the `content/` path.

Every text block below is to be inserted **as written**. Do not reword, extend or summarise.
Use `Edit` with a short unique `old_string`. Each step names its anchor line.

Two checks before you start:

- **The decision number.** Run `ls docs/decisions`. This runbook writes the new record as 0022.
  If a file `0022-…` already exists, use the next free number instead, in the file name and in
  every block below that says 0022 or ADR-0022.
- **Anchors.** Each anchor must be found exactly once in its file. If one is missing or appears
  twice, do not improvise a place: write a `blocked` note that quotes the anchor, skip that step
  and go on.

## 1. `docs/CONTENT_SPEC.md`

**1a. Layout (§1).** Replace the one line

```text
        └── fixtures/                  # optional: CSV / data files the tests load
```

with

```text
        ├── fixtures/                  # optional: CSV / data files the tests load
        └── variants/                  # proof for the tests (§6.8); written by scripts/exercise_gate.py
```

**1b. Lint (§6.4).** Directly after the line `- Fast: the whole run takes under 10 s on a laptop.`
insert:

~~~markdown
- **Lint.** `scripts/exercise_gate.py lint` checks four of these rules in `test_hidden.py`:
  `HIDDEN_PARAM_IDS` (every `parametrize` has `ids=`), `HIDDEN_BARE_RAISES` (no `pytest.raises`),
  `HIDDEN_GIVEN_SETTINGS` (every `@given` test has
  `@settings(deadline=None, report_multiple_bugs=False, …)`) and `HIDDEN_ASSERT_MESSAGE` (every
  `assert` has a message; every `np.testing` or `torch.testing` call has `err_msg` or `msg`).
  No digit may appear inside the `[...]` part of a hidden test name, dtype names excepted.
~~~

**1c. New sections.** Directly before the line `## 7. Verification and the learner's loop (`dogma`)`
insert:

~~~markdown
### 6.7 Designing tests

Follow these steps in order. The reasons and examples are in the
[test strategy](superpowers/specs/2026-11-09-test-strategy-design.md).

1. **Contract first.** Every test checks a sentence in Contract or Constraints. If a behaviour
   matters and no sentence states it, add the sentence, then the test. If you are unsure whether
   the Contract promises it, it does not.
2. **One row per Contract bullet:** what a wrong solution would do, the oracle, the data, public
   or hidden. A bullet with no row is a gap. A test with no bullet is over-reach.
3. **Oracle,** the first that fits: a library call on the same inputs; a brute-force reference
   in the test file, shaped differently from `solution.py`; a property that needs no expected
   value; a literal worked out by hand for a tiny case. Never the reference's own output. If a
   library oracle disagrees with `solution.py`, stop and report it.
4. **Inputs,** in this order: hand-built edge cases; a property or seeded loop over the stated
   domain; realistic data (§6.6); scale only when Constraints promise a complexity. Choose inputs
   for which a wrong step changes the result: a test of "does not mutate" needs an input the
   wrong code would actually alter.
5. **Messages.** The first line names the rule that was broken and holds no hidden value. Order
   the file from the most basic behaviour to the most exotic.
6. **Public and hidden.** Hidden tests repeat every family the public tests show, with other
   values.
7. **Proof** (§6.8). No test without a wrong solution it fails. No wrong solution without a
   Contract clause.

### 6.8 Variants: proof that the tests work

`variants/` holds solutions the tests must accept and wrong solutions they must reject.

| Entry | File | Must |
|---|---|---|
| `accept`, `origin: blind` | `accept_blind.py` | Pass every test. Written from `problem.md`, the starter and the public tests only |
| `accept`, `origin: freedom` | `accept_<what>.py` | Pass every test. Correct, and different from the reference where the Contract allows |
| `reject` | `reject_<mistake>.py` | Fail the test named in `killed_by`. The reference with one mistake that breaks the quoted `clause` |
| `uncovered` | none | Name a Contract bullet that has no reject, with the reason |

`manifest.yaml` is written only by `scripts/exercise_gate.py` (`add-accept`, `add-reject`,
`add-uncovered`, `remove-variant`, `adopt-blind`, `add-test`, `mark-swept`), from what the grader
showed. Never edit it by hand. A new hidden test enters through `add-test`: it appends the test,
bumps the version, runs `gate` and records the wrong solution the test catches, or restores every
file. A variant is evidence, not reference code: when one is installed, style findings are marked
in the file with `noqa`, while syntax errors and undefined names refuse it. Static rules, checked
by `dogma content verify`:

| Code | Rule |
|---|---|
| `VARIANT_MANIFEST_MISSING` | `variants/` has files but no `manifest.yaml` |
| `VARIANT_FILE_UNLISTED`, `VARIANT_FILE_MISSING` | The manifest and the directory list the same files |
| `VARIANT_SUFFIX` | `.py` for Python problems, `.sql` for SQL problems |
| `VARIANT_CLAUSE_NOT_IN_STATEMENT` | Every `clause` is in `problem.md`, word for word |
| `VARIANT_KILLED_BY_UNKNOWN` | `killed_by` is a test function of the file that `visible` names |
| `VARIANT_VERSION_STALE` | `problem_version` equals `meta.yaml: version` |

`make test-all` replays every manifest through the grader (`scripts/exercise_gate.py variants`).
After any edit to a problem's tests, run

    uv run --frozen --group torch python scripts/exercise_gate.py gate --out var/gate <problem id> --mode <mode>

and keep the edit only on `GATE: PASS`. Modes: `add` for new tests (no line removed), `text` for
messages, ids and Hypothesis reporting flags, `refactor` for any other edit that must not change
an outcome, `none` when the tests did not change. A failing accept is never fixed by editing the
test: report it.

~~~

## 2. `docs/ENGINEERING_STANDARDS.md`

**2a. §5.2.** Directly after the checklist item that ends
`or a `dogma_data.synth` generator; see CONTENT_SPEC §6.6.` insert:

~~~markdown
- [ ] Hidden tests pass the lint (CONTENT_SPEC §6.4), and `variants/` holds one blind accept
      and, for every Contract bullet, a reject or an `uncovered` reason (CONTENT_SPEC §6.8).
~~~

**2b. §13.** Replace the one line

```text
make test-all = check + Python e2e/slow suites (excluding llm) + Playwright browser smoke
```

with

```text
make test-all = check + verify-variants (every problem's variants replayed through the grader)
              + Python e2e/slow suites (excluding llm) + Playwright browser smoke
```

**2c. §16, rule 3.** Directly after the line
`   pass.** If a test is wrong, say so and fix it in a separate, explained commit.` insert:

```text
   Check every test edit with `scripts/exercise_gate.py gate` (CONTENT_SPEC §6.8).
```

## 3. `docs/ARCHITECTURE.md`

**3a. §4.1 tree.** Replace the one line

```text
                └── hints.md              # 3-step hint ladder: concept → formula → code shape
```

with

```text
                ├── hints.md              # 3-step hint ladder: concept → formula → code shape
                └── variants/             # solutions the tests must accept or reject (CONTENT_SPEC §6.8)
```

**3b. §4.3.** In the list under `**Kinds of check** (combine as needed):`, directly after the
bullet that starts `- **Test data** from` (it ends with `[DATASETS.md](DATASETS.md).`), insert:

~~~markdown
- **Variants** (`problems/<p>/variants/`): solutions the tests must accept and wrong solutions
  they must reject, each tied to a Contract clause. `scripts/exercise_gate.py` writes the
  manifest, checks every test edit and replays all variants in `make test-all` (ADR-0022).
~~~

**3c. §8.** In the `scripts/` line of the layout, add `exercise_audit.py  exercise_gate.py` after
`check_content_types.py` (same line, two spaces between names).

**3d. §9.** Append one row to the table, after the last row:

```text
| [0022](decisions/0022-problem-variants-and-test-gate.md) | Each problem carries `variants/`: solutions its tests must accept and reject, written by a gate tool that also checks every test edit | Proof kept in `var/` was lost after I-28; a mirror tree under `tests/` would drift from the content |
```

## 4. `docs/decisions/0022-problem-variants-and-test-gate.md` (new file, `Write`)

~~~markdown
# 0022. Problem variants and a gate for test edits

Date: 2026-11-09 · Status: accepted

## Context

The evidence that a problem's tests work was produced by one-off audits and kept in git-ignored
`var/`: blind solutions, wrong solutions, triage verdicts. The I-28 set no longer exists, so a
later test edit could only be re-checked by a new audit. Mutants are one-token edits of the
reference and miss conceptual mistakes: on 2026-11-10 a hand-written solution that rescaled `b`
in place passed every test of the example problem, which the mutation audit had scored as having
no gap. The rules for hidden tests in CONTENT_SPEC §6.4 had no check, and 214 sites in 73
problems broke them.

## Decision

Each problem package carries `variants/`: solutions its tests must accept (one written blind,
plus correct solutions that differ from the reference where the Contract allows) and wrong
solutions they must reject, each tied to a Contract clause and to the test that kills it. The
manifest is written only by `scripts/exercise_gate.py`, from what the grader showed. The same
tool lints hidden tests and checks every test edit in one of three modes (add, text, refactor).
Static manifest rules run in `make check`. The replay through the grader runs in `make test-all`.

## Consequences

A test edit is checked in about a minute instead of by a full audit, and a correct solution that
starts failing is noticed. The repository gains roughly 2,000 small files. A version bump or a
statement edit must be followed by a replay, which `VARIANT_VERSION_STALE` and the clause check
force. Variants show known mistakes. A mistake nobody wrote down is still unmeasured, so the
sampled mutation audit stays.

## Alternatives

- Keep the evidence in `var/`: free today, lost again later.
- A mirror tree under `tests/`: `content/` stays unchanged, but two trees drift apart and a
  content commit no longer carries its proof.
- A clause-by-test matrix for all hidden tests: complete, but too costly to write and to keep.
~~~

## 5. Template

In `templates/topic/problems/p01-todo/test_hidden.py.tmpl`, directly after the third comment line
(it ends `never the values (CONTENT_SPEC §6.4).`), insert:

```text
# Lint: every assert carries a message, every parametrize has neutral ids=, every @given test has
# @settings(deadline=None, report_multiple_bugs=False), and there is no pytest.raises (use a
# helper that names the mistake). Prove each test with scripts/exercise_gate.py (CONTENT_SPEC §6.8).
```

## 6. `Makefile`

Recipe lines start with a TAB character, not spaces.

- In the `.PHONY` line, add `verify-variants` after `verify-content`.
- Change `test-all: check` to `test-all: check verify-variants`.
- After the `verify-content` target and its recipe, add a blank line and:

```make
verify-variants:
	uv run --frozen --group torch python scripts/exercise_gate.py variants --out var/audit-variants
```

Check with `make -n verify-variants`: it must print the command and no error.

## 7. Variants for the example problem

Problem id: `cml/vector-operations/p01-vector-projection`. OUT: `var/audit-i32/a4`.
Read its `solution.py` (30 lines). Do not read its `test_hidden.py`. The problem already has
`variants/manifest.yaml` with its blind accept. The commands below add to it.

Make four candidates. For each: copy the solution, then make exactly the edit described, and
change the docstring to start with `Wrong on purpose:` (rejects) or leave it (accept).

```bash
mkdir -p var/audit-i32/a4/candidates
cp content/cml/vector-operations/problems/p01-vector-projection/solution.py var/audit-i32/a4/candidates/<name>
```

| Name | Edit | Command after | Expected |
|---|---|---|---|
| `reject_no_zero_guard.py` | Delete the two lines `if scale == 0.0:` and its `raise` | `add-reject … --mistake "no check for an all-zero b" --clause C3` | `killed_by test_zero_vector_b_raises_value_error (public)` |
| `reject_squared_length_overflow.py` | Replace the last three code lines (`unit = …`, `coefficient = …`, `return …`) with `coefficient = float(a_vec @ b_vec) / float(b_vec @ b_vec)` and `return coefficient * b_vec` | `add-reject … --mistake "forms b dot b directly, which overflows or underflows" --clause C3` | killed by a hidden test about a huge or tiny `b` |
| `reject_no_dimension_check.py` | Delete the two lines `if a_vec.ndim != 1 or b_vec.ndim != 1:` and its `raise` | `add-reject … --mistake "no check that the inputs are 1-D" --clause C3` | `killed_by test_two_dimensional_input_raises_value_error (hidden)` |
| `accept_unit_length_direction.py` | Replace the last three code lines with `unit = b_vec / scale`, `direction = unit / np.sqrt(float(unit @ unit))`, `return float(a_vec @ direction) * direction` | `add-accept … --origin freedom --freedom "projects onto a unit-length direction, so the last digits differ"` | `added accept` |

The full form of each command:

```bash
uv run --frozen --group torch python scripts/exercise_gate.py add-reject --out var/audit-i32/a4 cml/vector-operations/p01-vector-projection var/audit-i32/a4/candidates/<name> --mistake "<text>" --clause C3
uv run --frozen --group torch python scripts/exercise_gate.py add-accept --out var/audit-i32/a4 cml/vector-operations/p01-vector-projection var/audit-i32/a4/candidates/accept_unit_length_direction.py --origin freedom --freedom "<text>"
```

These four outcomes were checked against the real grader on 2026-11-10. If one differs, do not
adjust the candidate until it fits. Write a `question` note with the output and go on.

Do **not** add a variant that divides `b_vec` in place. It survives the current tests (the test
for unchanged inputs uses a `b` whose largest entry is 1). That gap is closed in I-32d, so that
the problem's version is bumped once.

## Accept

```bash
uv run --frozen --group torch python scripts/exercise_gate.py gate --out var/audit-i32/a4 cml/vector-operations/p01-vector-projection --mode none
uv run --frozen --group torch dogma content verify cml/vector-operations
uv run --frozen pytest tests/unit/cli/test_cli_new.py tests/unit/content -q
uv run --frozen pytest -m e2e tests/e2e/cli -q
make -n test-all
uv run --frozen python scripts/audit_privacy.py
git status --short
```

Expected: `GATE: PASS`; `All 1 topic(s) verified.`; tests pass; `make -n test-all` lists the
`exercise_gate.py variants` command; the audit exits 0; `git status --short` lists only the
owned paths.

## Traps

- The blocks in steps 1 to 5 are final text. A "small improvement" to them is a defect.
- Do not add a `variants/` template to the scaffold. A scaffolded problem must still fail
  verification with `TOPIC_TODO` only.
- Do not touch `test_hidden.py`, `meta.yaml` or `problem.md` of the example problem.
- The example's blind accept is already there. Do not remove or replace it.
