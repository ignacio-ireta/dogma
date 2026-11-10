# I-32d: reject sweep (wrong solutions per Contract bullet, and the gaps they expose)

**You own:** one topic. In each of its problems: `variants/` (through the tool), `test_hidden.py`
(through `add-test` only) and the version in `meta.yaml` (through the tool).
**Outcome per problem:** every Contract bullet has a wrong solution that the tests reject, or a
written reason why not. Each gap found is closed by one new test, proven by the tool.

Read [00-protocol](00-protocol.md) first. In this sweep you write wrong solutions and, rarely, a
test. You never edit an existing test, and you never type into `test_hidden.py` yourself.

Names used below: `<id>` the problem id, `<dir>` its directory
(`content/<track>/<topic>/problems/<pNN-slug>`), `<cand>` your scratch directory
`<OUT>/candidates/<pNN-slug>`. Files there are not part of the repository until the tool copies
them. `.py` becomes `.sql` for SQL problems.

Every tool command has this form. Only the part after `exercise_gate.py` changes:

```bash
uv run --frozen --group torch python scripts/exercise_gate.py <command> --out <OUT> <id> …
```

## For each problem of the topic

### Step 1. Read, in this order, and nothing more

```bash
mkdir -p <cand>
uv run --frozen --group torch python scripts/exercise_gate.py clauses --out <OUT> <id>
```

Then read `<dir>/problem.md`, `<dir>/solution.py` and `<dir>/test_public.py`. Read **one**
section of [catalogue.md](catalogue.md), chosen by `runtime` in `<dir>/meta.yaml`:

| `runtime` | Section |
|---|---|
| `py-numpy` | Numeric |
| `py-torch` | Torch |
| `py-stdlib` | Plain Python and algorithms |
| `py-data` | pandas |
| `sql-duckdb` | SQL |

Also read "Random draws" if the entry takes `rng` or `seed`, and "Classes" if the entry is a
class. **Do not open `test_hidden.py` in this step.** Wrong solutions written after reading the
tests tend to be the ones the tests already catch.

### Step 2. Write one wrong solution per Contract bullet

For each label `C1`, `C2`, … that `clauses` printed (at most eight; skip `C0`):

```bash
cp <dir>/solution.py <cand>/reject_<short_name>.py
```

Then make **one small edit** that breaks that bullet and nothing else, and change the docstring's
first line to start with `Wrong on purpose:`. The name is lowercase words joined by `_` that
say the mistake (`reject_population_variance.py`).

| The bullet says | A wrong solution |
|---|---|
| What inputs are valid (types, shapes, ranges) | Mishandles one valid form: works for arrays but not lists, for floats but not ints, for many rows but not one |
| What is returned (value, shape, dtype, order) | Makes one mistake from the catalogue: wrong divisor, wrong axis, wrong dtype, wrong tie order |
| `Raises X when …` | Drops that check, or checks a narrower condition (`<` where the bullet needs `<=`), or raises another exception type |
| Must not mutate inputs | Changes an input in place, in a way that alters its values for ordinary inputs |
| A complexity or a ban (Constraints) | Only if a `K` bullet promises a complexity: the naive slower algorithm |

Rules for every wrong solution:

- It must still run and keep the signature. It is the reference with one mistake, not a new
  program.
- Prefer the **subtle** mistake to the obvious one: the one that still gives the right answer on
  the simplest inputs. Example from `cml/vector-operations`: dividing `b` in place by its
  largest entry breaks "must not mutate", yet changes nothing when that entry is 1, which is the
  only case the tests tried. That wrong solution passed every test.
- One bullet, one file. If the bullet has several parts (three reasons to raise), pick the part
  a learner is most likely to forget.
- If a bullet cannot be broken without breaking another bullet too, do not force it:

```bash
uv run --frozen --group torch python scripts/exercise_gate.py add-uncovered --out <OUT> <id> --clause C4 --reason "inputs are tuples, which cannot be changed in place"
```

Then add up to **two** more wrong solutions from the catalogue section, if they break a bullet
in a way your first ones did not. If `var/audit-i32/harvest/<id>/wrong/` exists, copy up to three
of its files into `<cand>` as `reject_<name>.py` as well.

All free text you pass to the tool (`--mistake`, `--reason`, `--freedom`, `--text`) is plain
words: letters, digits, spaces, commas, hyphens. No backticks, quotes or dollar signs.

### Step 3. Probe each wrong solution

```bash
uv run --frozen --group torch python scripts/exercise_gate.py probe --out <OUT> <id> <cand>/reject_<short_name>.py
```

| `VERDICT:` | Do |
|---|---|
| `killed` | Record it: `add-reject --out <OUT> <id> <cand>/reject_<short_name>.py --mistake "<plain words>" --clause <label>` |
| `survived` | Step 4 |
| `broken (…)` | Your file does not run or passes nothing. Fix it once. Still broken: drop the file |
| `infra (…)` | Run the probe once more. Same again: note `blocked` for this file and go on |

If `add-reject` prints ruff errors, fix the file (usually an unused name) and run it again, once.

### Step 4. A wrong solution that survived

This is what the sweep is for. Decide with this table, top to bottom:

| Question | If no |
|---|---|
| 1. Is the file really wrong? Read the bullet's text again. Does the Contract forbid what your file does, in words? | `note --kind unpromised --text "<what the file does>, no Contract sentence forbids it"`. Add no test. Next file |
| 2. Does `<dir>/variants/` contain `accept_blind.py`? | `note --kind no-witness --text "<file name> survives and breaks <label>, no blind accept so no test added"`. Next file |

When unsure about question 1, the answer is no.

Two yeses: write **one** test in `<cand>/new_test_<short_name>.py`.

1. Now open `<dir>/test_hidden.py`, in parts: the first 60 lines (imports and helpers), then the
   test nearest in subject (`grep -n "def test_" <dir>/test_hidden.py` lists them).
2. Write a single `def test_<behaviour>()` of at most 30 lines that follows this:
   - It checks the quoted bullet on an input inside the domain the Contract states.
   - Choose the input so that the wrong step **matters** (a `b` whose largest entry is not 1).
   - The expected value comes from a library call, a property, or a tiny case worked by hand
     with the arithmetic in a comment. Never from running the reference or your wrong file.
   - Use the helpers the file already has for raising and for calling the learner's code.
   - Every `assert` has a message. Its first line says which rule was broken, with no value.
   - Use only names the file already imports. If you need another module, import it as the
     first line inside the test: `import math  # noqa: PLC0415`.
   - No `parametrize`, no Hypothesis, no `pytest.raises`, no randomness without a fixed seed.
3. Hand it to the tool, which appends it, bumps the version, runs the gate and records the wrong
   solution, or changes nothing:

```bash
uv run --frozen --group torch python scripts/exercise_gate.py add-test --out <OUT> <id> <cand>/new_test_<short_name>.py --wrong <cand>/reject_<short_name>.py --mistake "<plain words>" --clause <label>
```

| Last line | Meaning | Do |
|---|---|---|
| `ADD-TEST: added …` | The gap is closed | Next file |
| `FAILED`, and `reference FAIL` above it | Your test fails for the correct solution | If your expected value came from a library and you checked its arguments against the Contract: `note --kind reference-doubt` with the call. Otherwise rewrite the test once |
| `FAILED`, and `accepts FAIL` above it | An independent correct solution fails your test: you asked for more than the Contract | `note --kind overreach --text "<test idea>, fails accept_blind"`. No retry |
| `FAILED`, and `wrong FAIL` above it | Your test does not catch the wrong file | Rewrite the test once with an input where the mistake matters |
| `FAILED`, and `seeds`, `time`, `lint` or `names FAIL` | The test is flaky, slow or breaks a lint rule | Fix that and run again |
| a refusal before the gate ran | The message says why (two tests in the file, a name that exists, over 40 lines) | Fix the file and run again |

At most three `add-test` runs per wrong file, and at most three new tests per problem. After
that: `note --kind no-witness --text "<file name> survives and breaks <label>, no test found"`.

### Step 5. Correct solutions that differ (at most two per problem)

From the "Freedoms" list of your catalogue section, pick a choice that **no** clause fixes.
Write `<cand>/accept_<short_name>.py`: the reference, making the other choice.

```bash
uv run --frozen --group torch python scripts/exercise_gate.py add-accept --out <OUT> <id> <cand>/accept_<short_name>.py --origin freedom --freedom "<what it does differently, plain words>"
```

If it prints `NOT ACCEPTED`, a test demands that choice. Do not edit the test or the statement:

```bash
uv run --frozen --group torch python scripts/exercise_gate.py note --out <OUT> <id> --kind overreach --text "<file name>: <failing test name>, no clause fixes this choice"
```

If no freedom applies, skip this step.

### Step 6. Account for every bullet, then mark the problem

Every label `C1…` needs a recorded reject or an `add-uncovered` entry. For a bullet whose wrong
solution survived without a test, the reason is `wrong solution survives, see notes`. For a
bullet beyond the eighth, it is `cap of eight wrong solutions per problem`.

```bash
uv run --frozen --group torch python scripts/exercise_gate.py mark-swept --out <OUT> <id> --kind reject --mode <add if an add-test succeeded for this problem, else none>
```

It must end `GATE: PASS` and `marked reject`. If it fails, protocol §6.

## After the last problem of the topic

```bash
uv run --frozen --group torch dogma content verify <track>/<topic>
uv run --frozen ruff check content/<track>/<topic>
uv run --frozen ruff format --check content/<track>/<topic>
```

All three must succeed. Then the report of protocol §7. For each problem add one line:
`rejects <n>, survived <n>, tests added <n>, accepts <n>, uncovered <n>`.

## Never, in this sweep

- Never open `test_hidden.py` before step 4, and never edit it yourself.
- Never make a wrong solution pass or fail by changing a test, the reference or the statement.
- Never write a test for behaviour you could not point to with a label.
- Never keep a wrong solution that fails only because it crashes on import.
