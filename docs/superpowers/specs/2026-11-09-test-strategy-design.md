# Test strategy: how a problem's tests are designed and proven

*Design for increment I-32, 2026-11-09. Status: proposed. The rules here become binding when
I-32a4 copies them into [CONTENT_SPEC](../../CONTENT_SPEC.md). The work plan is
[2026-11-09-test-strength.md](../plans/2026-11-09-test-strength.md).*

This document answers two questions: what must the tests of a problem guarantee, and how does an
author (a person or a model) build tests that give those guarantees. It is written so that it can
be followed step by step. Where a rule needs judgement, the rule says what to do when unsure.

---

## 1. What a submission goes through

Nothing in this design changes the pipeline. It is listed so that every rule below has a place.

| Stage | Where | Stops the run when | The learner sees |
|---|---|---|---|
| Static checks | `src/dogma/grading/static.py` `check_submission` | The file does not parse, the entry is missing, or it imports something the problem does not allow | The finding in plain words; nothing executes |
| Staging | `src/dogma/runner/staging.py` | Fixtures contain a symlink | An infrastructure error |
| Public tests | `test_public.py`, run by `src/dogma_harness/` | Never stops the run | Test name and the full message |
| Hidden tests | `test_hidden.py`, on Submit only | Never stops the run | Test name and the **first line** of the message that names a mistake (`src/dogma/grading/display.py` `hidden_summary`) |
| Limits | `src/dogma/runner/sandbox.py` | Time (`meta.yaml: timeout_s`), memory, disk | One sentence naming the usual cause |

Fixed facts an author can rely on:

- Pytest runs with `--hypothesis-seed=0` and `PYTHONHASHSEED=0`. A submission sees the same
  examples on every Submit.
- Tests run in file order. The learner sees every failing test, in that order.
- One grader run costs about one second (`dogma content verify --all` took 774 s for 753 runs,
  ADR-0020).
- `meta.yaml: tolerance` is not read by grading. Each test carries its own tolerance.

## 2. What a suite must get right

A suite can fail a learner in five ways. Each has a measure and a guard.

| Failure | Example | Measure | Guard |
|---|---|---|---|
| **Rejects correct work** | A hidden test demands `np.float64` when the Contract says "a float" | An independent solution fails | Accept variants (§5): solutions that must pass |
| **Accepts wrong work** | Population variance passes because every test uses `n = 1` | A wrong solution passes | Reject variants (§5): wrong solutions that must fail; mutants |
| **Names the wrong mistake** | "shape mismatch" when the shape is right and the values are wrong | The message shown for a known wrong solution | `killed_by` and `message_has` in the manifest |
| **Passes or fails by luck** | A statistical bound that holds only for seed 0 | The reference under other seeds | `exercise_audit.py seeds`; `dogma_data.rng` salt |
| **Leaks hidden values** | A parametrized test named `test_rate[180.0]` | Hidden test names and first message lines | Lint rules (§7) |

The first failure is the worst one. A learner who is right and is told they are wrong loses time
and trust. When two rules conflict, prefer the one that keeps correct work passing.

## 3. Where the tests stand (2026-11-09)

Measured read-only on the working tree, which includes uncommitted I-31b edits. The numbers will
move; every sweep regenerates its own list by tool.

- 251 problems (231 Python, 20 SQL). 890 public test functions, 3,741 hidden ones. The median
  problem has 14 hidden tests.
- I-28 (2026-10-31): 98.6% of non-equivalent sampled mutants killed, 251/251 blind solves, 96% of
  hidden-only kills show a message that names a mistake.
- 152 problems use Hypothesis. 127 of 231 Python problems import a library oracle in their hidden
  tests; the other 104 rely on a brute-force reference, properties or literals.
- 214 sites in 73 problems break a CONTENT_SPEC §6.4 rule in hidden tests (§7).
- The proof behind those numbers is not in git. The I-28 blind solutions and triage verdicts are
  gone. `var/audit-i31/` holds 28 blind and 33 killed wrong files, for 28 problems.
- One probe of the method, on the example problem `cml/vector-operations/p01-vector-projection`
  (2026-11-10, real grader): six hand-written variants. Three wrong ones were rejected. One
  correct rewrite was accepted. One meant to be wrong still met the Contract (without the length
  check, NumPy raises the same error type on its own). And one that divides `b` in place passed
  all 17 tests, although the Contract says inputs must not
  change: the test for it uses a `b` whose largest entry is 1. I-28 had scored this problem as
  having no gap.

The tests are good. What is missing is durable proof, coverage of conceptual mistakes, and a gate
for the mechanical rules.

## 4. The design method

Seven steps, in this order. Do not start a step before the one above it is finished.

### Step 1. Contract first

Read `problem.md`. Every test must check a sentence in **Contract** or **Constraints**.

- If a behaviour matters and no sentence states it, add the sentence first. Then test it.
- If you are unsure whether the Contract promises a behaviour, it does not. Do not test it.
- Extreme values count. A test near `1e308` needs a Contract sentence that says such values are
  valid (13 of the 16 blind retries in I-28 came from this).

### Step 2. One row per Contract bullet

Before writing a test, write this table in your working notes (not in the repo):

| Bullet | What a wrong solution would do | Oracle (step 3) | Data (step 4) | Public or hidden |
|---|---|---|---|---|

A bullet with no row is a gap. A test with no bullet is over-reach. Both are defects.

### Step 3. Choose the oracle, from the top of this ladder

1. **A library call on the same inputs** (`numpy`, `scipy`, `sklearn`, `statsmodels`, `torch`,
   the stdlib, DuckDB or pandas for SQL). Every argument that selects a convention (`ddof`,
   `average`, `axis`) must match a Contract sentence.
2. **A brute-force reference written in the test file**, built from the definition and shaped
   differently from `solution.py` (a double loop where the solution is vectorised, a sort where
   the solution uses a heap).
3. **A property that needs no expected value** (§6).
4. **A literal worked out by hand**, for tiny cases only, with the arithmetic in a comment.

Never use the reference's own output as the expected value. A number read off `solution.py`
passes by construction and proves nothing.

If a library oracle disagrees with `solution.py`, stop. Do not change arguments until they agree.
Report it: either the Contract and the library use different conventions, or the reference is
wrong. The second case is the most valuable finding this method can produce.

### Step 4. Choose the inputs, in this order

1. **Hand-built edge cases** for exact rules: empty, single element, ties, duplicates, all equal,
   invalid input. Expected values are literals.
2. **A Hypothesis property or a seeded loop** over the domain the Contract states, checked
   against the oracle.
3. **Realistic data** where the algorithm depends on the data's structure
   ([CONTENT_SPEC §6.6](../../CONTENT_SPEC.md)).
4. **Scale**, only when Constraints promise a complexity. Prefer counting operations (a sequence
   that counts reads) over wall-clock time. A wall-clock limit needs a tenfold margin over a
   naive correct solution.

Vary the form of valid input wherever the Contract allows it: list and array, int and float
dtype, a non-contiguous view, a read-only array.

### Step 5. Write the assertion so that its first line teaches

- The first line of every hidden message is a plain sentence that names the rule that was broken
  and, where possible, the usual cause. It holds no input, no expected value and no array.
- Route `raises` checks through a helper that says what should have been raised. Never a bare
  `pytest.raises` in a hidden test.
- Wrap a learner call whose crash would otherwise show as a raw `TypeError` or `IndexError`.
- Order the file from the most basic behaviour to the most exotic. The first failing test should
  be the root cause.
- Name each test after the behaviour. Give every hidden `parametrize` neutral `ids=`.

### Step 6. Split public and hidden

- Public: the examples in `problem.md` plus one edge case. Literal values.
- Hidden: every family the public tests show, again, with different values, so that hard-coding
  the public answers fails. Then the families only "Done looks like" names.

### Step 7. Prove it

A test is finished when the tool has seen all of this through the real grader (§5):

- the reference passes, twice, with the same outcomes;
- an independent solution passes;
- for each new test, a wrong solution that passed before now fails, with a named message;
- the reference passes under other seeds, and the run stays under 10 s.

**No test without a kill. No kill without a clause. No clause quoted from memory.**

## 5. Proof that stays: `variants/`

Each problem package gains one directory:

```text
problems/p01-vector-projection/
└── variants/
    ├── manifest.yaml
    ├── accept_blind.py             # written from the statement and public tests only
    ├── accept_unit_vector_form.py  # correct, and different from the reference where allowed
    ├── reject_divides_by_norm.py   # wrong in one named way
    └── reject_no_zero_guard.py
```

```yaml
# manifest.yaml: written only by scripts/exercise_gate.py, never by hand
problem_version: 2
accept:
  - file: accept_blind.py
    origin: blind
  - file: accept_unit_vector_form.py
    origin: freedom
    freedom: "normalises b first, so the last digits differ from the reference"
reject:
  - file: reject_no_zero_guard.py
    mistake: "no check for an all-zero b"
    clause: "when every entry of `b` is zero"
    killed_by: test_zero_vector_b_raises_value_error
    visible: public
uncovered: []
sweeps: [reject]
```

Rules:

- **Accept variants must pass every test.** `origin: blind` is one solution written by an agent
  that never saw `solution.py`, the hidden tests or the hints. `origin: freedom` is a correct
  solution that makes a different choice wherever the Contract leaves one open (§6 of the
  [catalogue](../plans/2026-11-09-test-strength/catalogue.md)).
- **Reject variants must fail.** Each is the reference with one mistake. It must run, pass at
  least one test, and fail `killed_by` on an assertion. `clause` is copied from `problem.md`, and
  validation refuses a clause that is not there word for word.
- **The tool writes the manifest from what it observed.** An author never types a test name or
  a clause. The clause is picked by its label (`C3`) from the tool's own listing of the statement.
- **A new hidden test enters through one command** (`add-test`) that appends it, bumps the
  version, runs every check and records the wrong solution it catches, or restores every file.
- **A failing accept is never fixed by editing the test.** It is reported: either the statement
  must say what the test demands, or the test is wrong. A person decides.
- Static checks on the manifest run in `make check`. The replay through the grader runs in
  `make test-all`.

Variants are checked into the repo because every later test edit needs them. With them, "did this
edit reject correct work or stop catching wrong work?" is one command.

## 6. Properties that need no expected value

Use these where no library oracle exists, and next to an oracle where they are cheap.

| Relation | Holds for |
|---|---|
| Reordering rows or samples does not change the result (or reorders it the same way) | Metrics, statistics, losses, set-like queries |
| Scaling the input by `c` scales the result by `c`, `c**2` or not at all | Norms, variances, similarities, standardised scores |
| Adding a constant changes nothing, or adds the constant | Softmax, variance, centred statistics, means |
| Applying it twice equals applying it once | Projections, normalisers, clipping, deduplication |
| Two outputs of the same call agree with each other | Precision and recall against the confusion matrix; probabilities sum to 1 |
| A swap gives the mirror result | `AUC(1 - s) = 1 - AUC(s)`; swapping classes; negating scores |
| A limit case has a known value | One class, one cluster, zero learning rate, `k = n` |
| The analytic gradient matches a finite difference | Every backward pass and loss |
| The whole equals the sum of its parts | Batched against one at a time; streaming against batch |

A property must be able to fail. Before keeping one, name the wrong solution it catches.

## 7. Mechanical rules, checked by lint

`src/dogma/content/testlint.py` (I-32a1) checks `test_hidden.py`. Counts are from §3.

| Rule | What it requires | Sites | Problems |
|---|---|---:|---:|
| `HIDDEN_PARAM_IDS` | Every `parametrize` has `ids=` | 69 | 42 |
| `HIDDEN_BARE_RAISES` | No `pytest.raises` | 10 | 5 |
| `HIDDEN_GIVEN_SETTINGS` | Every `@given` test has `@settings(deadline=None, report_multiple_bugs=False, …)` | 73 | 31 |
| `HIDDEN_ASSERT_MESSAGE` | Every `assert` has a message; every `np.testing` or `torch.testing` call has `err_msg` or `msg` | 62 | 23 |

One rule is checked on the run instead: no digit inside the `[...]` part of a hidden test name,
dtype names such as `float32` excepted.

Not lint, because each case needs a look: long float literals (19 in 10 problems), `except: pass`
(5), global `np.random` (1). `max_examples` above 100 (11 tests) stays, because lowering it would
weaken a test.

## 8. Randomness and time

- Seed everything. Data comes from `dogma_data.rng("<problem id>:<purpose>:<i>")`, so the audit
  can vary the draw.
- A function that draws random numbers is tested in one of two ways. If the Contract fixes the
  order of draws from a given generator, compare with the exact stream. Otherwise test the
  distribution, with a stated bound of at least 4 standard errors.
- A test must pass for the reference under 20 other seeds. One lucky draw is a defect.
- The whole run stays under 10 s for a naive correct solution, not only for the reference.
- Never assert on the last digits of a BLAS result. Compare fitted values or objectives when the
  problem is ill-conditioned.

## 9. Definition of strong

A problem is strong when `scripts/exercise_gate.py gate <id>` passes and the manifest shows:

1. one `accept_blind` variant;
2. for every Contract bullet, a reject variant or an `uncovered` entry with a reason;
3. both sweeps recorded (`reject`, `mutants`);
4. no open `blocked` or `question` note for the problem.

Across the repo, the close-out report gives the mutation score, the message kinds for every kill,
seed results and run times, generated by tool.

## 10. Limits

- Variants prove what was tried. A mistake nobody wrote down is not covered.
- A blind solution written by a model is a floor. It shows a capable reader can succeed, not how
  hard the problem is for a person.
- Deliberate cheating is out of scope. The sandbox is not a security boundary
  (ENGINEERING_STANDARDS §11).
- "Names a mistake" is checked by pattern. Whether it names the right one is checked only where
  a manifest entry carries `message_has`.
