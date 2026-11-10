# I-32b: hygiene sweep (messages, ids, Hypothesis flags, `pytest.raises`)

**You own:** the `test_hidden.py` of each problem the coordinator listed. Nothing else.
**Outcome per problem:** `lint` reports nothing and `gate` passes, with no test made weaker and
no outcome changed. No version bump. No manifest.

Read [00-protocol](00-protocol.md) first. The edits here are small and exact. The risk is not
difficulty, it is changing something by accident. Use `Edit` on single statements. Never rewrite
a function, never reorder code, never "tidy".

## For each problem, in the order given

Let `<id>` be the problem id and `<dir>` its directory
(`content/<track>/<topic>/problems/<pNN-slug>`).

### Step 1. List the findings

```bash
uv run --frozen --group torch python scripts/exercise_gate.py lint --out <OUT> --only <id>
```

Each line is `problem, rule, line, message`. Exit code 1 is expected. If it prints
`lint: 0 findings`, write a `done` note (step 7) and go to the next problem.

Read only the lines you need: `Read` `<dir>/test_hidden.py` with `offset` near each finding and
`limit` 30. To word a message you may also run `clauses` (it prints the Contract):

```bash
uv run --frozen --group torch python scripts/exercise_gate.py clauses --out <OUT> <id>
```

### Step 2. Fix each finding by its rule

**`HIDDEN_ASSERT_MESSAGE` on an `assert`.** Add a message. Change nothing in the condition.

```python
assert out.shape == (n, k)                       # before
assert out.shape == (n, k), "wrong output shape: one row per sample, one column per class"
```

For an assert that spans lines in parentheses, the message goes after the closing parenthesis:
`assert (…), "message"`.

**`HIDDEN_ASSERT_MESSAGE` on a testing call.** Add `err_msg=` for `np.testing.*`, `msg=` for
`torch.testing.assert_close`. Change no other argument.

```python
np.testing.assert_allclose(got, want, rtol=1e-9)                                   # before
np.testing.assert_allclose(got, want, rtol=1e-9, err_msg="values differ from the reference")
```

**`HIDDEN_PARAM_IDS`.** Add `ids=` with exactly one id per case. Count the cases in the list.

```python
@pytest.mark.parametrize("lam", [0.7, 7.5, 120.0])                                  # before
@pytest.mark.parametrize("lam", [0.7, 7.5, 120.0], ids=["small-rate", "medium-rate", "large-rate"])
```

- An id is lowercase words joined by hyphens. It says what kind of case this is. It never
  contains a number or a value from the case. Dtype names (`float32`) are allowed.
- When the cases are just repeated draws (seeds), use letters: `ids=list("abcd")` for four.
- When the cases are a named list, `ids=[f"case-{chr(97 + i)}" for i in range(len(CASES))]`.
- When the cases come from an expression whose length you cannot read with certainty, do not
  guess. Leave the site and write a `question` note. The problem will not pass step 5; that is
  the correct result.
- Two `parametrize` decorators on one test each need their own `ids=`.

**`HIDDEN_GIVEN_SETTINGS`.** If the test has `@settings(...)`, add the missing keywords and keep
the others exactly (`max_examples` stays as it is):

```python
@settings(max_examples=60, deadline=None)                                           # before
@settings(max_examples=60, deadline=None, report_multiple_bugs=False)
```

If it has no `@settings`, add this line directly above `@given(...)`, and make sure the file's
`from hypothesis import …` line includes `settings`:

```python
@settings(deadline=None, report_multiple_bugs=False)
```

**`HIDDEN_BARE_RAISES`.** This is the one fix that changes code. Use it only for the plain form:

```python
with pytest.raises(ValueError):                                                     # before
    solve(bad_input)
```

```python
_raises(ValueError, lambda: solve(bad_input), "a negative size is invalid: raise ValueError")
```

If the file already has a helper for this (search for `def _rais`), use that helper with its own
signature. Otherwise add this one once, below the imports:

```python
def _raises(expected, call, mistake):
    try:
        call()
    except expected:
        return
    except Exception as exc:
        pytest.fail(f"{mistake}; raised {type(exc).__name__} instead")
    pytest.fail(mistake)
```

Do **not** convert, and write a `question` note instead, when the `pytest.raises` has `match=`,
is used `as` a name that is read afterwards, or wraps more than one statement. Converting those
would drop a check.

### Step 3. How to word a message

- One plain string. No f-string, no number or value taken from the test.
- It says which rule was broken, in the learner's terms. Build it from the test's name and the
  two things being compared: test `test_rows_sum_to_one` → `"each row of probabilities must sum
  to one"`.
- If you cannot tell what the assertion checks, use the test's name as a sentence. Do not invent
  a cause.
- Under 100 characters. No test input, no expected value, no array.

### Step 4. Look, do not touch

```bash
grep -nE "^\s*pass\s*$|np\.random\.(seed|rand|randn|randint|normal|uniform|choice|shuffle|permutation)\(|[0-9]\.[0-9]{8,}" <dir>/test_hidden.py
```

For each hit, read ten lines around it and write one `inspect` note. Do not edit.

| Hit | Note text |
|---|---|
| `pass` under an `except` | Which exception is swallowed, in which test, and whether a failure could hide behind it |
| `np.random.<function>(` on the global generator | Which test uses the global generator |
| A float with eight or more decimals | Which test compares against this literal, and whether a comment says where the number comes from |

### Step 5. Format, then gate

```bash
uv run --frozen ruff format <dir>/test_hidden.py
uv run --frozen ruff check <dir>/test_hidden.py
uv run --frozen --group torch python scripts/exercise_gate.py gate --out <OUT> <id> --mode text
```

Use `--mode refactor` instead of `--mode text` if, and only if, you replaced a `pytest.raises` in
this problem. It takes a few minutes longer.

| `gate` says | Cause | Do |
|---|---|---|
| `GATE: PASS` | | Step 7 |
| `change FAIL` (mode `text`) | You changed something besides messages, ids and the two flags | Run `git diff <dir>/test_hidden.py`, find the line that is not one of those, restore it with `Edit`, run step 5 again |
| `change FAIL` (mode `refactor`) | A wrong solution that used to fail now passes | Put every `pytest.raises` of this problem back as it was (`git diff` shows them), note `question`, and run step 5 again with `--mode text` |
| `reference FAIL` | Usually the number of ids does not match the number of cases, or a syntax slip | Fix that site, run step 5 again |
| `lint FAIL` | A finding is left | Fix it, or if it is a site you had to skip, go to step 6 |
| `names FAIL` | An id contains a digit | Rename the id |
| `frozen FAIL` | You touched another file | `git checkout -- <that file>` |
| anything else | | Protocol §6 |

Three failed `gate` runs for one problem: protocol §6, "Giving up on a problem".

### Step 6. A problem you could only partly fix

Keep the fixes that are done only if `lint` is the single check that fails. If any other check
fails, undo the file (`git checkout -- <dir>/test_hidden.py`). Either way, write a `question`
note naming each site left. The coordinator commits a problem in this state because of that note.

### Step 7. Record

```bash
uv run --frozen --group torch python scripts/exercise_gate.py note --out <OUT> <id> --kind done --text "hygiene: <n> findings fixed, mode <text or refactor>"
```

## Never, in this sweep

- Never change a condition, a tolerance, a value, `max_examples`, a test name or an import other
  than adding `settings`.
- Never touch `test_public.py`, `meta.yaml` or anything under `variants/`.
- Never fix something the lint did not report, however wrong it looks. Write an `inspect` note.
