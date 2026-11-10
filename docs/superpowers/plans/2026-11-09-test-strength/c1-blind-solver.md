# I-32c: instructions for a blind solver

*This file is everything you need. Do not read any other file in this folder.*

You solve practice problems the way a careful learner would: from the statement and the public
tests alone. Your solutions become the proof that each problem can be solved from its statement,
so it matters that you never see the answer.

## What you get

The coordinator gives you a list of problem ids and the directory you work from. For each id
there is a bundle:

```text
var/audit-i32/blind-run/blind/<problem id>/
├── problem.md         the statement
├── starter.py         the signature to implement   (starter.sql for SQL problems)
├── test_public.py     the public tests
└── fixtures/          data files, if the problem has any
```

## What you may and may not open

You may read and write **only inside the bundles of your problems**.

Never open, list or search anything under `content/`, `src/`, `tests/`, `docs/`, `scripts/`,
`workspace/`, or any other directory under `var/`. Never run `git` (history shows answers). Never
search the web. If you open one of these by mistake, stop and say so in your report: that
problem must be solved again by someone else.

Never commit, and never change a file outside your bundles.

## For each problem

1. Read `problem.md`, the starter and `test_public.py` in the bundle.
2. Read the **Constraints** section twice. Import only what it allows. A solution that imports
   anything else is rejected before it runs, and the rejection still costs one of your three
   submissions. Never import `dogma_data`.
3. Write your solution in the bundle as `solution.py` (or `solution.sql`), keeping the starter's
   exact signature. Write a general solution. Do not special-case the inputs you see in the
   public tests.
4. Run the public tests as often as you like:

```bash
uv run --frozen --group torch python scripts/exercise_audit.py public --out var/audit-i32/blind-run <problem id> var/audit-i32/blind-run/blind/<problem id>/solution.py
```

5. When the public tests pass, submit. You have **three submissions** per problem:

```bash
uv run --frozen --group torch python scripts/exercise_audit.py submit --out var/audit-i32/blind-run <problem id> var/audit-i32/blind-run/blind/<problem id>/solution.py
```

   The output is what a learner sees: each test's name, and for a failing hidden test one line
   that names the kind of mistake. You cannot see the hidden tests. Use the names and that line.
6. If a hidden test fails, re-read the Contract for the rule the message points at, fix the
   solution, run the public tests again, and submit again.
7. After a pass, or after the third submission, go to the next problem. Do not keep working on a
   problem that has no submissions left (`no attempts left`).

For SQL problems use `solution.sql` in both commands.

## Rules about what you report

- Report only what the commands printed. Copy the last line of the last `submit` for each
  problem.
- If a problem did not pass, say which hidden tests failed and, in one sentence, what in the
  statement you found unclear or missing. That sentence is useful. Do not guess at what the
  hidden test contains.
- If a command printed an error you do not understand, copy it. Do not try other commands.

End with this report and nothing after it:

```text
BLIND SOLVER REPORT
<problem id>: PASSED on submission <n>
<problem id>: FAILED after 3 submissions; failing: <test names>; unclear: <one sentence>
FILES OPENED OUTSIDE MY BUNDLES: none | <paths>
```
