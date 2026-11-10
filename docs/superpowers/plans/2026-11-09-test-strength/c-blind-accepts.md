# I-32c: one blind solution per problem, adopted as `accept_blind`

*Coordinator runbook. The solvers read only [c1-blind-solver](c1-blind-solver.md).*

**Outcome:** every problem has `variants/accept_blind.py` (or `.sql`): a solution written without
sight of the reference, the hidden tests or the hints, that passes every test. It is the guard
against tests that reject correct work. No test changes. No version bump.

All commands run from the worktree. The results directory for this sweep is
`var/audit-i32/blind-run`.

## Step 1. Make the bundles (once)

```bash
uv run --frozen --group torch python scripts/exercise_audit.py bundle --out var/audit-i32/blind-run
ls var/audit-i32/blind-run/blind | head
```

Each bundle holds `problem.md`, the starter, `test_public.py` and fixtures. It never holds the
solution, the hidden tests or the hints.

## Step 2. Use harvested solutions first (only if `var/audit-i32/harvest` exists)

For each file `var/audit-i32/harvest/<id>/blind/<name>`:

```bash
cp var/audit-i32/harvest/<id>/blind/<name> var/audit-i32/blind-run/blind/<id>/solution.py
uv run --frozen --group torch python scripts/exercise_audit.py submit --out var/audit-i32/blind-run <id> var/audit-i32/blind-run/blind/<id>/solution.py
```

A pass prints a line that ends `· passed`. If there is no such line, delete that one file with
`rm var/audit-i32/blind-run/blind/<id>/solution.py`, so that a solver starts clean. Use
`solution.sql` for SQL problems. These solutions were written blind during I-31.

## Step 3. The list

```bash
uv run --frozen --group torch python scripts/exercise_gate.py worklist --out var/audit-i32/coord --kind blind
```

Skip ids whose bundle already has a passing `solution.py` from step 2 (they are adopted in
step 5). First batch: the five pilot problems of the master plan.

## Step 4. Dispatch solvers

Five problems per solver. At most four solvers at a time. Prompt (`model: sonnet`):

```text
You are a blind solver for the dogma repository.
First run: cd <worktree path> && pwd
Read docs/superpowers/plans/2026-11-09-test-strength/c1-blind-solver.md and follow it exactly.
Do not read any other file in docs/.
Your problems:
  <problem id>
  <problem id>
Your bundles are under var/audit-i32/blind-run/blind/<problem id>/.
End with the BLIND SOLVER REPORT that file describes.
```

Do not put anything else in the prompt. In particular, no hint about a problem and no path under
`content/`.

## Step 5. Adopt what passed

After each solver returns, for its problems:

```bash
uv run --frozen --group torch python scripts/exercise_gate.py adopt-blind --out var/audit-i32/blind-run --only <id> --only <id>
```

`adopt-blind` trusts nothing in the report. It adopts a file only when `blind.jsonl` shows a
passing submit of exactly that file, and it grades the file again.

Then for each adopted problem:

```bash
uv run --frozen --group torch python scripts/exercise_gate.py gate --out var/audit-i32/coord <id> --mode none
```

## Step 6. Commit a batch

```bash
git status --short | grep -v "/variants/"       # must print nothing
uv run --frozen ruff check content
uv run --frozen ruff format --check content
git add content
git commit -m "content: adopt blind solutions as accept variants, batch <n> (I-32c)"
```

If the first command prints a path, something outside `variants/` changed. Do not commit. Undo
that path with `git checkout -- <path>` and find out which agent touched it before going on.

## Step 7. What to do with each outcome

| Outcome | Do |
|---|---|
| Solver reports a file opened outside its bundles | Discard its solution for that problem: `rm` the bundle's `solution.py`. Solve again with a fresh solver in a second results directory (below) |
| `adopt-blind` wrote a `question` note "similar to the reference" | Once: `remove-variant <id> accept_blind.py`, then solve again with a fresh solver in the second results directory. If the new one is flagged too, keep it and leave the note |
| Solver failed after three submissions | Write a note (`--kind question --text "blind solve failed: <test names>; unclear: <the solver's sentence>"`). Solve once more with a fresh solver in the second results directory |
| Second solver fails too | Leave the problem without a blind accept. It stays on the worklist with its note. In I-32d it gets no new hidden test (master plan, decision 5) |
| `gate --mode none` fails for an adopted problem | `remove-variant <id> accept_blind.py`, note `blocked` with the failing check |

**Second results directory.** `submit` allows three attempts per problem per directory, so a
second try needs its own:

```bash
uv run --frozen --group torch python scripts/exercise_audit.py bundle --out var/audit-i32/blind-run-2 --only <id>
```

Dispatch as in step 4 with `blind-run-2` in place of `blind-run` in the prompt, and tell the
solver to use `--out var/audit-i32/blind-run-2` in both commands. Adopt with
`--out var/audit-i32/blind-run-2`.

## Step 8. End of the sweep

The sweep is done when `worklist --kind blind` prints only problems that have a `question` or
`blocked` note. Then follow [01-coordinator](01-coordinator.md) §5.6 (end of a wave) and §7
(merge).

Put in the handoff entry: problems solved on the first, second and third submission, problems
solved by the second solver, problems left without a blind accept, and every "unclear" sentence
the solvers wrote. Those sentences point at statements that need a clearer Contract.
