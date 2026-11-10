# I-32d2: triage of surviving mutants

**You own:** one topic, exactly as in [d-reject-sweep](d-reject-sweep.md): `variants/` and
`test_hidden.py` through the tool only.
**Outcome per problem:** every sampled one-token mutant of the reference that still passes the
tests has been looked at once. A real gap got a test. The rest got one line saying why not.

Read [00-protocol](00-protocol.md) first. This runs after the reject sweep, so most mutants are
already caught. Expect zero to five survivors per problem.

Names: `<id>`, `<dir>`, `<cand>` (`<OUT>/candidates/<pNN-slug>`) as in the reject sweep.

## Step 1. Generate and grade the mutants (once per topic)

```bash
uv run --frozen --group torch python scripts/exercise_audit.py mutate --out <OUT> --only <track>/<topic>/
uv run --frozen --group torch python scripts/exercise_audit.py summary --out <OUT>
uv run --frozen --group torch python scripts/exercise_audit.py survivors --out <OUT> --only <track>/<topic>/ --cap 10
```

Keep the final `/` after the topic: `--only` matches by prefix. `mutate` takes a few minutes. `survivors` writes, for each problem that has any,
`<OUT>/triage/<id>/survivors.json` and one file per survivor,
`<OUT>/triage/<id>/mutant_<key>.py`. A problem with no `survivors.json` has none: go to step 4
for it.

## Step 2. For each problem with survivors

Read `<dir>/problem.md` and `<dir>/solution.py`, then run `clauses`:

```bash
uv run --frozen --group torch python scripts/exercise_gate.py clauses --out <OUT> <id>
```

Read `<OUT>/triage/<id>/survivors.json`. Each entry has `key`, `op`, `line` and `change` (the
line before and after, joined by `|`). Take them in the order given, at most ten.

## Step 3. Decide each survivor with this table, top to bottom

| Question | If yes |
|---|---|
| 1. Can the change alter any return value or raised error, for some input? Typical "no": a constant inside a message, a copy that is never written to, a comparison that is equal on both sides for every valid input | Go on to question 2. If **no**: count it as `same behaviour`. No note, no test |
| 2. Can you point to the label of a clause that the mutant breaks, for an input inside the stated domain? | Go on to question 3. If **no**: `note --kind unpromised --text "mutant <key>: <what differs>, no clause covers it"` |
| 3. Does `<dir>/variants/` contain `accept_blind.py`? | Write a test (below). If **no**: `note --kind no-witness --text "mutant <key> breaks <label>, no blind accept so no test added"` |

When unsure about question 1 or 2, the answer is no. A mutant you cannot tell apart from the
reference by the Contract is not a gap.

**Writing the test.** Copy the mutant so that it has a name that says the mistake, then follow
step 4 of the reject sweep exactly (the rules for the test, the `add-test` command and its
result table):

```bash
mkdir -p <cand>
cp <OUT>/triage/<id>/mutant_<key>.py <cand>/reject_<short_name>.py
```

```bash
uv run --frozen --group torch python scripts/exercise_gate.py add-test --out <OUT> <id> <cand>/new_test_<short_name>.py --wrong <cand>/reject_<short_name>.py --mistake "<plain words>" --clause <label>
```

Limits: one rewrite per survivor, and at most three new tests per problem in this sweep. Beyond
that: `note --kind no-witness --text "mutant <key> breaks <label>, no test found"`.

## Step 4. Mark the problem

```bash
uv run --frozen --group torch python scripts/exercise_gate.py mark-swept --out <OUT> <id> --kind mutants --mode <add if an add-test succeeded for this problem in this sweep, else none>
uv run --frozen --group torch python scripts/exercise_gate.py note --out <OUT> <id> --kind done --text "mutants: survivors <n>, same behaviour <n>, unpromised <n>, no-witness <n>, tests added <n>"
```

`mark-swept` must end `GATE: PASS` and `marked mutants`. If it fails, protocol §6.

## After the last problem of the topic

```bash
uv run --frozen --group torch dogma content verify <track>/<topic>
uv run --frozen ruff check content/<track>/<topic>
uv run --frozen ruff format --check content/<track>/<topic>
```

Then the report of protocol §7, with the `done` line of each problem.

## Never, in this sweep

- Never call a mutant "same behaviour" because a test would be hard to write. That answer is
  only for changes that cannot alter a result.
- Never write a test for a tie that depends on the last digit of a float (a strict against a
  non-strict comparison at an exact threshold). I-28 dropped five such tests on purpose: note
  `unpromised` instead.
- Never edit `survivors.json` or a mutant file under `<OUT>/triage/`.
