# I-32 protocol: rules for every executor

*Read this first, then your runbook. Read nothing else in this folder. If a rule here and your
runbook disagree, this file wins and you write a `blocked` note.*

## 1. Your scope

The coordinator's message gives you three things: a **runbook**, a list of **problem ids or
paths you own**, and an **OUT** directory such as `var/audit-i32/d-cml-softmax`.

- Change only the paths you own. Do not create, edit or delete anything else.
- Work in the directory the coordinator names. Run `pwd` first and check it ends with
  `.claude/worktrees/i32`.
- Problem id form: `<track>/<topic>/<pNN-slug>`. Its directory is
  `content/<track>/<topic>/problems/<pNN-slug>/`.

## 2. What you may read

- This file, your runbook, and the files your runbook lists for each problem.
- Read a file over 300 lines in parts (`offset`, `limit`). Never print a whole test file into
  your context when you need one function.
- Line numbers in a runbook were right on 2026-11-09. If the file has changed, find the named
  function with a search (`grep -n "def <name>" <file>`) and read from there.
- Do not read: other problems' files, `docs/ROADMAP.md`, `docs/audits/`, `private/`, `var/`
  outside your OUT, anything outside the repository. One exception: when your runbook names
  `var/audit-i32/harvest/<your problem id>/`, you may read the files in it.

## 3. Commands

Use only the commands written in your runbook, with your ids and paths filled in. Copy them
whole. A shell variable set in one call is gone in the next, so never shorten a command into a
variable.

Never run any of these:

```text
git commit   git add   git stash   git reset   git clean   git restore   git rebase   git merge
git push     git checkout <branch>            rm -r        uv add        uv sync    uv lock
ruff format .   ruff check --fix .   (repo-wide)           make fmt      pip        curl
```

The only `git checkout` you may run is `git checkout -- <one file you own>`, to undo your own
edit of that file.

## 4. Never

1. Never edit `solution.*`, `starter.*`, `problem.md`, `hints.md`, `test_public.py`, `topic.yaml`,
   `lesson.md`, `cards.yaml` or `interview.md`.
2. Never delete an existing test. Never change what one checks: its condition, its values, its
   tolerance, its `max_examples`. Never add `skip`, `xfail` or a broad `except` to a test. The
   one exception is a helper your runbook gives you word for word.
3. Never use the `Write` tool on a file that already exists. Use `Edit` with the smallest
   `old_string` that is unique.
4. Never type into `variants/manifest.yaml`. The tool writes it.
5. Never edit `meta.yaml` by hand. The tool bumps the version.
6. Never change the reference so that a test passes, and never change an oracle's arguments
   until it agrees with the reference. If they disagree, that is a finding (§6).
7. Never invent a requirement. A test needs a sentence in `problem.md`. If you cannot copy the
   sentence from the output of `clauses`, there is no requirement.
8. No names of people, employers, clients or projects in anything you write.

## 5. Truth

- Report only what a command printed in this session. Copy the `GATE:` line, the `VERDICT:` line
  or the last line of output.
- If you did not run something, write "not run". If it failed, write "failed" and the line.
- "It should pass" is not a result.

## 6. When something goes wrong

You have **three attempts** at any gate for one problem. An attempt is one run of `gate` that
ends in `GATE: FAIL`.

| You see | Do |
|---|---|
| The same failure text twice in a row | Stop work on this problem. Go to "Giving up on a problem" |
| Third `GATE: FAIL` on one problem | Same |
| `reference FAIL` after you added a test | Your test is wrong, or the statement is unclear. Remove your test. Do not touch the reference. Note `reference-doubt` if a library disagrees with the reference |
| `accepts FAIL` after you added a test | Your test asks for something a correct solution need not do. Remove your test. Note `overreach` |
| A command in the runbook does not exist or prints a usage error | Do not guess another command. Note `blocked` with the exact output |
| Your case is not in the runbook's decision table | Do not decide. Note `question` and go to the next problem |
| `timeout`, `oom` or `error` on a file that should run | Run the same command once more, alone. If it repeats, note `blocked` |

**Giving up on a problem:**

1. Undo your edits to its tracked files, one file at a time:
   `git checkout -- content/<track>/<topic>/problems/<pNN-slug>/test_hidden.py`
2. Remove variants you added for it with the tool's `remove-variant` command.
3. Write the note:

```bash
uv run --frozen --group torch python scripts/exercise_gate.py note --out <OUT> <problem id> \
  --kind blocked --text "<what you tried, and the last failing line>"
```

4. Go to the next problem. A blocked problem is an acceptable result. A weakened test is not.

Note kinds: `blocked`, `question`, `unpromised`, `no-witness`, `overreach`, `reference-doubt`,
`message-defect`, `inspect`, `done`.

## 7. Before you finish

Run, and copy the last line of each into your report:

```bash
git status --short
uv run --frozen --group torch python scripts/exercise_gate.py ledger --out <OUT>
```

`git status --short` must list only paths you own. If it lists anything else that you changed,
undo it with `git checkout -- <file>` and say so. Paths another executor owns may show up too:
leave them alone and name them in your report.

Runbooks a1, a2 and a3a run before `scripts/exercise_gate.py` has a `ledger` command. There,
skip the second command and write "ledger: not available yet" in the report.

End with this report and nothing after it:

```text
RUNBOOK: <file name>
OUT: <directory>
PROBLEMS:
  <problem id>: <GATE: PASS | blocked | question>  <one line>
FILES CHANGED: <output of git status --short>
NOTES WRITTEN: <count by kind>
NOT RUN: <anything in the runbook you skipped, and why>
```
