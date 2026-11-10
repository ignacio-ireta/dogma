# I-32 coordinator runbook

*For the one session that runs the program. Read [00-protocol](00-protocol.md) and the
[master plan](../2026-11-09-test-strength.md) first. You dispatch executors, check their work
with commands, commit, and write things down. You do not write tests or variants yourself, and
you do not settle `question` notes.*

## Rules for you

- Never `git add -A`, `git add .`, `git commit -a`, `git push`, `git reset`, `git stash`,
  `git rebase`, or anything with `--force`.
- Never commit a problem whose `gate` you have not seen end in `GATE: PASS` in this session.
- Never edit a runbook, a test, a variant or a statement. If a runbook is wrong, stop (§8).
- Never open `var/dogma.db` or run a `dogma` command that reads learner progress.
- After a context summary or a restart, do not trust your memory of what is done. Run the
  `worklist` command for the current sweep and continue from its output.

## 1. Preconditions

Run in the main checkout. Stop and tell the owner if a line does not match.

```bash
uname                                   # Darwin (the runner needs macOS Seatbelt)
ls /usr/bin/sandbox-exec                # exists
grep -n "^| I-31a \|^| I-31c " docs/ROADMAP.md
```

I-31a must read `done` before the platform phase (§3). I-31c must read `done` before any sweep
(§5 to §7).

## 2. Set up the worktree (once)

```bash
git worktree add -b i32-test-strength .claude/worktrees/i32 main
cd .claude/worktrees/i32
mkdir -p private
ln -s "$(cd "$(git rev-parse --git-common-dir)/.." && pwd)/private/denylist.txt" private/denylist.txt
uv run --frozen --group torch python -c "import torch, dogma, dogma_data; print('env ok')"
cd web && pnpm install --frozen-lockfile && cd ..
pwd
```

The link gives the privacy audit its rules without anyone reading them. If `pnpm install` cannot
finish, stop and ask. Every later command in this file runs from the path `pwd` printed. Give that
path to every executor.

If the branch or the worktree already exists, do not recreate it. `cd` into it and run
`git status --short` and `git log --oneline -5` to see where things stand.

## 3. Platform phase (I-32a)

Order: a1 and a2 together, then a3a, a3b, a3c, a3d, a4, one at a time.

Dispatch one agent per runbook (`model: sonnet`) with this prompt, filled in:

```text
You are a platform executor for increment I-32 of the dogma repository.
First run: cd <worktree path> && pwd
Read docs/superpowers/plans/2026-11-09-test-strength/00-protocol.md, then
docs/superpowers/plans/2026-11-09-test-strength/<runbook file>. Read nothing else in that folder.
You own exactly the paths in the runbook's "Owns" line.
OUT: var/audit-i32/<runbook name>
Follow the runbook in order. Do not commit. End with the report from protocol section 7.
```

When the agent returns, run the runbook's **Accept** commands yourself. If all pass:

```bash
git status --short                      # only the runbook's owned paths
git add <each owned path that changed>
git commit -m "<the commit message the runbook gives>"
```

If an Accept command fails, send the agent the failing output once (continue the same agent).
If it fails again, stop (§8).

After a4 is committed:

```bash
make check
git tag i32-start
```

Then merge (§7).

## 4. Harvest what I-31 left (once, optional)

If the main checkout still has `var/audit-i31/`, copy its proven files where the sweeps will
look for them:

```bash
uv run --frozen python - "$(cd "$(git rev-parse --git-common-dir)/.." && pwd)" var/audit-i32/harvest <<'EOF'
import hashlib, json, pathlib, shutil, sys
main, out = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2])
src = main / "var" / "audit-i31"
want = {}
for ledger in src.rglob("realism.jsonl"):
    for line in ledger.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            if (row["kind"], row["verdict"]) in {("blind", "survived"), ("wrong", "killed")}:
                want[row["file_sha"]] = (row["problem"], row["kind"])
done = set()
for path in sorted(src.rglob("*")):
    if path.suffix not in {".py", ".sql"} or not path.is_file():
        continue
    sha = hashlib.sha256(path.read_text(encoding="utf-8").encode()).hexdigest()[:16]
    if sha in want and sha not in done:
        problem, kind = want[sha]
        target = out / problem / kind / f"{sha}_{path.name}"
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
        done.add(sha)
print(f"harvested {len(done)} files")
EOF
```

On 2026-11-09 this found 28 blind and 33 wrong files for 28 problems. If the directory is gone,
skip this step. Nothing depends on it.

## 5. A sweep, in general

Sweeps: I-32b ([b-hygiene](b-hygiene.md)), I-32c ([c-blind-accepts](c-blind-accepts.md)),
I-32d ([d-reject-sweep](d-reject-sweep.md)), I-32d2 ([d2-mutant-triage](d2-mutant-triage.md)).
I-32c has its own steps in its runbook. The other three follow this loop.

**5.1 Get the list.**

```bash
uv run --frozen --group torch python scripts/exercise_gate.py worklist --out var/audit-i32/coord --kind <hygiene|reject|mutants>
```

It prints problem ids in order, grouped by path week. An empty list means the sweep is done.

**5.2 Pilot first.** The first batch of every sweep is the five pilot problems in the master
plan (those that appear in the list). Finish §5.3 to §5.5 for them before any other batch. If a
pilot executor wrote a `blocked` or `question` note about the runbook itself, stop (§8).

**5.3 Dispatch.** Batch size: 5 problems for hygiene, one topic for reject and mutants. At most
4 agents at a time. No two agents may own the same topic. Prompt:

```text
You are a <hygiene executor | variant author> for increment I-32 of the dogma repository.
First run: cd <worktree path> && pwd
Read docs/superpowers/plans/2026-11-09-test-strength/00-protocol.md, then
docs/superpowers/plans/2026-11-09-test-strength/<runbook file>. Read nothing else in that folder
unless the runbook names it.
You own these problems and nothing else:
  <problem id>
  <problem id>
OUT: var/audit-i32/<sweep letter>-<batch number>
Follow the runbook for each problem, in the order listed. Do not commit.
End with the report from protocol section 7.
```

**5.4 Check each returned problem yourself.** Mode: `text` for hygiene (or `refactor` where the
report says a `pytest.raises` was replaced); for reject and mutants, `add` if
`git status --short` lists that problem's `test_hidden.py`, else `none`.

```bash
uv run --frozen --group torch python scripts/exercise_gate.py gate --out var/audit-i32/coord <problem id> --mode <mode>
uv run --frozen --group torch dogma content verify <topic id>
uv run --frozen ruff check content/<track>/<topic>
uv run --frozen ruff format --check content/<track>/<topic>
git status --short
```

All four commands must succeed, and `git status --short` must list only paths inside topics you
dispatched in this batch.

- **All pass:** commit the topic (§5.5).
- **Hygiene only: `lint` is the single failing check, and the executor wrote a `question` note
  for the problem:** commit it. Some sites cannot be fixed without a decision.
- **`gate` fails otherwise:** do not commit. Undo the problem: `git checkout -- content/<track>/<topic>`,
  then `git status --short content/<track>/<topic>` and `rm` each untracked file it lists, one
  path per command. Write a note with `--kind blocked` and the failing check. The problem stays
  on the worklist. Dispatch it once more in a later batch, with the failing line added to the
  prompt. After a second failure, leave it for the decisions file.
- **A path outside the batch changed:** undo that file with `git checkout -- <file>` and tell the
  owner at the end of the wave.

For hygiene, also read the diff of one problem in five (`git diff content/<…>/test_hidden.py`)
and check three things: each new message says what rule was broken, none contains a number from
the test, and none contradicts the test's name. One bad message means the whole batch goes back
to its agent with the example.

**5.5 Commit one topic at a time.**

```bash
git add content/<track>/<topic>
git commit -m "content(<track>): <hygiene fixes | reject variants and gap tests | mutant triage> for <topic> (I-32<letter>)"
```

Do not change `status` in any `meta.yaml` (master plan, decision 11).

**5.6 End of a wave** (a wave is five batches for hygiene, one path week for reject and mutants):

```bash
git status --short                      # must print nothing
make check
uv run --frozen --group torch python scripts/exercise_gate.py ledger --out var/audit-i32/coord --all \
  --kind question --kind overreach --kind reference-doubt --kind blocked --kind message-defect
```

Append the ledger output to `docs/audits/test-strength-decisions.md` under a heading with the
date and the wave. Create the file on first use with one line of title. Commit it alone:
`git add docs/audits/test-strength-decisions.md` and
`git commit -m "docs: record I-32 decisions for <wave>"`. Then merge (§7).

## 6. Close-out

When every worklist is empty, or every problem left on one has a note in the decisions file:
dispatch Part A of [e-closeout](e-closeout.md) as a platform executor (§3 prompt), commit it,
then do Part B yourself.

## 7. Merge to main

At the end of I-32a, of I-32b, of I-32c, of each I-32d wave, of I-32d2 and of I-32e:

```bash
git merge main                          # in the worktree; brings main in
make test-all
cd "$(cd "$(git rev-parse --git-common-dir)/.." && pwd)" && git merge --ff-only i32-test-strength
```

If `git merge main` reports a conflict, run `git merge --abort` and stop (§8). If `make test-all`
fails, do not merge. If the fast-forward is refused for any reason, stop (§8). After a merge,
`cd` back into the worktree.

Then append a handoff entry to `docs/ROADMAP.md` §4 (newest first): what changed, the commands
you ran with their exit codes, what is left. Set the increment's status in the table. Commit it
in the worktree as `docs: I-32<letter> handoff`.

## 8. Stop and ask the owner

Stop dispatching, write what happened at the top of `docs/audits/test-strength-decisions.md`, and
end your turn when:

- a precondition does not hold;
- a pilot executor could not follow its runbook;
- any `reference-doubt` note appears (finish the running batch first);
- more than one problem in five of a wave ends `blocked`;
- `make check` fails twice for a reason you cannot tie to a problem committed in this wave;
- a merge conflicts or a fast-forward is refused;
- a platform Accept command fails twice.

## 9. Resuming

Nothing about progress is stored outside git. To resume: `cd` into the worktree, run
`git status --short` (undo or finish what is listed, using §5.4), then run each sweep's
`worklist`. Whatever it prints is what is left.
