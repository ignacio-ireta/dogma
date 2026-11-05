# Runner Hardening Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking. The coordinator reviews this focused plan before implementation; execution stays within the already approved maintenance scope.

**Goal:** Complete I-21 with enforced macOS network/write denial, protected test staging, and conservative run-artifact retention that preserves recorded learner attempts.

**Architecture:** Keep the existing subprocess, process group, watchdogs, GradeReport, and CLI/API grading seams. Wrap the harness in macOS Seatbelt, stage a temporary read-only test bundle outside the writable attempt directory, and expose explicit maintenance that removes only old, completed, recognizable, unreferenced run directories. No background cleanup or learner-state migration.

**Tech Stack:** Python 3.12+, stdlib subprocess/sqlite3/filesystem APIs, existing psutil/pytest/Typer; `/usr/bin/sandbox-exec` on Darwin. No dependency or lockfile changes.

**Spec:** `docs/superpowers/specs/2026-10-24-learning-sequence-design.md` (maintenance and verification), `docs/superpowers/plans/2026-10-24-learning-sequence.md` Task 7, `docs/ROADMAP.md` I-21, `docs/ARCHITECTURE.md` §§4.2–4.3 and ADR-0006/0015 decision-table entries, `docs/ENGINEERING_STANDARDS.md` §§6/11/16.

## Global Constraints

- "I-21 hardens execution and retention without deleting learner attempts."
- "All development databases and runs use scratch settings. Never read private learner data or mutate real progress for testing. Preserve no-record semantics."
- "Threat model: one trusted user on their own machine." No hostile-code security-boundary claim.
- "Use `uv run --frozen`." Keep existing imports, static rejections, deterministic hash seed, public/hidden visibility, SQL, timeout/OOM/disk/output guards, and recording defaults.
- No writes to real `var/`, `workspace/`, or `private/` while developing or verifying. No live model/browser gate, content edits, Git mutations, runtime dependency changes, or unrelated refactor.
- Fail closed when OS enforcement is unsupported or cannot start; never retry learner execution without the wrapper.

## Review Focus

- Executable presence can hide a failed Seatbelt launch: test unsupported OS, missing executable, and rejected `sandbox_apply`, with no unsandboxed retry.
- Existing SQL tests resolve submission and fixtures beside `__file__`: test an external test bundle with unchanged SQL fixture problems.
- Symlinks and subprocess descendants can bypass naive path/string checks: exercise outside-write denial through a scratch symlink and a spawned child.
- Cancellation, staging errors, and watchdog kills can leak processes or protected bundles: test cleanup and preserve original failures.
- Old, malformed, linked, or concurrently active artifact directories may contain learner work: retention skips them and every database-referenced attempt, including malformed `run_dir` values protected by attempt id.

## Evidence and decisions

On 2026-10-25 this host is Darwin arm64 and has `/usr/bin/sandbox-exec`. An inert profile running `/usr/bin/true` fails inside the managed tool sandbox with exit 71 (`sandbox_apply: Operation not permitted`) and succeeds with authorized host execution (exit 0). Therefore enforcement integration tests must run in the host execution context, not weaken the profile to accommodate nested sandbox restrictions. An existence-only platform check is insufficient.

ADR-0006 exists as a decision-table entry in `ARCHITECTURE.md`; there is no separate 0006 ADR file. Its trusted-user and defense-in-depth language remains true. This change deliberately makes execution unsupported outside working Darwin Seatbelt; unsupported platform errors occur before learner execution. Document this narrower support contract alongside the existing decision. Static rejection may still return its normal report without launching anything.

This profile prevents network operations and file writes outside the canonical attempt directory. It allows ordinary file reads so Python, dependencies, and the protected tests work. Same-user readable repository/private/home paths are consequently still readable by learner code; test source is readable, and learner code shares the pytest process/report channel. Relocation protects test-file integrity, not secrecy or report authenticity. Existing process-group cleanup also does not contain an adversarial child that escapes its session. Strong hostile-code isolation requires a different architecture and is outside I-21.

The retention decision is explicit `dogma runs prune --older-than-days 30`, dry-run by default, with `--apply` to delete. The minimum accepted age is one day. Preserve every recorded attempt directory and DB row; skip all unknown legacy directories. New completed unrecorded runs can expire. No automatic cleanup is added to grading or startup, and retention never touches learner workspace files.

## File ownership for implementation

| Files | Responsibility |
|---|---|
| `src/dogma/runner/policy.py` (new), `sandbox.py` | Fixed Seatbelt profile, platform detection, wrapped argv, lifecycle preservation |
| `src/dogma/runner/staging.py` (new), `run.py` | Protected temporary test bundle and completion metadata |
| `src/dogma_harness/cli.py` | Optional external test directory while retaining standalone CLI compatibility |
| `src/dogma/runner/retention.py` (new) | Pure eligibility plus bounded filesystem cleanup, no DB imports |
| `src/dogma/db/run_artifacts.py` (new) | Read-only snapshot of protected attempt ids/run paths; no migration or initialization |
| `src/dogma/cli/runs.py` (new), `cli/main.py` | Explicit dry-run/apply maintenance orchestration |
| Runner/harness tests; `tests/unit/db/test_run_artifacts.py`, `tests/unit/cli/test_cli_runs.py` (new) | Scratch-only enforcement, compatibility, lifecycle, preservation tests |
| `docs/ARCHITECTURE.md`, `docs/CONTENT_SPEC.md`, `docs/ROADMAP.md` | Accurate support/staging/retention contract and increment handoff |

The small DB/CLI additions are necessary to protect recorded artifacts without coupling the runner to storage. They require coordinator ownership assignment before implementation. `run.py` is shared with the coordinator's I-19 runtime guard: request ownership after that edit lands, preserve that guard, and do not edit concurrently. No `Settings`, dependency, API, or DB schema change is needed.

### Task 1: Enforced subprocess policy

**Files:** Create `src/dogma/runner/policy.py`, `tests/unit/runner/test_runner_policy.py`, `tests/integration/runner/test_runner_policy.py`; modify `sandbox.py` and existing environment helper tests.

**Interfaces:**
- Produce `sandbox_command(command: list[str], run_dir: Path) -> list[str]` in `policy.py`.
- Consume it immediately before `_spawn`; return argv using a fixed executable/profile and `-D RUN_DIR=<resolved path>` parameter. Do not interpolate paths into the policy language or invoke a shell.
- Preserve `execute(...) -> SandboxResult` and all current kill reasons. Use existing `RunnerError(ErrorCode.RUNNER_CRASHED, ...)` for unavailable enforcement.

- [ ] Write failing unit cases for non-Darwin, missing/non-executable fixed `/usr/bin/sandbox-exec`, unusual path characters, and rejected wrapper startup. Assert there is exactly one wrapped spawn and no raw-python fallback. Keep errors explicit about unsupported enforcement.
- [ ] Write failing host integration cases with scratch `Settings`: in-run file write succeeds; sibling sentinel cannot be changed/created; a symlink inside the run pointing to that sibling cannot be used to write; a spawned Python child inherits denial. Assert actual file bytes, not just argv/profile text.
- [ ] Write a loopback TCP test: start a short-lived parent listener on `127.0.0.1:0`; prove an ordinary parent connection works, then assert the harness's connection raises a permission error and the listener receives no learner connection. Use bounded socket waits and `finally` close/join. No internet or live service needed.
- [ ] Run new tests before implementation and observe failures attributable to missing enforcement.
- [ ] Implement the minimal policy: allow baseline execution/reads, deny `network*` and `file-write*`, allow writes only beneath the canonical attempt directory. Permit `/dev/null` writes only if required by a demonstrated runtime; do not grant other mutable paths. Set `PYTHONDONTWRITEBYTECODE=1` in the scrubbed environment so imports need no writable package/test paths.
- [ ] Retain the exact process session, watchdog sampling, stdout/stderr sinks, timeout, join, kill and wait behavior. A failed wrapper launch must surface an infrastructure error and never execute the submission. Nonzero/no-report sandbox-application failure is an explicit infrastructure failure; ordinary learner failures keep GradeReports.
- [ ] Run the unit and actual-denial integration tests in the authorized host context. Unsupported-host tests exercise failure behavior; supported-host actual-denial tests must pass, not be skipped because the managed sandbox prevents nesting.

### Task 2: Protected staging with Python/SQL compatibility

**Files:** Create `src/dogma/runner/staging.py`; modify `runner/run.py`, `dogma_harness/cli.py`, runner guard/integration tests and `test_harness_cli.py`.

**Interfaces:**
- Produce `staged_tests(run_dir: Path, submission: Path, problem_dir: Path, meta: ProblemMeta, tests: tuple[str, ...]) -> AbstractContextManager[Path]` in `staging.py`.
- Extend `sandbox.execute(..., tests_dir: Path | None = None)`; pass `--tests-dir` only when supplied.
- Harness `--tests-dir` defaults to `--run-dir`. `_run(run_dir: Path, tests: tuple[str, ...], tests_dir: Path | None = None)` uses absolute requested test paths and explicit `-c <run_dir>/pytest.ini`. `Collector` already identifies visibility by test basename.

- [ ] Add failing cases proving `test_hidden.py` is absent from the writable run dir, is collected from the protected bundle, and cannot be overwritten, chmodded, renamed or unlinked by learner code. An in-run fake `test_hidden.py` must not replace the actual hidden suite.
- [ ] Add passing-contract expectations for unchanged Python and SQL fixture problems, public-only runs, fixture paths next to `__file__`, neutral test names/visibility, standalone harness calls without `--tests-dir`, missing test files and collection failures. Keep expected test counts and verdicts unchanged.
- [ ] Implement the context manager: copy submission plus fixtures into the normal run dir for preserved artifacts, and a separate temporary sibling bundle containing only selected tests, the submitted source snapshot, and fixture snapshots. The bundle sits outside the permitted write subtree and supplies the import/SQL sibling-file layout expected by current content. Do not copy solutions or unrelated files. Apply read-only modes to the bundle files/directories as defense in depth; Seatbelt is the actual write barrier.
- [ ] Protect staging from fixture symlinks resolving outside the problem tree; reject such input before following it. Normal copied fixtures retain their existing semantics. Make cleanup restore directory modes as needed and remove the bundle after all process-group cleanup, including timeout/OOM/disk/cancellation/staging/spawn failures. Never delete the retained run dir on a failed attempt, and never mask its original exception with a cleanup error.
- [ ] Verify no bytecode, pytest cache, or assertion rewriting requires bundle writes, and rerun the existing timeout, child kill, OOM, disk, huge stdout, scrubbed-env, SQL and example tests. Preserve I-19 runtime validation in `run.py`.

### Task 3: Conservative explicit artifact retention

**Files:** Create `runner/retention.py`, `db/run_artifacts.py`, `cli/runs.py` and their focused tests; modify `run.py` and `cli/main.py`.

**Interfaces:**
- `mark_completed(run_dir: Path, *, completed_at: datetime) -> None`: parent writes `.dogma-completed.json` atomically only after grading and child cleanup; JSON contains `schema_version=1`, matching `attempt_id`, aware UTC `completed_at`. It records artifact lifecycle only, never mastery or recording truth. Incomplete/interrupted runs remain unmarked and are skipped.
- `protected_runs(settings: Settings) -> tuple[frozenset[str], frozenset[str]]`: attempt ids plus non-null stored run-dir strings. Read existing SQLite via a read-only URI and SELECT only; missing DB returns empty sets without creating anything; malformed/unreadable/schema-incompatible DB aborts pruning. Do not use `open_session`, which initializes/migrates.
- `prune_runs(runs_dir: Path, *, protected_ids: frozenset[str], protected_paths: frozenset[Path], before: datetime, apply: bool = False) -> tuple[Path, ...]`: returns candidates for dry-run and successfully removed candidates for apply. CLI resolves stored relative paths against `settings.repo_root`; the runner module does not read the database.
- `dogma runs prune --older-than-days INTEGER [--apply]`: default age 30 days, minimum 1, aware UTC cutoff, dry-run by default. Protect all database ids even when their `run_dir` is null/malformed, and all resolvable stored paths. Show candidate/removal count and never record a practice attempt.

- [ ] Add failing scratch cases: only an old, completed, matching-marker, unreferenced ordinary immediate child is eligible; recent/future-dated/unknown-marker/legacy/unfinished dirs are skipped. Protect DB-referenced paths and ids, rejected-attempt ids, and inconsistent stored paths. No DB row, version, mastery, submission, timestamp, or recorded artifact may change.
- [ ] Test no-record repositories with no DB: dry-run and apply create no DB/catalog/workspace; ordinary grading and `DOGMA_RECORD_ATTEMPTS=false` retain current behavior. A bad existing DB aborts before any deletion. Retention must not initialize or migrate storage.
- [ ] Test symlink runs roots, symlink candidate directories/markers, path escapes, candidate replacement before deletion and nested symlinks pointing at scratch sentinels. Skip unsafe candidates; deletion never follows a link outside the chosen directory. Require symlink-resistant `shutil.rmtree` support for apply; otherwise fail closed. Recheck marker, containment, ordinary-directory identity and protection immediately before removal.
- [ ] Test an active run remains unmarked and ineligible, even if old by directory mtime. A completed run uses parent completion time rather than user-controlled directory mtime. Restrict candidate names to the existing attempt-id rule; exclude temporary bundle names. Legacy and interrupted directories require a later separately reviewed recovery/import tool; do not infer safe deletion from age alone.
- [ ] Implement the helpers and CLI wiring. Obtain the DB protection snapshot before enumeration; abort on partial enumeration/read failures rather than silently treating the protected set as empty. Positive minimum age and post-run completion timestamps prevent the normal grade-to-record gap from making a current run eligible; no cleanup runs inside that gap.
- [ ] Verify CLI help has a concrete dry-run example, apply is explicit, and filesystem errors are reported with stable existing domain/storage errors. Test retained recorded attempts and no-record behavior through the existing CLI/API test seams without a live server.

### Task 4: Documentation and focused verification

**Files:** `docs/ARCHITECTURE.md`, `docs/CONTENT_SPEC.md`, `docs/ROADMAP.md`; plan/report progress only.

- [ ] Update architecture to remove the completed I-21 gaps, specify supported Darwin/fail-closed behavior, describe protected snapshots versus writable cwd, preserve candid same-user/read/report limitations, and explain dry-run retention/legacy skips.
- [ ] Update CONTENT_SPEC §6.4 so authors still use unchanged `from submission import ...`, sibling SQL submission reads, and sibling fixtures; runner staging remains an implementation detail and standalone harness defaults remain valid. No content version bumps are required because tests and problem contracts do not change.
- [ ] Run focused verification once: `uv run --frozen pytest tests/unit/runner tests/unit/grading tests/integration/runner tests/unit/db/test_run_artifacts.py tests/unit/cli/test_cli_runs.py -q`, then existing relevant CLI/API recording regressions, ruff, mypy and import-linter for modified code. Run actual Seatbelt enforcement tests in host context.
- [ ] Report exact results to the coordinator, including any unavailable host enforcement. The coordinator owns the one consolidated `make check`/full gate and whole-branch review; do not duplicate a broad browser/model gate here. I-21 stays incomplete if actual write/network-denial tests have not passed on this supported host.
- [ ] Update I-21 status and roadmap handoff only after passing evidence; no Git commits/staging until coordinator authorization.

## Migration and residual risks

No Alembic migration, dependency change, learner-state rewrite, or new progress source is introduced. Existing GradeReports and stored `attempt.run_dir` values remain compatible. Existing run directories remain intact and intentionally ineligible for automatic deletion. Standalone harness usage continues without the new option. Fail-closed unsupported platforms are a deliberate behavior change and must be visible in docs/errors.

Protected bundle snapshots temporarily duplicate submitted source/fixtures and consume extra disk; ensure staging failures clean their partial bundle before launch. The writable-directory disk watchdog keeps its established growth semantics and does not treat parent-staged fixture bytes as learner writes. Recorded run directories are retained indefinitely; the command bounds disposable unrecorded artifacts only. Cleanup cannot prove safety against a hostile same-user actor racing filesystem/DB state, and this implementation must not claim it does.
