"""Python problems through ``runner.run``: real subprocesses, real limits."""

from __future__ import annotations

import json
import textwrap
import time
from pathlib import Path

import psutil
import pytest

from dogma.core.config import Settings
from dogma.grading.models import RunStatus
from dogma.runner import run, sandbox

pytestmark = pytest.mark.integration

REPO_ROOT = Path(__file__).resolve().parents[3]
PY_ADD = REPO_ROOT / "tests" / "fixtures" / "problems" / "py-add"
GOOD = "def add(a, b):\n    return a + b\n"


def _gone(pid: int, timeout_s: float = 5.0) -> bool:
    """True once the process has exited (or is a zombie awaiting reaping)."""
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        try:
            if psutil.Process(pid).status() == psutil.STATUS_ZOMBIE:
                return True
        except psutil.NoSuchProcess:
            return True
        time.sleep(0.1)
    return False


def test_correct_solution_passes_public_and_hidden_tests(settings, submit):
    report = run(PY_ADD, submit(GOOD), include_hidden=True, settings=settings)
    assert report.status is RunStatus.PASSED
    assert report.passed
    assert report.counts("public") == (3, 3)
    assert report.counts("hidden") == (5, 5)
    assert report.problem_id == "fixtures/py-add/p01-add"
    assert report.duration_ms > 0


def test_wrong_solution_fails_and_says_what_went_wrong(settings, submit):
    report = run(
        PY_ADD, submit("def add(a, b):\n    return a - b\n"), include_hidden=True, settings=settings
    )
    assert report.status is RunStatus.FAILED
    assert not report.passed
    by_name = {t.name: t for t in report.tests}
    assert by_name["test_adds_two_positive_integers"].outcome == "fail"
    assert "add(2, 3) should be 5" in by_name["test_adds_two_positive_integers"].message
    hidden_failures = [t for t in report.tests if t.file == "hidden" and t.outcome == "fail"]
    assert hidden_failures, "the hidden tests must also catch subtraction"
    assert "add(" in hidden_failures[0].message, "the hidden message should show the input"


def test_public_only_run_never_executes_the_hidden_tests(settings, submit):
    report = run(PY_ADD, submit(GOOD), include_hidden=False, settings=settings)
    assert report.status is RunStatus.PASSED
    assert {t.file for t in report.tests} == {"public"}
    assert report.counts("hidden") == (0, 0)


def test_syntax_error_is_rejected_and_nothing_runs(settings, submit):
    report = run(
        PY_ADD, submit("def add(a, b) return a + b\n"), include_hidden=True, settings=settings
    )
    assert report.status is RunStatus.REJECTED
    assert [f.rule for f in report.static] == ["syntax"]
    assert report.tests == ()
    assert not settings.runs_dir.exists(), "a rejected attempt must not create a run directory"


def test_missing_entry_is_rejected(settings, submit):
    report = run(
        PY_ADD,
        submit("def plus(a, b):\n    return a + b\n"),
        include_hidden=True,
        settings=settings,
    )
    assert report.status is RunStatus.REJECTED
    assert [f.rule for f in report.static] == ["missing-entry"]


def test_banned_import_is_rejected(settings, submit):
    source = "import numpy.linalg\n\ndef add(a, b):\n    return a + b\n"
    report = run(PY_ADD, submit(source), include_hidden=True, settings=settings)
    assert report.status is RunStatus.REJECTED
    assert [f.rule for f in report.static] == ["banned-import"]
    assert report.static[0].line == 1


def test_banned_attribute_access_is_rejected(settings, submit):
    source = "import numpy as np\n\ndef add(a, b):\n    _ = np.linalg\n    return a + b\n"
    report = run(PY_ADD, submit(source), include_hidden=True, settings=settings)
    assert report.status is RunStatus.REJECTED
    assert report.static[0].rule == "banned-import"


def test_infinite_loop_is_stopped_at_the_time_limit(settings, submit, problem_copy):
    problem = problem_copy(PY_ADD, timeout_s=2)
    source = "def add(a, b):\n    while True:\n        pass\n"
    started = time.monotonic()
    report = run(problem, submit(source), include_hidden=True, settings=settings)
    elapsed = time.monotonic() - started
    assert report.status is RunStatus.TIMEOUT
    assert "2 s" in report.message
    assert elapsed < 20, f"a 2 s limit took {elapsed:.1f} s to enforce"


def test_timeout_kills_the_whole_process_group_including_children(settings, submit, problem_copy):
    problem = problem_copy(PY_ADD, timeout_s=2)
    source = textwrap.dedent(
        """\
        import subprocess
        import sys
        import time

        child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
        with open("child.pid", "w", encoding="utf-8") as handle:
            handle.write(str(child.pid))

        def add(a, b):
            while True:
                time.sleep(0.05)
        """
    )
    report = run(problem, submit(source), include_hidden=False, settings=settings)
    assert report.status is RunStatus.TIMEOUT
    child_pid = int((settings.runs_dir / report.attempt_id / "child.pid").read_text())
    assert _gone(child_pid), "a child of the timed-out run must not survive it"


def test_memory_hog_is_stopped_by_the_rss_watchdog(settings, submit):
    # The learner holds the buffer while it sleeps: the watchdog samples every 50 ms, so a spike
    # freed in under that window would not be seen (a documented limit of polling).
    source = textwrap.dedent(
        """\
        import time

        def add(a, b):
            hog = bytearray(600 * 1024 * 1024)
            for offset in range(0, len(hog), 4096):
                hog[offset] = 1
            time.sleep(30)
            return a + b
        """
    )
    report = run(
        PY_ADD, submit(source), include_hidden=False, settings=settings, memory_limit_mb=200
    )
    assert report.status is RunStatus.OOM
    assert "200 MB" in report.message


def test_huge_stdout_is_truncated_to_64_kb(settings, submit):
    source = "def add(a, b):\n    print('x' * 1_000_000)\n    return a + b\n"
    report = run(PY_ADD, submit(source), include_hidden=False, settings=settings)
    assert report.status is RunStatus.PASSED
    assert len(report.stdout.encode("utf-8")) <= 64 * 1024
    assert "truncated" in report.stdout


def test_endless_printing_costs_time_not_memory(settings, submit, problem_copy):
    # Uncapped, this loop reaches the 4 GB default in under a second. Bounded capture keeps
    # memory flat, so the run ends at its time limit (timeout) and never as an oom.
    problem = problem_copy(PY_ADD, timeout_s=3)
    source = "def add(a, b):\n    while True:\n        print('x' * 100_000)\n"
    report = run(problem, submit(source), include_hidden=False, settings=settings)
    assert report.status is RunStatus.TIMEOUT


def test_writing_too_much_to_disk_stops_the_run(settings, submit, monkeypatch):
    monkeypatch.setattr(sandbox, "MAX_RUN_DIR_BYTES", 4 * 1024 * 1024)
    source = textwrap.dedent(
        """\
        import pathlib
        import time

        def add(a, b):
            pathlib.Path("big.bin").write_bytes(b"0" * (8 * 1024 * 1024))
            time.sleep(30)
            return a + b
        """
    )
    report = run(PY_ADD, submit(source), include_hidden=False, settings=settings)
    assert report.status is RunStatus.ERROR
    assert "wrote more than" in report.message


def test_staged_fixtures_do_not_count_against_the_disk_limit(
    settings, submit, problem_copy, monkeypatch
):
    monkeypatch.setattr(sandbox, "MAX_RUN_DIR_BYTES", 4 * 1024 * 1024)
    problem = problem_copy(PY_ADD)
    (problem / "fixtures").mkdir()
    (problem / "fixtures" / "big.bin").write_bytes(b"0" * (6 * 1024 * 1024))
    report = run(problem, submit(GOOD), include_hidden=False, settings=settings)
    assert report.status is RunStatus.PASSED, report.message


def test_exit_before_reporting_is_an_error_with_no_tests(settings, submit):
    source = "import os\n\nos._exit(0)\n\ndef add(a, b):\n    return a + b\n"
    report = run(PY_ADD, submit(source), include_hidden=True, settings=settings)
    assert report.status is RunStatus.ERROR
    assert report.tests == ()
    assert "without a report" in report.message


def test_unreadable_report_is_an_error_with_the_stderr_tail(settings, submit):
    # Learner code shares the run directory, so it can overwrite report.json. A report that does
    # not match the schema must read as an error, never as a pass.
    source = textwrap.dedent(
        """\
        import os
        import sys

        with open("report.json", "w", encoding="utf-8") as handle:
            handle.write('{"status": "passed", "tests": "not-a-list"}')
        sys.stderr.write("harness trace line\\n")
        sys.stderr.flush()
        os._exit(0)

        def add(a, b):
            return a + b
        """
    )
    report = run(PY_ADD, submit(source), include_hidden=True, settings=settings)
    assert report.status is RunStatus.ERROR
    assert report.tests == ()
    assert "unreadable report" in report.message


def test_error_while_importing_the_submission_names_the_exception(settings, submit):
    source = "raise RuntimeError('boom at import')\n\ndef add(a, b):\n    return a + b\n"
    report = run(PY_ADD, submit(source), include_hidden=True, settings=settings)
    assert report.status is RunStatus.ERROR
    assert report.tests == ()
    assert "boom at import" in report.message


def test_learner_code_runs_in_its_own_scrubbed_directory(settings, submit):
    source = textwrap.dedent(
        """\
        import json
        import os
        import sys
        import tempfile

        def add(a, b):
            print(json.dumps({
                "hash_randomization": sys.flags.hash_randomization,
                "env": sorted(os.environ),
                "home": os.path.realpath(os.environ.get("HOME", "")),
                "cwd": os.path.realpath(os.getcwd()),
                "tmp": os.path.realpath(tempfile.gettempdir()),
            }))
            return a + b
        """
    )
    report = run(
        PY_ADD, submit(source), include_hidden=False, settings=settings, attempt_id="env-check"
    )
    line = next(line for line in report.stdout.splitlines() if line.startswith("{"))
    seen = json.loads(line)
    run_dir = (settings.runs_dir / "env-check").resolve()
    assert "PYTHONPATH" not in seen["env"]
    assert seen["hash_randomization"] == 0, "PYTHONHASHSEED=0 must make hashing deterministic"
    assert seen["cwd"] == str(run_dir)
    assert seen["home"] == str(run_dir)
    assert seen["tmp"] == str(run_dir), "temporary files must stay in the run directory"


def test_attempts_land_in_the_given_repo_root_and_never_in_the_real_var(settings, submit):
    report = run(PY_ADD, submit(GOOD), include_hidden=False, settings=settings)
    assert report.status is RunStatus.PASSED
    assert (settings.runs_dir / report.attempt_id).is_dir()
    assert not (REPO_ROOT / "var" / "runs" / report.attempt_id).exists()


def test_settings_run_limits_apply_when_meta_has_none(tmp_path, submit):
    # run_timeout_s from Settings is the fallback when meta.yaml sets no timeout_s.
    settings = Settings(repo_root=tmp_path / "repo", run_timeout_s=2)
    problem = tmp_path / "problem"
    problem.mkdir()
    (problem / "meta.yaml").write_text(
        "id: x/p01\nversion: 1\nlanguage: python\nruntime: py-numpy\nentry: add\n",
        encoding="utf-8",
    )
    (problem / "test_public.py").write_text(
        "from submission import add\n\ndef test_a():\n    assert add(1, 2) == 3\n",
        encoding="utf-8",
    )
    report = run(
        problem,
        submit("def add(a, b):\n    while True:\n        pass\n"),
        include_hidden=False,
        settings=settings,
    )
    assert report.status is RunStatus.TIMEOUT
    assert "2 s" in report.message
