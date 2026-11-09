"""A problem that uses ``dogma_data`` graded through the real runner, under enforcement.

The problem reads real and semi-synthetic data in module-scoped fixtures. These tests check
that such a problem passes in time, that the sandbox still confines it, and that a submission
may not import the test-data package.
"""

from __future__ import annotations

import shutil
import socket
import sys
import textwrap
import time
from pathlib import Path

import pytest

from dogma.grading.models import RunStatus
from dogma.grading.static import _PLATFORM_MESSAGE
from dogma.runner import policy, run

pytestmark = pytest.mark.integration

REPO_ROOT = Path(__file__).resolve().parents[3]
PY_REALISM = REPO_ROOT / "tests" / "fixtures" / "problems" / "py-realism"
REFERENCE = "import numpy as np\n\ndef column_means(X):\n    return np.nanmean(X, axis=0)\n"
TIME_BUDGET_MS = 10_000
_SUPPORTED = sys.platform == "darwin" and policy.SANDBOX_EXEC.is_file()
_NEEDS_SEATBELT = pytest.mark.skipif(
    not _SUPPORTED, reason="requires Darwin with /usr/bin/sandbox-exec"
)


def _probe_problem(tmp_path: Path, name: str, body: str) -> Path:
    """Copy the realism problem and replace its public tests with ``body``."""
    dest = tmp_path / "problems" / name / "py-realism"
    shutil.copytree(PY_REALISM, dest)
    (dest / "test_public.py").write_text(textwrap.dedent(body), encoding="utf-8")
    return dest


def test_reference_solution_passes_public_and_hidden_tests_within_budget(settings, submit):
    # The first run may pay for cold imports and caches; the timed run is the second one.
    warm = run(PY_REALISM, submit(REFERENCE), include_hidden=True, settings=settings)
    assert warm.status is RunStatus.PASSED, warm.message

    started = time.monotonic()
    report = run(PY_REALISM, submit(REFERENCE), include_hidden=True, settings=settings)
    elapsed_ms = (time.monotonic() - started) * 1000

    assert report.status is RunStatus.PASSED, [t.message for t in report.tests if t.outcome]
    assert report.counts("public") == (2, 2)
    assert report.counts("hidden") == (3, 3)
    assert report.problem_id == "fixtures/py-realism/p01-column-means"
    assert report.duration_ms < TIME_BUDGET_MS, f"graded in {report.duration_ms} ms"
    assert elapsed_ms < TIME_BUDGET_MS, f"the whole run took {elapsed_ms:.0f} ms"


@_NEEDS_SEATBELT
def test_nothing_is_written_outside_the_run_dir_while_dogma_data_is_loaded(
    settings, submit, tmp_path
):
    settings.runs_dir.mkdir(parents=True)
    sentinel = settings.runs_dir / "sentinel.txt"
    sentinel.write_bytes(b"original")
    problem = _probe_problem(
        tmp_path,
        "write-probe",
        f"""\
        import numpy as np
        import pytest
        from dogma_data import load
        from submission import column_means

        SENTINEL = {str(sentinel)!r}

        def test_data_loads_and_means_hold():
            assert column_means(np.array([[1.0], [3.0]])).tolist() == [2.0]
            assert load("iris").X.shape == (150, 4)

        def test_sibling_file_cannot_be_rewritten_or_created():
            with pytest.raises(PermissionError):
                open(SENTINEL, "wb").write(b"hacked")
            with pytest.raises(PermissionError):
                open(SENTINEL + ".new", "wb").write(b"x")

        def test_sibling_file_cannot_be_removed():
            import os
            with pytest.raises(PermissionError):
                os.unlink(SENTINEL)
        """,
    )
    report = run(problem, submit(REFERENCE), include_hidden=False, settings=settings)
    assert report.status is RunStatus.PASSED, report.message or [t.message for t in report.tests]
    assert report.counts("public") == (3, 3)
    assert sentinel.read_bytes() == b"original"
    names = {p.name for p in settings.runs_dir.iterdir()}
    assert names == {"sentinel.txt", report.attempt_id}, names


@_NEEDS_SEATBELT
def test_network_is_denied_while_dogma_data_is_loaded(settings, submit, tmp_path):
    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    listener.bind(("127.0.0.1", 0))
    listener.listen(4)
    listener.settimeout(0.5)
    port = listener.getsockname()[1]
    try:
        problem = _probe_problem(
            tmp_path,
            "network-probe",
            f"""\
            import socket
            from dogma_data import load

            def test_data_loads_before_the_probe():
                assert load("wine").X.shape == (178, 13)

            def test_loopback_connection_is_denied():
                probe = socket.socket()
                probe.settimeout(5)
                try:
                    probe.connect(("127.0.0.1", {port}))
                except PermissionError:
                    return
                finally:
                    probe.close()
                raise AssertionError("the sandbox let a learner connection through")
            """,
        )
        report = run(problem, submit(REFERENCE), include_hidden=False, settings=settings)
        assert report.status is RunStatus.PASSED, report.message or [
            t.message for t in report.tests
        ]
        with pytest.raises(TimeoutError):
            listener.accept()  # no connection from the learner reached the listener
    finally:
        listener.close()


@pytest.mark.parametrize(
    "source",
    [
        "import dogma_data\n\ndef column_means(X):\n    return X.mean(axis=0)\n",
        "from dogma_data import synth\n\ndef column_means(X):\n    return X.mean(axis=0)\n",
    ],
    ids=["import", "from-import"],
)
def test_submission_importing_dogma_data_is_rejected_before_it_runs(settings, submit, source):
    report = run(PY_REALISM, submit(source), include_hidden=True, settings=settings)
    assert report.status is RunStatus.REJECTED
    assert report.tests == ()
    assert [f.rule for f in report.static] == ["banned-import"]
    assert report.static[0].message == f"{_PLATFORM_MESSAGE} (line 1)"
    assert not settings.runs_dir.exists(), "a rejected submission must not create a run dir"
