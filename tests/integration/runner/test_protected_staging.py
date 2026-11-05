"""Protected staging through ``runner.run``: the hidden suite comes from the bundle."""

from __future__ import annotations

from pathlib import Path

import pytest

from dogma.grading.models import RunStatus
from dogma.runner import new_attempt_id, run

pytestmark = pytest.mark.integration

REPO_ROOT = Path(__file__).resolve().parents[3]
PY_ADD = REPO_ROOT / "tests" / "fixtures" / "problems" / "py-add"
GOOD = "def add(a, b):\n    return a + b\n"


def _problem(root: Path, public: str, hidden: str, **meta: str) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    fields = {
        "id": "fixtures/staged/p01",
        "version": "1",
        "language": "python",
        "runtime": "py-numpy",
        "entry": "add",
        "timeout_s": "30",
        **meta,
    }
    (root / "meta.yaml").write_text(
        "".join(f"{key}: {value}\n" for key, value in fields.items()), encoding="utf-8"
    )
    (root / "test_public.py").write_text(public, encoding="utf-8")
    (root / "test_hidden.py").write_text(hidden, encoding="utf-8")
    return root


def _leftover_bundles(settings) -> list[str]:  # type: ignore[no-untyped-def]
    if not settings.runs_dir.exists():
        return []
    return [p.name for p in settings.runs_dir.iterdir() if ".tests-" in p.name]


def test_hidden_suite_is_not_in_the_writable_run_dir_and_still_runs(settings, submit):
    report = run(
        PY_ADD, submit(GOOD), include_hidden=True, settings=settings, attempt_id="staged-a"
    )
    assert report.status is RunStatus.PASSED
    assert report.counts("hidden") == (5, 5)
    run_dir = settings.runs_dir / "staged-a"
    assert (run_dir / "submission.py").is_file()
    assert not (run_dir / "test_hidden.py").exists()
    assert not (run_dir / "test_public.py").exists()
    assert _leftover_bundles(settings) == []


def test_fake_hidden_file_written_by_the_learner_does_not_replace_the_suite(
    settings, tmp_path, submit
):
    problem = _problem(
        tmp_path / "problem",
        public="from submission import add\n\ndef test_add():\n    assert add(1, 1) == 2\n",
        hidden="from submission import add\n\ndef test_real_hidden():\n    assert add(2, 2) == 4\n",
    )
    learner = (
        "import os\n"
        "open(os.path.join(os.getcwd(), 'test_hidden.py'), 'w').write(\n"
        "    'def test_fake():\\n    pass\\n')\n"
        "def add(a, b):\n    return a + b\n"
    )
    report = run(problem, submit(learner), include_hidden=True, settings=settings)
    names = {test.name for test in report.tests}
    assert "test_fake" not in names
    assert ("test_real_hidden", "hidden", "pass") in {
        (t.name, t.file, t.outcome) for t in report.tests
    }
    run_dir = settings.runs_dir / report.attempt_id
    assert (run_dir / "test_hidden.py").exists(), "the learner could write its own copy, unused"


def test_learner_cannot_tamper_with_the_bundle_during_the_run(settings, tmp_path, submit):
    problem = _problem(
        tmp_path / "problem",
        public=(
            "from submission import TAMPER\n\n"
            "def test_bundle_files_are_not_learner_writable():\n"
            "    assert TAMPER == {k: 'PermissionError' for k in TAMPER}, TAMPER\n"
        ),
        hidden="from submission import add\n\ndef test_hidden_adds():\n    assert add(2, 2) == 4\n",
    )
    learner = (
        "import os\n"
        "from pathlib import Path\n"
        "BUNDLE = Path(__file__).resolve().parent\n"
        "HIDDEN = BUNDLE / 'test_hidden.py'\n"
        "def _attempt(action):\n"
        "    try:\n"
        "        action()\n"
        "    except OSError as exc:\n"
        "        return type(exc).__name__\n"
        "    return 'allowed'\n"
        "TAMPER = {\n"
        "    'rewrite': _attempt(lambda: HIDDEN.write_text('def test_fake():\\n    pass\\n')),\n"
        "    'unlink': _attempt(lambda: os.unlink(HIDDEN)),\n"
        "    'rename': _attempt(lambda: os.rename(HIDDEN, BUNDLE / 'moved.py')),\n"
        "    'chmod': _attempt(lambda: os.chmod(HIDDEN, 0o666)),\n"
        "    'create': _attempt(lambda: (BUNDLE / 'extra_test.py').write_text('x')),\n"
        "}\n"
        "def add(a, b):\n    return a + b\n"
    )
    report = run(problem, submit(learner), include_hidden=True, settings=settings)
    assert report.counts("public") == (1, 1), [t.message for t in report.tests]
    assert report.counts("hidden") == (1, 1), "the real hidden suite still ran, unchanged"
    assert report.status is RunStatus.PASSED
    assert _leftover_bundles(settings) == []


def test_bundle_is_removed_after_a_timeout_and_the_attempt_is_kept(settings, tmp_path, submit):
    problem = _problem(
        tmp_path / "problem",
        public="from submission import add\n\ndef test_add():\n    assert add(1, 1) == 2\n",
        hidden="from submission import add\n\ndef test_hidden():\n    assert add(1, 1) == 2\n",
        timeout_s="1",
    )
    attempt = new_attempt_id()
    report = run(
        problem,
        submit("import time\ntime.sleep(60)\ndef add(a, b):\n    return a + b\n"),
        include_hidden=True,
        settings=settings,
        attempt_id=attempt,
    )
    assert report.status is RunStatus.TIMEOUT
    assert _leftover_bundles(settings) == []
    assert (settings.runs_dir / attempt).is_dir()
