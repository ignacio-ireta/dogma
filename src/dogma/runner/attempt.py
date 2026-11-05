"""The one place where the app asks the runner to grade an attempt (ARCHITECTURE §4.2, §4.7).

The CLI and the API call ``grade_attempt`` through this module at call time, so tests can replace
it with a fake that returns crafted GradeReports, without touching the sandbox. The module has no
CLI or HTTP code, so both layers can depend on it.
"""

from __future__ import annotations

from pathlib import Path

from dogma.core.config import Settings
from dogma.grading.models import GradeReport
from dogma.runner.run import run


def grade_attempt(
    problem_dir: Path,
    submission: Path,
    *,
    include_hidden: bool,
    settings: Settings,
) -> GradeReport:
    """Grade ``submission`` against the problem in ``problem_dir`` (ARCHITECTURE §4.3)."""
    return run(problem_dir, submission, include_hidden=include_hidden, settings=settings)
