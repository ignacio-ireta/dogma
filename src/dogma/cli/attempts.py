"""How the CLI stores a graded attempt and shows a warning when storage fails.

Grading itself is in ``dogma.runner.attempt`` (shared with the API). This module only prints.
"""

from __future__ import annotations

from datetime import datetime

from rich.markup import escape

from dogma.cli.context import err
from dogma.core.config import Settings
from dogma.db.errors import StorageError
from dogma.db.models import AttemptMode
from dogma.db.store import record_attempt
from dogma.grading.models import GradeReport
from dogma.learning.resolves import ClosedWindow


def record(
    settings: Settings,
    report: GradeReport,
    *,
    mode: AttemptMode,
    core: bool,
    started_at: datetime,
    submitted_at: datetime,
) -> ClosedWindow | None:
    """Store the attempt. A storage failure is one warning line; the grade is still the answer.

    Returns the verdict of the timed re-solve this submit closed, if one was open.
    """
    try:
        return record_attempt(
            settings,
            report,
            mode=mode,
            core=core,
            started_at=started_at,
            submitted_at=submitted_at,
        )
    except StorageError as exc:
        err().print(
            f"[bold yellow]warning[/] {escape(f'[{exc.code}]')} "
            f"attempt not recorded: {escape(exc.message)}",
            soft_wrap=True,
        )
        return None
