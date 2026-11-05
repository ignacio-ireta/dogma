"""Progress reporting: rich bar for TTY runs, no-op for tests and pipes."""

from __future__ import annotations

from typing import Protocol

from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TaskID,
    TextColumn,
    TimeElapsedColumn,
)

from docmd.pipeline.state import FileRecord


class ProgressReporter(Protocol):
    def start(self, total: int) -> None: ...

    def advance(self, record: FileRecord) -> None: ...

    def stop(self) -> None: ...


class NullProgress:
    def start(self, total: int) -> None:
        pass

    def advance(self, record: FileRecord) -> None:
        pass

    def stop(self) -> None:
        pass


class RichProgress:
    def __init__(self) -> None:
        self._progress = Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TimeElapsedColumn(),
            transient=True,
        )
        self._task_id: TaskID | None = None

    def start(self, total: int) -> None:
        self._progress.start()
        self._task_id = self._progress.add_task("converting", total=total)

    def advance(self, record: FileRecord) -> None:
        if self._task_id is not None:
            self._progress.update(self._task_id, advance=1, description=record.source[-40:])

    def stop(self) -> None:
        self._progress.stop()
