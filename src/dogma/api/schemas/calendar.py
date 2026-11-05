"""The personal course calendar: the shifted dates, the warp request and the reset (ADR-0019)."""

from __future__ import annotations

from datetime import date, datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

WarpKindOut = Literal["sync", "to_week", "ahead", "back", "original", "course_reset"]
WarpKindIn = Literal["sync", "to_week", "ahead", "back", "original"]


class CalendarWeekOut(BaseModel):
    week: int
    starts: date
    theme: str


class LastWarpOut(BaseModel):
    kind: WarpKindOut
    created_at: datetime
    offset_weeks: int


class CalendarOut(BaseModel):
    """The calendar as the learner sees it: the offset in force, this week, and the course."""

    offset_weeks: int
    current_week: int
    anchor_week: int
    anchor_problem_id: str | None
    start: date
    end: date
    weeks: list[CalendarWeekOut]
    last_warp: LastWarpOut | None


class CalendarWarpRequest(BaseModel):
    """A move. ``week`` goes with to_week; ``weeks`` with ahead and back; neither with the rest."""

    model_config = ConfigDict(extra="forbid")

    kind: WarpKindIn
    week: int | None = Field(default=None, ge=1)
    weeks: int | None = Field(default=None, ge=1)
    dry_run: bool = False


class CalendarPointOut(BaseModel):
    offset_weeks: int
    current_week: int
    current_starts: date
    mocks_from: date
    interview_week: date
    end: date


class CalendarWarpOut(BaseModel):
    """What the move does. ``applied`` is true only when a row was written."""

    applied: bool
    changed: bool
    before: CalendarPointOut
    after: CalendarPointOut
    anchor_week: int
    anchor_problem_id: str | None


class ResetRequest(BaseModel):
    """Starting over needs the literal word ``reset``. Anything else is a 422."""

    model_config = ConfigDict(extra="forbid")

    confirm: Literal["reset"]


class ResetOut(BaseModel):
    backup_path: str
    snapshot_path: str | None
    rows_cleared: dict[str, int]
    files_set_aside: int
    history_dir: str | None
    workspace_error: str | None
    week_one_starts: date
    course_end: date
