"""The course calendar over HTTP: the same use cases as ``dogma warp`` and ``dogma course reset``.

The rules live in ``dogma.learning.calendar``. This module maps them to the wire shapes.
"""

from __future__ import annotations

from datetime import datetime

from dogma.api.schemas.calendar import (
    CalendarOut,
    CalendarPointOut,
    CalendarWarpOut,
    CalendarWarpRequest,
    CalendarWeekOut,
    LastWarpOut,
    ResetOut,
)
from dogma.core.config import Settings
from dogma.learning import calendar
from dogma.learning.calendar import CalendarPoint, WarpKind


def status(settings: Settings, *, now: datetime) -> CalendarOut:
    view = calendar.status(settings, now=now)
    path = calendar.load_calendar(settings)
    return CalendarOut(
        offset_weeks=view.offset_weeks,
        current_week=view.current_week.week,
        anchor_week=view.anchor.week,
        anchor_problem_id=view.anchor.problem_id,
        start=view.start,
        end=view.end,
        weeks=[
            CalendarWeekOut(week=item.week, starts=item.starts, theme=item.theme)
            for item in sorted(path.weeks, key=lambda item: item.week)
        ],
        last_warp=(
            None
            if view.last_warp is None
            else LastWarpOut(
                kind=view.last_warp.kind.value,
                created_at=view.last_warp.created_at,
                offset_weeks=view.last_warp.offset_weeks,
            )
        ),
    )


def warp(settings: Settings, body: CalendarWarpRequest, *, now: datetime) -> CalendarWarpOut:
    request = calendar.WarpRequest(WarpKind(body.kind), week=body.week, weeks=body.weeks)
    result = calendar.preview(settings, request, now=now)
    # A dry run, or a move that changes nothing, writes nothing and needs no recording.
    if not body.dry_run and result.changed:
        result = calendar.apply(settings, request, now=now)
    found = calendar.status(settings, now=now).anchor
    return CalendarWarpOut(
        applied=result.written,
        changed=result.changed,
        before=_point(result.before),
        after=_point(result.after),
        anchor_week=found.week,
        anchor_problem_id=found.problem_id,
    )


def reset(settings: Settings, *, now: datetime) -> ResetOut:
    result = calendar.reset_course(settings, now=now)
    return ResetOut(
        backup_path=str(result.backup_path),
        snapshot_path=None if result.snapshot_path is None else str(result.snapshot_path),
        rows_cleared=dict(result.rows_cleared),
        files_set_aside=result.files_set_aside,
        history_dir=None if result.history_dir is None else str(result.history_dir),
        workspace_error=result.workspace_error,
        week_one_starts=result.week_one_starts,
        course_end=result.course_end,
    )


def _point(point: CalendarPoint) -> CalendarPointOut:
    return CalendarPointOut(
        offset_weeks=point.offset_weeks,
        current_week=point.current_week,
        current_starts=point.current_starts,
        mocks_from=point.mocks_from,
        interview_week=point.interview_week,
        end=point.end,
    )
