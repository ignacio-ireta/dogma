"""The personal course calendar (I-30a): the shift, the anchor, previews, warps and course reset.

Every test runs on a synthetic 14-week path under tmp_path, or on the real authored path read
read-only for the arithmetic. Nothing here touches a real ``var/`` or ``workspace/``.
"""

from __future__ import annotations

import json
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import pytest
import yaml
from sqlalchemy import select, text
from tests.unit.db.support import make_report
from tests.unit.learning.support import (  # noqa: F401  autouse: pins the zone to UTC
    ALPHA,
    BETA,
    DELTA,
    NOON,
    build_repo,
    core_problem,
    solve,
    utc_timezone,
)

from dogma.content.loader import load_core_path
from dogma.content.schemas import CorePath
from dogma.core.config import Settings
from dogma.core.errors import DogmaError, ErrorCode
from dogma.db.backup import write_backup
from dogma.db.base import Base
from dogma.db.calendar_models import CalendarWarp
from dogma.db.errors import StorageError, StorageErrorCode
from dogma.db.models import Attempt, AttemptMode
from dogma.db.session import database_path, open_session
from dogma.db.store import record_attempt
from dogma.grading.models import RunStatus
from dogma.learning import calendar, sessions
from dogma.learning.calendar import Anchor, WarpKind, WarpRequest
from dogma.learning.path import current_week
from dogma.learning.planner import SessionKind
from dogma.learning.sessions import CloseNote

AUTHORED_START = date(2026, 10, 26)  # a Monday: week 1 of the synthetic path
TUESDAY_3_NOV = datetime(2026, 11, 3, 12, tzinfo=UTC)  # authored week 2
TUESDAY_27_OCT = datetime(2026, 10, 27, 12, tzinfo=UTC)  # authored week 1
GAMMA = "tst/gamma"  # a draft topic, listed in the weeks that have no other topic
WEEK_TOPICS = {1: (ALPHA,), 2: (BETA,), 3: (DELTA,)}


def _weeks() -> list[dict[str, object]]:
    weeks: list[dict[str, object]] = []
    for number in range(1, 15):
        topics = WEEK_TOPICS.get(number, (GAMMA,))
        weeks.append(
            {
                "week": number,
                "starts": (AUTHORED_START + timedelta(weeks=number - 1)).isoformat(),
                "theme": f"Synthetic week {number}",
                "topics": list(topics),
            }
        )
    return weeks


@pytest.fixture
def calendar_repo(tmp_path: Path) -> Settings:
    """A 14-week synthetic path, so the planner's final week (14) exists."""
    settings = build_repo(tmp_path / "repo")
    core = {
        "id": "test",
        "title": "Synthetic calendar path",
        "start": AUTHORED_START.isoformat(),
        "end": "2027-01-31",
        "sessions_per_day": 3,
        "session_minutes": 45,
        "days_per_week": 6,
        "weeks": _weeks(),
    }
    (settings.curriculum_dir / "paths" / "core.yaml").write_text(
        yaml.safe_dump(core, sort_keys=False), encoding="utf-8"
    )
    return settings


def _authored(settings: Settings) -> CorePath:
    return load_core_path(settings.curriculum_dir / "paths" / "core.yaml")


def _warp_rows(settings: Settings) -> list[tuple[str, int]]:
    if not database_path(settings).is_file():
        return []
    with open_session(settings, on_failure=StorageErrorCode.DB_READ_FAILED) as db:
        rows = db.scalars(select(CalendarWarp).order_by(CalendarWarp.id)).all()
        return [(row.kind, row.offset_weeks) for row in rows]


def _count(settings: Settings, table: str) -> int:
    with open_session(settings, on_failure=StorageErrorCode.DB_READ_FAILED) as db:
        return int(db.execute(text(f"SELECT COUNT(*) FROM {table}")).scalar_one())  # noqa: S608


# --- the shift (pure arithmetic) ---------------------------------------------------------------


def test_the_offset_puts_the_asked_week_on_today_on_every_day_of_the_course(
    calendar_repo: Settings,
) -> None:
    core = _authored(calendar_repo)
    day = AUTHORED_START - timedelta(days=20)
    while day < AUTHORED_START + timedelta(weeks=20):
        for week in range(1, 15):
            shifted = calendar.shift_path(core, calendar.offset_for_week(core, week, day))
            assert current_week(shifted.weeks, day).week == week, (day, week)
        day += timedelta(days=1)


def test_a_sync_or_a_move_on_a_tuesday_follows_the_worked_example(calendar_repo: Settings) -> None:
    core = _authored(calendar_repo)
    tuesday = date(2026, 11, 3)

    offset = calendar.offset_for_week(core, 1, tuesday)
    shifted = calendar.shift_path(core, offset)

    assert offset == 1  # (3 Nov - 26 Oct).days == 8, 8 // 7 == 1
    assert shifted.weeks[0].starts == date(2026, 11, 2)
    assert shifted.end == date(2027, 2, 7)


def test_the_offset_is_correct_across_the_year_boundary(calendar_repo: Settings) -> None:
    core = _authored(calendar_repo)
    new_year = date(2027, 1, 4)  # a Monday

    offset = calendar.offset_for_week(core, 1, new_year)

    assert offset == 10
    assert calendar.shift_path(core, offset).weeks[0].starts == new_year
    assert calendar.offset_for_week(core, 1, date(2027, 1, 1)) == 9  # Friday, earlier week


def test_a_week_number_outside_the_course_is_clamped(calendar_repo: Settings) -> None:
    core = _authored(calendar_repo)
    today = date(2026, 11, 3)

    assert calendar.offset_for_week(core, 0, today) == calendar.offset_for_week(core, 1, today)
    assert calendar.offset_for_week(core, 99, today) == calendar.offset_for_week(core, 14, today)


def test_shifting_moves_every_date_by_whole_weeks_and_keeps_mondays(
    calendar_repo: Settings,
) -> None:
    core = _authored(calendar_repo)

    shifted = calendar.shift_path(core, 3)

    assert shifted.start == core.start + timedelta(weeks=3)
    assert shifted.end == core.end + timedelta(weeks=3)
    assert all(item.starts.weekday() == 0 for item in shifted.weeks)
    assert calendar.shift_path(core, 0) is core


def test_the_offset_note_is_only_shown_while_the_calendar_is_moved() -> None:
    assert calendar.offset_note(0) is None
    assert calendar.offset_note(1) == (
        "Dates moved 1 week later by time-warp. `dogma warp --original` restores them."
    )
    assert calendar.offset_note(-3) == (
        "Dates moved 3 weeks earlier by time-warp. `dogma warp --original` restores them."
    )


# --- the anchor -------------------------------------------------------------------------------

WEEKS_OF = {ALPHA: 1, BETA: 2, DELTA: 3}


def test_with_no_attempts_the_anchor_is_week_one() -> None:
    assert calendar.furthest_anchor([], WEEKS_OF) == Anchor(week=1, problem_id=None)


def test_the_anchor_is_the_furthest_week_not_the_latest_attempt() -> None:
    attempts = [core_problem(DELTA), core_problem(ALPHA)]  # week 3, then an older week 1

    assert calendar.furthest_anchor(attempts, WEEKS_OF) == Anchor(
        week=3, problem_id=core_problem(DELTA)
    )


def test_a_problem_outside_the_path_never_moves_the_anchor() -> None:
    attempts = [core_problem(ALPHA), "sys/a-case/p01-question"]

    assert calendar.furthest_anchor(attempts, WEEKS_OF) == Anchor(
        week=1, problem_id=core_problem(ALPHA)
    )


def test_on_a_tie_the_later_attempt_names_the_problem() -> None:
    attempts = [core_problem(BETA), f"{BETA}/p02-extra"]

    assert calendar.furthest_anchor(attempts, WEEKS_OF).problem_id == f"{BETA}/p02-extra"


def test_runs_count_toward_the_anchor_as_well_as_submits(calendar_repo: Settings) -> None:
    report = make_report(RunStatus.PASSED, public=(3, 3), problem_id=core_problem(BETA))
    record_attempt(
        calendar_repo,
        report,
        mode=AttemptMode.RUN,
        core=False,
        started_at=NOON,
        submitted_at=NOON + timedelta(minutes=1),
    )

    found = calendar.anchor(calendar_repo, _authored(calendar_repo))

    assert found == Anchor(week=2, problem_id=core_problem(BETA))


def test_re_solving_an_older_problem_does_not_pull_the_calendar_back(
    calendar_repo: Settings,
) -> None:
    solve(calendar_repo, DELTA, when=NOON)  # week 3
    solve(calendar_repo, ALPHA, when=NOON + timedelta(days=5))  # week 1, later

    found = calendar.anchor(calendar_repo, _authored(calendar_repo))

    assert found == Anchor(week=3, problem_id=core_problem(DELTA))


# --- previews, warps and their writes ---------------------------------------------------------


def test_a_sync_with_no_attempts_makes_this_week_week_one(calendar_repo: Settings) -> None:
    result = calendar.preview(calendar_repo, WarpRequest(WarpKind.SYNC), now=TUESDAY_3_NOV)

    assert result.anchor == Anchor(week=1, problem_id=None)
    assert result.before.current_week == 2
    assert result.after.current_week == 1
    assert result.changed is True
    assert result.written is False


def test_a_preview_never_writes_and_creates_no_database(calendar_repo: Settings) -> None:
    calendar.preview(calendar_repo, WarpRequest(WarpKind.TO_WEEK, week=1), now=TUESDAY_3_NOV)

    assert not database_path(calendar_repo).is_file()


def test_a_sync_moves_the_calendar_to_the_furthest_exercise(calendar_repo: Settings) -> None:
    solve(calendar_repo, DELTA, when=NOON)  # week 3, on a Tuesday in week 2 of the authored path

    result = calendar.apply(calendar_repo, WarpRequest(WarpKind.SYNC), now=TUESDAY_3_NOV)

    assert result.written is True
    assert result.after.current_week == 3
    assert _warp_rows(calendar_repo) == [("sync", -1)]  # week 3 is one week earlier than week 2
    status = calendar.status(calendar_repo, now=TUESDAY_3_NOV)
    assert status.offset_weeks == -1
    assert status.current_week.week == 3
    assert status.last_warp is not None and status.last_warp.kind is WarpKind.SYNC


def test_a_move_to_a_week_loads_on_the_next_read(calendar_repo: Settings) -> None:
    calendar.apply(calendar_repo, WarpRequest(WarpKind.TO_WEEK, week=1), now=TUESDAY_3_NOV)

    assert calendar.current_offset(calendar_repo) == 1
    loaded = sessions._load_core(calendar_repo)
    assert loaded.weeks[0].starts == date(2026, 11, 2)
    assert loaded.end == date(2027, 2, 7)


def test_moving_ahead_and_back_clamps_to_the_course(calendar_repo: Settings) -> None:
    back = calendar.preview(calendar_repo, WarpRequest(WarpKind.BACK, weeks=5), now=TUESDAY_3_NOV)
    ahead = calendar.preview(
        calendar_repo, WarpRequest(WarpKind.AHEAD, weeks=99), now=TUESDAY_3_NOV
    )
    one_ahead = calendar.preview(
        calendar_repo, WarpRequest(WarpKind.AHEAD, weeks=1), now=TUESDAY_3_NOV
    )

    assert back.after.current_week == 1  # week 2 - 5, clamped to week 1
    assert ahead.after.current_week == 14  # week 2 + 99, clamped to the last week
    assert one_ahead.after.current_week == 3


def test_the_original_dates_come_back_with_one_row(calendar_repo: Settings) -> None:
    calendar.apply(calendar_repo, WarpRequest(WarpKind.TO_WEEK, week=1), now=TUESDAY_3_NOV)

    result = calendar.apply(calendar_repo, WarpRequest(WarpKind.ORIGINAL), now=TUESDAY_3_NOV)

    assert result.after.offset_weeks == 0
    assert result.after.current_week == 2
    assert _warp_rows(calendar_repo) == [("to_week", 1), ("original", 0)]


def test_a_move_that_changes_nothing_writes_nothing(calendar_repo: Settings) -> None:
    result = calendar.apply(calendar_repo, WarpRequest(WarpKind.ORIGINAL), now=TUESDAY_3_NOV)

    assert result.changed is False
    assert result.written is False
    assert _warp_rows(calendar_repo) == []


def test_recording_off_refuses_a_move_and_a_reset_before_touching_anything(
    calendar_repo: Settings,
) -> None:
    off = Settings(repo_root=calendar_repo.repo_root, record_attempts=False)

    with pytest.raises(DogmaError) as moved:
        calendar.apply(off, WarpRequest(WarpKind.TO_WEEK, week=1), now=TUESDAY_3_NOV)
    with pytest.raises(DogmaError) as reset:
        calendar.reset_course(off, now=TUESDAY_3_NOV)

    assert moved.value.code is ErrorCode.CALENDAR_RECORDING_OFF
    assert reset.value.code is ErrorCode.CALENDAR_RECORDING_OFF
    assert not database_path(calendar_repo).is_file()
    assert not (calendar_repo.var_dir / "backups").exists()


@pytest.mark.parametrize(
    "request_",
    [
        {"kind": WarpKind.TO_WEEK},
        {"kind": WarpKind.AHEAD, "weeks": 0},
        {"kind": WarpKind.SYNC, "week": 2},
        {"kind": WarpKind.COURSE_RESET},
    ],
)
def test_malformed_requests_are_refused(request_: dict[str, object]) -> None:
    with pytest.raises(DogmaError) as caught:
        WarpRequest(**request_)  # type: ignore[arg-type]

    assert caught.value.code is ErrorCode.CALENDAR_REQUEST_INVALID


# --- the planner follows the calendar ---------------------------------------------------------


def _third_session_kind(settings: Settings, when: datetime) -> SessionKind:
    """Plan and close the first two sessions of the day, then report the third one's kind."""
    for _ in range(2):
        current = sessions.today(settings, now=when).current
        assert current is not None
        sessions.close_session(
            settings,
            current.session_id,
            CloseNote(stopped_where="here", next_step="next"),
            answer=None,
            now=when,
        )
    third = sessions.today(settings, now=when).current
    assert third is not None and third.slot == 3
    return third.plan.kind


def test_without_a_warp_the_third_session_of_week_one_is_a_consolidate(
    calendar_repo: Settings,
) -> None:
    assert _third_session_kind(calendar_repo, TUESDAY_27_OCT) is SessionKind.CONSOLIDATE


def test_weekly_interviews_follow_a_warp_into_week_three(calendar_repo: Settings) -> None:
    calendar.apply(calendar_repo, WarpRequest(WarpKind.TO_WEEK, week=3), now=TUESDAY_27_OCT)

    assert calendar.current_offset(calendar_repo) == -2
    assert _third_session_kind(calendar_repo, TUESDAY_27_OCT) is SessionKind.INTERVIEW


def test_the_final_interview_week_follows_a_warp(calendar_repo: Settings) -> None:
    calendar.apply(calendar_repo, WarpRequest(WarpKind.TO_WEEK, week=14), now=TUESDAY_27_OCT)

    view = sessions.today(calendar_repo, now=TUESDAY_27_OCT)

    assert view.current is not None
    assert view.current.plan.kind is SessionKind.INTERVIEW  # every session of week 14


# --- course reset ----------------------------------------------------------------------------


@pytest.fixture
def started(calendar_repo: Settings) -> Settings:
    """A learner with attempts, sessions, a warp and a workspace file, ready to start over."""
    solve(calendar_repo, ALPHA, when=NOON)
    sessions.today(calendar_repo, now=NOON)
    calendar.apply(calendar_repo, WarpRequest(WarpKind.TO_WEEK, week=1), now=TUESDAY_3_NOV)
    workspace = calendar_repo.workspace_dir / "tst" / "alpha"
    workspace.mkdir(parents=True)
    (workspace / "p01-core.py").write_text("def solve(x):\n    return x\n", encoding="utf-8")
    return calendar_repo


def test_a_reset_clears_every_progress_table_and_keeps_the_warp_log(started: Settings) -> None:
    result = calendar.reset_course(started, now=TUESDAY_3_NOV)

    cleared = [table.name for table in Base.metadata.sorted_tables if table.name != "calendar_warp"]
    assert {name: _count(started, name) for name in cleared} == dict.fromkeys(cleared, 0)
    assert result.rows_cleared["attempt"] == 1
    assert result.rows_cleared["topic_state"] == 1
    assert set(result.rows_cleared) == set(cleared)
    assert _warp_rows(started)[-2:] == [("to_week", 1), ("course_reset", 1)]


def test_a_reset_makes_this_week_week_one_and_reports_the_dates(started: Settings) -> None:
    result = calendar.reset_course(started, now=TUESDAY_3_NOV)

    status = calendar.status(started, now=TUESDAY_3_NOV)
    assert status.current_week.week == 1
    assert result.week_one_starts == date(2026, 11, 2)
    assert result.course_end == date(2027, 2, 7)
    assert status.anchor == Anchor(week=1, problem_id=None)


def test_a_reset_writes_a_backup_before_it_clears(started: Settings) -> None:
    result = calendar.reset_course(started, now=TUESDAY_3_NOV)

    payload = json.loads(result.backup_path.read_text(encoding="utf-8"))
    assert len(payload["tables"]["attempt"]) == 1
    assert len(payload["tables"]["calendar_warp"]) == 1  # the warp made before the reset
    assert result.snapshot_path is not None and result.snapshot_path.is_file()


def test_a_reset_moves_workspace_files_into_history_keeping_their_paths(started: Settings) -> None:
    result = calendar.reset_course(started, now=TUESDAY_3_NOV)

    assert result.files_set_aside == 1
    assert result.history_dir is not None
    kept = result.history_dir / "tst" / "alpha" / "p01-core.py"
    assert kept.read_text(encoding="utf-8").startswith("def solve")
    assert not (started.workspace_dir / "tst" / "alpha" / "p01-core.py").exists()
    assert result.history_dir.parent == started.var_dir / "workspace-history"


def test_a_failed_backup_stops_the_reset_and_clears_nothing(
    started: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    def broken(settings: Settings, *, now: datetime) -> None:
        raise StorageError(StorageErrorCode.BACKUP_WRITE_FAILED, "disk full")

    monkeypatch.setattr(calendar, "write_backup", broken)

    with pytest.raises(StorageError):
        calendar.reset_course(started, now=TUESDAY_3_NOV)

    assert _count(started, "attempt") == 1
    assert _count(started, "topic_state") == 1
    assert _warp_rows(started)[-1] == ("to_week", 1)
    assert (started.workspace_dir / "tst" / "alpha" / "p01-core.py").is_file()


def test_the_backup_lists_the_warp_log(started: Settings) -> None:
    calendar.apply(started, WarpRequest(WarpKind.ORIGINAL), now=TUESDAY_3_NOV)

    result = write_backup(started, now=TUESDAY_3_NOV)

    payload = json.loads(result.path.read_text(encoding="utf-8"))
    assert [row["kind"] for row in payload["tables"]["calendar_warp"]] == ["to_week", "original"]


def test_a_reset_with_no_database_still_makes_this_week_week_one(calendar_repo: Settings) -> None:
    result = calendar.reset_course(calendar_repo, now=TUESDAY_3_NOV)

    assert result.offset_weeks == 1
    assert calendar.current_offset(calendar_repo) == 1
    assert _count(calendar_repo, Attempt.__tablename__) == 0
    assert _count(calendar_repo, "calendar_warp") == 1
