"""The course calendar over HTTP: the shifted dates, every warp kind, the preview, and the reset.

The rules are the domain's (tests/unit/learning/test_calendar*.py). These tests check the wire
shapes, the status codes and that the progress view follows a move. The clock is frozen at
2026-10-26 (week 1), so the expected weeks are exact.
"""

from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient
from tests.unit.api.conftest import FakeGrader
from tests.unit.api.support import PROBLEM, FrozenClock

from dogma.core.config import Settings

CALENDAR = "/api/v1/calendar"
WARP = f"{CALENDAR}/warp"
RESET = "/api/v1/course/reset"
PROGRESS = "/api/v1/progress"
ATTEMPTS = "/api/v1/attempts"


def _warp(client: TestClient, **payload: object) -> dict[str, object]:
    response = client.post(WARP, json=payload)
    assert response.status_code == 200, response.text
    body: dict[str, object] = response.json()
    return body


def _submit_passing(settings: Settings, client: TestClient, grader: FakeGrader) -> None:
    path = settings.workspace_dir / "cml" / "vector-operations" / "p01-vector-projection.py"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("def vector_projection(a, b):\n    return a\n", encoding="utf-8")
    client.post(ATTEMPTS, json={"problem_id": PROBLEM, "mode": "submit"})
    assert grader.calls


def test_get_calendar_shows_the_authored_course_at_the_start(client: TestClient) -> None:
    body = client.get(CALENDAR).json()
    assert body["offset_weeks"] == 0
    assert body["current_week"] == 1
    assert body["anchor_week"] == 1
    assert body["anchor_problem_id"] is None
    assert body["start"] == "2026-10-26"
    assert body["end"] == "2027-01-31"
    assert body["last_warp"] is None
    assert len(body["weeks"]) == 14
    assert body["weeks"][0] == {
        "week": 1,
        "starts": "2026-10-26",
        "theme": body["weeks"][0]["theme"],
    }
    assert body["weeks"][0]["theme"]


def test_reading_the_calendar_creates_no_database(settings: Settings, client: TestClient) -> None:
    client.get(CALENDAR)
    client.post(WARP, json={"kind": "sync", "dry_run": True})
    assert not (settings.var_dir / "dogma.db").exists()


def test_a_move_to_week_three_shifts_the_calendar_and_is_logged(client: TestClient) -> None:
    body = _warp(client, kind="to_week", week=3)
    assert body["applied"] is True
    assert body["changed"] is True
    before, after = body["before"], body["after"]
    assert before["current_week"] == 1
    assert after["offset_weeks"] == -2
    assert after["current_week"] == 3
    assert after["current_starts"] == "2026-10-26"
    assert after["mocks_from"] == "2026-10-26"
    assert after["end"] == "2027-01-17"
    calendar = client.get(CALENDAR).json()
    assert calendar["offset_weeks"] == -2
    assert calendar["current_week"] == 3
    assert calendar["last_warp"]["kind"] == "to_week"
    assert calendar["last_warp"]["offset_weeks"] == -2


def test_ahead_and_back_move_by_whole_weeks_and_clamp(client: TestClient) -> None:
    ahead = _warp(client, kind="ahead", weeks=2)
    assert ahead["after"]["current_week"] == 3
    back = _warp(client, kind="back", weeks=5)
    assert back["after"]["current_week"] == 1
    assert back["after"]["offset_weeks"] == 0
    far = _warp(client, kind="ahead", weeks=40)
    assert far["after"]["current_week"] == 14


def test_original_restores_the_authored_dates(client: TestClient) -> None:
    _warp(client, kind="to_week", week=5)
    body = _warp(client, kind="original")
    assert body["after"]["offset_weeks"] == 0
    assert body["applied"] is True
    assert client.get(CALENDAR).json()["offset_weeks"] == 0


def test_sync_with_no_exercises_makes_this_week_week_one(client: TestClient) -> None:
    _warp(client, kind="to_week", week=4)
    body = _warp(client, kind="sync")
    assert body["after"]["current_week"] == 1
    assert body["anchor_week"] == 1
    assert body["anchor_problem_id"] is None


def test_sync_follows_the_furthest_exercise(
    settings: Settings, client: TestClient, grader: FakeGrader
) -> None:
    _submit_passing(settings, client, grader)
    calendar = client.get(CALENDAR).json()
    assert calendar["anchor_week"] == 1
    assert calendar["anchor_problem_id"] == PROBLEM
    body = _warp(client, kind="sync", dry_run=True)
    assert body["changed"] is False
    assert body["applied"] is False


def test_a_dry_run_previews_and_writes_nothing(client: TestClient) -> None:
    body = _warp(client, kind="to_week", week=6, dry_run=True)
    assert body["applied"] is False
    assert body["changed"] is True
    assert body["after"]["current_week"] == 6
    calendar = client.get(CALENDAR).json()
    assert calendar["offset_weeks"] == 0
    assert calendar["last_warp"] is None


def test_a_move_that_changes_nothing_writes_nothing(client: TestClient) -> None:
    body = _warp(client, kind="original")
    assert body["changed"] is False
    assert body["applied"] is False
    assert client.get(CALENDAR).json()["last_warp"] is None


def test_the_progress_view_follows_a_move(client: TestClient) -> None:
    assert client.get(PROGRESS).json()["summary"]["current_week"] == 1
    _warp(client, kind="to_week", week=3)
    summary = client.get(PROGRESS).json()["summary"]
    assert summary["current_week"] == 3
    assert summary["week_start"] == "2026-10-26"
    assert client.get(PROGRESS).json()["course"]["end"] == "2027-01-17"


def test_bad_combinations_are_unprocessable(client: TestClient) -> None:
    cases: list[dict[str, object]] = [
        {"kind": "to_week"},
        {"kind": "to_week", "week": 2, "weeks": 1},
        {"kind": "ahead"},
        {"kind": "back", "week": 2},
        {"kind": "sync", "week": 2},
        {"kind": "original", "weeks": 1},
    ]
    for payload in cases:
        response = client.post(WARP, json=payload)
        assert response.status_code == 422, payload
        assert response.json()["code"] == "CALENDAR_REQUEST_INVALID", payload


def test_out_of_shape_bodies_are_validation_errors(client: TestClient) -> None:
    for payload in (
        {"kind": "course_reset"},
        {"kind": "sideways"},
        {"kind": "to_week", "week": 0},
        {"kind": "ahead", "weeks": 0},
        {"kind": "sync", "extra": 1},
    ):
        response = client.post(WARP, json=payload)
        assert response.status_code == 422, payload
        assert response.json()["code"] == "API_VALIDATION_FAILED", payload


def test_recording_off_refuses_a_real_move_but_not_a_preview(
    repo: Path, make_client, clock: FrozenClock
) -> None:
    off = Settings(repo_root=repo, record_attempts=False)
    with make_client(app_settings=off) as client:
        preview = client.post(WARP, json={"kind": "to_week", "week": 3, "dry_run": True})
        assert preview.status_code == 200
        response = client.post(WARP, json={"kind": "to_week", "week": 3})
        assert response.status_code == 409
        assert response.json()["code"] == "CALENDAR_RECORDING_OFF"
        assert client.post(WARP, json={"kind": "original"}).status_code == 200
        reset = client.post(RESET, json={"confirm": "reset"})
        assert reset.status_code == 409
        assert reset.json()["code"] == "CALENDAR_RECORDING_OFF"
    assert not (off.var_dir / "dogma.db").exists()


def test_reset_needs_the_typed_word(client: TestClient) -> None:
    for payload in ({}, {"confirm": ""}, {"confirm": "Reset"}, {"confirm": "yes"}):
        response = client.post(RESET, json=payload)
        assert response.status_code == 422, payload
        assert response.json()["code"] == "API_VALIDATION_FAILED"


def test_reset_backs_up_clears_progress_and_makes_this_week_week_one(
    settings: Settings, client: TestClient, grader: FakeGrader
) -> None:
    _submit_passing(settings, client, grader)
    _warp(client, kind="to_week", week=4)
    assert client.get(PROGRESS).json()["summary"]["solved_topics"] == 1

    response = client.post(RESET, json={"confirm": "reset"})
    assert response.status_code == 200, response.text
    body = response.json()
    assert Path(body["backup_path"]).is_file()
    assert body["snapshot_path"] is not None
    assert Path(body["snapshot_path"]).is_file()
    assert body["rows_cleared"]["attempt"] == 1
    assert "calendar_warp" not in body["rows_cleared"]
    assert body["week_one_starts"] == "2026-10-26"
    assert body["course_end"] == "2027-01-31"
    assert body["files_set_aside"] == 1
    workspace_file = (
        settings.workspace_dir / "cml" / "vector-operations" / "p01-vector-projection.py"
    )
    assert not workspace_file.exists()

    progress = client.get(PROGRESS).json()
    assert progress["summary"]["solved_topics"] == 0
    assert progress["summary"]["current_week"] == 1
    calendar = client.get(CALENDAR).json()
    assert calendar["last_warp"]["kind"] == "course_reset"
    assert calendar["offset_weeks"] == 0
