"""POST /attempts never returns a hidden test's raw message (Step 0 safety, ROADMAP I-15).

The grader is replaced with a report that carries raw hidden text, including a sentinel value that
must never appear in any response body.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient
from tests.unit.api.support import PROBLEM

from dogma.core.config import Settings
from dogma.grading.display import NO_DETAIL
from dogma.grading.models import GradeReport, RunStatus, StaticFinding
from dogma.grading.models import TestResult as Result
from dogma.runner import attempt

URL = "/api/v1/attempts"
SENTINEL = "SECRET_HIDDEN_VALUE_7731"
RAW_HIDDEN = (
    "AssertionError: wrong sign convention\n"
    "Arrays are not almost equal to 7 decimals\n"
    f"x: array([{SENTINEL}, 2.0])\n"
    "ACTUAL: array([9.0, 1.0])\n"
    "Max absolute difference: 4.0\n"
    f"Falsifying example: test_property(\n    a=array([{SENTINEL}]),\n)\n"
)
PUBLIC_MESSAGE = "AssertionError: shape mismatch\nx: array([1.0, 2.0])\ny: array([1.0])"


def _workspace(settings: Settings) -> None:
    path = settings.workspace_dir / "cml" / "vector-operations" / "p01-vector-projection.py"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("def vector_projection(a, b):\n    return a\n", encoding="utf-8")


def _grade_with(monkeypatch: pytest.MonkeyPatch, report: GradeReport) -> None:
    def fake(
        problem_dir: Path,
        submission: Path,
        *,
        include_hidden: bool,
        settings: Settings,
    ) -> GradeReport:
        return report

    monkeypatch.setattr(attempt, "grade_attempt", fake)


def _report(
    *tests: Result,
    status: RunStatus = RunStatus.FAILED,
    static: tuple[StaticFinding, ...] = (),
    message: str = "",
) -> GradeReport:
    return GradeReport(
        attempt_id="t" + "a" * 12,
        problem_id=PROBLEM,
        problem_version=2,
        runtime="py-numpy",
        status=status,
        tests=tests,
        static=static,
        message=message,
    )


def _submit(client: TestClient) -> tuple[dict[str, Any], str]:
    response = client.post(URL, json={"problem_id": PROBLEM, "mode": "submit"})
    assert response.status_code == 201, response.text
    return response.json(), response.text


def test_a_hidden_failure_is_one_safe_line_and_the_raw_text_never_leaves(
    settings: Settings, client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    _workspace(settings)
    _grade_with(
        monkeypatch,
        _report(
            Result(name="test_pub_0", file="public", outcome="pass"),
            Result(name="test_hid_0", file="hidden", outcome="fail", message=RAW_HIDDEN),
        ),
    )
    body, text = _submit(client)
    hidden = next(t for t in body["report"]["tests"] if t["file"] == "hidden")
    assert hidden["message"] == "wrong sign convention"
    assert hidden["outcome"] == "fail"
    assert SENTINEL not in text
    assert RAW_HIDDEN not in text
    assert "ACTUAL" not in text


def test_a_hidden_error_is_summarised_the_same_way(
    settings: Settings, client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    _workspace(settings)
    _grade_with(
        monkeypatch,
        _report(Result(name="test_hid_0", file="hidden", outcome="error", message=RAW_HIDDEN)),
    )
    body, text = _submit(client)
    assert body["report"]["tests"][0]["message"] == "wrong sign convention"
    assert SENTINEL not in text


@pytest.mark.parametrize(
    "only_inputs",
    [
        f"Falsifying example: test_property(\n    a=array([{SENTINEL}]),\n)\n",
        f"AssertionError: assert 42 == {SENTINEL}",
        f"AssertionError: ACTUAL: array([{SENTINEL}])",
        f"Failed: AssertionError: x: array([{SENTINEL}])",
    ],
    ids=["falsifier", "wrapped-assert", "wrapped-actual", "nested-wrappers"],
)
def test_hidden_output_without_a_safe_mistake_line_uses_the_fixed_sentence(
    settings: Settings, client: TestClient, monkeypatch: pytest.MonkeyPatch, only_inputs: str
) -> None:
    _workspace(settings)
    _grade_with(
        monkeypatch,
        _report(Result(name="test_hid_0", file="hidden", outcome="fail", message=only_inputs)),
    )
    body, text = _submit(client)
    assert body["report"]["tests"][0]["message"] == NO_DETAIL
    assert SENTINEL not in text


def test_a_passing_hidden_test_carries_no_message_at_all(
    settings: Settings, client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    _workspace(settings)
    _grade_with(
        monkeypatch,
        _report(
            Result(name="test_hid_0", file="hidden", outcome="pass", message=RAW_HIDDEN),
            status=RunStatus.PASSED,
        ),
    )
    body, text = _submit(client)
    assert body["report"]["tests"][0]["message"] == ""
    assert SENTINEL not in text


def test_public_failures_keep_their_full_message(
    settings: Settings, client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    _workspace(settings)
    _grade_with(
        monkeypatch,
        _report(Result(name="test_pub_0", file="public", outcome="fail", message=PUBLIC_MESSAGE)),
    )
    body, _ = _submit(client)
    assert body["report"]["tests"][0]["message"] == PUBLIC_MESSAGE


def test_static_findings_come_back_in_plain_words_next_to_the_raw_ones(
    settings: Settings, client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    _workspace(settings)
    finding = StaticFinding(
        rule="banned-import", message="`numpy.linalg` is not allowed here", line=3
    )
    _grade_with(monkeypatch, _report(status=RunStatus.REJECTED, static=(finding,)))
    body, _ = _submit(client)
    report = body["report"]
    assert report["findings"] == ["line 3 uses numpy.linalg, which this problem bans"]
    assert report["static"][0]["rule"] == "banned-import"


def test_a_timeout_carries_its_headline_and_no_test_detail(
    settings: Settings, client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    _workspace(settings)
    _grade_with(
        monkeypatch,
        _report(status=RunStatus.TIMEOUT, message="the run exceeded its 30 s time limit"),
    )
    body, _ = _submit(client)
    report = body["report"]
    assert report["status"] == "timeout"
    assert report["headline"].startswith("Timed out")
    assert report["message"] == "the run exceeded its 30 s time limit"
    assert report["tests"] == []


def test_a_passing_run_has_no_headline_and_no_findings(
    settings: Settings, client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    _workspace(settings)
    _grade_with(
        monkeypatch,
        _report(
            Result(name="test_pub_0", file="public", outcome="pass"),
            status=RunStatus.PASSED,
        ),
    )
    body, _ = _submit(client)
    report = body["report"]
    assert report["headline"] is None
    assert report["findings"] == []
