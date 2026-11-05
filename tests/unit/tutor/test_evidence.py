"""Evidence gathering against a real temporary repo and learner database.

The privacy sentinels are written into private/, the reference solution and the hidden tests. None
of them may appear in any prompt built from the gathered evidence.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest
from tests.unit.tutor.support import (
    HIDDEN_SENTINEL,
    HIDDEN_VALUE,
    LEARNER_CODE,
    PROBLEM_ID,
    PROFILE_SENTINEL,
    SOLUTION_SENTINEL,
    failing_tests,
    record_attempt_with_report,
)

from dogma.content.loader import load_problem
from dogma.core.config import Settings
from dogma.core.errors import ErrorCode
from dogma.corpus.models import Chunk, SearchHit
from dogma.db.models import AttemptMode
from dogma.grading.models import RunStatus
from dogma.tutor import evidence as evidence_module
from dogma.tutor.context import assemble_prompt
from dogma.tutor.evidence import FAILED_STATUSES, gather_evidence
from dogma.tutor.models import CallMode

T0 = datetime(2026, 10, 24, 9, 0, tzinfo=UTC)


def _gather(settings: Settings):  # type: ignore[no-untyped-def]
    problem_dir = settings.content_dir / "tst" / "alpha" / "problems" / "p01-double"
    meta = load_problem(problem_dir).meta
    assert meta is not None
    return gather_evidence(settings, problem_dir=problem_dir, problem_id=PROBLEM_ID, meta=meta)


def test_gathering_reads_only_problem_lesson_workspace_and_history(settings: Settings) -> None:
    gathered = _gather(settings)

    assert gathered.workspace_text == LEARNER_CODE
    assert gathered.lesson_title == "Alpha lesson"
    assert [section.heading for section in gathered.lesson_sections] == ["Doubling"]
    assert gathered.history.known is True
    assert gathered.history.failed_submits == 0
    assert gathered.history.latest is None
    assert any("corpus" in note for note in gathered.notes)


def test_the_reviewed_hints_are_gathered_and_a_missing_file_is_a_note(settings: Settings) -> None:
    gathered = _gather(settings)

    assert [hint.level for hint in gathered.reviewed.hints] == [1, 2, 3]
    assert gathered.reviewed.note is None
    (settings.content_dir / "tst" / "alpha" / "problems" / "p01-double" / "hints.md").unlink()

    without = _gather(settings)

    assert without.reviewed.hints == ()
    assert without.reviewed.note is not None
    assert "missing" in without.reviewed.note
    assert without.notes == gathered.notes


def test_no_private_solution_or_hidden_test_text_reaches_any_prompt(settings: Settings) -> None:
    assert (
        (settings.repo_root / "private" / "learner.yaml").read_text().startswith(PROFILE_SENTINEL)
    )
    gathered = _gather(settings)

    for mode in (CallMode.HINT, CallMode.DEBUG):
        level = 2 if mode is CallMode.HINT else None
        text = "\n".join(
            message.content
            for message in assemble_prompt(gathered, mode=mode, level=level).prompt.messages
        )
        for sentinel in (PROFILE_SENTINEL, SOLUTION_SENTINEL, HIDDEN_SENTINEL):
            assert sentinel not in text


def test_failed_submits_count_only_recorded_submits_that_did_not_pass(settings: Settings) -> None:
    record_attempt_with_report(settings, attempt_id="a1-failed", status=RunStatus.FAILED, when=T0)
    record_attempt_with_report(
        settings,
        attempt_id="a2-timeout",
        status=RunStatus.TIMEOUT,
        when=T0 + timedelta(minutes=1),
    )
    record_attempt_with_report(
        settings,
        attempt_id="a3-rejected",
        status=RunStatus.REJECTED,
        when=T0 + timedelta(minutes=2),
    )
    record_attempt_with_report(
        settings,
        attempt_id="a4-run-failed",
        status=RunStatus.FAILED,
        mode=AttemptMode.RUN,
        when=T0 + timedelta(minutes=3),
    )
    record_attempt_with_report(
        settings,
        attempt_id="a5-passed",
        status=RunStatus.PASSED,
        when=T0 + timedelta(minutes=4),
    )

    gathered = _gather(settings)

    assert gathered.history.failed_submits == 2
    assert gathered.history.latest is not None
    assert gathered.history.latest.attempt_id == "a5-passed"
    assert RunStatus.REJECTED not in FAILED_STATUSES


def test_the_latest_report_is_rebuilt_and_its_hidden_detail_redacted(settings: Settings) -> None:
    record_attempt_with_report(
        settings,
        attempt_id="a6-latest",
        status=RunStatus.FAILED,
        tests=failing_tests(),
        when=T0,
    )

    gathered = _gather(settings)

    latest = gathered.history.latest
    assert latest is not None
    assert latest.status is RunStatus.FAILED
    assert latest.attempt_id == "a6-latest"
    hidden = [test for test in latest.tests if test.file == "hidden" and test.outcome == "fail"]
    assert hidden and HIDDEN_VALUE not in hidden[0].message
    assert "AssertionError" in hidden[0].message
    public = [test for test in latest.tests if test.file == "public" and test.outcome == "fail"]
    assert "expected [-2] got [0]" in public[0].message


def test_a_report_without_a_harness_file_degrades_to_the_recorded_status(
    settings: Settings,
) -> None:
    record_attempt_with_report(settings, attempt_id="a7-oom", status=RunStatus.OOM, when=T0)

    gathered = _gather(settings)

    assert gathered.history.latest is not None
    assert gathered.history.latest.status is RunStatus.OOM
    assert gathered.history.latest.tests == ()


def test_an_unreadable_learner_database_is_reported_not_raised(settings: Settings) -> None:
    settings.var_dir.mkdir(parents=True)
    (settings.var_dir / "dogma.db").mkdir()

    gathered = _gather(settings)

    assert gathered.history.known is False
    assert any("learner history unavailable" in note for note in gathered.notes)


def test_corpus_hits_drop_the_lesson_already_included(
    settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    def chunk(source_id: str) -> SearchHit:
        return SearchHit(
            Chunk(source_id, "p", None, (), 0, "text about doubling", ("tst/alpha",)),
            f"[{source_id}]",
            -1.0,
        )

    hits = (chunk("content:tst/alpha/lesson"), chunk("sources:notes/doubling.md"))
    monkeypatch.setattr(evidence_module, "corpus_search", lambda *_a, **_k: hits)

    gathered = _gather(settings)

    assert [hit.chunk.source_id for hit in gathered.corpus_hits] == ["sources:notes/doubling.md"]


def test_the_workspace_file_is_optional(settings: Settings) -> None:
    (settings.workspace_dir / "tst" / "alpha" / "p01-double.py").unlink()

    gathered = _gather(settings)

    assert gathered.workspace_text is None
    assert (
        "no workspace file yet"
        in assemble_prompt(gathered, mode=CallMode.HINT, level=1).prompt.messages[1].content
    )


def test_error_codes_used_by_gathering_are_stable() -> None:
    assert ErrorCode.TUTOR_NO_FAILURE == "TUTOR_NO_FAILURE"
