"""hint and debug: the L4 unlock rule, output guards, and degradation. Every model call is faked."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest
from tests.unit.tutor.support import (
    HIDDEN_SENTINEL,
    HIDDEN_VALUE,
    PROFILE_SENTINEL,
    SOLUTION_SENTINEL,
    failing_tests,
    record_attempt_with_report,
)

from dogma.content.loader import load_problem
from dogma.core.config import Settings
from dogma.core.errors import DogmaError, ErrorCode
from dogma.grading.models import RunStatus
from dogma.tutor import service
from dogma.tutor.gateway import Gateway
from dogma.tutor.models import CallMode, Mode, ProviderError, Source
from dogma.tutor.prompts import load_template
from dogma.tutor.providers.fake import FakeProvider
from dogma.tutor.service import Target, cites, guard_verdicts, omit_long_code

T0 = datetime(2026, 10, 24, 9, 0, tzinfo=UTC)


def _target(settings: Settings) -> Target:
    problem_dir = settings.content_dir / "tst" / "alpha" / "problems" / "p01-double"
    meta = load_problem(problem_dir).meta
    assert meta is not None
    return Target(problem_dir=problem_dir, problem_id="tst/alpha/p01-double", meta=meta)


def _gateway(settings: Settings, *replies: str | ProviderError) -> tuple[Gateway, FakeProvider]:
    provider = FakeProvider(list(replies))
    return Gateway(settings, provider), provider


def _no_question(_question: str) -> bool:
    raise AssertionError("the walkthrough must not ask when it is already unlocked")


def _fail_twice(settings: Settings) -> None:
    for index in range(2):
        record_attempt_with_report(
            settings,
            attempt_id=f"fail-{index}",
            status=RunStatus.FAILED,
            tests=failing_tests(),
            when=T0 + timedelta(minutes=index),
        )


def test_an_l1_hint_is_returned_with_its_sources(settings: Settings) -> None:
    gateway, provider = _gateway(settings, "Name the concept: the mean of a list [S1].")

    answer = service.hint(
        settings, _target(settings), level=1, yes=False, ask=_no_question, gateway=gateway
    )

    assert answer.reviewed is not None
    assert answer.reviewed.level == 1
    assert answer.nudge is not None
    assert answer.nudge.text == "Name the concept: the mean of a list [S1]."
    assert answer.nudge.cited is True
    # No corpus index in the synthetic repo: the reviewed hint and the lesson are the sources.
    assert [source.tag for source in answer.nudge.sources] == ["S1", "S2"]
    assert [source.kind for source in answer.nudge.sources] == ["reviewed", "lesson"]
    assert any("corpus" in note for note in answer.nudge.notes)
    assert answer.nudge.prompt_version == load_template(CallMode.HINT).version
    assert answer.nudge.model == settings.llm_model
    assert len(provider.requests) == 1


def test_an_answer_without_a_source_is_marked_unsourced(settings: Settings) -> None:
    gateway, _ = _gateway(settings, "Think about the shape of the input.")

    answer = service.hint(
        settings, _target(settings), level=1, yes=False, ask=_no_question, gateway=gateway
    )

    assert answer.nudge is not None
    assert answer.nudge.cited is False
    assert "Unsourced" in answer.nudge.text


def test_the_model_never_receives_private_solution_or_hidden_text(settings: Settings) -> None:
    gateway, provider = _gateway(settings, "Concept [S1].")

    service.hint(settings, _target(settings), level=3, yes=False, ask=_no_question, gateway=gateway)

    sent = "\n".join(message.content for message in provider.requests[0].messages)
    for sentinel in (PROFILE_SENTINEL, SOLUTION_SENTINEL, HIDDEN_SENTINEL):
        assert sentinel not in sent
    assert "Double all" in sent


def test_l4_is_locked_until_two_recorded_submits_fail(settings: Settings) -> None:
    gateway, provider = _gateway(settings, "never sent")
    asked: list[str] = []

    def decline(question: str) -> bool:
        asked.append(question)
        return False

    with pytest.raises(DogmaError) as locked:
        service.hint(settings, _target(settings), level=4, yes=False, ask=decline, gateway=gateway)

    assert locked.value.code is ErrorCode.TUTOR_LOCKED
    assert "2 recorded submits" in locked.value.message
    assert len(asked) == 1
    assert provider.requests == []


def test_l4_opens_when_the_learner_confirms(settings: Settings) -> None:
    gateway, provider = _gateway(settings, "Walk through it [S1].")

    answer = service.hint(
        settings, _target(settings), level=4, yes=False, ask=lambda _q: True, gateway=gateway
    )

    assert answer.level == 4
    assert len(provider.requests) == 1


def test_l4_opens_without_asking_after_two_failed_submits(settings: Settings) -> None:
    _fail_twice(settings)
    gateway, provider = _gateway(settings, "Walk through it [S1].")

    answer = service.hint(
        settings, _target(settings), level=4, yes=False, ask=_no_question, gateway=gateway
    )

    assert answer.level == 4
    assert len(provider.requests) == 1


def test_the_yes_flag_opens_l4_without_asking(settings: Settings) -> None:
    gateway, provider = _gateway(settings, "Walk through it [S1].")

    service.hint(settings, _target(settings), level=4, yes=True, ask=_no_question, gateway=gateway)

    assert len(provider.requests) == 1


def test_a_later_passing_submit_does_not_unlock_l4_on_its_own(settings: Settings) -> None:
    record_attempt_with_report(
        settings,
        attempt_id="one-failure",
        status=RunStatus.FAILED,
        tests=failing_tests(),
        when=T0,
    )
    record_attempt_with_report(
        settings, attempt_id="now-passing", status=RunStatus.PASSED, when=T0 + timedelta(hours=1)
    )
    gateway, provider = _gateway(settings, "never sent")

    with pytest.raises(DogmaError) as locked:
        service.hint(
            settings, _target(settings), level=4, yes=False, ask=lambda _q: False, gateway=gateway
        )

    assert locked.value.code is ErrorCode.TUTOR_LOCKED
    assert provider.requests == []


def test_sentences_judging_the_learner_code_are_removed(settings: Settings) -> None:
    gateway, _ = _gateway(
        settings, "Your code is correct. The concept is the mean [S1]. This is wrong, too."
    )

    answer = service.hint(
        settings, _target(settings), level=1, yes=False, ask=_no_question, gateway=gateway
    )

    assert answer.nudge is not None
    assert "Your code is correct" not in answer.nudge.text
    assert "This is wrong" not in answer.nudge.text
    assert "The concept is the mean [S1]." in answer.nudge.text
    assert "grader" in answer.nudge.text


def test_debug_without_any_recorded_attempt_is_a_domain_error(settings: Settings) -> None:
    gateway, provider = _gateway(settings, "never sent")

    with pytest.raises(DogmaError) as missing:
        service.debug(settings, _target(settings), gateway=gateway)

    assert missing.value.code is ErrorCode.TUTOR_NO_FAILURE
    assert provider.requests == []


def test_debug_says_when_the_learner_database_is_unreadable(settings: Settings) -> None:
    settings.var_dir.mkdir(parents=True)
    (settings.var_dir / "dogma.db").mkdir()
    gateway, provider = _gateway(settings, "never sent")

    with pytest.raises(DogmaError) as missing:
        service.debug(settings, _target(settings), gateway=gateway)

    assert missing.value.code is ErrorCode.TUTOR_NO_FAILURE
    assert "cannot be read" in missing.value.message
    assert provider.requests == []


def test_debug_after_a_passing_latest_attempt_is_refused(settings: Settings) -> None:
    record_attempt_with_report(settings, attempt_id="ok-1", status=RunStatus.PASSED, when=T0)
    gateway, provider = _gateway(settings, "never sent")

    with pytest.raises(DogmaError) as missing:
        service.debug(settings, _target(settings), gateway=gateway)

    assert missing.value.code is ErrorCode.TUTOR_NO_FAILURE
    assert provider.requests == []


def test_debug_names_the_failure_without_hidden_values(settings: Settings) -> None:
    _fail_twice(settings)
    gateway, provider = _gateway(
        settings,
        "First wrong: the second element is read from the wrong row [S1].\n"
        "Error type: AssertionError.\nKind: shape bug.\nWhat does xs[0] hold?",
    )

    answer = service.debug(settings, _target(settings), gateway=gateway)

    assert answer.mode is Mode.DEBUG
    assert "First wrong" in answer.text
    sent = "\n".join(message.content for message in provider.requests[0].messages)
    assert "error type: AssertionError" in sent
    assert HIDDEN_VALUE not in sent


def test_debug_does_not_rewrite_long_code(settings: Settings) -> None:
    _fail_twice(settings)
    code = "```python\n" + "\n".join(f"line_{n} = {n}" for n in range(8)) + "\n```"
    gateway, _ = _gateway(settings, f"The fault is on line 2 [S1].\n{code}")

    answer = service.debug(settings, _target(settings), gateway=gateway)

    assert "line_5 = 5" not in answer.text
    assert "[code omitted" in answer.text


def test_a_down_model_leaves_the_reviewed_hint_and_one_note(settings: Settings) -> None:
    gateway, _ = _gateway(
        settings, ProviderError(ErrorCode.LLM_UNAVAILABLE, "unavailable", "server is down")
    )

    answer = service.hint(
        settings, _target(settings), level=1, yes=False, ask=_no_question, gateway=gateway
    )

    assert answer.reviewed is not None
    assert answer.nudge is None
    assert answer.nudge_note == "Personalised nudge needs the local model: server is down"


def test_without_a_reviewed_hint_a_down_model_is_still_an_error(settings: Settings) -> None:
    (_target(settings).problem_dir / "hints.md").unlink()
    gateway, _ = _gateway(
        settings, ProviderError(ErrorCode.LLM_UNAVAILABLE, "unavailable", "server is down")
    )

    with pytest.raises(DogmaError) as unavailable:
        service.hint(
            settings, _target(settings), level=1, yes=False, ask=_no_question, gateway=gateway
        )

    assert unavailable.value.code is ErrorCode.LLM_UNAVAILABLE


def test_the_reviewed_hint_is_ready_before_the_model_is_called(settings: Settings) -> None:
    _, provider = _gateway(settings, "never sent")

    prepared = service.prepare_hint(
        settings, _target(settings), level=2, yes=False, ask=_no_question
    )

    assert prepared.reviewed is not None
    assert prepared.reviewed.text.startswith("Each output value")
    assert prepared.hints_note is None
    assert provider.requests == []


def test_the_model_sees_the_reviewed_hints_up_to_its_level_and_no_further(
    settings: Settings,
) -> None:
    gateway, provider = _gateway(settings, "Concept [S1].")

    service.hint(settings, _target(settings), level=1, yes=False, ask=_no_question, gateway=gateway)

    sent = "\n".join(message.content for message in provider.requests[0].messages)
    assert "Doubling is multiplication by two" in sent
    assert "Each output value" not in sent
    assert "The new value" not in sent


def test_level_four_is_model_only_and_never_sees_a_reviewed_hint(settings: Settings) -> None:
    gateway, provider = _gateway(settings, "Walk through it [S1].")

    answer = service.hint(
        settings, _target(settings), level=4, yes=True, ask=_no_question, gateway=gateway
    )

    assert answer.reviewed is None
    assert answer.nudge is not None
    sent = "\n".join(message.content for message in provider.requests[0].messages)
    assert "REVIEWED HINT" not in sent


def test_a_missing_hints_file_falls_back_to_the_model_with_a_note(settings: Settings) -> None:
    (_target(settings).problem_dir / "hints.md").unlink()
    gateway, _ = _gateway(settings, "Think about the mean [S1].")

    prepared = service.prepare_hint(
        settings, _target(settings), level=1, yes=False, ask=_no_question
    )
    answer = prepared.complete(gateway)

    assert prepared.reviewed is None
    assert prepared.hints_note is not None
    assert "missing" in prepared.hints_note
    assert answer.reviewed is None
    assert answer.nudge is not None
    assert answer.nudge.text == "Think about the mean [S1]."
    assert answer.nudge_note is None


def test_a_malformed_hints_file_falls_back_with_a_note(settings: Settings) -> None:
    (_target(settings).problem_dir / "hints.md").write_text(
        "## Hint 1: only\n\nA.\n", encoding="utf-8"
    )

    prepared = service.prepare_hint(
        settings, _target(settings), level=1, yes=False, ask=_no_question
    )

    assert prepared.reviewed is None
    assert prepared.hints_note is not None
    assert "malformed" in prepared.hints_note


def test_guard_verdicts_only_touches_judgements_about_code() -> None:
    text = (
        "That is correct reasoning about the mean. Your solution is incorrect here.\n"
        "The correct approach divides by n.\n"
        "The code passes all tests."
    )

    guarded, changed = guard_verdicts(text)

    assert changed is True
    assert "Your solution is incorrect" not in guarded
    assert "The code passes" not in guarded
    assert "The correct approach divides by n." in guarded


def test_guard_verdicts_reports_no_change_for_neutral_text() -> None:
    assert guard_verdicts("Name the concept of an average.") == (
        "Name the concept of an average.",
        False,
    )


def test_omit_long_code_keeps_short_blocks() -> None:
    short = "```python\nx = 1\n```"
    long = "```python\n" + "\n".join("y = 2" for _ in range(6)) + "\n```"

    assert omit_long_code(short) == (short, False)
    assert omit_long_code(long)[1] is True


def test_citations_must_name_a_real_source() -> None:
    sources = (Source(tag="S1", citation="[lesson]", kind="lesson"),)

    assert cites("see [S1]", sources) is True
    assert cites("see [S9]", sources) is False
    assert cites("see [S1]", ()) is False
