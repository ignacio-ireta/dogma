"""dogma hint and dogma debug: output, degradation, the L4 gate and usage errors.

The model is always a FakeProvider behind a real Gateway. Nothing here reaches a server.
"""

from __future__ import annotations

from pathlib import Path

import httpx
import pytest
from tests.unit.cli.support import PROBLEM
from tests.unit.tutor.support import HIDDEN_VALUE, failing_tests, record_attempt_with_report

from dogma.cli import app
from dogma.cli import tutor as tutor_cli
from dogma.core.config import Settings
from dogma.core.errors import ErrorCode
from dogma.grading.models import RunStatus
from dogma.tutor.gateway import Gateway
from dogma.tutor.models import ProviderError
from dogma.tutor.providers import local
from dogma.tutor.providers.fake import FakeProvider

PROBLEM_DIR = "content/cml/vector-operations/problems/p01-vector-projection"


@pytest.fixture
def model(monkeypatch: pytest.MonkeyPatch, settings: Settings) -> FakeProvider:
    provider = FakeProvider([])
    monkeypatch.setattr(tutor_cli, "make_gateway", lambda s: Gateway(s, provider))
    return provider


def test_hint_prints_the_answer_its_level_and_its_sources(invoke, model: FakeProvider) -> None:
    model.queue("Name the concept: linear combinations of arrows [S1].")

    outcome = invoke("hint", PROBLEM)

    assert outcome.exit_code == 0, outcome.output
    assert "Reviewed hint · L1 · concept" in outcome.output
    assert "Personalised nudge · L1" in outcome.output
    assert "Name the concept" in outcome.output
    assert "Sources" in outcome.output
    assert "[S1]" in outcome.output
    assert "Traceback" not in outcome.output


def test_hint_levels_two_and_three_are_selected(invoke, model: FakeProvider) -> None:
    model.queue("Approach [S1].")
    model.queue("Shape [S1].")

    assert "L2" in invoke("hint", PROBLEM, "--level", "2").output
    assert "L3" in invoke("hint", PROBLEM, "--level", "3").output


def test_model_text_is_printed_literally(invoke, model: FakeProvider) -> None:
    model.queue("Try [bold]this[/bold] shape [S1].")

    outcome = invoke("hint", PROBLEM)

    assert "[bold]this[/bold]" in outcome.output


def test_a_down_model_keeps_the_reviewed_hint_and_notes_the_missing_nudge(
    invoke, model: FakeProvider
) -> None:
    model.queue(
        ProviderError(
            ErrorCode.LLM_UNAVAILABLE,
            "unavailable",
            "the local model server is not reachable; start it or mount the External SSD",
        )
    )

    outcome = invoke("hint", PROBLEM)

    assert outcome.exit_code == 0, outcome.output
    assert "Reviewed hint · L1 · concept" in outcome.output
    assert "Personalised nudge needs the local model" in outcome.output
    assert "error [" not in outcome.output
    assert "Traceback" not in outcome.output
    assert invoke("open", PROBLEM).exit_code == 0


def test_a_missing_hints_file_gives_the_model_only_hint_with_a_note(
    invoke, repo: Path, model: FakeProvider
) -> None:
    (repo / PROBLEM_DIR / "hints.md").unlink()
    model.queue("Name the concept of linear combinations [S1].")

    outcome = invoke("hint", PROBLEM)

    assert outcome.exit_code == 0, outcome.output
    assert "hints.md is missing" in outcome.output
    assert "Hint · L1 · concept" in outcome.output
    assert "Reviewed hint" not in outcome.output


def test_a_down_model_without_reviewed_hints_is_still_an_error(
    invoke, repo: Path, model: FakeProvider
) -> None:
    (repo / PROBLEM_DIR / "hints.md").unlink()
    model.queue(ProviderError(ErrorCode.LLM_UNAVAILABLE, "unavailable", "not reachable"))

    outcome = invoke("hint", PROBLEM)

    assert outcome.exit_code == 1
    assert "error [LLM_UNAVAILABLE]" in outcome.output


def test_a_later_level_names_the_earlier_hints(invoke, model: FakeProvider) -> None:
    model.queue("Approach [S1].")

    outcome = invoke("hint", PROBLEM, "--level", "2")

    assert outcome.exit_code == 0, outcome.output
    assert "Reviewed hint · L2 · formula / approach" in outcome.output
    assert "Earlier hints: --level 1." in outcome.output


def test_an_unreachable_server_is_refused_without_any_request(
    cli, settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    def boom(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("a refused endpoint must not open an HTTP client")

    monkeypatch.setattr(local.httpx, "Client", boom)
    external = Settings(repo_root=settings.repo_root, llm_base_url="https://api.example.com/v1")

    outcome = cli.invoke(app, ["hint", PROBLEM], obj=external, env={"COLUMNS": "200"})

    assert outcome.exit_code == 1
    assert "error [LLM_ENDPOINT_REFUSED]" in outcome.output
    assert "Reviewed hint" in outcome.output, "the reviewed hint shows before any model call"
    assert "Traceback" not in outcome.output


def test_a_transport_failure_from_the_real_provider_degrades(
    cli, settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    def refuse(_request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused")

    provider = local.LocalProvider(settings.llm_base_url, transport=httpx.MockTransport(refuse))
    monkeypatch.setattr(tutor_cli, "make_gateway", lambda s: Gateway(s, provider))

    outcome = cli.invoke(app, ["hint", PROBLEM], obj=settings, env={"COLUMNS": "200"})

    assert outcome.exit_code == 0, outcome.output
    assert "Reviewed hint" in outcome.output
    assert "Personalised nudge needs the local model" in outcome.output
    assert "not reachable" in outcome.output
    assert "error [" not in outcome.output


def test_l4_needs_confirmation_and_declining_is_a_clean_refusal(
    cli, settings: Settings, model: FakeProvider
) -> None:
    outcome = cli.invoke(
        app, ["hint", PROBLEM, "--level", "4"], obj=settings, input="n\n", env={"COLUMNS": "200"}
    )

    assert outcome.exit_code == 1
    assert "error [TUTOR_LOCKED]" in outcome.output
    assert model.requests == []


def test_l4_opens_when_the_learner_answers_yes(
    cli, settings: Settings, model: FakeProvider
) -> None:
    model.queue("Walk through it [S1].")

    outcome = cli.invoke(
        app, ["hint", PROBLEM, "--level", "4"], obj=settings, input="y\n", env={"COLUMNS": "200"}
    )

    assert outcome.exit_code == 0, outcome.output
    assert len(model.requests) == 1


def test_l4_opens_with_the_yes_flag(invoke, model: FakeProvider) -> None:
    model.queue("Walk through it [S1].")

    outcome = invoke("hint", PROBLEM, "--level", "4", "--yes")

    assert outcome.exit_code == 0, outcome.output
    assert "L4" in outcome.output
    assert len(model.requests) == 1


def test_l4_opens_after_two_recorded_failed_submits(
    invoke, settings: Settings, model: FakeProvider
) -> None:
    for index in range(2):
        record_attempt_with_report(
            settings,
            attempt_id=f"cli-fail-{index}",
            status=RunStatus.FAILED,
            tests=failing_tests(),
            problem_id=PROBLEM,
        )
    model.queue("Walk through it [S1].")

    outcome = invoke("hint", PROBLEM, "--level", "4")

    assert outcome.exit_code == 0, outcome.output


@pytest.mark.parametrize("level", ["0", "5"])
def test_levels_outside_the_ladder_are_usage_errors(invoke, level: str) -> None:
    outcome = invoke("hint", PROBLEM, "--level", level)

    assert outcome.exit_code == 2


def test_debug_without_a_recorded_failure_says_what_to_do(invoke, model: FakeProvider) -> None:
    outcome = invoke("debug", PROBLEM)

    assert outcome.exit_code == 1
    assert "error [TUTOR_NO_FAILURE]" in outcome.output
    assert model.requests == []


def test_debug_names_the_first_wrong_line_without_hidden_values(
    invoke, settings: Settings, model: FakeProvider
) -> None:
    record_attempt_with_report(
        settings,
        attempt_id="cli-debug",
        status=RunStatus.FAILED,
        tests=failing_tests(),
        problem_id=PROBLEM,
    )
    model.queue("First wrong: the indexing on line 2 [S1].\nError type: AssertionError.")

    outcome = invoke("debug", PROBLEM)

    assert outcome.exit_code == 0, outcome.output
    assert "First wrong" in outcome.output
    assert HIDDEN_VALUE not in outcome.output
    sent = "\n".join(message.content for message in model.requests[0].messages)
    assert HIDDEN_VALUE not in sent


def test_hint_and_debug_help_have_an_example(cli, settings: Settings) -> None:
    for command in ("hint", "debug"):
        outcome = cli.invoke(app, [command, "--help"], obj=settings)

        assert outcome.exit_code == 0, outcome.output
        assert "Example:" in outcome.output


def test_root_help_lists_hint_and_debug(cli, settings: Settings) -> None:
    outcome = cli.invoke(app, ["--help"], obj=settings)

    assert "hint" in outcome.output
    assert "debug" in outcome.output
