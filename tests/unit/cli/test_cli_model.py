"""Model features in the terminal: advisory scores beside self-scores, and the interviewer.

The model is always a FakeProvider behind a real Gateway. A down model is a ProviderError, so the
outage path is tested without any socket. Every command shows its learner-facing output only.
"""

# ruff: noqa: F811  the open-mock fixture is imported once and requested by name in each test
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType

import pytest
from sqlalchemy import select
from tests.unit.cli.test_cli_mock import mock_settings  # noqa: F401  the open-mock fixture
from typer.testing import CliRunner

from dogma.cli import answer_cmd, app, mock_cmd
from dogma.content.rubrics import load_rubric_version
from dogma.core.config import Settings
from dogma.core.errors import ErrorCode
from dogma.db.answer_models import Answer
from dogma.db.errors import StorageErrorCode
from dogma.db.session import open_session
from dogma.tutor import gateway as gateway_module
from dogma.tutor.gateway import Gateway
from dogma.tutor.models import ProviderError
from dogma.tutor.providers.fake import FakeProvider

REPO_ROOT = Path(__file__).resolve().parents[3]
TOPIC = "cml/vector-operations"
READY = "accuracy=3,structure=3,trade-offs=2,communication=4"
ANSWER_TEXT = (
    "Use a split by time, not a random split, because the data is ordered. "
    "Then check the metric on the newest month only."
)
DEBRIEF_LINE = "Tier by lifetime value times risk."
WIDE = {"COLUMNS": "200"}


@dataclass(frozen=True)
class StoredAnswer:
    scores: dict[str, int]
    model_score: dict[str, object] | None


def _words(output: str) -> str:
    return " ".join(output.split())


def _answer_module() -> ModuleType:
    return answer_cmd


def _mock_module() -> ModuleType:
    return mock_cmd


def _model(
    monkeypatch: pytest.MonkeyPatch, module: ModuleType, settings: Settings, *replies: object
) -> FakeProvider:
    """Put a FakeProvider behind the real Gateway for one command module."""
    provider = FakeProvider([reply for reply in replies if isinstance(reply, (str, ProviderError))])
    monkeypatch.setattr(module, "make_gateway", lambda s: Gateway(s, provider))
    return provider


def _scores_file(tmp_path: Path, scores: dict[str, int]) -> Path:
    path = tmp_path / "scores.json"
    path.write_text(json.dumps(scores), encoding="utf-8")
    return path


def _down() -> ProviderError:
    return ProviderError(ErrorCode.LLM_UNAVAILABLE, "unavailable", "the local model is down")


def _answer_file(tmp_path: Path) -> Path:
    path = tmp_path / "answer.txt"
    path.write_text(ANSWER_TEXT, encoding="utf-8")
    return path


def _stored_answer(settings: Settings) -> StoredAnswer:
    """The one stored answer, read inside its session so no ORM object outlives it."""
    with open_session(settings, on_failure=StorageErrorCode.DB_READ_FAILED) as session:
        row = session.scalars(select(Answer)).one()
        return StoredAnswer(scores=dict(row.scores), model_score=row.model_score)


def _answer(cli: CliRunner, settings: Settings, file_path: Path, *extra: str) -> object:
    args = ["answer", TOPIC, "--file", str(file_path), "--scores", READY, *extra]
    return cli.invoke(app, args, obj=settings, env=WIDE)


# --- dogma answer --model-score -------------------------------------------------------------


def test_answer_model_score_prints_the_advisory_beside_the_self_scores(
    cli: CliRunner, settings: Settings, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    reply = json.dumps({"scores": [{"id": "accuracy", "score": 2, "quote": "a split by time"}]})
    _model(monkeypatch, _answer_module(), settings, reply)

    result = _answer(cli, settings, _answer_file(tmp_path), "--model-score")

    assert result.exit_code == 0, result.output
    output = _words(result.output)
    assert "accuracy: you 3, model 2" in output
    assert "a split by time" in output
    assert "differs from you by 1 on accuracy" in output
    assert "Your scores stand" in output


def test_answer_model_score_stores_the_advisory_and_not_a_new_self_score(
    cli: CliRunner, settings: Settings, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    reply = json.dumps({"scores": [{"id": "accuracy", "score": 1, "quote": "random split"}]})
    _model(monkeypatch, _answer_module(), settings, reply)

    _answer(cli, settings, _answer_file(tmp_path), "--model-score")

    stored = _stored_answer(settings)
    assert stored.scores == {"accuracy": 3, "structure": 3, "trade-offs": 2, "communication": 4}
    assert stored.model_score is not None
    assert stored.model_score["scores"] == {"accuracy": {"score": 1, "quote": "random split"}}


def test_a_down_model_keeps_the_self_scores_and_exits_zero(
    cli: CliRunner, settings: Settings, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _model(monkeypatch, _answer_module(), settings, _down())

    result = _answer(cli, settings, _answer_file(tmp_path), "--model-score")

    assert result.exit_code == 0, result.output
    assert "Model score unavailable" in _words(result.output)
    assert "Your self-scores are stored" in _words(result.output)
    stored = _stored_answer(settings)
    assert stored.scores["accuracy"] == 3
    assert stored.model_score is None


def test_a_spoken_answer_has_no_text_to_score(
    cli: CliRunner, settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    provider = _model(monkeypatch, _answer_module(), settings)  # any call would fail the test

    result = cli.invoke(
        app,
        ["answer", TOPIC, "--spoken", "--scores", READY, "--model-score"],
        obj=settings,
        env=WIDE,
    )

    assert result.exit_code == 0, result.output
    assert "no text for the model to score" in _words(result.output)
    assert provider.requests == []


def test_a_reply_with_no_usable_score_is_a_note_and_the_self_scores_stand(
    cli: CliRunner, settings: Settings, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _model(monkeypatch, _answer_module(), settings, "Looks correct to me!")

    result = _answer(cli, settings, _answer_file(tmp_path), "--model-score")

    assert result.exit_code == 0, result.output
    assert "Model score unavailable" in _words(result.output)
    assert _stored_answer(settings).scores["accuracy"] == 3


def test_without_the_flag_no_model_is_called(
    cli: CliRunner, settings: Settings, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    provider = _model(monkeypatch, _answer_module(), settings)

    result = _answer(cli, settings, _answer_file(tmp_path))

    assert result.exit_code == 0, result.output
    assert provider.requests == []


# --- dogma mock score --model-score ---------------------------------------------------------


def _mock_score_input() -> str:
    """Thirty items: a score each, and a note on the first item only (the model reads notes)."""
    rubric = load_rubric_version(REPO_ROOT / "curriculum", "mock", 1)
    lines: list[str] = []
    for index, _ in enumerate(rubric.criteria_ids):
        lines.append("3")
        lines.append("I explained the split by time before writing it." if index == 0 else "")
    return "\n".join(lines) + "\n"


def test_mock_score_model_prints_advisory_items_and_stores_them(
    cli: CliRunner, mock_settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    assert cli.invoke(app, ["mock", "start"], obj=mock_settings, env=WIDE).exit_code == 0
    reply = json.dumps(
        {"scores": [{"id": "coding.narration", "score": 2, "quote": "the split by time"}]}
    )
    _model(monkeypatch, _mock_module(), mock_settings, reply)

    result = cli.invoke(
        app,
        ["mock", "score", "--model-score"],
        obj=mock_settings,
        input=_mock_score_input(),
        env=WIDE,
    )

    assert result.exit_code == 0, result.output
    output = _words(result.output)
    assert "coding.narration: you 3, model 2" in output
    assert "differs from you by 1 on coding.narration" in output
    assert "Your scores stand" in output


def test_mock_score_model_without_notes_says_there_is_nothing_to_read(
    cli: CliRunner, mock_settings: Settings, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    assert cli.invoke(app, ["mock", "start"], obj=mock_settings, env=WIDE).exit_code == 0
    provider = _model(monkeypatch, _mock_module(), mock_settings)
    rubric = load_rubric_version(REPO_ROOT / "curriculum", "mock", 1)
    scores = _scores_file(tmp_path, dict.fromkeys(rubric.criteria_ids, 3))

    result = cli.invoke(
        app,
        ["mock", "score", "--scores-file", str(scores), "--model-score"],
        obj=mock_settings,
        env=WIDE,
    )

    assert result.exit_code == 0, result.output
    assert "No item notes to score" in _words(result.output)
    assert provider.requests == []


# --- dogma mock ask -------------------------------------------------------------------------


def _start_churn(cli: CliRunner, mock_settings: Settings) -> None:
    result = cli.invoke(
        app,
        ["mock", "start", "--case", "sys/churn-prediction-case"],
        obj=mock_settings,
        env=WIDE,
    )
    assert result.exit_code == 0, result.output


def test_mock_ask_releases_the_reveal_the_model_picks(
    cli: CliRunner, mock_settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    _start_churn(cli, mock_settings)
    provider = _model(monkeypatch, _mock_module(), mock_settings, '{"reveal": 2}')

    result = cli.invoke(
        app, ["mock", "ask", "How do the segments differ?"], obj=mock_settings, env=WIDE
    )

    assert result.exit_code == 0, result.output
    output = _words(result.output)
    assert "Prepaid is 60 percent of subscribers" in output
    assert "content/sys/churn-prediction-case/exhibits/segment-shares.csv" in output
    assert len(provider.requests) == 1


def test_mock_ask_says_nothing_when_the_model_picks_none(
    cli: CliRunner, mock_settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    _start_churn(cli, mock_settings)
    _model(monkeypatch, _mock_module(), mock_settings, '{"reveal": null}')

    result = cli.invoke(app, ["mock", "ask", "What is the weather?"], obj=mock_settings, env=WIDE)

    assert result.exit_code == 0, result.output
    assert "The interviewer has nothing on that." in _words(result.output)


def test_a_pick_whose_question_shares_no_keyword_is_not_released(
    cli: CliRunner, mock_settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    _start_churn(cli, mock_settings)
    _model(monkeypatch, _mock_module(), mock_settings, '{"reveal": 2}')

    result = cli.invoke(app, ["mock", "ask", "What is the weather?"], obj=mock_settings, env=WIDE)

    assert result.exit_code == 0, result.output
    output = _words(result.output)
    assert "Prepaid is 60 percent" not in output
    assert "nothing on that" in output


def test_a_down_model_falls_back_to_the_keyword_match(
    cli: CliRunner, mock_settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    _start_churn(cli, mock_settings)
    _model(monkeypatch, _mock_module(), mock_settings, _down())

    result = cli.invoke(
        app, ["mock", "ask", "How do the segments differ?"], obj=mock_settings, env=WIDE
    )

    assert result.exit_code == 0, result.output
    output = _words(result.output)
    assert "Prepaid is 60 percent of subscribers" in output
    assert "keyword match" in output


def test_the_debrief_never_reaches_the_model_or_the_terminal(
    cli: CliRunner, mock_settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    _start_churn(cli, mock_settings)
    provider = _model(monkeypatch, _mock_module(), mock_settings, '{"reveal": null}')

    result = cli.invoke(app, ["mock", "ask", "What is the tiering?"], obj=mock_settings, env=WIDE)

    assert DEBRIEF_LINE not in result.output
    for request in provider.requests:
        assert all(DEBRIEF_LINE not in message.content for message in request.messages)


def test_mock_ask_without_an_open_mock_is_a_clear_error(
    cli: CliRunner, mock_settings: Settings
) -> None:
    result = cli.invoke(app, ["mock", "ask", "anything"], obj=mock_settings, env=WIDE)

    assert result.exit_code == 1
    assert "No mock is open" in _words(result.output)


def test_an_empty_question_never_reaches_the_model(
    cli: CliRunner, mock_settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    _start_churn(cli, mock_settings)
    provider = _model(monkeypatch, _mock_module(), mock_settings)

    result = cli.invoke(app, ["mock", "ask", "   "], obj=mock_settings, env=WIDE)

    assert result.exit_code == 1
    assert "1 to 500 characters" in _words(result.output)
    assert provider.requests == []


def test_the_model_is_called_through_the_gateway_only(
    cli: CliRunner, mock_settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The interviewer's calls land in the call log as metadata, mode interviewer."""
    _start_churn(cli, mock_settings)
    _model(monkeypatch, _mock_module(), mock_settings, '{"reveal": null}')

    cli.invoke(app, ["mock", "ask", "What is the weather?"], obj=mock_settings, env=WIDE)

    log = gateway_module.call_log_path(mock_settings).read_text(encoding="utf-8")
    assert '"mode": "interviewer"' in log
    assert "What is the weather?" not in log
