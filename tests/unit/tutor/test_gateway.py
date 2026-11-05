"""The gateway: the one path to a model. It logs metadata only and refuses other hosts."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import pytest
from tests.unit.tutor.support import PROBLEM_ID

from dogma.core.config import Settings
from dogma.core.errors import DogmaError, ErrorCode
from dogma.tutor.gateway import CallMeta, Gateway, call_log_path, strip_thinking
from dogma.tutor.models import CallMode, ProviderError
from dogma.tutor.prompts import load_template, render_prompt
from dogma.tutor.providers import local
from dogma.tutor.providers.fake import FakeProvider

PROMPT_BODY = "PROMPT_BODY_SENTINEL_0d9c user text the log must never contain"


def _meta(level: int = 1) -> CallMeta:
    prompt = render_prompt(CallMode.HINT, level, PROMPT_BODY)
    return CallMeta(mode=CallMode.HINT, prompt=prompt, max_tokens=300, problem_id=PROBLEM_ID)


def _log_lines(settings: Settings) -> list[dict[str, object]]:
    path = call_log_path(settings)
    if not path.exists():
        return []
    rows: list[dict[str, object]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        assert isinstance(row, dict)
        rows.append(row)
    return rows


def test_a_successful_call_appends_one_metadata_line(settings: Settings) -> None:
    meta = _meta(level=1)
    gateway = Gateway(settings, FakeProvider(["Think about shape [S1]."]))

    result = gateway.complete(meta)

    assert result.text == "Think about shape [S1]."
    [row] = _log_lines(settings)
    assert row["mode"] == "hint"
    assert row["level"] == 1
    assert row["problem"] == PROBLEM_ID
    assert row["prompt_version"] == load_template(CallMode.HINT).version
    assert row["prompt_sha256"] == meta.prompt.sha256
    assert row["model"] == settings.llm_model
    assert row["provider"] == "fake"
    assert row["outcome"] == "ok"
    assert isinstance(row["latency_ms"], int) and row["latency_ms"] >= 0
    datetime.fromisoformat(str(row["ts"]))


def test_the_log_never_holds_prompt_or_answer_text(settings: Settings) -> None:
    gateway = Gateway(settings, FakeProvider(["ANSWER_BODY_SENTINEL_44aa"]))

    gateway.complete(_meta())

    raw = call_log_path(settings).read_text(encoding="utf-8")
    assert PROMPT_BODY not in raw
    assert "ANSWER_BODY_SENTINEL_44aa" not in raw


def test_each_call_adds_exactly_one_line(settings: Settings) -> None:
    gateway = Gateway(settings, FakeProvider(["one", "two"]))

    gateway.complete(_meta())
    gateway.complete(_meta(level=2))

    assert [row["level"] for row in _log_lines(settings)] == [1, 2]


def test_a_failed_call_is_logged_with_its_outcome(settings: Settings) -> None:
    gateway = Gateway(
        settings,
        FakeProvider([ProviderError(ErrorCode.LLM_UNAVAILABLE, "timeout", "too slow")]),
    )

    with pytest.raises(DogmaError) as failed:
        gateway.complete(_meta())

    assert failed.value.code is ErrorCode.LLM_UNAVAILABLE
    [row] = _log_lines(settings)
    assert row["outcome"] == "timeout"


def test_an_empty_answer_is_an_invalid_response(settings: Settings) -> None:
    gateway = Gateway(settings, FakeProvider(["<think>only reasoning</think>  "]))

    with pytest.raises(DogmaError) as failed:
        gateway.complete(_meta())

    assert failed.value.code is ErrorCode.LLM_RESPONSE_INVALID
    assert [row["outcome"] for row in _log_lines(settings)] == ["invalid"]


def test_a_non_loopback_endpoint_is_refused_before_any_request(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def boom(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("a refused endpoint must not open an HTTP client")

    monkeypatch.setattr(local.httpx, "Client", boom)
    settings = Settings(repo_root=tmp_path, llm_base_url="http://10.0.0.7:8080/v1")

    with pytest.raises(DogmaError) as refused:
        Gateway(settings).complete(_meta())

    assert refused.value.code is ErrorCode.LLM_ENDPOINT_REFUSED
    assert [row["outcome"] for row in _log_lines(settings)] == ["refused"]


def test_an_unwritable_log_is_a_domain_error(tmp_path: Path) -> None:
    (tmp_path / "var").write_text("a file, not a directory")
    settings = Settings(repo_root=tmp_path)

    with pytest.raises(DogmaError) as failed:
        Gateway(settings, FakeProvider(["answer"])).complete(_meta())

    assert failed.value.code is ErrorCode.LLM_LOG_FAILED


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("<think>plan the answer</think>Use the mean.", "Use the mean."),
        ("<think>first\nsecond</think>\n\nOK", "OK"),
        ("No reasoning block here.", "No reasoning block here."),
        ("<think>never closed", ""),
    ],
)
def test_reasoning_blocks_are_removed_from_answers(raw: str, expected: str) -> None:
    assert strip_thinking(raw).strip() == expected
