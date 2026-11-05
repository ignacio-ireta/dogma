"""Context assembly: the token budget, deterministic truncation, source tags and redaction."""

from __future__ import annotations

from dataclasses import replace

import pytest
from tests.unit.tutor.support import (
    HIDDEN_VALUE,
    PROBLEM_ID,
    failing_tests,
    sample_evidence,
)

from dogma.corpus.models import Chunk, SearchHit
from dogma.grading.models import GradeReport, RunStatus
from dogma.tutor import context
from dogma.tutor.context import MAX_PROMPT_CHARS, assemble_prompt, render_report
from dogma.tutor.evidence import LessonSection, redact_hidden_message
from dogma.tutor.models import CallMode, ReviewedHints


def test_the_whole_prompt_fits_the_token_budget_even_for_huge_inputs() -> None:
    huge = sample_evidence()
    evidence = replace(
        huge,
        problem_text="# Huge\n" + "lorem ipsum dolor sit amet\n" * 20_000,
        workspace_text="x = 1\n" * 20_000,
        lesson_sections=tuple(LessonSection(f"Section {n}", "word " * 5_000) for n in range(6)),
    )

    assembled = assemble_prompt(evidence, mode=CallMode.HINT, level=2)

    total = sum(len(message.content) for message in assembled.prompt.messages)
    assert total <= MAX_PROMPT_CHARS
    assert MAX_PROMPT_CHARS <= 12_000 * 3  # 12K tokens at 3 characters per token
    assert "[truncated]" in assembled.prompt.messages[1].content


def test_assembly_is_deterministic() -> None:
    evidence = sample_evidence()

    first = assemble_prompt(evidence, mode=CallMode.DEBUG)
    second = assemble_prompt(evidence, mode=CallMode.DEBUG)

    assert first == second


def test_sources_are_tagged_in_prompt_order_and_listed() -> None:
    assembled = assemble_prompt(sample_evidence(), mode=CallMode.HINT, level=1)
    user = assembled.prompt.messages[1].content

    tags = [source.tag for source in assembled.sources]
    assert tags == ["S1", "S2", "S3"]
    assert [source.kind for source in assembled.sources] == ["reviewed", "lesson", "corpus"]
    for tag in tags:
        assert f" {tag}: " in user


def test_level_four_has_no_reviewed_hint_and_keeps_the_old_sources() -> None:
    assembled = assemble_prompt(sample_evidence(), mode=CallMode.HINT, level=4)

    assert [source.kind for source in assembled.sources] == ["lesson", "corpus"]
    assert "REVIEWED HINT" not in assembled.prompt.messages[1].content


@pytest.mark.parametrize(
    ("level", "shown", "hidden"),
    [
        (1, ["Doubling is multiplication"], ["Each output value", "The new value"]),
        (2, ["Doubling is multiplication", "Each output value"], ["The new value"]),
        (3, ["Doubling is multiplication", "Each output value", "The new value"], []),
    ],
)
def test_the_reviewed_hints_up_to_the_level_are_the_ceiling(
    level: int, shown: list[str], hidden: list[str]
) -> None:
    assembled = assemble_prompt(sample_evidence(), mode=CallMode.HINT, level=level)
    text = "\n".join(message.content for message in assembled.prompt.messages)

    for fragment in shown:
        assert fragment in text
    for fragment in hidden:
        assert fragment not in text


def test_debug_never_carries_reviewed_hints() -> None:
    assembled = assemble_prompt(sample_evidence(), mode=CallMode.DEBUG)
    text = "\n".join(message.content for message in assembled.prompt.messages)

    assert "Each output value" not in text
    assert all(source.kind != "reviewed" for source in assembled.sources)


def _hit(source_id: str, ordinal: int) -> SearchHit:
    chunk = Chunk(
        source_id=source_id,
        path=source_id.split(":", 1)[1],
        page=None,
        headings=(),
        ordinal=ordinal,
        text=f"passage {ordinal} of {source_id}",
    )
    return SearchHit(chunk=chunk, citation=f"[{source_id}]", score=float(ordinal))


def test_selection_keeps_the_best_passage_of_each_document_in_order() -> None:
    hits = (
        _hit("sources:papers/a.pdf", 0),
        _hit("sources:papers/a.pdf", 1),
        _hit("sources:notes/b.md", 0),
        _hit("sources:papers/a.pdf", 2),
        _hit("sources:notes/c.md", 0),
        _hit("sources:notes/d.md", 0),
        _hit("sources:notes/e.md", 0),
    )

    chosen = context.select_corpus(hits)

    assert [(hit.chunk.source_id, hit.chunk.ordinal) for hit in chosen] == [
        ("sources:papers/a.pdf", 0),
        ("sources:notes/b.md", 0),
        ("sources:notes/c.md", 0),
        ("sources:notes/d.md", 0),
    ]


def test_one_long_paper_cannot_fill_the_corpus_slots_of_a_prompt() -> None:
    hits = tuple(_hit("sources:papers/long.pdf", ordinal) for ordinal in range(6))
    evidence = replace(sample_evidence(), corpus_hits=hits)

    assembled = assemble_prompt(evidence, mode=CallMode.DEBUG)

    corpus = [source for source in assembled.sources if source.kind == "corpus"]
    assert len(corpus) == 1
    assert "passage 0 of sources:papers/long.pdf" in assembled.prompt.messages[1].content
    assert "passage 1 of sources:papers/long.pdf" not in assembled.prompt.messages[1].content


def test_with_no_sources_the_prompt_says_so() -> None:
    evidence = replace(
        sample_evidence(),
        lesson_sections=(),
        corpus_hits=(),
        reviewed=ReviewedHints(note="hints.md is missing"),
    )

    assembled = assemble_prompt(evidence, mode=CallMode.HINT, level=1)

    assert assembled.sources == ()
    assert "SOURCES: none" in assembled.prompt.messages[1].content


def test_hidden_values_never_reach_the_prompt() -> None:
    assembled = assemble_prompt(sample_evidence(), mode=CallMode.DEBUG)
    text = "\n".join(message.content for message in assembled.prompt.messages)

    assert HIDDEN_VALUE not in text
    assert "error type: AssertionError" in text
    assert "learner file line(s): 2" in text
    assert "test_hidden_mutation (hidden): fail" in text


def test_public_failures_keep_their_message() -> None:
    assembled = assemble_prompt(sample_evidence(), mode=CallMode.DEBUG)

    assert "expected [-2] got [0]" in assembled.prompt.messages[1].content


def test_the_problem_block_names_the_problem_and_its_text() -> None:
    assembled = assemble_prompt(sample_evidence(), mode=CallMode.HINT, level=3)
    user = assembled.prompt.messages[1].content

    assert f"PROBLEM ({PROBLEM_ID}, Double all, python)" in user
    assert "Implement `double_all(xs)`" in user
    assert "LEARNER FILE (workspace/tst/alpha/p01-double.py)" in user
    assert "return [x + x for x in xs[0]]" in user


def test_omitted_blocks_are_named_when_the_budget_is_tiny(monkeypatch) -> None:
    monkeypatch.setattr(context, "MAX_PROMPT_CHARS", 3_000)

    assembled = assemble_prompt(sample_evidence(), mode=CallMode.HINT, level=1)

    assert any("omitted for length" in note for note in assembled.notes)
    assert "PROBLEM" in assembled.prompt.messages[1].content


def test_report_rendering_shows_status_counts_and_no_hidden_detail() -> None:
    report = GradeReport(
        attempt_id="a",
        problem_id=PROBLEM_ID,
        problem_version=1,
        runtime="py-numpy",
        status=RunStatus.FAILED,
        tests=failing_tests(),
    )

    text = render_report(report, submitted_at="2026-10-24T12:00:00Z")

    assert "status: failed" in text
    assert "submitted: 2026-10-24T12:00:00Z" in text
    assert "public 1/2 passed; hidden 1/2 passed" in text


@pytest.mark.parametrize(
    ("message", "expected"),
    [
        ("AssertionError: value 999\nFalsifying example: xs=[9]", "error type: AssertionError"),
        ('ValueError raised\n  File "submission.py", line 7, in f', "learner file line(s): 7"),
        ("something went sideways with 12345", "details withheld (hidden test)"),
    ],
)
def test_hidden_messages_keep_only_error_type_and_learner_lines(
    message: str, expected: str
) -> None:
    assert expected in redact_hidden_message(message)
    assert "999" not in redact_hidden_message(message)
    assert "12345" not in redact_hidden_message(message)
