"""Chunk boundaries preserve provenance without interpreting fenced examples."""

from __future__ import annotations

import pytest

from dogma.corpus.chunker import chunk_markdown


def test_headings_pages_and_front_matter() -> None:
    chunks = chunk_markdown(
        "---\nsecret: omitted\n---\n# Paper\n## Content\n### Page 1\n#### A\nAlpha.\n"
        "##### B\nBeta.\n### Page 2\nGamma.\n",
        source_id="corpus:papers/a.pdf.md",
        path="corpus/md/papers/a.pdf.md",
        topic_ids=("cml/a",),
    )
    assert [(c.page, c.headings, c.text) for c in chunks] == [
        (1, ("Paper", "Content", "A"), "Alpha."),
        (1, ("Paper", "Content", "A", "B"), "Beta."),
        (2, ("Paper", "Content"), "Gamma."),
    ]
    assert [c.ordinal for c in chunks] == [0, 1, 2]
    assert all(c.topic_ids == ("cml/a",) for c in chunks)


@pytest.mark.parametrize("fence", ["```", "~~~~"])
def test_fenced_page_markers_are_data(fence: str) -> None:
    chunks = chunk_markdown(
        f"# Notes\n### Page 3\n{fence}markdown\n### Page 99\n# Fake\n{fence}\nReal.",
        source_id="x",
        path="x.md",
    )
    assert len(chunks) == 1
    assert chunks[0].page == 3
    assert chunks[0].headings == ("Notes",)
    assert "### Page 99" in chunks[0].text


def test_long_paragraph_is_bounded_and_lossless() -> None:
    body = " ".join(f"word{i}" for i in range(1800))
    chunks = chunk_markdown(body, source_id="x", path="x.md")
    assert len(chunks) > 1
    assert all(len(c.text) <= 2000 for c in chunks)
    assert " ".join(c.text for c in chunks) == body


@pytest.mark.parametrize("text", ["", "  \n", "# Heading\n", "---\nkey: value\n---\n"])
def test_empty_document(text: str) -> None:
    assert chunk_markdown(text, source_id="x", path="x.md") == ()


def test_unclosed_front_matter_is_a_parse_failure() -> None:
    with pytest.raises(ValueError, match="front matter"):
        chunk_markdown("---\nkey: value", source_id="x", path="x.md")


def test_heading_nesting_without_pages() -> None:
    chunks = chunk_markdown("# Root\nA\n### Deep\nB\n## Sibling\nC\n", source_id="x", path="x.md")
    assert [(c.headings, c.page) for c in chunks] == [
        (("Root",), None),
        (("Root", "Deep"), None),
        (("Root", "Sibling"), None),
    ]


def test_giant_unbroken_token_is_bounded() -> None:
    chunks = chunk_markdown("a" * 5001, source_id="x", path="x.md")
    assert [len(c.text) for c in chunks] == [2000, 2000, 1001]
