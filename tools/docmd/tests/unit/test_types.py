"""IR round-trips through JSON with the block discriminator preserved."""

from docmd.types import (
    CodeBlock,
    DocumentIR,
    Heading,
    ImageRef,
    ListItem,
    Paragraph,
    Section,
    SectionKind,
    SourceType,
    Table,
)


def build_kitchen_sink() -> DocumentIR:
    return DocumentIR(
        source_path="docs/sample.docx",
        source_type=SourceType.DOCX,
        title="Sample",
        properties={"Author": "Example Author"},
        sections=[
            Section(
                kind=SectionKind.BODY,
                blocks=[
                    Heading(level=1, text="Intro"),
                    Paragraph(text="Hello"),
                    ListItem(text="one", ordered=True),
                    Table(rows=[["a", "b"], ["1", "2"]]),
                    ImageRef(name="logo.png", content_type="image/png", alt="Logo"),
                    CodeBlock(text="print('hi')", language="python"),
                ],
            )
        ],
    )


def test_json_round_trip_preserves_block_types() -> None:
    doc = build_kitchen_sink()
    restored = DocumentIR.model_validate_json(doc.model_dump_json())
    assert restored == doc
    blocks = restored.sections[0].blocks
    assert [type(b).__name__ for b in blocks] == [
        "Heading",
        "Paragraph",
        "ListItem",
        "Table",
        "ImageRef",
        "CodeBlock",
    ]


def test_heading_level_bounds() -> None:
    import pytest
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        Heading(level=0, text="x")
    with pytest.raises(ValidationError):
        Heading(level=7, text="x")
