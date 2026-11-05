"""Per-block Markdown rendering and escaping rules."""

from docmd.markdown.writer import MarkdownWriter, escape_leading_marker
from docmd.types import (
    CodeBlock,
    Heading,
    ImageRef,
    ListItem,
    Paragraph,
    Section,
    SectionKind,
    Table,
)

writer = MarkdownWriter()


def test_heading_shifted_under_content() -> None:
    assert writer.render_block(Heading(level=1, text="Intro")) == "### Intro"
    assert writer.render_block(Heading(level=3, text="Deep")) == "##### Deep"


def test_heading_level_capped_at_six() -> None:
    assert writer.render_block(Heading(level=6, text="Deepest")) == "###### Deepest"


def test_paragraph_leading_marker_escaped() -> None:
    assert escape_leading_marker("# not a heading") == "\\# not a heading"
    assert escape_leading_marker("- not a list") == "\\- not a list"
    assert escape_leading_marker("1. not ordered") == "1\\. not ordered"
    assert escape_leading_marker("2) also not") == "2\\) also not"
    assert escape_leading_marker("normal text") == "normal text"
    assert escape_leading_marker("not#escaped mid-text") == "not#escaped mid-text"


def test_table_basic() -> None:
    table = Table(rows=[["Name", "Qty"], ["Apples", "3"]])
    assert writer.render_table(table) == ("| Name | Qty |\n| --- | --- |\n| Apples | 3 |")


def test_table_escapes_pipes_and_newlines() -> None:
    table = Table(rows=[["a|b", "c\nd"]], has_header=False)
    out = writer.render_table(table)
    assert "a\\|b" in out
    assert "c<br>d" in out


def test_table_without_header_synthesizes_columns() -> None:
    table = Table(rows=[["x", "y"]], has_header=False)
    assert writer.render_table(table).startswith("| Column 1 | Column 2 |")


def test_table_ragged_rows_padded() -> None:
    table = Table(rows=[["a", "b", "c"], ["only"]])
    assert writer.render_table(table).endswith("| only |  |  |")


def test_table_caption() -> None:
    table = Table(rows=[["h"], ["v"]], caption="Quarterly totals")
    assert writer.render_table(table).startswith("*Quarterly totals*\n\n")


def test_list_items_glued_and_nested() -> None:
    section = Section(
        kind=SectionKind.BODY,
        blocks=[
            ListItem(text="first", ordered=True),
            ListItem(text="nested", ordered=True, level=1),
            Paragraph(text="after"),
        ],
    )
    rendered = writer.render_section(section)
    assert "1. first\n  1. nested\n\nafter" in rendered


def test_image_marker() -> None:
    img = ImageRef(name="logo.png", content_type="image/png", alt="Logo")
    assert writer.render_block(img) == "`[image: logo.png (image/png); alt: Logo]`"
    assert writer.render_block(ImageRef()) == "`[image: unnamed]`"


def test_code_fence_lengthens_when_needed() -> None:
    code = CodeBlock(text="``` inside", language="md")
    rendered = writer.render_block(code)
    assert rendered.startswith("````md\n")
    assert rendered.endswith("\n````")


def test_section_labels() -> None:
    assert writer.render_section(Section(kind=SectionKind.SLIDE, index=2, title="Plan")).startswith(
        "### Slide 2: Plan"
    )
    assert writer.render_section(Section(kind=SectionKind.SHEET, title="Data")).startswith(
        "### Sheet: Data"
    )
    assert writer.render_section(Section(kind=SectionKind.PAGE, index=4)).startswith("### Page 4")
    assert writer.render_section(Section(kind=SectionKind.NOTES, index=2)).startswith(
        "### Speaker notes (slide 2)"
    )
    assert writer.render_section(Section(kind=SectionKind.BODY)) == ""
