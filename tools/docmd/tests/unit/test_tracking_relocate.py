"""Relocating outputs for moved sources: frontmatter rewrite + pair move."""

import json
from datetime import UTC, datetime
from pathlib import Path

from docmd.markdown.metadata import SidecarMeta, sidecar_path_for, write_sidecar
from docmd.tracking.relocate import relocate_output, rewrite_frontmatter_source


def md_text(source_file: str) -> str:
    return (
        "---\n"
        f"source_file: {json.dumps(source_file, ensure_ascii=False)}\n"
        'source_type: "docx"\n'
        'extractor_version: "docx/0.1.0"\n'
        'status: "success"\n'
        "---\n"
        "\n"
        "# Report\n"
        "\n"
        'source_file: "decoy-in-body.docx"\n'
    )


def make_meta(source_file: str, created_at: datetime | None = None) -> SidecarMeta:
    return SidecarMeta(
        source_file=source_file,
        source_sha256="ab" * 32,
        source_size=1234,
        extractor="docx",
        extractor_version="0.1.0",
        tool_version="0.2.0",
        pipeline_steps=["extract", "normalize", "write"],
        created_at=created_at,
    )


def write_pair(md_path: Path, source_file: str, created_at: datetime | None = None) -> Path:
    md_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.write_text(md_text(source_file), encoding="utf-8")
    return write_sidecar(md_path, make_meta(source_file, created_at=created_at))


# --- rewrite_frontmatter_source ---


def test_rewrite_frontmatter_replaces_only_source_file_line() -> None:
    original = md_text("a/b.docx")
    text, ok = rewrite_frontmatter_source(original, "c/d.docx")
    assert ok is True
    # Only the first (frontmatter) source_file line changes; every other byte,
    # including the body decoy line, is identical.
    assert text == original.replace('source_file: "a/b.docx"', 'source_file: "c/d.docx"', 1)
    assert 'source_file: "decoy-in-body.docx"' in text


def test_rewrite_frontmatter_json_encodes_quote_and_non_ascii() -> None:
    text, ok = rewrite_frontmatter_source(md_text("a/b.docx"), 'wé"ird/påth.docx')
    assert ok is True
    assert text.split("\n")[1] == 'source_file: "wé\\"ird/påth.docx"'


def test_rewrite_no_frontmatter_returns_unchanged_false() -> None:
    original = '# Plain doc\n\nsource_file: "x.docx"\n'
    assert rewrite_frontmatter_source(original, "new.docx") == (original, False)


def test_rewrite_unterminated_frontmatter_returns_unchanged_false() -> None:
    original = '---\nsource_file: "a/b.docx"\nbody with no closing delimiter\n'
    assert rewrite_frontmatter_source(original, "new.docx") == (original, False)


def test_rewrite_frontmatter_without_source_file_returns_unchanged_false() -> None:
    original = '---\nsource_type: "docx"\nstatus: "success"\n---\nsource_file: "body.docx"\n'
    assert rewrite_frontmatter_source(original, "new.docx") == (original, False)


# --- relocate_output ---


def test_relocate_output_moves_pair_and_rewrites_source_references(tmp_path: Path) -> None:
    old_md = tmp_path / "notes" / "a" / "b.docx.md"
    old_sidecar = write_pair(old_md, "a/b.docx", created_at=datetime(2026, 7, 1, 9, tzinfo=UTC))
    new_md = tmp_path / "notes" / "c" / "d.docx.md"

    assert relocate_output(old_md, new_md, "c/d.docx") is True

    assert not old_md.exists()
    assert not old_sidecar.exists()
    assert new_md.read_text(encoding="utf-8") == md_text("c/d.docx")
    data = json.loads(sidecar_path_for(new_md).read_text(encoding="utf-8"))
    assert data["source_file"] == "c/d.docx"
    assert data["source_sha256"] == "ab" * 32
    assert data["created_at"].startswith("2026-07-01")


def test_relocate_output_old_md_missing_returns_false_untouched(tmp_path: Path) -> None:
    old_md = tmp_path / "a.docx.md"  # .md never written
    old_sidecar = write_sidecar(old_md, make_meta("a.docx"))
    new_md = tmp_path / "moved" / "a.docx.md"
    sidecar_before = old_sidecar.read_text(encoding="utf-8")

    assert relocate_output(old_md, new_md, "moved/a.docx") is False

    assert old_sidecar.read_text(encoding="utf-8") == sidecar_before
    assert not new_md.exists()
    assert not sidecar_path_for(new_md).exists()


def test_relocate_output_sidecar_missing_returns_false_untouched(tmp_path: Path) -> None:
    old_md = tmp_path / "a.docx.md"
    old_md.write_text(md_text("a.docx"), encoding="utf-8")
    new_md = tmp_path / "moved" / "a.docx.md"

    assert relocate_output(old_md, new_md, "moved/a.docx") is False

    assert old_md.read_text(encoding="utf-8") == md_text("a.docx")
    assert not new_md.exists()
    assert not sidecar_path_for(new_md).exists()


def test_relocate_output_corrupt_sidecar_returns_false_untouched(tmp_path: Path) -> None:
    old_md = tmp_path / "a.docx.md"
    old_md.write_text(md_text("a.docx"), encoding="utf-8")
    old_sidecar = sidecar_path_for(old_md)
    old_sidecar.write_text("{this is not json", encoding="utf-8")
    new_md = tmp_path / "moved" / "a.docx.md"

    assert relocate_output(old_md, new_md, "moved/a.docx") is False

    assert old_md.read_text(encoding="utf-8") == md_text("a.docx")
    assert old_sidecar.read_text(encoding="utf-8") == "{this is not json"
    assert not new_md.exists()
    assert not sidecar_path_for(new_md).exists()


def test_relocate_output_sidecar_without_created_at_stays_omitted(tmp_path: Path) -> None:
    old_md = tmp_path / "a.docx.md"
    write_pair(old_md, "a.docx", created_at=None)
    new_md = tmp_path / "moved" / "a.docx.md"

    assert relocate_output(old_md, new_md, "moved/a.docx") is True

    data = json.loads(sidecar_path_for(new_md).read_text(encoding="utf-8"))
    assert "created_at" not in data
    assert data["source_file"] == "moved/a.docx"
