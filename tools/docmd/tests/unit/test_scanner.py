"""Scanner: deterministic ordering, skip rules, recursion, single-file roots."""

from pathlib import Path

import pytest

from docmd.discovery.scanner import scan_with_skips
from docmd.errors import DiscoveryError

ZIP_HEAD = b"PK\x03\x04" + b"\x00" * 8
PDF_HEAD = b"%PDF-1.7\n"


@pytest.fixture
def tree(tmp_path: Path) -> Path:
    (tmp_path / "sub" / "deep").mkdir(parents=True)
    (tmp_path / "b.docx").write_bytes(ZIP_HEAD)
    (tmp_path / "a.pdf").write_bytes(PDF_HEAD)
    (tmp_path / "sub" / "c.xlsx").write_bytes(ZIP_HEAD)
    (tmp_path / "sub" / "deep" / "d.pptx").write_bytes(ZIP_HEAD)
    (tmp_path / "notes.txt").write_text("x")
    (tmp_path / "~$b.docx").write_bytes(b"\x00")
    (tmp_path / ".hidden.docx").write_bytes(ZIP_HEAD)
    (tmp_path / "sub" / ".secret").mkdir()
    (tmp_path / "sub" / ".secret" / "e.docx").write_bytes(ZIP_HEAD)
    return tmp_path


def test_recursive_scan_sorted(tree: Path) -> None:
    found, skipped = scan_with_skips(tree, recursive=True)
    assert [f.relative_path for f in found] == [
        "a.pdf",
        "b.docx",
        "sub/c.xlsx",
        "sub/deep/d.pptx",
    ]
    reasons = {s.relative_path: s.reason for s in skipped}
    assert reasons[".hidden.docx"] == "hidden file"
    assert reasons["sub/.secret/e.docx"] == "hidden file"
    assert reasons["~$b.docx"] == "office lock file"
    assert reasons["notes.txt"] == "unsupported extension"


def test_non_recursive_scan(tree: Path) -> None:
    found, _ = scan_with_skips(tree, recursive=False)
    assert [f.relative_path for f in found] == ["a.pdf", "b.docx"]


def test_single_file_root(tree: Path) -> None:
    found, skipped = scan_with_skips(tree / "b.docx")
    assert len(found) == 1 and not skipped
    assert found[0].relative_path == "b.docx"


def test_output_root_excluded_when_nested(tree: Path) -> None:
    out = tree / "markdown"
    out.mkdir()
    (out / "stale.docx").write_bytes(ZIP_HEAD)
    found, skipped = scan_with_skips(tree, recursive=True, out_root=out)
    all_paths = [f.relative_path for f in found] + [s.relative_path for s in skipped]
    assert not any(p.startswith("markdown/") for p in all_paths)


def test_missing_root_raises(tmp_path: Path) -> None:
    with pytest.raises(DiscoveryError):
        scan_with_skips(tmp_path / "nope")


def test_sizes_and_detection_warning(tmp_path: Path) -> None:
    (tmp_path / "fake.pdf").write_bytes(ZIP_HEAD)
    found, _ = scan_with_skips(tmp_path)
    assert found[0].size == len(ZIP_HEAD)
    assert found[0].detection_warning is not None
