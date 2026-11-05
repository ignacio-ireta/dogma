"""Magic-byte and extension detection table."""

from pathlib import Path

from docmd.discovery.file_types import detect_file_type
from docmd.errors import OLE_MAGIC
from docmd.types import SourceType

ZIP_HEAD = b"PK\x03\x04" + b"\x00" * 8
PDF_HEAD = b"%PDF-1.7\n"


def write(tmp_path: Path, name: str, content: bytes) -> Path:
    p = tmp_path / name
    p.write_bytes(content)
    return p


def test_supported_extensions(tmp_path: Path) -> None:
    for name, expected in [
        ("a.docx", SourceType.DOCX),
        ("b.pptx", SourceType.PPTX),
        ("c.xlsx", SourceType.XLSX),
        ("d.PDF", SourceType.PDF),
    ]:
        content = PDF_HEAD if expected is SourceType.PDF else ZIP_HEAD
        detection = detect_file_type(write(tmp_path, name, content))
        assert detection.file_type is expected
        assert detection.warning is None


def test_macro_enabled_skipped(tmp_path: Path) -> None:
    detection = detect_file_type(write(tmp_path, "m.docm", ZIP_HEAD))
    assert detection.file_type is None
    assert detection.skip_reason == "macro-enabled file"


def test_unsupported_extension(tmp_path: Path) -> None:
    detection = detect_file_type(write(tmp_path, "n.txt", b"hello"))
    assert detection.file_type is None
    assert detection.skip_reason == "unsupported extension"


def test_ole_under_docx_is_still_docx(tmp_path: Path) -> None:
    detection = detect_file_type(write(tmp_path, "locked.docx", OLE_MAGIC + b"\x00" * 8))
    assert detection.file_type is SourceType.DOCX
    assert detection.warning is None  # encrypted Office is expected, not a mismatch


def test_pdf_extension_zip_magic_mismatch(tmp_path: Path) -> None:
    detection = detect_file_type(write(tmp_path, "fake.pdf", ZIP_HEAD))
    assert detection.file_type is SourceType.PDF
    assert detection.warning is not None and "mismatch" in detection.warning


def test_docx_extension_pdf_magic_mismatch(tmp_path: Path) -> None:
    detection = detect_file_type(write(tmp_path, "fake.docx", PDF_HEAD))
    assert detection.file_type is SourceType.DOCX
    assert detection.warning is not None and "mismatch" in detection.warning


def test_tiny_file_flagged(tmp_path: Path) -> None:
    detection = detect_file_type(write(tmp_path, "tiny.docx", b"PK"))
    assert detection.file_type is SourceType.DOCX
    assert detection.warning is not None and "too short" in detection.warning
