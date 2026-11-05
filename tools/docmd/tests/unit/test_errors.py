"""Error hierarchy: codes, exit codes, serialization, and library-error mapping."""

import zipfile
from pathlib import Path

from docmd.errors import (
    OLE_MAGIC,
    ConfigError,
    DocmdError,
    ExtractionError,
    FileCorruptionError,
    PasswordProtectedError,
    RunInterrupted,
    ScannedPdfError,
    UnsupportedFileTypeError,
    wrap_library_error,
)


def test_exit_codes() -> None:
    assert DocmdError("x").exit_code == 3
    assert ConfigError("x").exit_code == 2
    assert RunInterrupted("x").exit_code == 130
    assert FileCorruptionError("x").exit_code == 3


def test_error_codes_are_distinct() -> None:
    codes = {
        cls.error_code
        for cls in (
            ConfigError,
            ExtractionError,
            UnsupportedFileTypeError,
            FileCorruptionError,
            PasswordProtectedError,
            ScannedPdfError,
            RunInterrupted,
        )
    }
    assert len(codes) == 7


def test_to_error_info() -> None:
    err = FileCorruptionError("bad zip", path=Path("a.docx"), detail="zipfile.BadZipFile: boom")
    info = err.to_error_info()
    assert info.error_code == "CORRUPT_FILE"
    assert info.error_type == "FileCorruptionError"
    assert info.message == "bad zip"
    assert info.detail is not None and "BadZipFile" in info.detail


def test_wrap_badzip_becomes_corruption(tmp_path: Path) -> None:
    f = tmp_path / "x.docx"
    f.write_bytes(b"PK\x03\x04 truncated")
    wrapped = wrap_library_error(zipfile.BadZipFile("File is not a zip file"), f)
    assert isinstance(wrapped, FileCorruptionError)
    assert wrapped.detail is not None and "BadZipFile" in wrapped.detail


def test_wrap_ole_becomes_password_protected(tmp_path: Path) -> None:
    f = tmp_path / "locked.docx"
    f.write_bytes(OLE_MAGIC + b"\x00" * 64)
    wrapped = wrap_library_error(zipfile.BadZipFile("File is not a zip file"), f)
    assert isinstance(wrapped, PasswordProtectedError)


def test_wrap_passes_through_domain_errors(tmp_path: Path) -> None:
    original = ScannedPdfError("scanned", path=tmp_path / "s.pdf")
    assert wrap_library_error(original, tmp_path / "s.pdf") is original


def test_wrap_unknown_exception(tmp_path: Path) -> None:
    f = tmp_path / "x.docx"
    f.write_bytes(b"PK\x03\x04")
    wrapped = wrap_library_error(RuntimeError("???"), f)
    assert isinstance(wrapped, ExtractionError)
    assert not isinstance(wrapped, FileCorruptionError)
