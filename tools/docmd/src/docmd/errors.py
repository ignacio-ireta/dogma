"""Domain exception hierarchy and mapping from third-party library errors.

Every failure docmd can report flows through a ``DocmdError`` subclass so that
error codes, exit codes, and the structured ``errors.json`` report stay
consistent across the pipeline.
"""

from __future__ import annotations

import zipfile
from pathlib import Path
from typing import ClassVar

from pydantic import BaseModel

#: Magic bytes of an OLE2/CFB container. Password-protected Office files are
#: OLE containers (not zips), so this distinguishes "encrypted" from "corrupt".
OLE_MAGIC = bytes.fromhex("D0CF11E0A1B11AE1")


class ErrorInfo(BaseModel):
    """Serializable error description used in manifests and errors.json."""

    error_code: str
    error_type: str
    message: str
    detail: str | None = None


class DocmdError(Exception):
    """Base class for all docmd failures."""

    error_code: ClassVar[str] = "DOCMD_ERROR"
    exit_code: ClassVar[int] = 3

    def __init__(
        self,
        message: str,
        *,
        path: Path | None = None,
        detail: str | None = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.path = path
        self.detail = detail

    def to_error_info(self) -> ErrorInfo:
        return ErrorInfo(
            error_code=self.error_code,
            error_type=type(self).__name__,
            message=self.message,
            detail=self.detail,
        )


class ConfigError(DocmdError):
    error_code = "CONFIG_INVALID"
    exit_code = 2


class DiscoveryError(DocmdError):
    error_code = "DISCOVERY_FAILED"


class OutputWriteError(DocmdError):
    error_code = "OUTPUT_WRITE_FAILED"


class ExtractionError(DocmdError):
    error_code = "EXTRACTION_FAILED"


class UnsupportedFileTypeError(ExtractionError):
    error_code = "UNSUPPORTED_TYPE"


class FileCorruptionError(ExtractionError):
    error_code = "CORRUPT_FILE"


class PasswordProtectedError(ExtractionError):
    error_code = "PASSWORD_PROTECTED"


class ScannedPdfError(ExtractionError):
    error_code = "SCANNED_PDF_OCR_DISABLED"


class OcrError(ExtractionError):
    error_code = "OCR_FAILED"


class OcrNoTextError(ExtractionError):
    error_code = "OCR_NO_TEXT"


class RunInterrupted(DocmdError):
    error_code = "INTERRUPTED"
    exit_code = 130


class ReservedRepoNameError(ConfigError):
    error_code = "REPO_NAME_RESERVED"


class RegistryCorruptError(DocmdError):
    error_code = "REGISTRY_CORRUPT"


class SnapshotCorruptError(DocmdError):
    error_code = "SNAPSHOT_CORRUPT"


class RepoLockedError(DocmdError):
    error_code = "REPO_LOCKED"
    exit_code = 2


def _is_ole_container(path: Path) -> bool:
    try:
        with path.open("rb") as fh:
            return fh.read(8) == OLE_MAGIC
    except OSError:
        return False


def wrap_library_error(exc: Exception, path: Path) -> ExtractionError:
    """Map third-party parser exceptions onto the docmd error hierarchy.

    Imports of optional parser exception types happen lazily so this module
    stays importable even if an extractor dependency is missing.
    """
    if isinstance(exc, ExtractionError):
        return exc

    detail = f"{type(exc).__module__}.{type(exc).__qualname__}: {exc}"

    if _is_ole_container(path):
        return PasswordProtectedError(
            "file is an OLE container; likely a password-protected Office document",
            path=path,
            detail=detail,
        )

    corrupt_types: list[type[Exception]] = [zipfile.BadZipFile, KeyError, ValueError, OSError]
    try:
        from docx.opc.exceptions import PackageNotFoundError as DocxPackageNotFoundError

        corrupt_types.append(DocxPackageNotFoundError)
    except ImportError:  # pragma: no cover
        pass
    try:
        from pptx.exc import PackageNotFoundError as PptxPackageNotFoundError

        corrupt_types.append(PptxPackageNotFoundError)
    except ImportError:  # pragma: no cover
        pass
    try:
        from openpyxl.utils.exceptions import InvalidFileException

        corrupt_types.append(InvalidFileException)
    except ImportError:  # pragma: no cover
        pass
    try:
        import pymupdf

        corrupt_types.append(pymupdf.FileDataError)
        corrupt_types.append(pymupdf.EmptyFileError)
    except ImportError:  # pragma: no cover
        pass

    if isinstance(exc, tuple(corrupt_types)):
        return FileCorruptionError(
            "file appears corrupted or is not a valid document of its type",
            path=path,
            detail=detail,
        )

    return ExtractionError("unexpected extraction failure", path=path, detail=detail)
