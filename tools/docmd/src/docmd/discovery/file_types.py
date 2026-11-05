"""File type detection: extension map first, then magic-byte sniffing.

Macro-enabled Office formats (.docm/.pptm/.xlsm) are skipped outright as a
security posture — docmd never executes macros, but declining to parse them at
all keeps the contract simple. An OLE container under an OOXML extension is
NOT a mismatch: that is the on-disk shape of a password-protected Office file,
and the extractor surfaces it as PasswordProtectedError.
"""

from __future__ import annotations

from pathlib import Path
from typing import NamedTuple

from docmd.errors import OLE_MAGIC
from docmd.types import SourceType

MAGIC_ZIP = b"PK\x03\x04"
MAGIC_PDF = b"%PDF"

EXTENSION_MAP: dict[str, SourceType] = {
    ".docx": SourceType.DOCX,
    ".pptx": SourceType.PPTX,
    ".xlsx": SourceType.XLSX,
    ".pdf": SourceType.PDF,
}

MACRO_EXTENSIONS = frozenset({".docm", ".pptm", ".xlsm"})


class Detection(NamedTuple):
    file_type: SourceType | None
    skip_reason: str | None = None
    warning: str | None = None


def detect_file_type(path: Path) -> Detection:
    ext = path.suffix.lower()
    if ext in MACRO_EXTENSIONS:
        return Detection(None, skip_reason="macro-enabled file")
    file_type = EXTENSION_MAP.get(ext)
    if file_type is None:
        return Detection(None, skip_reason="unsupported extension")

    try:
        with path.open("rb") as fh:
            head = fh.read(8)
    except OSError as exc:
        return Detection(file_type, warning=f"could not read file header: {exc}")

    if len(head) < 4:
        return Detection(file_type, warning="file too short to contain a valid header")
    if file_type is SourceType.PDF:
        if not head.startswith(MAGIC_PDF):
            return Detection(file_type, warning="extension/magic mismatch: not a PDF header")
        return Detection(file_type)
    # OOXML formats are zip containers; OLE means password-protected (handled
    # downstream), anything else is a mismatch worth flagging.
    if head.startswith(MAGIC_ZIP) or head == OLE_MAGIC:
        return Detection(file_type)
    return Detection(file_type, warning="extension/magic mismatch: not a zip/OLE container")
