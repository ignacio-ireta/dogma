"""Extractor contract: every format maps a file to DocumentIR + warnings.

Extractors carry their own version (independent of the package version) so
golden files and sidecars only churn when the responsible extractor actually
changes behavior. Partial extraction must return a result with warnings —
never silently drop content; unrecoverable problems raise ExtractionError
subclasses (see docmd.errors.wrap_library_error).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import ClassVar, Literal

from pydantic import BaseModel, Field

from docmd.types import DocumentIR, ExtractionWarning, SourceType


class ExtractionResult(BaseModel):
    document: DocumentIR
    warnings: list[ExtractionWarning] = Field(default_factory=list)

    @property
    def status(self) -> Literal["success", "warning"]:
        return "warning" if self.warnings else "success"


class Extractor(ABC):
    name: ClassVar[str]
    version: ClassVar[str]
    file_type: ClassVar[SourceType]

    @abstractmethod
    def extract(self, path: Path, *, relative_path: str | None = None) -> ExtractionResult:
        """Extract ``path`` into the IR.

        ``relative_path`` (posix, relative to the scan root) becomes
        ``DocumentIR.source_path``; defaults to the bare filename.
        """

    @property
    def versioned_name(self) -> str:
        return f"{self.name}/{self.version}"
