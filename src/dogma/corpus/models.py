"""Detached corpus records shared by the CLI and future tutor."""

from __future__ import annotations

from dataclasses import dataclass

from dogma.core.errors import ErrorCode


@dataclass(frozen=True)
class Chunk:
    """A bounded text passage with stable source and structural provenance."""

    source_id: str
    path: str
    page: int | None
    headings: tuple[str, ...]
    ordinal: int
    text: str
    topic_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class SearchHit:
    """A detached matching passage; lower BM25 scores rank first."""

    chunk: Chunk
    citation: str
    score: float


@dataclass(frozen=True)
class CorpusIssue:
    """A source-level infrastructure failure without document contents."""

    path: str
    code: ErrorCode
    message: str


@dataclass(frozen=True)
class SyncReport:
    """Incremental source counts; failed sources retain their last good chunks."""

    added: int = 0
    updated: int = 0
    removed: int = 0
    unchanged: int = 0
    issues: tuple[CorpusIssue, ...] = ()


@dataclass(frozen=True)
class ConversionReport:
    """Number of source directories converted, plus per-directory failures."""

    converted: int = 0
    issues: tuple[CorpusIssue, ...] = ()


@dataclass(frozen=True)
class CorpusSyncReport:
    """Combined conversion and indexing results."""

    conversion: ConversionReport
    index: SyncReport

    @property
    def issues(self) -> tuple[CorpusIssue, ...]:
        return self.conversion.issues + self.index.issues
