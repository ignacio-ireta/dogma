"""Frontmatter and sidecar metadata for every Markdown output.

Frontmatter YAML is hand-rolled with a fixed key order: all values are emitted
as JSON scalars (valid YAML), which avoids a pyyaml dependency and keeps output
byte-stable. ``extraction_timestamp``/``created_at`` set to ``None`` means
deterministic mode — the key is omitted entirely, not emitted as null.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

from docmd.fsio import atomic_write_text
from docmd.types import ExtractionWarning, SourceType

Status = Literal["success", "warning"]


class FrontMatter(BaseModel):
    source_file: str
    source_type: SourceType
    extraction_timestamp: datetime | None = None
    extractor_version: str
    status: Status


def _scalar(value: str) -> str:
    return json.dumps(value, ensure_ascii=False)


def render_frontmatter(fm: FrontMatter) -> str:
    lines = [
        "---",
        f"source_file: {_scalar(fm.source_file)}",
        f"source_type: {_scalar(fm.source_type.value)}",
    ]
    if fm.extraction_timestamp is not None:
        lines.append(f"extraction_timestamp: {_scalar(fm.extraction_timestamp.isoformat())}")
    lines.append(f"extractor_version: {_scalar(fm.extractor_version)}")
    lines.append(f"status: {_scalar(fm.status)}")
    lines.append("---")
    return "\n".join(lines) + "\n"


class SidecarMeta(BaseModel):
    """Per-output traceability record written next to each Markdown file."""

    schema_version: int = 1
    source_file: str
    source_sha256: str
    source_size: int
    extractor: str
    extractor_version: str
    tool_version: str
    pipeline_steps: list[str]
    warnings: list[ExtractionWarning] = Field(default_factory=list)
    created_at: datetime | None = None


def sidecar_path_for(md_path: Path) -> Path:
    return md_path.with_name(md_path.name + ".meta.json")


def read_sidecar_source_hash(md_path: Path) -> str | None:
    """The ``source_sha256`` recorded next to an output, or None if unavailable.

    The single reader for sidecar-based change detection — the runner's
    skip-unchanged and sync's move/orphan adoption must agree on what counts
    as a readable sidecar.
    """
    try:
        raw = json.loads(sidecar_path_for(md_path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    value = raw.get("source_sha256")
    return value if isinstance(value, str) else None


def write_sidecar(md_path: Path, meta: SidecarMeta) -> Path:
    path = sidecar_path_for(md_path)
    data = meta.model_dump(mode="json")
    if meta.created_at is None:
        del data["created_at"]
    atomic_write_text(path, json.dumps(data, indent=2, ensure_ascii=False) + "\n")
    return path
