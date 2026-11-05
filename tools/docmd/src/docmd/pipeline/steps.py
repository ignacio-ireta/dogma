"""Per-file pipeline steps: extract -> normalize -> write.

Steps mutate a FileContext; the runner (and the `process` command) execute
them in order with cancellation checkpoints in between. Output naming is
``<name>.<ext>.md`` mirroring the source tree — collision-proof when the same
basename exists in several formats.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Protocol

from docmd import __version__
from docmd.config import AppConfig
from docmd.discovery.scanner import DiscoveredFile
from docmd.errors import ConfigError, OutputWriteError
from docmd.extractors import get_extractor
from docmd.extractors.base import ExtractionResult, Extractor
from docmd.fsio import atomic_write_text, compute_sha256
from docmd.markdown.metadata import FrontMatter, SidecarMeta, write_sidecar
from docmd.markdown.normalizer import normalize
from docmd.markdown.writer import MarkdownWriter
from docmd.types import ExtractionWarning

DEFAULT_STEP_NAMES = ("extract", "normalize", "write")


def output_path_for(relative_path: str, out_root: Path) -> Path:
    """Mirror the source tree; `reports/q3.docx` -> `<out>/reports/q3.docx.md`."""
    rel = PurePosixPath(relative_path)
    target = out_root.joinpath(*rel.parts)
    return target.with_name(target.name + ".md")


@dataclass
class FileContext:
    source: DiscoveredFile
    out_root: Path
    config: AppConfig
    extractor: Extractor | None = None
    result: ExtractionResult | None = None
    markdown: str | None = None
    output_path: Path | None = None
    steps_run: list[str] = field(default_factory=list)


class Step(Protocol):
    name: str

    def run(self, ctx: FileContext) -> None: ...


class ExtractStep:
    name = "extract"

    def run(self, ctx: FileContext) -> None:
        extractor = get_extractor(ctx.source.file_type, ctx.config)
        result = extractor.extract(ctx.source.path, relative_path=ctx.source.relative_path)
        if ctx.source.detection_warning:
            result.warnings.insert(
                0,
                ExtractionWarning(code="DETECTION_MISMATCH", message=ctx.source.detection_warning),
            )
        ctx.extractor = extractor
        ctx.result = result
        ctx.steps_run.append(self.name)


class NormalizeStep:
    name = "normalize"

    def run(self, ctx: FileContext) -> None:
        if ctx.result is None:
            raise ConfigError("normalize step requires extract to run first")
        ctx.result.document = normalize(ctx.result.document)
        ctx.steps_run.append(self.name)


class WriteStep:
    name = "write"

    def run(self, ctx: FileContext) -> None:
        if ctx.result is None or ctx.extractor is None:
            raise ConfigError("write step requires extract to run first")
        out_path = output_path_for(ctx.source.relative_path, ctx.out_root)
        now = None if ctx.config.deterministic else datetime.now(UTC)
        front = FrontMatter(
            source_file=ctx.source.relative_path,
            source_type=ctx.source.file_type,
            extraction_timestamp=now,
            extractor_version=ctx.extractor.versioned_name,
            status=ctx.result.status,
        )
        markdown = MarkdownWriter().render(ctx.result.document, front, ctx.result.warnings)
        ctx.steps_run.append(self.name)
        try:
            atomic_write_text(out_path, markdown)
            write_sidecar(
                out_path,
                SidecarMeta(
                    source_file=ctx.source.relative_path,
                    source_sha256=ctx.source.sha256 or compute_sha256(ctx.source.path),
                    source_size=ctx.source.size,
                    extractor=ctx.extractor.name,
                    extractor_version=ctx.extractor.version,
                    tool_version=__version__,
                    pipeline_steps=list(ctx.steps_run),
                    warnings=ctx.result.warnings,
                    created_at=now,
                ),
            )
        except OSError as exc:
            raise OutputWriteError(
                f"failed to write output: {exc}", path=out_path, detail=str(exc)
            ) from exc
        ctx.markdown = markdown
        ctx.output_path = out_path


_STEP_CLASSES: dict[str, type[ExtractStep] | type[NormalizeStep] | type[WriteStep]] = {
    "extract": ExtractStep,
    "normalize": NormalizeStep,
    "write": WriteStep,
}


def steps_from_names(names: list[str] | tuple[str, ...] = DEFAULT_STEP_NAMES) -> list[Step]:
    unknown = [n for n in names if n not in _STEP_CLASSES]
    if unknown:
        raise ConfigError(
            f"unknown step(s): {', '.join(unknown)}; valid: {', '.join(_STEP_CLASSES)}"
        )
    ordered = [n for n in DEFAULT_STEP_NAMES if n in names]
    return [_STEP_CLASSES[name]() for name in ordered]


def run_steps(ctx: FileContext, steps: list[Step]) -> None:
    for step in steps:
        step.run(ctx)
