"""Invoke the separately locked vendored converter without importing or executing data."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from pydantic import BaseModel, Field, ValidationError

from dogma.core.config import Settings
from dogma.core.errors import DogmaError, ErrorCode
from dogma.corpus.index import safe_path
from dogma.corpus.models import ConversionReport, CorpusIssue, CorpusSyncReport
from dogma.corpus.sync import sync_index

CONVERSION_TIMEOUT_S = 3600
_OCR_POLICIES = frozenset({"off", "auto", "force"})
_OCR_GUIDANCE = (
    "Scanned PDFs need OCR: retry --ocr auto (macOS; installs the vendored "
    "project's locked ocr extra). Use --ocr force for garbled text."
)


class _Error(BaseModel):
    message: str = "Conversion failed"


class _Change(BaseModel):
    source: str
    status: str
    error: _Error | None = None


class _Changes(BaseModel):
    repo: str
    changes: tuple[_Change, ...] = ()
    notes: tuple[str, ...] = ()
    counts: dict[str, int] = Field(default_factory=dict)


def _no_links(root: Path) -> None:
    """Docmd follows file symlinks, so refuse unsafe trees before handing them over."""

    def fail(exc: OSError) -> None:
        raise exc

    if not root.exists():
        return
    for directory, dirs, files in root.walk(on_error=fail):
        if any((directory / name).is_symlink() for name in [*dirs, *files]):
            raise DogmaError(
                ErrorCode.CORPUS_PATH_INVALID,
                "Remove symlinks from conversion input and output trees.",
            )


def _run_reports(output: Path) -> dict[Path, int]:
    return {p: p.stat().st_mtime_ns for p in (output / "runs").glob("*/changes.json")}


def _conversion_issues(
    source: Path, root: Path, output: Path, before: dict[Path, int]
) -> tuple[CorpusIssue, ...]:
    issues: list[CorpusIssue] = []
    for path, modified in sorted(_run_reports(output).items()):
        if before.get(path) == modified:
            continue
        try:
            report = _Changes.model_validate_json(path.read_text(encoding="utf-8"))
        except (OSError, ValueError, ValidationError):
            issues.append(
                CorpusIssue(
                    path.relative_to(root).as_posix(),
                    ErrorCode.CORPUS_CONVERSION_FAILED,
                    "Cannot read docmd conversion report.",
                )
            )
            continue
        if report.repo != source.name:
            continue
        for change in report.changes:
            if change.status in {"failed", "interrupted", "planned"}:
                issues.append(
                    CorpusIssue(
                        f"{source.relative_to(root).as_posix()}/{change.source}",
                        ErrorCode.CORPUS_CONVERSION_FAILED,
                        (change.error.message if change.error else "Conversion did not complete.")
                        + " "
                        + _OCR_GUIDANCE,
                    )
                )
        if report.notes:
            issues.append(
                CorpusIssue(
                    source.relative_to(root).as_posix(),
                    ErrorCode.CORPUS_CONVERSION_FAILED,
                    " ".join(report.notes),
                )
            )
    return tuple(issues)


def _convert_one(
    settings: Settings, source: Path, project: Path, output: Path, ocr: str
) -> tuple[CorpusIssue, ...]:
    root = settings.repo_root.resolve()
    rel = source.relative_to(root).as_posix()
    argv = ["uv", "run", "--frozen", "--project", str(project)]
    if ocr != "off" and sys.platform == "darwin":
        argv.extend(["--extra", "ocr"])
    argv.extend(
        [
            "docmd",
            "sync",
            str(source),
            "--out",
            str(output),
            "--deterministic",
            "--ocr",
            ocr,
            "--workers",
            "1",
            "--no-progress",
        ]
    )
    try:
        _no_links(source)
        before = _run_reports(output)
        completed = subprocess.run(  # noqa: S603 - fixed executable and literal flags; no shell.
            argv,
            cwd=root,
            capture_output=True,
            text=True,
            timeout=CONVERSION_TIMEOUT_S,
            check=False,
        )
        return _completed_issues(completed, source, root, output, before)
    except FileNotFoundError:
        return (
            CorpusIssue(
                rel,
                ErrorCode.CORPUS_CONVERTER_MISSING,
                "Install uv and restore tools/docmd before conversion.",
            ),
        )
    except subprocess.TimeoutExpired:
        return (
            CorpusIssue(
                rel,
                ErrorCode.CORPUS_CONVERSION_TIMEOUT,
                "Conversion exceeded the one-hour source-directory budget; retry sync.",
            ),
        )
    except DogmaError as exc:
        return (CorpusIssue(rel, exc.code, exc.message),)
    except OSError:
        return (
            CorpusIssue(
                rel,
                ErrorCode.CORPUS_CONVERSION_FAILED,
                "Cannot launch or read the converter; check source permissions.",
            ),
        )


def _completed_issues(
    completed: subprocess.CompletedProcess[str],
    source: Path,
    root: Path,
    output: Path,
    before: dict[Path, int],
) -> tuple[CorpusIssue, ...]:
    rel = source.relative_to(root).as_posix()
    issues = _conversion_issues(source, root, output, before)
    if "deletions were NOT processed" in completed.stdout + completed.stderr:
        return (
            *issues,
            CorpusIssue(
                rel,
                ErrorCode.CORPUS_CONVERSION_FAILED,
                "Docmd retained old conversions because the source tree is empty; "
                "deletions were not processed. Restore the source tree or review stale outputs.",
            ),
        )
    if completed.returncode and not issues:
        return (
            CorpusIssue(
                rel,
                ErrorCode.CORPUS_CONVERSION_FAILED,
                f"Docmd exited {completed.returncode}. " + _OCR_GUIDANCE,
            ),
        )
    return issues


def convert_sources(settings: Settings, *, ocr: str = "off") -> ConversionReport:
    """Convert raw-source directories through docmd with a one-hour budget each.

    OCR is explicitly off by default. ``auto``/``force`` opt into the vendor's
    locked OCR extra on macOS. No shell, imports from docmd, or learner DB writes.
    Docmd owns incremental hashes, live Markdown, and its own archive/history.
    """
    if ocr not in _OCR_POLICIES:
        raise DogmaError(ErrorCode.CORPUS_QUERY_INVALID, "OCR must be off, auto, or force.")
    raw = safe_path(settings, "corpus/raw")
    output = safe_path(settings, "corpus/md")
    project = safe_path(settings, "tools/docmd")
    if not raw.exists():
        return ConversionReport()
    if not (project / "pyproject.toml").is_file():
        return ConversionReport(
            issues=(
                CorpusIssue(
                    "tools/docmd",
                    ErrorCode.CORPUS_CONVERTER_MISSING,
                    "Restore the vendored tools/docmd project before conversion.",
                ),
            )
        )
    _no_links(output)
    issues: list[CorpusIssue] = []
    converted = 0
    for source in sorted(raw.iterdir()):
        if source.name.startswith("."):
            continue
        if source.is_symlink():
            issues.append(
                CorpusIssue(
                    source.relative_to(settings.repo_root).as_posix(),
                    ErrorCode.CORPUS_PATH_INVALID,
                    "Raw source directories cannot be symlinks.",
                )
            )
        elif source.is_dir():
            failures = _convert_one(settings, source, project, output, ocr)
            issues.extend(failures)
            if not failures:
                converted += 1
    return ConversionReport(converted, tuple(issues))


def sync_corpus(settings: Settings, *, convert: bool = True, ocr: str = "off") -> CorpusSyncReport:
    """Convert if requested, then index all good live Markdown even after partial failure."""
    conversion = convert_sources(settings, ocr=ocr) if convert else ConversionReport()
    return CorpusSyncReport(conversion, sync_index(settings))
