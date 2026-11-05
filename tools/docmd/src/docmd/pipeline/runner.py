"""Parallel pipeline runner with failure isolation and graceful interruption.

Worker bodies never raise: every outcome becomes a FileRecord, persisted
write-through to the manifest. One bad file cannot kill a folder run (unless
--fail-fast). SIGINT protocol: the handler only sets the cancellation token;
in-flight workers hit a checkpoint between steps and finish as INTERRUPTED;
queued files return INTERRUPTED immediately; all printing happens on the main
thread after the progress display stops.
"""

from __future__ import annotations

import signal
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, datetime
from pathlib import Path
from threading import Event

from docmd.config import AppConfig
from docmd.discovery.scanner import DiscoveredFile, SkippedFile
from docmd.errors import DocmdError, RunInterrupted
from docmd.fsio import compute_sha256
from docmd.logging_config import get_logger
from docmd.markdown.metadata import read_sidecar_source_hash
from docmd.pipeline.progress import NullProgress, ProgressReporter
from docmd.pipeline.state import (
    FileRecord,
    FileStatus,
    ManifestStore,
    RunSummary,
)
from docmd.pipeline.steps import FileContext, Step, output_path_for

log = get_logger("runner")


class CancellationToken:
    def __init__(self) -> None:
        self._event = Event()

    def cancel(self) -> None:
        self._event.set()

    def cancelled(self) -> bool:
        return self._event.is_set()

    def checkpoint(self) -> None:
        if self._event.is_set():
            raise RunInterrupted("run interrupted")


def install_sigint_handler(token: CancellationToken) -> None:
    """First Ctrl+C: graceful stop. Second Ctrl+C: default (hard) behavior."""

    def handler(signum: int, frame: object) -> None:
        if token.cancelled():
            signal.signal(signal.SIGINT, signal.default_int_handler)
            raise KeyboardInterrupt
        token.cancel()
        sys.stderr.write(
            "\nInterrupted: finishing in-flight files, accepting no new ones. "
            "Press Ctrl+C again to force quit.\n"
        )

    signal.signal(signal.SIGINT, handler)


class PipelineRunner:
    def __init__(
        self,
        config: AppConfig,
        steps: list[Step],
        store: ManifestStore,
        out_root: Path,
        progress: ProgressReporter | None = None,
        token: CancellationToken | None = None,
    ) -> None:
        self.config = config
        self.steps = steps
        self.store = store
        self.out_root = out_root
        self.progress = progress or NullProgress()
        self.token = token or CancellationToken()

    def run(self, files: list[DiscoveredFile], skipped: list[SkippedFile]) -> RunSummary:
        t0 = time.monotonic()
        files, collisions = self._preflight_collisions(files)
        pending = [
            FileRecord(source=f.relative_path, size=f.size, status=FileStatus.PENDING)
            for f in files
        ]
        scan_skips = [
            FileRecord(
                source=s.relative_path,
                status=FileStatus.SKIPPED,
                skip_reason=s.reason,
            )
            for s in skipped
        ]
        self.store.update_many(pending + scan_skips + collisions)

        self.progress.start(len(files))
        try:
            with ThreadPoolExecutor(max_workers=self.config.workers) as pool:
                futures = {pool.submit(self._process_file, f): f for f in files}
                for future in as_completed(futures):
                    record = future.result()
                    self.store.update(record)
                    self.progress.advance(record)
                    if record.status is FileStatus.FAILED and self.config.fail_fast:
                        log.error("fail-fast: stopping run after failure path=%s", record.source)
                        self.token.cancel()
        finally:
            self.progress.stop()

        summary = RunSummary.from_records(self.store.records, time.monotonic() - t0)
        log.info(
            "run completed success=%d warning=%d failed=%d skipped=%d interrupted=%d "
            "duration=%.1fs",
            summary.success,
            summary.warning,
            summary.failed,
            summary.skipped,
            summary.interrupted,
            summary.duration_s,
        )
        return summary

    def _preflight_collisions(
        self, files: list[DiscoveredFile]
    ) -> tuple[list[DiscoveredFile], list[FileRecord]]:
        """Fail outputs that collide under casefolding (case-insensitive fs)."""
        from docmd.errors import ErrorInfo

        seen: dict[str, str] = {}
        ok: list[DiscoveredFile] = []
        bad: list[FileRecord] = []
        for file in files:
            key = str(output_path_for(file.relative_path, self.out_root)).casefold()
            holder = seen.get(key)
            if holder is None:
                seen[key] = file.relative_path
                ok.append(file)
                continue
            log.error("output collision path=%s collides_with=%s", file.relative_path, holder)
            bad.append(
                FileRecord(
                    source=file.relative_path,
                    size=file.size,
                    status=FileStatus.FAILED,
                    error=ErrorInfo(
                        error_code="OUTPUT_COLLISION",
                        error_type="OutputCollision",
                        message=(
                            f"output path collides with that of {holder!r} "
                            "on case-insensitive filesystems"
                        ),
                    ),
                )
            )
        return ok, bad

    def _process_file(self, file: DiscoveredFile) -> FileRecord:
        """Worker body — must never raise."""
        if self.token.cancelled():
            return FileRecord(
                source=file.relative_path,
                size=file.size,
                status=FileStatus.INTERRUPTED,
                skip_reason="not started before interruption",
            )

        started = datetime.now(UTC)
        t0 = time.monotonic()
        record = FileRecord(
            source=file.relative_path,
            size=file.size,
            status=FileStatus.RUNNING,
            started_at=started,
        )
        self.store.update(record)
        log.info("processing file path=%s type=%s", file.relative_path, file.file_type.value)

        def finish(**changes: object) -> FileRecord:
            return record.model_copy(
                update={
                    "finished_at": datetime.now(UTC),
                    "duration_ms": int((time.monotonic() - t0) * 1000),
                    **changes,
                }
            )

        try:
            sha256 = file.sha256 or compute_sha256(file.path)
            if not self.config.force:
                skip = self._unchanged_output(file, sha256)
                if skip is not None:
                    log.info("skipping unchanged file path=%s", file.relative_path)
                    return finish(
                        status=FileStatus.SKIPPED,
                        sha256=sha256,
                        skip_reason="unchanged",
                        output=skip,
                    )

            ctx = FileContext(source=file, out_root=self.out_root, config=self.config)
            for step in self.steps:
                self.token.checkpoint()
                step.run(ctx)

            warnings = ctx.result.warnings if ctx.result else []
            for warning in warnings:
                log.warning(
                    "extraction warning path=%s code=%s location=%s",
                    file.relative_path,
                    warning.code,
                    warning.location or "-",
                )
            return finish(
                status=FileStatus.WARNING if warnings else FileStatus.SUCCESS,
                sha256=sha256,
                output=str(ctx.output_path) if ctx.output_path else None,
                extractor_version=ctx.extractor.versioned_name if ctx.extractor else None,
                warning_count=len(warnings),
            )
        except RunInterrupted:
            log.info("interrupted mid-file path=%s", file.relative_path)
            return finish(status=FileStatus.INTERRUPTED)
        except DocmdError as err:
            log.error(
                "failed file path=%s code=%s reason=%s",
                file.relative_path,
                err.error_code,
                err.message,
            )
            return finish(status=FileStatus.FAILED, error=err.to_error_info())
        except Exception as exc:
            log.error("unexpected failure path=%s error=%r", file.relative_path, exc)
            from docmd.errors import ErrorInfo

            return finish(
                status=FileStatus.FAILED,
                error=ErrorInfo(
                    error_code="UNEXPECTED",
                    error_type=type(exc).__name__,
                    message=str(exc) or repr(exc),
                ),
            )

    def _unchanged_output(self, file: DiscoveredFile, sha256: str) -> str | None:
        """Output path if a current output for this exact content exists."""
        out_path = output_path_for(file.relative_path, self.out_root)
        if not out_path.is_file():
            return None
        if read_sidecar_source_hash(out_path) == sha256:
            return str(out_path)
        return None
