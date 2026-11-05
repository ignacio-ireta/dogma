"""Sync orchestrator: tracked, diff-driven conversion of a document repo.

Ordering is the crash-safety contract: relocations and archives happen before
conversion; previous versions of updated files are COPIED (never moved) so a
failed conversion cannot destroy the only good output; the snapshot is written
last and only reflects completed work. There is no ``--resume`` — re-running
sync IS resume: the diff is recomputed against the (possibly partially
advanced) snapshot, and every disk mutation is atomic or idempotent under
re-classification.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath

from docmd import __version__
from docmd.config import AppConfig
from docmd.discovery.scanner import DiscoveredFile, scan_with_skips
from docmd.errors import ErrorInfo, ReservedRepoNameError, SnapshotCorruptError
from docmd.fsio import compute_sha256
from docmd.logging_config import get_logger, setup_logging
from docmd.markdown.metadata import read_sidecar_source_hash, sidecar_path_for
from docmd.pipeline.progress import NullProgress, RichProgress
from docmd.pipeline.runner import CancellationToken, PipelineRunner, install_sigint_handler
from docmd.pipeline.state import (
    MANIFEST_NAME,
    FileRecord,
    FileStatus,
    Manifest,
    ManifestStore,
    RunSummary,
    allocate_run_dir,
    new_run_id,
    write_errors_json,
)
from docmd.pipeline.steps import output_path_for, steps_from_names
from docmd.tracking.archive import ArchiveMode, archive_output, archive_path_for, prune_archive
from docmd.tracking.diff import HashedFile, classify_changes
from docmd.tracking.lock import REGISTRY_LOCK_NAME, StateLock
from docmd.tracking.registry import RepoEntry, load_registry, save_registry, validate_repo_name
from docmd.tracking.relocate import relocate_output
from docmd.tracking.report import (
    ChangeKind,
    ChangeRecord,
    ChangeStatus,
    SyncCounts,
    SyncReport,
    write_changes_json,
)
from docmd.tracking.snapshot import (
    SnapshotEntry,
    apply_run_results,
    load_snapshot,
    output_relpath_for,
    save_snapshot,
)

log = get_logger("sync")


@dataclass
class SyncOutcome:
    exit_code: int  # 0 ok, 1 failed actions, 4 nothing to do, 130 interrupted
    repo: str
    known: bool
    run_dir: Path
    report: SyncReport | None = None
    summary: RunSummary | None = None
    records: list[FileRecord] = field(default_factory=list)


def _hash_all(
    files: list[DiscoveredFile], *, workers: int
) -> tuple[dict[str, HashedFile], list[FileRecord]]:
    """Hash every discovered file up front; unreadable files fail, not abort.

    These hashes are the single source of truth for the whole run — diffing,
    skip-unchanged, sidecars, and the snapshot all reuse them. A source edited
    mid-run is therefore recorded under its pre-pass hash; the next sync
    re-hashes everything and reconciles it.
    """
    from concurrent.futures import ThreadPoolExecutor

    hashed: dict[str, HashedFile] = {}
    failures: list[FileRecord] = []

    def one(f: DiscoveredFile) -> tuple[DiscoveredFile, str | None, str | None]:
        try:
            return f, compute_sha256(f.path), None
        except OSError as exc:
            return f, None, str(exc)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        for f, sha256, error in pool.map(one, files):
            if sha256 is None:
                log.error("hash failed path=%s error=%s", f.relative_path, error)
                failures.append(
                    FileRecord(
                        source=f.relative_path,
                        size=f.size,
                        status=FileStatus.FAILED,
                        error=ErrorInfo(
                            error_code="HASH_FAILED",
                            error_type="OSError",
                            message=f"failed to hash source: {error}",
                        ),
                    )
                )
            else:
                f.sha256 = sha256  # reused by the runner and WriteStep; no re-hashing
                hashed[f.relative_path] = HashedFile(
                    relative_path=f.relative_path, sha256=sha256, size=f.size
                )
    return hashed, failures


def _collision_error(holder: str) -> ErrorInfo:
    return ErrorInfo(
        error_code="OUTPUT_COLLISION",
        error_type="OutputCollision",
        message=(f"output path collides with that of {holder!r} on case-insensitive filesystems"),
    )


def _prune_empty_dirs(dirs: set[Path], repo_root: Path) -> None:
    """Remove directories emptied by relocations/deletions, up to the repo root."""
    for directory in dirs:
        current = directory
        while current != repo_root and current.is_relative_to(repo_root):
            try:
                current.rmdir()
            except OSError:
                break  # not empty, or already gone with its parent
            current = current.parent


def _conversion_status(record: FileRecord | None) -> tuple[ChangeStatus, ErrorInfo | None]:
    if record is None:
        return ChangeStatus.INTERRUPTED, None
    if record.status in (FileStatus.SUCCESS, FileStatus.WARNING):
        return ChangeStatus.COMPLETED, None
    if record.status is FileStatus.SKIPPED and record.output is not None:
        return ChangeStatus.COMPLETED, None  # skip-unchanged: output already current
    if record.status is FileStatus.FAILED:
        return ChangeStatus.FAILED, record.error
    return ChangeStatus.INTERRUPTED, None


def run_sync(
    *,
    source: Path,
    out_root: Path,
    config: AppConfig,
    command: list[str],
    dry_run: bool = False,
    show_progress: bool = True,
) -> SyncOutcome:
    source = source.resolve()
    repo = source.name
    validate_repo_name(repo)
    if dry_run:
        # A dry run is an advisory, read-only view. Taking StateLock would create
        # and unlink a lockfile (and mutate the locks directory's mtime), which
        # violates the command's "change nothing" contract. A concurrent real
        # sync can therefore make this plan stale; the real run always re-plans
        # under its lock before applying anything.
        return _run_sync_locked(
            source=source,
            repo=repo,
            out_root=out_root,
            config=config,
            command=command,
            dry_run=True,
            show_progress=show_progress,
        )
    # One sync per repo at a time: concurrent runs would race the snapshot,
    # the archive, and each other's relocations. Different repos may sync
    # concurrently; only the registry upsert takes the shared registry lock.
    with StateLock(out_root, repo, run_id="startup"):
        return _run_sync_locked(
            source=source,
            repo=repo,
            out_root=out_root,
            config=config,
            command=command,
            dry_run=dry_run,
            show_progress=show_progress,
        )


def _run_sync_locked(
    *,
    source: Path,
    repo: str,
    out_root: Path,
    config: AppConfig,
    command: list[str],
    dry_run: bool,
    show_progress: bool,
) -> SyncOutcome:
    repo_root = out_root / repo

    if dry_run:
        run_id = new_run_id()
        run_dir = out_root / "runs" / run_id
        setup_logging(config.log_level)
    else:
        run_id, run_dir = allocate_run_dir(out_root / "runs")
        setup_logging(config.log_level, run_dir / "run.log")

    registry = load_registry(out_root)
    # On case-insensitive filesystems (default macOS/Windows) two repos whose
    # names differ only in case share one output directory and one snapshot
    # path — silently corrupting each other. Refuse the ambiguity everywhere.
    clash = next((n for n in registry.repos if n != repo and n.casefold() == repo.casefold()), None)
    if clash is not None:
        raise ReservedRepoNameError(
            f"repo name {repo!r} collides with already-tracked repo {clash!r} "
            "on case-insensitive filesystems; homogenize the root folder name"
        )
    repo_entry = registry.repos.get(repo)
    known = repo_entry is not None
    log.info(
        "sync start repo=%s known=%s run_id=%s source=%s dry_run=%s",
        repo,
        known,
        run_id,
        source,
        dry_run,
    )

    notes: list[str] = []
    if repo_entry is not None and repo_entry.source_root != str(source):
        log.warning("repo source path changed old=%s new=%s", repo_entry.source_root, source)
        notes.append(f"source path changed from {repo_entry.source_root} to {source}")

    try:
        snapshot = load_snapshot(out_root, repo)
    except SnapshotCorruptError as err:
        log.warning("snapshot corrupt; treating repo as new detail=%s", err.detail or "-")
        notes.append(f"snapshot was corrupt ({err.message}); treated repo as new")
        snapshot = None
    if known and snapshot is None:
        log.warning("no usable snapshot for known repo; rebuilding from scratch")

    files, skipped = scan_with_skips(source, recursive=config.recursive, out_root=out_root)
    log.info(
        "discovered files total=%d supported=%d unsupported=%d",
        len(files) + len(skipped),
        len(files),
        len(skipped),
    )
    if not files:
        # Refuse to infer whole-repo deletion from an empty tree: an unmounted
        # or wrong directory must not archive away an entire repo.
        log.warning("no supported files found; deletions were NOT processed")
        return SyncOutcome(exit_code=4, repo=repo, known=known, run_dir=run_dir)

    hashed, hash_failures = _hash_all(files, workers=config.workers)
    if hash_failures:
        notes.append(f"{len(hash_failures)} file(s) could not be hashed and were not classified")
    previous_files: dict[str, SnapshotEntry] = dict(snapshot.files) if snapshot else {}
    # A tracked file that exists but could not be hashed must classify as
    # nothing at all: absent from the current side it would read as deleted and
    # its output would be archived away over a transient read error. Dropping
    # it from both sides keeps its snapshot entry and live output untouched.
    failed_paths = {record.source for record in hash_failures}
    previous_for_diff = {p: e for p, e in previous_files.items() if p not in failed_paths}
    plan = classify_changes(previous_for_diff, hashed)
    log.info(
        "classified added=%d updated=%d unchanged=%d moved=%d deleted=%d",
        len(plan.added),
        len(plan.updated),
        len(plan.unchanged),
        len(plan.moved),
        len(plan.deleted),
    )

    # Guard 1 — scan depth changed: absence is meaningless when the previous
    # snapshot saw a different slice of the tree. A path missing from a
    # shallower scan is probably still there, and relocating "moves" against an
    # invisible old path would strand live outputs. Downgrade moves to plain
    # adds and skip deletions; a second run at the same depth reconciles fully.
    snapshot_recursive = snapshot.recursive if snapshot is not None else None
    if (
        snapshot_recursive is not None
        and snapshot_recursive != config.recursive
        and (plan.moved or plan.deleted)
    ):
        log.warning(
            "scan depth changed (recursive %s -> %s); skipping %d deletion(s) and "
            "downgrading %d move(s) to adds this run",
            snapshot_recursive,
            config.recursive,
            len(plan.deleted),
            len(plan.moved),
        )
        notes.append(
            f"scan depth changed (recursive {snapshot_recursive} -> {config.recursive}): "
            f"{len(plan.deleted)} deletion(s) skipped and {len(plan.moved)} move(s) "
            "re-converted as adds; re-run at the same depth to reconcile deletions"
        )
        plan = replace(
            plan,
            added=sorted(plan.added + [new for _, new in plan.moved]),
            moved=[],
            deleted=[],
        )

    # Guard 2 — mass deletion: archiving away a large slice of a known repo is
    # more often an unmounted share or a wrong --recursive than real intent.
    delete_threshold = max(10, len(previous_for_diff) // 4)
    if len(plan.deleted) > delete_threshold and not config.confirm_deletes:
        log.warning(
            "%d deletion(s) exceed the safety threshold (%d); not applied "
            "(pass --confirm-deletes to archive them)",
            len(plan.deleted),
            delete_threshold,
        )
        notes.append(
            f"{len(plan.deleted)} deletion(s) exceed the safety threshold "
            f"({delete_threshold}) and were NOT applied; re-run with --confirm-deletes "
            "to archive them"
        )
        plan = replace(plan, deleted=[])

    def live_output(entry: SnapshotEntry) -> Path:
        return repo_root.joinpath(*PurePosixPath(entry.output).parts)

    # Self-heal: an "unchanged" file whose output or sidecar vanished (manual
    # deletion, partial restore) is re-converted rather than reported healthy
    # forever with its provenance gone.
    unchanged: list[str] = []
    reconvert: list[str] = []
    for path in plan.unchanged:
        target = live_output(previous_files[path])
        healthy = target.is_file() and sidecar_path_for(target).is_file()
        (unchanged if healthy else reconvert).append(path)
    if reconvert:
        log.warning("re-converting %d unchanged file(s) with missing outputs", len(reconvert))
        notes.append(
            f"{len(reconvert)} unchanged output(s) were missing on disk and were re-converted"
        )

    # Preflight: every output path that will exist after this run, casefolded.
    # Kept outputs win; colliding moves and adds fail up front.
    taken: dict[str, str] = {}
    for path in unchanged + plan.updated + reconvert:
        taken[output_relpath_for(path).casefold()] = path
    # Every other tracked output that stays live this run is claimed too:
    # hash-failed files and guard-retained deletions keep theirs, and an add
    # must not be allowed to overwrite them on a case-insensitive filesystem.
    going_away = set(plan.deleted) | {old for old, _ in plan.moved}
    for path, entry in previous_files.items():
        if path not in going_away:
            taken.setdefault(entry.output.casefold(), path)
    collision_failures: list[FileRecord] = []
    changes: list[ChangeRecord] = []
    moves: list[tuple[str, str]] = []
    for old, new in plan.moved:
        key = output_relpath_for(new).casefold()
        holder = taken.get(key)
        record = ChangeRecord(
            action=ChangeKind.MOVED,
            source=new,
            previous_source=old,
            sha256=hashed[new].sha256,
            previous_sha256=previous_files[old].sha256,
            output=output_relpath_for(new),
            status=ChangeStatus.PLANNED,
        )
        changes.append(record)
        if holder is None:
            taken[key] = new
            moves.append((old, new))
        else:
            record.status = ChangeStatus.FAILED
            record.error = _collision_error(holder)
            log.error("output collision path=%s collides_with=%s", new, holder)
            collision_failures.append(
                FileRecord(
                    source=new,
                    sha256=hashed[new].sha256,
                    size=hashed[new].size,
                    status=FileStatus.FAILED,
                    error=record.error,
                )
            )
    for old in plan.deleted:
        changes.append(
            ChangeRecord(
                action=ChangeKind.DELETED,
                source=old,
                previous_sha256=previous_files[old].sha256,
                status=ChangeStatus.PLANNED,
            )
        )
    for path in plan.updated + reconvert:
        changes.append(
            ChangeRecord(
                action=ChangeKind.UPDATED,
                source=path,
                sha256=hashed[path].sha256,
                previous_sha256=previous_files[path].sha256,
                output=output_relpath_for(path),
                status=ChangeStatus.PLANNED,
            )
        )
    adds: list[str] = []
    for path in plan.added:
        key = output_relpath_for(path).casefold()
        holder = taken.get(key)
        record = ChangeRecord(
            action=ChangeKind.ADDED,
            source=path,
            sha256=hashed[path].sha256,
            output=output_relpath_for(path),
            status=ChangeStatus.PLANNED,
        )
        changes.append(record)
        if holder is None:
            taken[key] = path
            adds.append(path)
        else:
            record.status = ChangeStatus.FAILED
            record.error = _collision_error(holder)
            log.error("output collision path=%s collides_with=%s", path, holder)
            collision_failures.append(
                FileRecord(
                    source=path,
                    sha256=hashed[path].sha256,
                    size=hashed[path].size,
                    status=FileStatus.FAILED,
                    error=record.error,
                )
            )

    change_by_source = {c.source: c for c in changes}

    def counts_now() -> SyncCounts:
        return SyncCounts(
            added=len(plan.added),
            updated=len(plan.updated) + len(reconvert),
            unchanged=len(unchanged),
            moved=len(plan.moved),
            deleted=len(plan.deleted),
            failed=sum(1 for c in changes if c.status is ChangeStatus.FAILED),
            interrupted=sum(1 for c in changes if c.status is ChangeStatus.INTERRUPTED),
        )

    def build_report() -> SyncReport:
        return SyncReport(
            run_id=run_id,
            repo=repo,
            source_root=str(source),
            output_root=str(repo_root),
            generated_at=datetime.now(UTC),
            dry_run=dry_run,
            notes=notes,
            counts=counts_now(),
            changes=changes,
        )

    def rel_to_out(path: Path) -> str:
        return path.relative_to(out_root).as_posix()

    def archive_or_fail(
        record: ChangeRecord, live_md: Path, dest: Path, mode: ArchiveMode, what: str
    ) -> list[Path] | None:
        """Archive one output pair; on OSError mark the record failed, return None."""
        try:
            archived = archive_output(live_md, dest, mode=mode)
        except OSError as exc:
            record.status = ChangeStatus.FAILED
            record.error = ErrorInfo(
                error_code="ARCHIVE_FAILED",
                error_type=type(exc).__name__,
                message=f"failed to archive {what}: {exc}",
            )
            log.error("archive failed source=%s error=%s", record.source, exc)
            return None
        if archived:
            record.archived_to = rel_to_out(archived[0])
        return archived

    if dry_run:
        report = build_report()
        log.info("dry run complete; plan was not persisted and no changes were made")
        # A plan can already contain failures (output collisions); surface them.
        exit_code = 1 if report.counts.failed else 0
        return SyncOutcome(
            exit_code=exit_code, repo=repo, known=known, run_dir=run_dir, report=report
        )

    store = ManifestStore(
        Manifest(
            run_id=run_id,
            tool_version=__version__,
            command=command,
            input_root=str(source),
            output_root=str(repo_root),
            started_at=datetime.now(UTC),
        ),
        run_dir / MANIFEST_NAME,
    )
    if hash_failures or collision_failures:
        store.update_many(hash_failures + collision_failures)
    token = CancellationToken()
    install_sigint_handler(token)

    # (a) Relocations. Copy-then-delete per pair; anomalies archive what exists
    # and fall back to re-extraction so a move is never silently lossy.
    completed_moves: list[tuple[str, str]] = []
    downgrade_deletes: list[str] = []
    convert_paths: list[str] = []
    pending: dict[str, ChangeRecord] = {}  # source relpath -> record to fill after conversion
    emptied_dirs: set[Path] = set()
    for old, new in moves:
        record = change_by_source[new]
        if token.cancelled():
            record.status = ChangeStatus.INTERRUPTED
            continue
        old_md = output_path_for(old, repo_root)
        new_md = output_path_for(new, repo_root)
        try:
            if not old_md.is_file() and read_sidecar_source_hash(new_md) == hashed[new].sha256:
                # A previous crashed sync already relocated this pair; adopt it.
                completed_moves.append((old, new))
                record.status = ChangeStatus.COMPLETED
                log.info("adopted already-relocated output old=%s new=%s", old, new)
                continue
            # On a case-insensitive filesystem a case-only rename makes new_md
            # resolve to the live old output; that is the pair to relocate,
            # not a stray to archive away.
            same_pair = old_md.is_file() and new_md.exists() and old_md.samefile(new_md)
            if not same_pair and (new_md.is_file() or sidecar_path_for(new_md).is_file()):
                archive_output(
                    new_md,
                    archive_path_for(out_root, repo, run_id, output_relpath_for(new)),
                    mode="move",
                )
                log.warning("archived stray output at move target path=%s", new)
            if relocate_output(old_md, new_md, new):
                completed_moves.append((old, new))
                record.status = ChangeStatus.COMPLETED
                emptied_dirs.add(old_md.parent)
                log.info("relocated output old=%s new=%s", old, new)
            else:
                archived = archive_output(
                    old_md,
                    archive_path_for(out_root, repo, run_id, output_relpath_for(old)),
                    mode="move",
                )
                if archived:
                    record.archived_to = rel_to_out(archived[0])
                downgrade_deletes.append(old)
                convert_paths.append(new)
                pending[new] = record
                emptied_dirs.add(old_md.parent)
                log.warning("could not relocate cleanly; re-extracting old=%s new=%s", old, new)
        except OSError as exc:
            record.status = ChangeStatus.FAILED
            record.error = ErrorInfo(
                error_code="RELOCATE_FAILED",
                error_type=type(exc).__name__,
                message=f"failed to relocate output: {exc}",
            )
            log.error("relocate failed old=%s new=%s error=%s", old, new, exc)

    # (b) Archive deleted outputs — move; the source is gone, nothing stays live.
    completed_deletes: list[str] = []
    for old in plan.deleted:
        record = change_by_source[old]
        if token.cancelled():
            record.status = ChangeStatus.INTERRUPTED
            continue
        entry = previous_files[old]
        live_md = live_output(entry)
        sidecar_sha = read_sidecar_source_hash(live_md)
        if sidecar_sha is not None and sidecar_sha != entry.sha256:
            # Whatever lives at that path is not this entry's conversion (e.g.
            # a casefold-equal move target relocated there this run); untrack
            # the deletion but leave the other source's output alone.
            completed_deletes.append(old)
            record.status = ChangeStatus.COMPLETED
            log.warning(
                "output at %s belongs to another source; nothing archived for deleted source=%s",
                entry.output,
                old,
            )
            continue
        outcome = archive_or_fail(
            record,
            live_md,
            archive_path_for(out_root, repo, run_id, entry.output),
            "move",
            "deleted output",
        )
        if outcome is None:
            continue
        completed_deletes.append(old)
        record.status = ChangeStatus.COMPLETED
        emptied_dirs.add(live_md.parent)
        if outcome:
            log.info("archived deleted source=%s archived_to=%s", old, record.archived_to)
        else:
            log.warning("no output found to archive for deleted source=%s", old)

    _prune_empty_dirs(emptied_dirs, repo_root)

    # (c) Archive prior versions of updated files — copy, never move: a failed
    # conversion must leave the previous good output live.
    for path in plan.updated:
        record = change_by_source[path]
        if token.cancelled():
            record.status = ChangeStatus.INTERRUPTED
            continue
        entry = previous_files[path]
        outcome = archive_or_fail(
            record,
            live_output(entry),
            archive_path_for(out_root, repo, run_id, entry.output),
            "copy",
            "previous version",
        )
        if outcome is None:
            continue  # converting without preserved history would be lossy
        if outcome:
            log.info("archived previous version source=%s archived_to=%s", path, record.archived_to)
        convert_paths.append(path)
        pending[path] = record
    for path in reconvert:
        record = change_by_source[path]
        if token.cancelled():
            record.status = ChangeStatus.INTERRUPTED
            continue
        convert_paths.append(path)
        pending[path] = record

    # Adds: preserve any orphan output of unknown provenance at the target
    # (unless it already matches the incoming content — the runner adopts it).
    for path in adds:
        record = change_by_source[path]
        if token.cancelled():
            record.status = ChangeStatus.INTERRUPTED
            continue
        live_md = output_path_for(path, repo_root)
        has_orphan = live_md.is_file() or sidecar_path_for(live_md).is_file()
        if has_orphan and read_sidecar_source_hash(live_md) != hashed[path].sha256:
            outcome = archive_or_fail(
                record,
                live_md,
                archive_path_for(out_root, repo, run_id, output_relpath_for(path)),
                "copy",
                "orphan output",
            )
            if outcome is None:
                continue
            if outcome:
                log.warning("archived orphan output at add target path=%s", path)
        convert_paths.append(path)
        pending[path] = record

    # (d) Convert added + updated (+ downgraded moves) through the pipeline.
    by_rel = {f.relative_path: f for f in files}
    to_convert = [by_rel[p] for p in sorted(set(convert_paths))]
    reporter = RichProgress() if show_progress and sys.stderr.isatty() else NullProgress()
    runner = PipelineRunner(
        config=config,
        steps=steps_from_names(),
        store=store,
        out_root=repo_root,
        progress=reporter,
        token=token,
    )
    summary = runner.run(to_convert, skipped)

    # (e) Fill conversion outcomes into the change records; write changes.json.
    record_by_source = {r.source: r for r in store.records}
    for path, change in pending.items():
        change.status, change.error = _conversion_status(record_by_source.get(path))
    if config.archive_keep_runs > 0 and not token.cancelled():
        pruned = prune_archive(out_root, repo, keep_runs=config.archive_keep_runs)
        if pruned:
            log.info("archive pruned repo=%s runs=%s", repo, ",".join(pruned))
            notes.append(
                f"pruned {len(pruned)} archived run(s) beyond the newest {config.archive_keep_runs}"
            )
    report = build_report()
    write_changes_json(run_dir, report)

    # (f) Snapshot: only completed work moves the diff baseline.
    new_snapshot = apply_run_results(
        snapshot,
        repo=repo,
        source_root=str(source),
        run_id=run_id,
        recursive=config.recursive,
        completed_moves=completed_moves,
        completed_deletes=completed_deletes + downgrade_deletes,
        records=store.records,
    )
    save_snapshot(out_root, new_snapshot)
    log.info("snapshot updated files=%d", len(new_snapshot.files))

    # (g) Registry upsert. Re-read under the registry lock: a concurrent sync
    # of a *different* repo may have saved since our startup read, and a blind
    # write-back of the stale copy would drop its entry.
    now = datetime.now(UTC)
    with StateLock(out_root, REGISTRY_LOCK_NAME, run_id=run_id, wait_s=10.0):
        registry = load_registry(out_root)
        current = registry.repos.get(repo)
        registry.repos[repo] = RepoEntry(
            name=repo,
            source_root=str(source),
            first_synced=current.first_synced if current is not None else now,
            last_synced=now,
            last_run_id=run_id,
            run_count=(current.run_count if current is not None else 0) + 1,
            file_count=len(new_snapshot.files),
            last_counts=report.counts,
        )
        save_registry(out_root, registry)
    log.info(
        "registry updated repo=%s run_count=%d file_count=%d",
        repo,
        registry.repos[repo].run_count,
        registry.repos[repo].file_count,
    )

    write_errors_json(run_dir, store.records)
    store.finalize(summary)

    # token.cancelled() alone does not mean interrupted: --fail-fast cancels
    # the token after a failure, and with nothing left undone that run is a
    # plain failure (exit 1), not an interruption to resume (exit 130).
    interrupted = summary.interrupted > 0 or report.counts.interrupted > 0
    failed = summary.failed > 0 or report.counts.failed > 0
    exit_code = 130 if interrupted else (1 if failed else 0)
    return SyncOutcome(
        exit_code=exit_code,
        repo=repo,
        known=known,
        run_dir=run_dir,
        report=report,
        summary=summary,
        records=store.records,
    )
