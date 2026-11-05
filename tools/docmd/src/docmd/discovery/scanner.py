"""Discover supported files under a root, deterministically ordered.

Relative paths are stored as posix strings so manifests and sidecars stay
stable across platforms. Results are sorted by relative path; one bad or
unreadable entry never aborts the walk.
"""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel

from docmd.discovery.file_types import detect_file_type
from docmd.errors import DiscoveryError
from docmd.types import SourceType


class DiscoveredFile(BaseModel):
    path: Path
    relative_path: str  # posix, relative to scan root
    file_type: SourceType
    size: int
    detection_warning: str | None = None
    sha256: str | None = None  # filled by sync's hash pre-pass; reused downstream


class SkippedFile(BaseModel):
    path: Path
    relative_path: str
    reason: str


def _is_hidden(rel_parts: tuple[str, ...]) -> bool:
    return any(part.startswith(".") for part in rel_parts)


def scan_with_skips(
    root: Path,
    *,
    recursive: bool = False,
    out_root: Path | None = None,
) -> tuple[list[DiscoveredFile], list[SkippedFile]]:
    root = root.resolve()
    if not root.exists():
        raise DiscoveryError(f"input path does not exist: {root}", path=root)

    if root.is_file():
        candidates = [root]
        base = root.parent
    else:
        base = root
        try:
            iterator = root.rglob("*") if recursive else root.glob("*")
            candidates = sorted(p for p in iterator if p.is_file())
        except OSError as exc:
            raise DiscoveryError(f"failed to walk {root}: {exc}", path=root) from exc

    out_resolved = out_root.resolve() if out_root is not None else None

    discovered: list[DiscoveredFile] = []
    skipped: list[SkippedFile] = []
    for path in candidates:
        rel = path.relative_to(base)
        rel_posix = rel.as_posix()
        if out_resolved is not None and path.resolve().is_relative_to(out_resolved):
            continue  # never re-ingest our own output
        if _is_hidden(rel.parts):
            skipped.append(SkippedFile(path=path, relative_path=rel_posix, reason="hidden file"))
            continue
        if path.name.startswith("~$"):
            skipped.append(
                SkippedFile(path=path, relative_path=rel_posix, reason="office lock file")
            )
            continue
        detection = detect_file_type(path)
        if detection.file_type is None:
            assert detection.skip_reason is not None
            skipped.append(
                SkippedFile(path=path, relative_path=rel_posix, reason=detection.skip_reason)
            )
            continue
        try:
            size = path.stat().st_size
        except OSError as exc:
            skipped.append(
                SkippedFile(path=path, relative_path=rel_posix, reason=f"stat failed: {exc}")
            )
            continue
        discovered.append(
            DiscoveredFile(
                path=path,
                relative_path=rel_posix,
                file_type=detection.file_type,
                size=size,
                detection_warning=detection.warning,
            )
        )

    discovered.sort(key=lambda f: f.relative_path)
    skipped.sort(key=lambda f: f.relative_path)
    return discovered, skipped
