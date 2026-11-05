"""Registry of document repos synced into an output root.

``<out>/registry.json`` answers "is this a known repo?" in one place. Repo
identity is the source root folder name, which the user guarantees is stable.
Only per-repo metadata lives here so the registry stays O(#repos); per-file
state is in each repo's snapshot. A corrupt registry fails loudly instead of
being rebuilt: silently recreating it would orphan every repo's snapshot,
archive, and history.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

from pydantic import BaseModel, Field

from docmd.errors import RegistryCorruptError, ReservedRepoNameError
from docmd.fsio import atomic_write_text
from docmd.tracking.report import SyncCounts

REGISTRY_NAME = "registry.json"

#: Top-level names under an output root that belong to docmd, not to a repo.
#: "_registry" is reserved because repo locks share a directory with the
#: registry lock (see tracking.lock.REGISTRY_LOCK_NAME).
RESERVED_REPO_NAMES = frozenset({"runs", "registry.json", ".docmd", "_registry"})


class RepoEntry(BaseModel):
    name: str
    source_root: str  # absolute source path last seen; informational
    first_synced: datetime
    last_synced: datetime
    last_run_id: str
    run_count: int = 0
    file_count: int = 0  # tracked files after the last completed sync
    last_counts: SyncCounts | None = None


class Registry(BaseModel):
    schema_version: int = 1
    repos: dict[str, RepoEntry] = Field(default_factory=dict)


def registry_path(out_root: Path) -> Path:
    return out_root / REGISTRY_NAME


def validate_repo_name(name: str) -> None:
    """Reject repo names that collide with the output root's own structure."""
    if not name:
        raise ReservedRepoNameError(
            "cannot derive a repo name from the source path; "
            "sync a named folder, not a filesystem root"
        )
    if name.startswith("."):
        raise ReservedRepoNameError(
            f"repo name {name!r} is hidden and would be invisible to future scans"
        )
    if name.casefold() in RESERVED_REPO_NAMES:
        raise ReservedRepoNameError(
            f"repo name {name!r} is reserved for docmd's own files under the output root"
        )


def load_registry(out_root: Path) -> Registry:
    """Load the registry; a missing file is an empty registry, a bad one is fatal."""
    path = registry_path(out_root)
    if not path.exists():
        return Registry()
    try:
        return Registry.model_validate_json(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise RegistryCorruptError(
            f"registry is unreadable: {path}; restore it from backup or delete it "
            "deliberately to start a fresh knowledge base",
            path=path,
            detail=str(exc),
        ) from exc


def save_registry(out_root: Path, registry: Registry) -> Path:
    path = registry_path(out_root)
    atomic_write_text(path, registry.model_dump_json(indent=2) + "\n")
    return path
