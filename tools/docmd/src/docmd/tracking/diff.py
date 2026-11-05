"""Classify a repo tree against its last synced snapshot. Pure, no I/O.

Moves are matched strictly one-to-one by content hash: a hash pairs a deleted
path with an added path only when it identifies exactly one candidate on each
side. Duplicate content makes intent undecidable, so ambiguous sets fall back
to delete+add — the archive preserves the deleted outputs and re-extraction of
an add costs the same as any fresh add. Paths present in both trees resolve
first (unchanged/updated), so content that lands on a still-existing path can
never be mistaken for a move.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping
from dataclasses import dataclass

from pydantic import BaseModel

from docmd.tracking.snapshot import SnapshotEntry


class HashedFile(BaseModel):
    relative_path: str
    sha256: str
    size: int


@dataclass(frozen=True)
class SyncPlan:
    added: list[str]
    updated: list[str]  # same path, different hash
    unchanged: list[str]
    moved: list[tuple[str, str]]  # (old_relpath, new_relpath)
    deleted: list[str]


def classify_changes(
    previous: Mapping[str, SnapshotEntry],
    current: Mapping[str, HashedFile],
) -> SyncPlan:
    previous_keys = set(previous)
    current_keys = set(current)
    updated: list[str] = []
    unchanged: list[str] = []
    for path in sorted(previous_keys & current_keys):
        if previous[path].sha256 == current[path].sha256:
            unchanged.append(path)
        else:
            updated.append(path)

    added_candidates = sorted(current_keys - previous_keys)
    deleted_candidates = sorted(previous_keys - current_keys)

    deleted_by_hash: dict[str, list[str]] = defaultdict(list)
    for path in deleted_candidates:
        deleted_by_hash[previous[path].sha256].append(path)
    added_by_hash: dict[str, list[str]] = defaultdict(list)
    for path in added_candidates:
        added_by_hash[current[path].sha256].append(path)

    moved: list[tuple[str, str]] = []
    moved_from: set[str] = set()
    moved_to: set[str] = set()
    for sha256, olds in deleted_by_hash.items():
        news = added_by_hash.get(sha256, [])
        if len(olds) == 1 and len(news) == 1:
            moved.append((olds[0], news[0]))
            moved_from.add(olds[0])
            moved_to.add(news[0])
    moved.sort()

    return SyncPlan(
        added=[p for p in added_candidates if p not in moved_to],
        updated=updated,
        unchanged=unchanged,
        moved=moved,
        deleted=[p for p in deleted_candidates if p not in moved_from],
    )
