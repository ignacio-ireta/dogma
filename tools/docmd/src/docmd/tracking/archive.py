"""Archive displaced conversions instead of losing them.

Every superseded output — updated, deleted, or moved away — lands under
``<out>/.docmd/archive/<repo>/<run_id>/<original output relpath>``, grouped by
the run that displaced it. ``.docmd`` is hidden, so archived Markdown can
never be re-ingested by a scan of the output root. Deletions MOVE the pair
(nothing stays behind at the live path); updates COPY it, because a failed
re-conversion must never destroy the only good output.
"""

from __future__ import annotations

import shutil
from pathlib import Path, PurePosixPath
from typing import Literal

from docmd.fsio import atomic_write_bytes
from docmd.markdown.metadata import sidecar_path_for

ArchiveMode = Literal["move", "copy"]


def archive_root(out_root: Path, repo: str, run_id: str) -> Path:
    return out_root / ".docmd" / "archive" / repo / run_id


def archive_path_for(out_root: Path, repo: str, run_id: str, output_relpath: str) -> Path:
    rel = PurePosixPath(output_relpath)
    return archive_root(out_root, repo, run_id).joinpath(*rel.parts)


def archive_output(md_path: Path, archive_md_path: Path, *, mode: ArchiveMode) -> list[Path]:
    """Archive an output ``.md`` and its sidecar; returns what was archived.

    Missing files are tolerated (a partial prior crash may have left only one
    of the pair); filesystem failures raise ``OSError`` for the caller's
    per-action isolation.
    """
    archived: list[Path] = []
    pairs = (
        (md_path, archive_md_path),
        (sidecar_path_for(md_path), sidecar_path_for(archive_md_path)),
    )
    for live, dest in pairs:
        if not live.is_file():
            continue
        if mode == "move":
            dest.parent.mkdir(parents=True, exist_ok=True)
            live.replace(dest)
        else:
            atomic_write_bytes(dest, live.read_bytes())
        archived.append(dest)
    return archived


def prune_archive(out_root: Path, repo: str, *, keep_runs: int) -> list[str]:
    """Drop a repo's oldest archived runs beyond ``keep_runs``; 0 keeps all.

    Run ids sort lexicographically by age (they are UTC timestamps). Returns
    the pruned run ids; filesystem errors on one run dir do not stop the rest.
    """
    if keep_runs <= 0:
        return []
    repo_archive = out_root / ".docmd" / "archive" / repo
    if not repo_archive.is_dir():
        return []
    run_dirs = sorted((p for p in repo_archive.iterdir() if p.is_dir()), key=lambda p: p.name)
    pruned: list[str] = []
    for run_dir in run_dirs[:-keep_runs] if len(run_dirs) > keep_runs else []:
        try:
            shutil.rmtree(run_dir)
        except OSError:
            continue
        pruned.append(run_dir.name)
    return pruned
