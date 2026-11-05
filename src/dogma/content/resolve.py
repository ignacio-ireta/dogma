"""Turn what the learner typed into content: an id, or the path to its directory.

Topic ids are ``track/topic-slug``. Problem ids are ``track/topic-slug/pNN-slug``, and their
directories are ``content/<track>/<topic-slug>/problems/pNN-slug``. An existing directory is
accepted as a path. The workspace file for a problem is ``workspace/<track>/<topic>/pNN-slug.py``
(or ``.sql``), the same layout as scripts/grade.py, so earlier work carries over. A copy the
learner sets aside to start over goes to ``var/workspace-history/`` with the same layout.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from dogma.content.loader import load_problem
from dogma.content.schemas import ProblemMeta
from dogma.core.config import Settings
from dogma.core.errors import ContentError, ErrorCode
from dogma.core.fsio import atomic_write


@dataclass(frozen=True)
class ProblemRef:
    """A problem directory under content/, its id, and its validated meta.yaml."""

    dir: Path
    id: str
    meta: ProblemMeta

    @property
    def suffix(self) -> str:
        return ".sql" if self.meta.language == "sql" else ".py"

    @property
    def starter(self) -> Path:
        return self.dir / f"starter{self.suffix}"

    @property
    def problem_md(self) -> Path:
        return self.dir / "problem.md"


@dataclass(frozen=True)
class Target:
    """What ``content verify`` was pointed at: a whole topic, or one problem inside it."""

    topic_dir: Path
    problem_dir: Path | None


def resolve_problem(settings: Settings, ref: str) -> ProblemRef:
    """Find the problem named by ``ref`` (an id or a directory) under content/."""
    directory = _existing_dir(settings, ref)
    if directory is None:
        raise ContentError(
            ErrorCode.CONTENT_NOT_FOUND,
            f"no problem {ref!r}: give an id such as cml/vector-operations/p01-vector-projection"
            " or the problem's directory",
        )
    resolved = directory.resolve()
    parts = _content_parts(settings, resolved)
    if parts is None or len(parts) != 4 or parts[2] != "problems":
        raise ContentError(
            ErrorCode.CONTENT_NOT_FOUND,
            f"{resolved} is not a problem directory (content/<track>/<topic>/problems/pNN-slug)",
        )
    package = load_problem(resolved)
    if package.meta is None:
        raise ContentError(ErrorCode.CONTENT_NOT_FOUND, f"{resolved} has no meta.yaml")
    return ProblemRef(
        dir=resolved,
        id="/".join((parts[0], parts[1], parts[3])),
        meta=package.meta,
    )


def resolve_target(settings: Settings, ref: str) -> Target:
    """Find the topic or problem named by ``ref``; a problem maps to its topic."""
    directory = _existing_dir(settings, ref)
    if directory is None:
        raise ContentError(
            ErrorCode.CONTENT_NOT_FOUND,
            f"no topic or problem {ref!r}: give a topic id such as cml/vector-operations, "
            "or a directory under content/",
        )
    resolved = directory.resolve()
    if (resolved / "meta.yaml").is_file() and resolved.parent.name == "problems":
        return Target(topic_dir=resolved.parent.parent, problem_dir=resolved)
    if (resolved / "topic.yaml").is_file():
        return Target(topic_dir=resolved, problem_dir=None)
    raise ContentError(
        ErrorCode.CONTENT_NOT_FOUND,
        f"{resolved} is neither a topic (topic.yaml) nor a problem (meta.yaml) directory",
    )


def workspace_file(settings: Settings, ref: ProblemRef) -> Path:
    """Where the learner's own copy lives: ``workspace/<track>/<topic>/pNN-slug.py``."""
    track, topic, problem = ref.id.split("/")
    return settings.workspace_dir / track / topic / f"{problem}{ref.suffix}"


def set_aside_workspace(settings: Settings, ref: ProblemRef, *, at: datetime) -> Path | None:
    """Copy the learner's file to ``var/workspace-history/`` before a fresh start.

    Returns the copy's path, or None when there is no workspace file. The original is left in
    place: the caller overwrites it with the starter only after the copy is safely on disk.
    """
    source = workspace_file(settings, ref)
    if not source.is_file():
        return None
    track, topic, problem = ref.id.split("/")
    stamp = at.strftime("%Y%m%dT%H%M%S%fZ")
    kept = settings.var_dir / "workspace-history" / track / topic / f"{problem}.{stamp}{ref.suffix}"
    atomic_write(kept, source.read_bytes())
    return kept


def _existing_dir(settings: Settings, ref: str) -> Path | None:
    as_path = Path(ref)
    if as_path.is_dir():
        return as_path
    candidate = _id_dir(settings, ref)
    return candidate if candidate is not None and candidate.is_dir() else None


def _id_dir(settings: Settings, ref: str) -> Path | None:
    """The directory an id stands for: 2 parts is a topic, 3 parts is a problem."""
    parts = ref.strip("/").split("/")
    if not all(parts):
        return None
    if len(parts) == 2:
        return settings.content_dir.joinpath(*parts)
    if len(parts) == 3:
        return settings.content_dir.joinpath(parts[0], parts[1], "problems", parts[2])
    return None


def _content_parts(settings: Settings, resolved: Path) -> tuple[str, ...] | None:
    try:
        return resolved.relative_to(settings.content_dir.resolve()).parts
    except ValueError:
        return None
