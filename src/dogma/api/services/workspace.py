"""The learner's workspace file, the same file `dogma open/run/submit` use (ARCHITECTURE §4.7).

Saves are atomic (ENGINEERING_STANDARDS §4.7). The path comes from the problem id through the
content loader, never from the request, so a save cannot land outside ``workspace/``.
"""

from __future__ import annotations

from dogma.api.schemas.workspace import WorkspaceFile
from dogma.api.services.content import problem_ref
from dogma.content.resolve import workspace_file
from dogma.core.config import Settings
from dogma.core.errors import DogmaError, ErrorCode
from dogma.core.fsio import atomic_write

MAX_WORKSPACE_BYTES = 256 * 1024


def read(settings: Settings, problem_id: str) -> WorkspaceFile:
    ref = problem_ref(settings, problem_id)
    path = workspace_file(settings, ref)
    if not path.is_file():
        return WorkspaceFile(problem_id=ref.id, exists=False, content=None)
    try:
        text = path.read_bytes().decode("utf-8")
    except UnicodeDecodeError as exc:
        raise DogmaError(
            ErrorCode.API_WORKSPACE_INVALID,
            f"{path.name} is not UTF-8 text, so it is not shown or overwritten by the API",
        ) from exc
    return WorkspaceFile(problem_id=ref.id, exists=True, content=text)


def write(settings: Settings, problem_id: str, content: str) -> WorkspaceFile:
    ref = problem_ref(settings, problem_id)
    if len(content.encode("utf-8")) > MAX_WORKSPACE_BYTES:
        raise DogmaError(
            ErrorCode.API_WORKSPACE_TOO_LARGE,
            f"a workspace file is limited to {MAX_WORKSPACE_BYTES // 1024} KiB",
        )
    atomic_write(workspace_file(settings, ref), content)
    return WorkspaceFile(problem_id=ref.id, exists=True, content=content)
