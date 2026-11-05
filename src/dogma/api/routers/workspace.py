from __future__ import annotations

from fastapi import APIRouter

from dogma.api.deps import SettingsDep
from dogma.api.openapi import error_responses
from dogma.api.schemas.workspace import WorkspaceFile, WorkspaceWrite
from dogma.api.services import workspace

router = APIRouter(tags=["workspace"])


@router.get(
    "/workspace/{problem_id:path}",
    response_model=WorkspaceFile,
    summary="The learner's own file for a problem (what the CLI grades)",
    responses=error_responses(
        (404, "No problem with this id"),
        (422, "The file is not UTF-8 text"),
    ),
)
def get_workspace(problem_id: str, settings: SettingsDep) -> WorkspaceFile:
    return workspace.read(settings, problem_id)


@router.put(
    "/workspace/{problem_id:path}",
    response_model=WorkspaceFile,
    summary="Save the learner's file (atomic; shared with `dogma run/submit`)",
    responses=error_responses(
        (404, "No problem with this id"),
        (413, "The content is larger than 256 KiB"),
    ),
)
def put_workspace(problem_id: str, body: WorkspaceWrite, settings: SettingsDep) -> WorkspaceFile:
    return workspace.write(settings, problem_id, body.content)
