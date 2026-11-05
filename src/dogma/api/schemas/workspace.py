"""The learner's own file for a problem: the browser editor and the CLI both read and write it."""

from __future__ import annotations

from pydantic import BaseModel


class WorkspaceWrite(BaseModel):
    content: str


class WorkspaceFile(BaseModel):
    problem_id: str
    exists: bool
    content: str | None
