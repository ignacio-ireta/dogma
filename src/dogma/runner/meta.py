"""Typed loader for the fields of ``meta.yaml`` the runner needs (CONTENT_SPEC §6.2).

Deliberately independent of ``dogma.content``: the runner reads only what it executes with,
and ignores every other key.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from dogma.core.errors import ErrorCode, RunnerError

META_FILE = "meta.yaml"


class ProblemMeta(BaseModel):
    """The runner's view of one problem's ``meta.yaml``."""

    model_config = ConfigDict(extra="ignore", frozen=True)

    id: str = Field(min_length=1)
    version: int = Field(ge=1)
    language: Literal["python", "sql"]
    runtime: str = Field(min_length=1)
    entry: str = Field(min_length=1)
    banned_imports: tuple[str, ...] = ()
    # Whole-run wall-clock limit; None means "use the Settings default".
    timeout_s: int | None = Field(default=None, gt=0)


def load_meta(problem_dir: Path) -> ProblemMeta:
    """Read and validate ``problem_dir/meta.yaml``.

    Raises ``RunnerError`` (CONTENT_NOT_FOUND or CONTENT_SCHEMA_INVALID) for a broken problem
    directory: an infrastructure problem, not a learner failure.
    """
    path = problem_dir / META_FILE
    if not path.is_file():
        raise RunnerError(ErrorCode.CONTENT_NOT_FOUND, f"no {META_FILE} in {problem_dir}")
    try:
        raw: object = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise RunnerError(ErrorCode.CONTENT_SCHEMA_INVALID, f"cannot parse {path}: {exc}") from exc
    try:
        return ProblemMeta.model_validate(raw)
    except ValidationError as exc:
        raise RunnerError(
            ErrorCode.CONTENT_SCHEMA_INVALID, f"invalid {path}: {_summary(exc)}"
        ) from exc


def _summary(exc: ValidationError) -> str:
    parts = [
        f"{'.'.join(str(p) for p in err['loc']) or 'meta'}: {err['msg']}" for err in exc.errors()
    ]
    return "; ".join(parts)
