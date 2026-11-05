"""Grading: result models and static checks. Pure: no I/O beyond reading the submission."""

from dogma.grading.models import (
    FileKind,
    GradeReport,
    Outcome,
    RunStatus,
    StaticFinding,
    TestResult,
)
from dogma.grading.static import Language, check_submission

__all__ = [
    "FileKind",
    "GradeReport",
    "Language",
    "Outcome",
    "RunStatus",
    "StaticFinding",
    "TestResult",
    "check_submission",
]
