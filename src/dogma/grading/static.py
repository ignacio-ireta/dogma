"""Static checks that run before any learner code executes (ARCHITECTURE §4.2).

Python: the file parses, the entry point is a top-level ``def``/``class``, and no banned module
is imported or reached through attribute access (``np.linalg`` counts). The banned-import rule is
ported from ``scripts/grade.py``. SQL: only a non-empty check.
"""

from __future__ import annotations

import ast
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Literal

from dogma.grading.models import StaticFinding

Language = Literal["python", "sql"]

_FUNCTION_OR_CLASS = (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
# Test-data helpers and dataset downloads belong to the tests alone (I-31a). This applies to every
# problem, so it is not repeated in each meta.yaml.
PLATFORM_BANNED_IMPORTS = ("dogma_data", "datasets", "huggingface_hub")
_PLATFORM_MESSAGE = "submissions may not import test-data helpers (dogma_data) or download datasets"


def check_submission(
    path: Path,
    *,
    language: Language,
    entry: str,
    banned_imports: Iterable[str],
) -> list[StaticFinding]:
    """Return every static problem in the submission at ``path``; an empty list means it may run."""
    try:
        source = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return [StaticFinding(rule="syntax", message="the file is not valid UTF-8 text")]
    if language == "sql":
        return _check_sql(source)
    return _check_python(source, entry=entry, banned=tuple(banned_imports))


def _check_sql(source: str) -> list[StaticFinding]:
    if not source.strip():
        return [StaticFinding(rule="empty", message="the query file is empty")]
    return []


def _check_python(source: str, *, entry: str, banned: Sequence[str]) -> list[StaticFinding]:
    try:
        tree = ast.parse(source, filename="submission.py")
    except SyntaxError as exc:
        where = f" (line {exc.lineno})" if exc.lineno is not None else ""
        return [StaticFinding(rule="syntax", message=f"{exc.msg}{where}", line=exc.lineno)]
    except (ValueError, RecursionError, MemoryError) as exc:  # MemoryError: parser stack overflow
        return [StaticFinding(rule="syntax", message=f"cannot parse the file: {exc}")]

    findings: list[StaticFinding] = []
    if not _defines_top_level(tree, entry):
        findings.append(
            StaticFinding(
                rule="missing-entry",
                message=f"define `{entry}` as a top-level function or class",
            )
        )
    findings.extend(_banned_findings(tree, banned))
    findings.extend(_platform_findings(tree))
    return findings


def _defines_top_level(tree: ast.Module, entry: str) -> bool:
    return any(isinstance(node, _FUNCTION_OR_CLASS) and node.name == entry for node in tree.body)


def _banned_findings(tree: ast.Module, banned: Sequence[str]) -> list[StaticFinding]:
    return [
        StaticFinding(
            rule="banned-import",
            message=f"`{module}` is not allowed in this problem (line {line})",
            line=line,
        )
        for line, module in sorted(_hits(tree, banned))
    ]


def _platform_findings(tree: ast.Module) -> list[StaticFinding]:
    lines = sorted({line for line, _ in _hits(tree, PLATFORM_BANNED_IMPORTS)})
    return [
        StaticFinding(
            rule="banned-import",
            message=f"{_PLATFORM_MESSAGE} (line {line})",
            line=line,
        )
        for line in lines
    ]


def _hits(tree: ast.Module, banned: Sequence[str]) -> set[tuple[int, str]]:
    hits: set[tuple[int, str]] = set()
    for node in ast.walk(tree):
        line = getattr(node, "lineno", None)
        if line is None:
            continue
        for name in _names_used(node):
            for module in banned:
                if _matches(name, module, is_attribute=isinstance(node, ast.Attribute)):
                    hits.add((line, module))
    return hits


def _names_used(node: ast.AST) -> list[str]:
    """Module names an import statement brings in, or the attribute name of an access."""
    if isinstance(node, ast.Import):
        return [alias.name for alias in node.names]
    if isinstance(node, ast.ImportFrom) and node.module:
        return [node.module, *(f"{node.module}.{alias.name}" for alias in node.names)]
    if isinstance(node, ast.Attribute):
        return [node.attr]
    return []


def _matches(name: str, module: str, *, is_attribute: bool) -> bool:
    if not is_attribute:
        return name == module or name.startswith(module + ".")
    # ``np.linalg`` is an attribute access, not an import: match the last component only when
    # the banned name is dotted (``numpy.linalg`` -> ``linalg``). A bare banned name such as
    # ``random`` must not match attributes like ``rng.random`` or ``np.random``.
    return "." in module and name == module.rsplit(".", 1)[-1]
