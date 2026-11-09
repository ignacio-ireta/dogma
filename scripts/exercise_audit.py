"""Measure how well the exercises reject wrong answers and accept right ones (ROADMAP I-28).

Every grade goes through ``dogma.runner.attempt.grade_attempt`` with a scratch ``repo_root``, so it
is the learner's real path (Seatbelt, fixed Hypothesis seed) and never touches ``var/``. Results are
appended to JSONL files under ``--out``, so a run can stop and resume. Nothing here edits content.

Usage (the torch group is needed for the py-torch runtime)::

    uv run --frozen --group torch python scripts/exercise_audit.py baseline  --out DIR
    uv run --frozen --group torch python scripts/exercise_audit.py mutate    --out DIR
    uv run --frozen --group torch python scripts/exercise_audit.py precision --out DIR
    uv run --frozen --group torch python scripts/exercise_audit.py coverage  --out DIR
    uv run --frozen --group torch python scripts/exercise_audit.py seeds     --out DIR
    uv run --frozen --group torch python scripts/exercise_audit.py bundle    --out DIR
    uv run --frozen --group torch python scripts/exercise_audit.py public    --out DIR ID FILE
    uv run --frozen --group torch python scripts/exercise_audit.py submit    --out DIR ID FILE
    uv run --frozen --group torch python scripts/exercise_audit.py summary   --out DIR
    uv run --frozen --group torch python scripts/exercise_audit.py report    --out DIR
    uv run --frozen --group torch python scripts/exercise_audit.py survivors --out DIR
    uv run --frozen --group torch python scripts/exercise_audit.py witness   --out DIR ID FILE
    uv run --frozen --group torch python scripts/exercise_audit.py realism  --out DIR ID WRONG...
        (optional: --base REV, default HEAD; --blind FILE)

``--only PREFIX`` (repeatable) limits a command to problems whose id starts with PREFIX.

``seeds`` also sets ``DOGMA_DATA_SALT`` to the seed, so tests that draw data through
``dogma_data.rng`` see a different draw per run. The default grader never sets the salt.

``realism`` grades the reference, the optional blind solution and each WRONG file against the
test files at git revision ``--base`` (default HEAD, the "base" suite) and against the working
tree (the "new" suite). Each WRONG file must pass the base suite and fail the new one (``killed``).
It exits 0 only when the reference and the blind solution pass the new suite and every WRONG file
is killed. Results go to ``DIR/realism.jsonl``, keyed by the file and suite hashes.
"""

from __future__ import annotations

import argparse
import ast
import difflib
import hashlib
import json
import os
import random
import re
import shutil
import subprocess
import sys
import tempfile
import threading
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable, Iterator, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from rich.console import Console

from dogma.cli.render import render_report
from dogma.core.config import Settings
from dogma.grading.display import NO_DETAIL, learner_message
from dogma.grading.models import GradeReport, RunStatus, TestResult
from dogma.grading.static import check_submission
from dogma.runner import attempt, load_meta

REPO = Path(__file__).resolve().parents[1]
MAX_MUTANTS = 60
MAX_BLIND_ATTEMPTS = 3  # one submission and up to two resubmits
BLIND_BATCH = 10
PRECISION_EPSILONS = (1e-3, 1e-6, 1e-8)
SEEDS = tuple(range(1, 21))
# Inside these, a mutation only changes an error message or a type hint, which no test should see.
_INERT_PARENTS = (ast.Raise, ast.Assert)


@dataclass(frozen=True)
class Problem:
    """One exercise, as the tool needs it."""

    id: str
    path: Path
    language: str
    entry: str
    timeout_s: int
    banned: tuple[str, ...]

    @property
    def suffix(self) -> str:
        return ".sql" if self.language == "sql" else ".py"

    @property
    def solution(self) -> Path:
        return self.path / f"solution{self.suffix}"

    @property
    def track(self) -> str:
        return self.id.split("/", 1)[0]


@dataclass(frozen=True)
class Mutant:
    """One deliberately wrong version of a reference solution."""

    op: str
    line: int
    source: str
    change: str

    @property
    def key(self) -> str:
        return hashlib.sha256(self.source.encode()).hexdigest()[:16]


def discover(repo: Path, only: Sequence[str]) -> list[Problem]:
    """Every problem under ``content/``, sorted by id, optionally limited to id prefixes."""
    found: list[Problem] = []
    for path in sorted(repo.glob("content/*/*/problems/*/meta.yaml")):
        meta = load_meta(path.parent)
        if only and not any(meta.id.startswith(prefix) for prefix in only):
            continue
        found.append(
            Problem(
                id=meta.id,
                path=path.parent,
                language=meta.language,
                entry=meta.entry,
                timeout_s=meta.timeout_s if meta.timeout_s is not None else 30,
                banned=tuple(meta.banned_imports),
            )
        )
    return found


# --- Python mutation operators ------------------------------------------------------------------

_COMPARE: dict[type[ast.cmpop], tuple[type[ast.cmpop], ...]] = {
    ast.Lt: (ast.LtE, ast.GtE),
    ast.LtE: (ast.Lt, ast.Gt),
    ast.Gt: (ast.GtE, ast.LtE),
    ast.GtE: (ast.Gt, ast.Lt),
    ast.Eq: (ast.NotEq,),
    ast.NotEq: (ast.Eq,),
    ast.Is: (ast.IsNot,),
    ast.IsNot: (ast.Is,),
    ast.In: (ast.NotIn,),
    ast.NotIn: (ast.In,),
}
_BINARY: dict[type[ast.operator], tuple[type[ast.operator], ...]] = {
    ast.Add: (ast.Sub,),
    ast.Sub: (ast.Add,),
    ast.Mult: (ast.Div,),
    ast.Div: (ast.Mult,),
    ast.FloorDiv: (ast.Div,),
    ast.Pow: (ast.Mult,),
    ast.MatMult: (ast.Mult,),
}
_CALL_SWAPS: dict[str, str] = {
    "min": "max",
    "max": "min",
    "amin": "amax",
    "amax": "amin",
    "nanmin": "nanmax",
    "nanmax": "nanmin",
    "argmin": "argmax",
    "argmax": "argmin",
    "sum": "mean",
    "mean": "sum",
    "all": "any",
    "any": "all",
    "floor": "ceil",
    "ceil": "floor",
    "log": "log1p",
    "exp": "expm1",
    "softmax": "log_softmax",
    "log_softmax": "softmax",
    "no_grad": "enable_grad",
    "array": "asarray",
}
_DROP_WRAPPER = frozenset({"abs", "sqrt", "reversed", "sorted"})
_DROP_METHOD = frozenset({"detach", "copy", "clone"})
_DTYPE_SWAPS = {"float64": "float32", "int64": "int32", "double": "float"}
_AXIS_KEYWORDS = frozenset({"axis", "dim"})

_Edit = Callable[[ast.AST, dict[int, ast.AST]], None]


def _replace(old: ast.AST, new: ast.AST, parents: dict[int, ast.AST]) -> None:
    """Put ``new`` where ``old`` sits in its parent, as a field or a list item."""
    parent = parents[id(old)]
    for field, value in ast.iter_fields(parent):
        if value is old:
            setattr(parent, field, new)
            return
        if isinstance(value, list) and any(item is old for item in value):
            value[[item is old for item in value].index(True)] = new
            return
    raise ValueError(f"{type(old).__name__} is not a direct child of {type(parent).__name__}")


def _parents(tree: ast.AST) -> dict[int, ast.AST]:
    return {id(child): node for node in ast.walk(tree) for child in ast.iter_child_nodes(node)}


def _inert(tree: ast.AST) -> set[int]:
    """Ids of nodes whose mutation cannot change behaviour: messages, hints and docstrings."""
    inert: set[int] = set()

    def mark(node: ast.AST | None) -> None:
        if node is not None:
            inert.update(id(child) for child in ast.walk(node))

    for node in ast.walk(tree):
        if isinstance(node, _INERT_PARENTS):
            mark(node)
        elif isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
            mark(node.returns)
        elif isinstance(node, (ast.arg, ast.AnnAssign)):
            mark(node.annotation)
        elif (
            isinstance(node, ast.Expr)
            and isinstance(node.value, ast.Constant)
            and isinstance(node.value.value, str)
        ):
            mark(node)
    return inert


def _sites(  # noqa: PLR0912 - one branch per node type reads better than a table of closures
    node: ast.AST, parent: ast.AST | None
) -> Iterator[tuple[str, _Edit]]:
    """Every way to mutate ``node``: pairs of an operator name and an in-place edit."""
    if isinstance(node, ast.Compare):
        yield from _compare_sites(node)
    elif isinstance(node, ast.BinOp | ast.AugAssign):
        for swap in _BINARY.get(type(node.op), ()):

            def set_operator(n: ast.AST, _: dict[int, ast.AST], op: type = swap) -> None:
                n.op = op()  # type: ignore[attr-defined]

            yield "arithmetic", set_operator
    elif isinstance(node, ast.BoolOp):

        def swap_bool(n: ast.AST, _: dict[int, ast.AST]) -> None:
            n.op = ast.Or() if isinstance(n.op, ast.And) else ast.And()  # type: ignore[attr-defined]

        yield "boolean", swap_bool
    elif isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub | ast.Not):
        if isinstance(parent, ast.keyword) and parent.arg in _AXIS_KEYWORDS:  # dim=-1
            yield "axis", lambda n, p: _replace(n, ast.Constant(0), p)
        else:
            yield "drop_negation", lambda n, p: _replace(n, n.operand, p)  # type: ignore[attr-defined]
    elif isinstance(node, ast.Constant):
        yield from _constant_sites(node, parent)
    elif isinstance(node, ast.Break | ast.Continue):
        swapped: type[ast.stmt] = ast.Continue if isinstance(node, ast.Break) else ast.Break
        yield "flow", lambda n, p: _replace(n, swapped(), p)
    elif isinstance(node, ast.Call):
        yield from _call_sites(node)
    elif isinstance(node, ast.Attribute):
        yield from _attribute_sites(node, parent)
    elif (
        isinstance(node, ast.If)
        and not node.orelse
        and all(isinstance(statement, ast.Raise) for statement in node.body)
    ):
        yield "guard", lambda n, p: _replace(n, ast.Pass(), p)
    elif isinstance(node, ast.Expr) and isinstance(node.value, ast.Call):
        yield "drop_statement", lambda n, p: _replace(n, ast.Pass(), p)


def _compare_sites(node: ast.Compare) -> Iterator[tuple[str, _Edit]]:
    for index, operator in enumerate(node.ops):
        for swap in _COMPARE.get(type(operator), ()):

            def set_compare(
                n: ast.AST, _: dict[int, ast.AST], i: int = index, op: type = swap
            ) -> None:
                n.ops[i] = op()  # type: ignore[attr-defined]

            yield "compare", set_compare


def _constant_sites(node: ast.Constant, parent: ast.AST | None) -> Iterator[tuple[str, _Edit]]:
    value = node.value

    def set_value(new: object) -> _Edit:
        def edit(n: ast.AST, _: dict[int, ast.AST]) -> None:
            n.value = new  # type: ignore[attr-defined]

        return edit

    if isinstance(value, bool):
        yield "bool", set_value(not value)
    elif isinstance(value, int):
        if isinstance(parent, ast.keyword) and parent.arg in _AXIS_KEYWORDS:
            yield "axis", set_value(0 if value in (1, -1) else 1)
        else:
            yield "constant", set_value(value + 1)
            yield "constant", set_value(value - 1)
    elif isinstance(value, float):
        yield "constant", set_value(value * 2)


def _call_sites(node: ast.Call) -> Iterator[tuple[str, _Edit]]:
    name = (
        node.func.id
        if isinstance(node.func, ast.Name)
        else node.func.attr
        if isinstance(node.func, ast.Attribute)
        else None
    )
    if name is None:
        return
    swap = _CALL_SWAPS.get(name)
    if swap is not None:

        def rename(n: ast.AST, _: dict[int, ast.AST], new: str = swap) -> None:
            func = n.func  # type: ignore[attr-defined]
            if isinstance(func, ast.Name):
                func.id = new
            else:
                func.attr = new

        yield "array_copy" if name == "array" else "call", rename
    if name in _DROP_WRAPPER and len(node.args) == 1 and not node.keywords:
        yield "drop_call", lambda n, p: _replace(n, n.args[0], p)  # type: ignore[attr-defined]
    if name in _DROP_METHOD and isinstance(node.func, ast.Attribute) and not node.args:
        yield "drop_call", lambda n, p: _replace(n, n.func.value, p)  # type: ignore[attr-defined]


def _attribute_sites(node: ast.Attribute, parent: ast.AST | None) -> Iterator[tuple[str, _Edit]]:
    swap = _DTYPE_SWAPS.get(node.attr)
    if swap is not None:

        def set_dtype(n: ast.AST, _: dict[int, ast.AST], new: str = swap) -> None:
            n.attr = new  # type: ignore[attr-defined]

        yield "dtype", set_dtype
    if node.attr == "T" and not isinstance(parent, ast.Call):
        yield "transpose", lambda n, p: _replace(n, n.value, p)  # type: ignore[attr-defined]


def python_mutants(source: str) -> list[Mutant]:
    """Every distinct mutant of ``source`` that still compiles."""
    base = ast.parse(source)
    baseline = ast.unparse(base)
    nodes = list(ast.walk(base))
    parents = _parents(base)
    inert = _inert(base)
    seen = {baseline}
    found: list[Mutant] = []
    for index, node in enumerate(nodes):
        if id(node) in inert:
            continue
        for op, edit in _sites(node, parents.get(id(node))):
            tree = ast.parse(source)
            target = list(ast.walk(tree))[index]
            try:
                edit(target, _parents(tree))
                code = ast.unparse(tree)
                compile(code, "submission.py", "exec")
            except (ValueError, SyntaxError, TypeError, AttributeError, IndexError):
                continue
            if code in seen:
                continue
            seen.add(code)
            line = getattr(node, "lineno", 0)
            found.append(Mutant(op=op, line=line, source=code, change=_change(baseline, code)))
    return found


def _change(before: str, after: str) -> str:
    """A one-line description of a mutation, for the report and for triage."""
    diff = difflib.unified_diff(before.splitlines(), after.splitlines(), lineterm="", n=0)
    changed = [line for line in diff if line[:1] in "+-" and line[:3] not in ("+++", "---")]
    return " | ".join(line[:140] for line in changed[:4])


# --- SQL mutation operators ---------------------------------------------------------------------

_SQL_RULES: tuple[tuple[str, str, str], ...] = (
    ("join", r"\bLEFT\s+JOIN\b", "JOIN"),
    ("join", r"\bINNER\s+JOIN\b", "LEFT JOIN"),
    ("distinct", r"\bDISTINCT\s+", ""),
    ("window", r"\bROW_NUMBER\b", "RANK"),
    ("window", r"\bDENSE_RANK\b", "RANK"),
    ("window", r"(?<![_A-Z])\bRANK\b", "DENSE_RANK"),
    ("window", r"\bLAG\b", "LEAD"),
    ("window", r"\bLEAD\b", "LAG"),
    ("window", r"\bFIRST_VALUE\b", "LAST_VALUE"),
    ("window", r"\bLAST_VALUE\b", "FIRST_VALUE"),
    ("frame", r"\bROWS\b", "RANGE"),
    ("frame", r"\bRANGE\b", "ROWS"),
    ("order", r"\bASC\b", "DESC"),
    ("order", r"\bDESC\b", "ASC"),
    ("order", r"\bNULLS\s+FIRST\b", "NULLS LAST"),
    ("order", r"\bNULLS\s+LAST\b", "NULLS FIRST"),
    ("aggregate", r"\bMIN\b", "MAX"),
    ("aggregate", r"\bMAX\b", "MIN"),
    ("aggregate", r"\bSUM\b", "AVG"),
    ("aggregate", r"\bAVG\b", "SUM"),
    ("set", r"\bUNION\s+ALL\b", "UNION"),
    ("boolean", r"\bAND\b", "OR"),
    ("boolean", r"\bOR\b", "AND"),
    ("null", r"\bIS\s+NOT\s+NULL\b", "IS NULL"),
    ("null", r"(?<!NOT )\bIS\s+NULL\b", "IS NOT NULL"),
    ("compare", r"<=", "<"),
    ("compare", r">=", ">"),
    ("compare", r"(?<![<>!=])<(?![=>])", "<="),
    ("compare", r"(?<![<>!=-])>(?![=])", ">="),
    ("compare", r"(?<![<>!=])=(?![=>])", "<>"),
)
_SQL_NUMBER = re.compile(r"(?<![\w.])\d+(?![\w.])")
_SQL_PLAIN_JOIN = re.compile(r"\bJOIN\b", re.IGNORECASE)
_SQL_JOIN_KINDS = ("LEFT", "RIGHT", "FULL", "CROSS", "INNER", "OUTER", "SEMI", "ANTI", "ASOF")


def mask_sql(source: str) -> str:
    """Blank string literals and comments, keeping offsets, so rules only match real tokens."""

    def blank(match: re.Match[str]) -> str:
        return " " * len(match.group())

    return re.sub(r"'(?:[^']|'')*'|--[^\n]*|/\*.*?\*/", blank, source, flags=re.DOTALL)


def _closing_scan(masked: str, start: int) -> tuple[int, int | None]:
    """From just inside a ``(``: the matching ``)`` offset and the first top-level comma, if any."""
    depth, comma = 0, None
    for index in range(start, len(masked)):
        char = masked[index]
        if char == "(":
            depth += 1
        elif char == ")":
            if depth == 0:
                return index, comma
            depth -= 1
        elif char == "," and depth == 0 and comma is None:
            comma = index
    return len(masked), comma


def _sql_spans(masked: str) -> Iterator[tuple[str, int, int, str]]:
    """Every SQL edit as ``(op, start, end, replacement text)``."""
    for op, pattern, replacement in _SQL_RULES:
        for match in re.finditer(pattern, masked, re.IGNORECASE):
            yield op, match.start(), match.end(), replacement
    for match in _SQL_PLAIN_JOIN.finditer(masked):
        before = masked[: match.start()].rstrip().upper()
        if not before.endswith(_SQL_JOIN_KINDS):
            yield "join", match.start(), match.end(), "LEFT JOIN"
    for match in _SQL_NUMBER.finditer(masked):
        value = int(match.group())
        yield "constant", match.start(), match.end(), str(value + 1)
        if value > 0:
            yield "constant", match.start(), match.end(), str(value - 1)
    for match in re.finditer(r"\bCOALESCE\s*\(", masked, re.IGNORECASE):
        close, comma = _closing_scan(masked, match.end())
        if comma is not None:
            yield "null", match.start(), close + 1, f"\0{match.end()}:{comma}"
    for match in re.finditer(r"\bPARTITION\s+BY\b", masked, re.IGNORECASE):
        stop = re.compile(r"\b(ORDER\s+BY|ROWS|RANGE)\b|\)", re.IGNORECASE).search(
            masked, match.end()
        )
        if stop is not None:
            yield "window", match.start(), stop.start(), ""


def sql_mutants(source: str) -> list[Mutant]:
    """Every distinct mutant of a SQL solution. String literals and comments stay untouched."""
    masked = mask_sql(source)
    seen = {source}
    found: list[Mutant] = []
    for op, start, end, replacement in _sql_spans(masked):
        text = replacement
        if text.startswith("\0"):  # keep the first COALESCE argument
            first, last = (int(part) for part in text[1:].split(":"))
            text = source[first:last].strip()
        code = source[:start] + text + source[end:]
        if code in seen:
            continue
        seen.add(code)
        line = source.count("\n", 0, start) + 1
        change = f"{' '.join(source[start:end].split())[:60]} -> {text[:60]}"
        found.append(Mutant(op=op, line=line, source=code, change=change))
    return found


def select_mutants(mutants: Sequence[Mutant], problem_id: str, limit: int) -> list[Mutant]:
    """A repeatable sample of at most ``limit``: all guards, then round-robin across operators."""
    rng = random.Random(problem_id)  # noqa: S311 - a repeatable sample, not security
    by_op: dict[str, list[Mutant]] = defaultdict(list)
    for mutant in mutants:
        by_op[mutant.op].append(mutant)
    for group in by_op.values():
        rng.shuffle(group)
    chosen = list(by_op.pop("guard", []))[:limit]
    ops = sorted(by_op)
    while len(chosen) < limit and any(by_op[op] for op in ops):
        for op in ops:
            if by_op[op] and len(chosen) < limit:
                chosen.append(by_op[op].pop())
    return chosen


def mutants_for(problem: Problem, limit: int) -> list[Mutant]:
    """The sampled mutants of one reference that a learner could actually submit."""
    source = problem.solution.read_text(encoding="utf-8")
    raw = sql_mutants(source) if problem.language == "sql" else python_mutants(source)
    allowed = [mutant for mutant in raw if _passes_static(problem, mutant.source)]
    return select_mutants(allowed, problem.id, limit)


def _passes_static(problem: Problem, source: str) -> bool:
    with tempfile.TemporaryDirectory(prefix="audit-static-") as tmp:
        path = Path(tmp) / f"submission{problem.suffix}"
        path.write_text(source, encoding="utf-8")
        findings = check_submission(
            path,
            language="sql" if problem.language == "sql" else "python",
            entry=problem.entry,
            banned_imports=problem.banned,
        )
    return not findings


# --- Grading ------------------------------------------------------------------------------------


def grade_source(
    problem: Problem, source: str, settings: Settings, *, include_hidden: bool = True
) -> GradeReport:
    """Grade ``source`` on the real path, in a scratch repo root."""
    with tempfile.TemporaryDirectory(prefix="audit-") as tmp:
        root = Path(tmp)
        submission = root / f"submission{problem.suffix}"
        submission.write_text(source, encoding="utf-8")
        scratch = settings.model_copy(update={"repo_root": root})
        return attempt.grade_attempt(
            problem.path, submission, include_hidden=include_hidden, settings=scratch
        )


def classify_message(text: str) -> str:
    """How useful a learner-visible failure message is (ENGINEERING_STANDARDS §5.6)."""
    first = text.strip().splitlines()[0].strip() if text.strip() else ""
    if not first or first == NO_DETAIL:
        return "no_detail"
    if re.match(r"^[\w.]*(Error|Exception|Warning)\b", first):
        return "bare_exception"
    if first.startswith(("assert ", "E ", "Falsifying")):
        return "raw_assert"
    return "named"


def _first_failure(report: GradeReport) -> TestResult | None:
    failing = [test for test in report.tests if test.outcome in ("fail", "error")]
    hidden = [test for test in failing if test.file == "hidden"]
    choices = hidden or failing
    return choices[0] if choices else None


def verdict(report: GradeReport) -> str:
    """One word: survived, or how the mutant died."""
    if report.status is RunStatus.PASSED:
        return "survived"
    if report.status in (RunStatus.TIMEOUT, RunStatus.OOM, RunStatus.ERROR, RunStatus.REJECTED):
        return report.status.value
    public_failed = any(t.file == "public" and t.outcome != "pass" for t in report.tests)
    return "killed_public" if public_failed else "killed_hidden"


def record(report: GradeReport) -> dict[str, Any]:
    """The fields of a report the audit keeps."""
    failure = _first_failure(report)
    message = learner_message(failure) if failure is not None else ""
    return {
        "verdict": verdict(report),
        "status": report.status.value,
        "public": report.counts("public"),
        "hidden": report.counts("hidden"),
        "duration_ms": report.duration_ms,
        "failed": [t.name for t in report.tests if t.outcome != "pass"][:6],
        "message": message[:240],
        "message_kind": classify_message(message) if failure is not None else None,
    }


class Log:
    """Thread-safe append-only JSONL, keyed for resuming."""

    def __init__(self, path: Path, key: Callable[[dict[str, Any]], str]) -> None:
        self.path = path
        self._key = key
        self._lock = threading.Lock()
        path.parent.mkdir(parents=True, exist_ok=True)
        self.done = {key(row) for row in read_jsonl(path)}

    def write(self, row: dict[str, Any]) -> None:
        with self._lock, self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row) + "\n")
            self.done.add(self._key(row))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _run_all(jobs: Iterable[Callable[[], None]], workers: int) -> int:
    """Run ``jobs`` on a thread pool (each one spawns its own process). Returns the number run."""
    count = 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for future in as_completed([pool.submit(job) for job in jobs]):
            future.result()
            count += 1
            if count % 100 == 0:
                print(f"  {count} done", flush=True)
    return count


# --- Commands -----------------------------------------------------------------------------------


def cmd_baseline(problems: Sequence[Problem], out: Path, workers: int) -> int:
    """The unmodified reference must pass through this tool, or no score can be trusted."""
    settings = Settings()
    log = Log(out / "baseline.jsonl", lambda row: row["problem"])

    def job(problem: Problem) -> Callable[[], None]:
        def run() -> None:
            source = problem.solution.read_text(encoding="utf-8")
            report = grade_source(problem, source, settings)
            log.write({"problem": problem.id, **record(report)})

        return run

    _run_all([job(p) for p in problems if p.id not in log.done], workers)
    rows = read_jsonl(log.path)
    bad = [row for row in rows if row["verdict"] != "survived"]
    print(f"baseline: {len(rows) - len(bad)}/{len(rows)} references pass")
    for row in bad:
        print(f"  {row['problem']}: {row['verdict']} {row['failed']} {row['message']}")
    return 1 if bad else 0


def cmd_mutate(problems: Sequence[Problem], out: Path, workers: int, limit: int) -> int:
    """Grade every sampled mutant, then re-run timeouts alone so load cannot fake a kill."""
    settings = Settings()
    log = Log(out / "mutants.jsonl", lambda row: f"{row['problem']}:{row['key']}")

    def job(problem: Problem, mutant: Mutant) -> Callable[[], None]:
        def run() -> None:
            report = grade_source(problem, mutant.source, settings)
            log.write(
                {
                    "problem": problem.id,
                    "key": mutant.key,
                    "op": mutant.op,
                    "line": mutant.line,
                    "change": mutant.change,
                    **record(report),
                }
            )

        return run

    jobs: list[Callable[[], None]] = []
    index: dict[str, tuple[Problem, Mutant]] = {}
    for problem in problems:
        for mutant in mutants_for(problem, limit):
            index[f"{problem.id}:{mutant.key}"] = (problem, mutant)
            if f"{problem.id}:{mutant.key}" not in log.done:
                jobs.append(job(problem, mutant))
    print(f"mutate: {len(index)} mutants, {len(jobs)} to run", flush=True)
    _run_all(jobs, workers)
    return _recheck_timeouts(out, index, settings)


def _recheck_timeouts(
    out: Path, index: dict[str, tuple[Problem, Mutant]], settings: Settings
) -> int:
    """A timeout under a loaded machine may be a slow survivor: grade it again almost alone."""
    log = Log(out / "mutants_rechecked.jsonl", lambda row: f"{row['problem']}:{row['key']}")
    rows = [row for row in read_jsonl(out / "mutants.jsonl") if row["verdict"] == "timeout"]
    todo = [r for r in rows if f"{r['problem']}:{r['key']}" not in log.done]

    def job(row: dict[str, Any]) -> Callable[[], None]:
        def run() -> None:
            pair = index.get(f"{row['problem']}:{row['key']}")
            if pair is None:
                return
            report = grade_source(pair[0], pair[1].source, settings)
            log.write({**row, **record(report)})

        return run

    print(f"recheck: {len(todo)} timeouts", flush=True)
    _run_all([job(row) for row in todo], 3)
    return 0


_PRECISION_WRAPPER = """

def _audit_scale(value):
    if isinstance(value, bool) or isinstance(value, int) or value is None:
        return value
    if isinstance(value, float):
        return value * {factor!r}
    if isinstance(value, tuple) and hasattr(value, "_fields"):
        return type(value)(*(_audit_scale(item) for item in value))
    if isinstance(value, (list, tuple)):
        return type(value)(_audit_scale(item) for item in value)
    if isinstance(value, dict):
        return {{key: _audit_scale(item) for key, item in value.items()}}
    kind = getattr(getattr(value, "dtype", None), "kind", None)
    floating = kind == "f" or (
        kind is None and callable(getattr(value, "is_floating_point", None))
        and value.is_floating_point()
    )
    return value * {factor!r} if floating else value


_audit_reference = {entry}


def {entry}(*args, **kwargs):
    return _audit_scale(_audit_reference(*args, **kwargs))
"""


def scaled_solution(problem: Problem, epsilon: float) -> str | None:
    """The reference with its returned floats scaled by ``1 + epsilon``, if that is possible."""
    source = problem.solution.read_text(encoding="utf-8")
    if problem.language == "sql":
        return None
    tree = ast.parse(source)
    if not any(isinstance(n, ast.FunctionDef) and n.name == problem.entry for n in tree.body):
        return None  # a class entry has no single return value to scale
    return source + _PRECISION_WRAPPER.format(factor=1.0 + epsilon, entry=problem.entry)


def cmd_precision(problems: Sequence[Problem], out: Path, workers: int) -> int:
    """How small an error do the tests notice? Compare it with the declared tolerance."""
    settings = Settings()
    log = Log(out / "precision.jsonl", lambda row: f"{row['problem']}:{row['epsilon']}")

    def job(problem: Problem, epsilon: float, source: str) -> Callable[[], None]:
        def run() -> None:
            report = grade_source(problem, source, settings)
            log.write({"problem": problem.id, "epsilon": epsilon, **record(report)})

        return run

    jobs = []
    for problem in problems:
        for epsilon in PRECISION_EPSILONS:
            source = scaled_solution(problem, epsilon)
            if source is not None and f"{problem.id}:{epsilon}" not in log.done:
                jobs.append(job(problem, epsilon, source))
    print(f"precision: {len(jobs)} runs", flush=True)
    _run_all(jobs, workers)
    return 0


def _plain_bundle(problem: Problem, root: Path) -> None:
    """A directory where plain pytest can run the tests against the reference."""
    shutil.copy(problem.solution, root / f"submission{problem.suffix}")
    for name in ("test_public.py", "test_hidden.py"):
        shutil.copy(problem.path / name, root / name)
    if (problem.path / "fixtures").is_dir():
        shutil.copytree(problem.path / "fixtures", root / "fixtures")


def cmd_coverage(problems: Sequence[Problem], out: Path, workers: int) -> int:
    """Branch coverage of each Python reference: paths no test ever takes."""
    log = Log(out / "coverage.jsonl", lambda row: row["problem"])

    def job(problem: Problem) -> Callable[[], None]:
        def run() -> None:
            with tempfile.TemporaryDirectory(prefix="audit-cov-") as tmp:
                root = Path(tmp)
                _plain_bundle(problem, root)
                base = [sys.executable, "-m", "coverage"]
                test_args = ["-m", "pytest", "-q", "-p", "no:cacheprovider"]
                seed = ["--hypothesis-seed=0", "test_public.py", "test_hidden.py"]
                command = [*base, "run", "--branch", "--include=submission.py", *test_args, *seed]
                subprocess.run(command, cwd=root, capture_output=True, timeout=300, check=False)
                report = root / "coverage.json"
                subprocess.run(
                    [*base, "json", "-o", str(report), "--include=submission.py"],
                    cwd=root,
                    capture_output=True,
                    check=False,
                )
                log.write({"problem": problem.id, **_coverage_row(report)})

        return run

    python = [p for p in problems if p.language != "sql" and p.id not in log.done]
    print(f"coverage: {len(python)} runs", flush=True)
    _run_all([job(p) for p in python], workers)
    return 0


def _coverage_row(report: Path) -> dict[str, Any]:
    if not report.is_file():
        return {"percent": None, "missing_lines": [], "missing_branches": []}
    data = json.loads(report.read_text(encoding="utf-8"))
    files = data.get("files", {})
    if not files:
        return {"percent": None, "missing_lines": [], "missing_branches": []}
    entry = next(iter(files.values()))
    return {
        "percent": round(entry["summary"]["percent_covered"], 1),
        "missing_lines": entry.get("missing_lines", []),
        "missing_branches": entry.get("missing_branches", []),
    }


def cmd_seeds(problems: Sequence[Problem], out: Path, workers: int) -> int:
    """Run references under other Hypothesis seeds: brittleness that seed 0 hides."""
    log = Log(out / "seeds.jsonl", lambda row: f"{row['problem']}:{row['seed']}")

    def job(problem: Problem, seed: int) -> Callable[[], None]:
        def run() -> None:
            with tempfile.TemporaryDirectory(prefix="audit-seed-") as tmp:
                root = Path(tmp)
                _plain_bundle(problem, root)
                command = [
                    sys.executable,
                    "-m",
                    "pytest",
                    "-q",
                    "--tb=line",
                    "-p",
                    "no:cacheprovider",
                    f"--hypothesis-seed={seed}",
                    "test_public.py",
                    "test_hidden.py",
                ]
                env = {
                    **os.environ,
                    "PYTHONHASHSEED": str(seed),
                    "DOGMA_DATA_SALT": str(seed),
                }
                try:
                    done = subprocess.run(
                        command,
                        cwd=root,
                        env=env,
                        capture_output=True,
                        text=True,
                        timeout=problem.timeout_s * 4,
                        check=False,
                    )
                    failed = re.findall(r"^FAILED (\S+)", done.stdout, re.MULTILINE)
                    passed = done.returncode == 0
                except subprocess.TimeoutExpired:
                    failed, passed = ["<timeout>"], False
                log.write(
                    {"problem": problem.id, "seed": seed, "passed": passed, "failed": failed[:6]}
                )

        return run

    jobs = [
        job(problem, seed)
        for problem in problems
        for seed in SEEDS
        if f"{problem.id}:{seed}" not in log.done
    ]
    print(f"seeds: {len(jobs)} runs", flush=True)
    _run_all(jobs, workers)
    bad = [row for row in read_jsonl(log.path) if not row["passed"]]
    print(f"seeds: {len(bad)} failing runs on {len({row['problem'] for row in bad})} problems")
    return 0


def cmd_bundle(problems: Sequence[Problem], out: Path) -> int:
    """Blind bundles: what a learner sees, and never the solution, hidden tests or hints."""
    blind = out / "blind"
    batches: dict[str, list[str]] = defaultdict(list)
    for problem in problems:
        target = blind / problem.id
        target.mkdir(parents=True, exist_ok=True)
        files = ["problem.md", f"starter{problem.suffix}", "test_public.py"]
        for name in files:
            shutil.copy(problem.path / name, target / name)
        if (problem.path / "fixtures").is_dir():
            shutil.copytree(problem.path / "fixtures", target / "fixtures", dirs_exist_ok=True)
        batches[problem.track].append(problem.id)
    numbered: dict[str, list[str]] = {}
    for track, ids in sorted(batches.items()):
        for start in range(0, len(ids), BLIND_BATCH):
            numbered[f"{track}-{start // BLIND_BATCH + 1:02d}"] = ids[start : start + BLIND_BATCH]
    (out / "batches.json").write_text(json.dumps(numbered, indent=1), encoding="utf-8")
    print(f"bundle: {len(problems)} problems in {len(numbered)} batches under {blind}")
    return 0


def cmd_public(problems: Sequence[Problem], problem_id: str, file: Path) -> int:
    """Run only the public tests, as often as a learner likes. Attempts are not counted."""
    match = [problem for problem in problems if problem.id == problem_id]
    if not match:
        print(f"no problem {problem_id}", file=sys.stderr)
        return 2
    problem = match[0]
    with tempfile.TemporaryDirectory(prefix="audit-") as tmp:
        root = Path(tmp)
        submission = root / f"submission{problem.suffix}"
        shutil.copy(file, submission)
        scratch = Settings().model_copy(update={"repo_root": root})
        report = attempt.grade_attempt(
            problem.path, submission, include_hidden=False, settings=scratch
        )
    render_report(Console(width=100), report, hidden=False)
    return 0


def cmd_submit(problems: Sequence[Problem], out: Path, problem_id: str, file: Path) -> int:
    """Grade a blind solution and print only what a learner would see."""
    match = [problem for problem in problems if problem.id == problem_id]
    if not match:
        print(f"no problem {problem_id}", file=sys.stderr)
        return 2
    log_path = out / "blind.jsonl"
    earlier = [row for row in read_jsonl(log_path) if row["problem"] == problem_id]
    if len(earlier) >= MAX_BLIND_ATTEMPTS or any(r["verdict"] == "survived" for r in earlier):
        print(f"no attempts left for {problem_id} (used {len(earlier)})")
        return 3
    source = file.read_text(encoding="utf-8")
    report = grade_source(match[0], source, Settings())
    render_report(Console(width=100), report, hidden=True)
    if report.static:
        print("(a rejected file still uses one attempt)")
    row = {"problem": problem_id, "attempt": len(earlier) + 1, **record(report)}
    row["source_sha"] = hashlib.sha256(source.encode()).hexdigest()[:16]
    row["failed_all"] = [t.name for t in report.tests if t.outcome != "pass"]
    Log(log_path, lambda r: f"{r['problem']}:{r['attempt']}").write(row)
    left = MAX_BLIND_ATTEMPTS - row["attempt"]
    print(f"attempts left: {0 if report.passed else left}")
    return 0


# Equivalent almost by construction: np.asarray only differs from np.array if code then writes in
# place, and that is what the input-mutation tests already catch.
_LOW_VALUE_OPS = frozenset({"array_copy"})
_TRIAGE_ORDER = (
    "guard",
    "compare",
    "boolean",
    "call",
    "drop_call",
    "flow",
    "bool",
    "axis",
    "dtype",
)


def triage_rank(row: dict[str, Any]) -> tuple[int, str]:
    """Contract-facing mutants first, then the rest, so a per-problem cap keeps the useful ones."""
    op = row["op"]
    rank = _TRIAGE_ORDER.index(op) if op in _TRIAGE_ORDER else len(_TRIAGE_ORDER)
    return rank, row["key"]


def cmd_survivors(problems: Sequence[Problem], out: Path, limit: int, cap: int) -> int:
    """Write each surviving mutant to disk for triage, with a batch file for the triage agents."""
    by_problem: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in effective_rows(out):
        if row["verdict"] == "survived" and row["op"] not in _LOW_VALUE_OPS:
            by_problem[row["problem"]].append(row)
    root = out / "triage"
    ids: list[str] = []
    for problem in problems:
        rows = sorted(by_problem.get(problem.id, []), key=triage_rank)[:cap]
        if not rows:
            continue
        ids.append(problem.id)
        target = root / problem.id
        target.mkdir(parents=True, exist_ok=True)
        wanted = {row["key"] for row in rows}
        sources = {m.key: m.source for m in mutants_for(problem, limit) if m.key in wanted}
        kept = []
        for row in rows:
            if row["key"] in sources:
                path = target / f"mutant_{row['key']}{problem.suffix}"
                path.write_text(sources[row["key"]], encoding="utf-8")
                kept.append({k: row[k] for k in ("key", "op", "line", "change")})
        (target / "survivors.json").write_text(json.dumps(kept, indent=1), encoding="utf-8")
    batches = {
        f"t-{start // BLIND_BATCH + 1:02d}": ids[start : start + BLIND_BATCH]
        for start in range(0, len(ids), BLIND_BATCH)
    }
    (out / "triage_batches.json").write_text(json.dumps(batches, indent=1), encoding="utf-8")
    print(f"survivors: {len(ids)} problems in {len(batches)} batches under {root}")
    return 0


def _with_extra_tests(problem: Problem, extra: str, root: Path) -> Problem:
    """A copy of ``problem`` whose hidden tests also hold ``extra``, for the real grader to run."""
    copied = root / "problem"
    shutil.copytree(problem.path, copied, ignore=shutil.ignore_patterns("__pycache__"))
    hidden = copied / "test_hidden.py"
    hidden.write_text(hidden.read_text(encoding="utf-8") + "\n\n" + extra, encoding="utf-8")
    return Problem(
        problem.id, copied, problem.language, problem.entry, problem.timeout_s, problem.banned
    )


def cmd_witness(problems: Sequence[Problem], out: Path, problem_id: str, file: Path) -> int:
    """Test proposed extra hidden tests. A witness counts only if the tool sees it work.

    The tests must pass on the reference and must fail on a survivor. Nothing is trusted from the
    proposer: this runs both through the real grader and reports which survivors are now killed.
    """
    match = [problem for problem in problems if problem.id == problem_id]
    if not match:
        print(f"no problem {problem_id}", file=sys.stderr)
        return 2
    problem = match[0]
    extra = file.read_text(encoding="utf-8")
    target = out / "triage" / problem.id
    survivors = json.loads((target / "survivors.json").read_text(encoding="utf-8"))
    settings = Settings()
    with tempfile.TemporaryDirectory(prefix="audit-witness-") as tmp:
        extended = _with_extra_tests(problem, extra, Path(tmp))
        reference = grade_source(extended, problem.solution.read_text(encoding="utf-8"), settings)
        if reference.status is not RunStatus.PASSED:
            print(
                f"reference FAILED your tests: {reference.status.value} {reference.message[:300]}"
            )
            for test in reference.tests:
                if test.outcome != "pass":
                    print(f"  {test.name}: {test.message[:300]}")
            return 1
        print(f"reference passes: {reference.counts('hidden')}")
        blind_file = out / "blind" / problem.id / f"solution{problem.suffix}"
        blind_ok = True
        if blind_file.is_file():
            # An independent solution that passed before must still pass, or the new test is
            # over-fitted to the reference and would reject a correct learner.
            independent = grade_source(extended, blind_file.read_text(encoding="utf-8"), settings)
            blind_ok = independent.status is RunStatus.PASSED
            print(f"independent solution: {'passes' if blind_ok else 'FAILS'}")
            for test in independent.tests:
                if test.outcome != "pass":
                    print(f"  over-fitted? {test.name}: {learner_message(test)[:160]}")
        killed: list[str] = []
        for row in survivors:
            source = (target / f"mutant_{row['key']}{problem.suffix}").read_text(encoding="utf-8")
            report = grade_source(extended, source, settings)
            result = verdict(report)
            print(f"  {row['key']} {row['op']:12} {result}")
            if result != "survived":
                killed.append(row["key"])
    entry = {
        "problem": problem.id,
        "file": str(file),
        "killed": killed,
        "total": len(survivors),
        "blind_ok": blind_ok,
    }
    Log(out / "witness.jsonl", lambda r: f"{r['problem']}:{r['file']}").write(entry)
    print(f"killed {len(killed)} of {len(survivors)}")
    return 0 if blind_ok else 1


# --- Realism: do the new realism tests catch a realistic wrong solution? -----------------------

_TEST_FILES = ("test_public.py", "test_hidden.py")
_CLEAN_VERDICTS = frozenset({"survived", "killed_public", "killed_hidden"})


@dataclass(frozen=True)
class Suite:
    """The two test files of one problem. ``None`` means the file is absent."""

    public: str | None
    hidden: str | None


@dataclass(frozen=True)
class _Check:
    """One file to grade: the reference, a blind solution, or a wrong one with its base suite."""

    kind: str
    path: Path
    against: Suite | None = None


class _UsageError(Exception):
    """An argument or a suite that cannot be graded; the message says why."""


def working_suite(problem: Problem) -> Suite:
    """The test files as they are in the working tree (the "new" suite)."""
    return Suite(*(_read_or_none(problem.path / name) for name in _TEST_FILES))


def base_suite(problem: Problem, rev: str) -> Suite:
    """The test files as they are at git revision ``rev`` (the "base" suite)."""
    rel = problem.path.relative_to(REPO).as_posix()
    return Suite(*(_git_show(rev, f"{rel}/{name}") for name in _TEST_FILES))


def _read_or_none(path: Path) -> str | None:
    return path.read_text(encoding="utf-8") if path.is_file() else None


def _git(*args: str) -> subprocess.CompletedProcess[bytes]:
    return subprocess.run(["git", "-C", str(REPO), *args], capture_output=True, check=False)


def _git_show(rev: str, path: str) -> str | None:
    done = _git("show", f"{rev}:{path}")
    return done.stdout.decode("utf-8") if done.returncode == 0 else None


def _suite_sha(suite: Suite) -> str:
    return hashlib.sha256(json.dumps([suite.public, suite.hidden]).encode()).hexdigest()[:16]


def _text_sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:16]


def grade_suite(problem: Problem, source: str, suite: Suite, settings: Settings) -> GradeReport:
    """Grade ``source`` through the real grader against ``suite``, in a scratch copy of the problem.

    A missing hidden file means the suite has no hidden tests, so only the public file runs.
    """
    with tempfile.TemporaryDirectory(prefix="audit-realism-") as tmp:
        copied = Path(tmp) / "problem"
        shutil.copytree(problem.path, copied, ignore=shutil.ignore_patterns("__pycache__"))
        for name, text in zip(_TEST_FILES, (suite.public, suite.hidden), strict=True):
            target = copied / name
            if text is None:
                target.unlink(missing_ok=True)
            else:
                target.write_text(text, encoding="utf-8")
        scoped = replace(problem, path=copied)
        return grade_source(scoped, source, settings, include_hidden=suite.hidden is not None)


def realism_verdict(base: str, new: str) -> str:
    """``killed``, ``already_caught`` or ``survives``; ``error`` if a run was neither pass nor fail.

    ``base`` and ``new`` are :func:`verdict` values of one wrong file on the two suites.
    """
    if base not in _CLEAN_VERDICTS or new not in _CLEAN_VERDICTS:
        return "error"
    if base != "survived":
        return "already_caught"
    return "killed" if new != "survived" else "survives"


def _realism_setup(
    problems: Sequence[Problem], words: Sequence[str], base: str, blind: Path | None
) -> tuple[Problem, list[Path], Suite, Suite]:
    """Check the arguments and load both suites, or raise ``_UsageError`` saying why not."""
    if len(words) < 2:
        raise _UsageError("usage: realism [--base REV] [--blind FILE] PROBLEM_ID WRONG [WRONG ...]")
    problem_id, *wrong_names = words
    match = [problem for problem in problems if problem.id == problem_id]
    if not match:
        raise _UsageError(f"no problem {problem_id}")
    wrongs = [Path(name) for name in wrong_names]
    for path in [*wrongs, *([blind] if blind is not None else [])]:
        if not path.is_file():
            raise _UsageError(f"no such file: {path}")
    if _git("rev-parse", "--verify", "--quiet", f"{base}^{{commit}}").returncode != 0:
        raise _UsageError(f"unknown git revision {base!r}")
    problem = match[0]
    new = working_suite(problem)
    if new.public is None or new.hidden is None:
        raise _UsageError(f"{problem.id}: the working tree needs test_public.py and test_hidden.py")
    old = base_suite(problem, base)
    if old.public is None:
        raise _UsageError(f"{problem.id}: {base} has no test_public.py for this problem")
    return problem, wrongs, new, old


def _grade_check(
    problem: Problem, source: str, new: Suite, against: Suite | None
) -> dict[str, Any]:
    """The verdicts of one file on the new suite and, for wrong files, on the base suite."""
    settings = Settings()
    new_row = record(grade_suite(problem, source, new, settings))
    fields: dict[str, Any] = {
        "new": new_row["verdict"],
        "new_message": new_row["message"],
        "base": None,
        "base_message": "",
        "verdict": new_row["verdict"],
    }
    if against is not None:
        base_row = record(grade_suite(problem, source, against, settings))
        fields.update(
            base=base_row["verdict"],
            base_message=base_row["message"],
            verdict=realism_verdict(base_row["verdict"], new_row["verdict"]),
        )
    return fields


def _run_check(
    problem: Problem, check: _Check, new: Suite, log: Log, cached: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    """Grade one check, or reuse its row when the file and both suites are unchanged."""
    source = check.path.read_text(encoding="utf-8")
    against_sha = _suite_sha(check.against) if check.against is not None else "-"
    key = f"{check.kind}:{problem.id}:{_text_sha(source)}:{_suite_sha(new)}:{against_sha}"
    if key not in cached:
        cached[key] = {
            "key": key,
            "kind": check.kind,
            "problem": problem.id,
            "file": check.path.name,
            "file_sha": _text_sha(source),
            "new_sha": _suite_sha(new),
            "base_sha": against_sha,
            **_grade_check(problem, source, new, check.against),
        }
        log.write(cached[key])
    return cached[key]


def _realism_failures(rows: Sequence[dict[str, Any]]) -> list[str]:
    failures = []
    for row in rows:
        if row["kind"] == "wrong" and row["verdict"] != "killed":
            failures.append(f"{row['file']}: {row['verdict']} (a wrong solution must be killed)")
        elif row["kind"] != "wrong" and row["verdict"] != "survived":
            failures.append(f"{row['kind']} {row['file']}: {row['verdict']} on the new tests")
    return failures


def cmd_realism(
    problems: Sequence[Problem],
    out: Path,
    words: Sequence[str],
    base: str,
    blind: Path | None,
) -> int:
    """Check that the new realism tests kill each realistic wrong solution the base missed."""
    try:
        problem, wrongs, new, old = _realism_setup(problems, words, base, blind)
    except _UsageError as exc:
        print(exc, file=sys.stderr)
        return 2
    checks = [_Check("reference", problem.solution)]
    if blind is not None:
        checks.append(_Check("blind", blind))
    checks.extend(_Check("wrong", path, old) for path in wrongs)
    log = Log(out / "realism.jsonl", lambda row: row["key"])
    cached = {row["key"]: row for row in read_jsonl(log.path)}
    rows = [_run_check(problem, check, new, log, cached) for check in checks]

    print(f"{'check':<10} {'file':<28} {'base':<16} {'new':<16} verdict")
    for row in rows:
        print(
            f"{row['kind']:<10} {row['file']:<28} {row['base'] or '-':<16} "
            f"{row['new']:<16} {row['verdict']}"
        )
    failures = _realism_failures(rows)
    if failures:
        print("realism: FAILED")
        for reason in failures:
            print(f"  {reason}")
        return 1
    print(f"realism: all {len(wrongs)} wrong solutions are killed by the new tests")
    return 0


def effective_rows(out: Path) -> list[dict[str, Any]]:
    """Mutant rows, with timeouts replaced by their quieter re-run where there is one."""
    quiet = read_jsonl(out / "mutants_rechecked.jsonl")
    rechecked = {f"{row['problem']}:{row['key']}": row for row in quiet}
    rows = read_jsonl(out / "mutants.jsonl")
    return [rechecked.get(f"{row['problem']}:{row['key']}", row) for row in rows]


def cmd_summary(out: Path) -> int:
    """Print the headline numbers from whatever has been collected so far."""
    rows = effective_rows(out)
    if rows:
        tally = Counter(row["verdict"] for row in rows)
        total = len(rows)
        print(f"mutants {total}: {dict(tally)}")
        print(f"killed {100 * (total - tally['survived']) / total:.1f}%")
        print("hidden-only kills", tally["killed_hidden"])
        tracks: dict[str, Counter[str]] = defaultdict(Counter)
        for row in rows:
            tracks[row["problem"].split("/")[0]][row["verdict"]] += 1
        for track, counts in sorted(tracks.items()):
            size = sum(counts.values())
            print(f"  {track:6} {size:6} killed {100 * (size - counts['survived']) / size:5.1f}%")
        ops: dict[str, Counter[str]] = defaultdict(Counter)
        for row in rows:
            ops[row["op"]][row["verdict"]] += 1
        for op, counts in sorted(ops.items(), key=lambda item: item[0]):
            size = sum(counts.values())
            print(f"  op {op:15} {size:6} survived {counts['survived']:5}")
    return 0


def _triage_rows(out: Path) -> dict[str, list[dict[str, Any]]]:
    """Survivor verdicts per problem, from the triage directory."""
    found: dict[str, list[dict[str, Any]]] = {}
    root = out / "triage"
    for path in root.rglob("triage.json"):
        found[str(path.parent.relative_to(root))] = json.loads(path.read_text(encoding="utf-8"))
    return found


def _precision_label(rows: Sequence[dict[str, Any]]) -> str:
    caught = [row["epsilon"] for row in rows if row["verdict"] != "survived"]
    if not rows:
        return "n/a"
    return f"{min(caught):.0e}" if caught else "n/a"


def scorecard_rows(out: Path) -> list[dict[str, Any]]:
    """One row per problem, from every result file, so the report is generated and not typed."""
    mutants: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in effective_rows(out):
        mutants[row["problem"]].append(row)
    coverage = {row["problem"]: row for row in read_jsonl(out / "coverage.jsonl")}
    blind: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in read_jsonl(out / "blind.jsonl"):
        blind[row["problem"]].append(row)
    seeds: Counter[str] = Counter(
        row["problem"] for row in read_jsonl(out / "seeds.jsonl") if not row["passed"]
    )
    precision: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in read_jsonl(out / "precision.jsonl"):
        precision[row["problem"]].append(row)
    triage = _triage_rows(out)
    rows = []
    for problem in sorted(mutants):
        verdicts = Counter(row["verdict"] for row in triage.get(problem, []))
        mine = mutants[problem]
        survived = sum(row["verdict"] == "survived" for row in mine)
        cover = coverage.get(problem, {}).get("percent")
        attempts = len(blind.get(problem, []))
        rows.append(
            {
                "problem": problem,
                "mutants": len(mine),
                "killed_raw": len(mine) - survived,
                "hidden_only": sum(row["verdict"] == "killed_hidden" for row in mine),
                "survived": survived,
                "gaps": verdicts["gap"],
                "equivalent": verdicts["equivalent"],
                "unpromised": verdicts["unpromised"],
                "coverage": cover,
                "attempts": attempts,
                "seed_failures": seeds[problem],
                "precision": _precision_label(precision.get(problem, [])),
            }
        )
    return rows


def _pct(part: int, whole: int) -> str:
    return f"{100 * part / whole:.1f}" if whole else "n/a"


def cmd_report(out: Path) -> int:
    """Print the per-track and per-problem tables as Markdown."""
    rows = scorecard_rows(out)
    tracks: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        tracks[row["problem"].split("/")[0]].append(row)
    head = "| Track | Problems | Mutants | Raw killed % | Real gaps | First-try blind solves |"
    print(head)
    print("|---|---:|---:|---:|---:|---:|")
    for track, group in sorted(tracks.items()):
        mutants = sum(row["mutants"] for row in group)
        killed = sum(row["killed_raw"] for row in group)
        first = sum(row["attempts"] == 1 for row in group)
        gaps = sum(row["gaps"] for row in group)
        print(
            f"| {track} | {len(group)} | {mutants} | {_pct(killed, mutants)} | {gaps} | "
            f"{first}/{len(group)} |"
        )
    print()
    print(
        "| Problem | Mutants | Raw killed % | Hidden-only | Gaps | Equiv. | Coverage % | Blind "
        "| Seeds |"
    )
    print("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for row in rows:
        cover = "sql" if row["coverage"] is None else row["coverage"]
        print(
            f"| `{row['problem']}` | {row['mutants']} | {_pct(row['killed_raw'], row['mutants'])} "
            f"| {row['hidden_only']} | {row['gaps']} | {row['equivalent']} | {cover} | "
            f"{row['attempts']} | {row['seed_failures']}/20 |"
        )
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "command",
        choices=[
            "baseline",
            "mutate",
            "precision",
            "coverage",
            "seeds",
            "bundle",
            "public",
            "report",
            "submit",
            "survivors",
            "summary",
            "witness",
            "realism",
        ],
    )
    parser.add_argument("args", nargs="*", help="submit: PROBLEM_ID FILE; realism: ID WRONG...")
    parser.add_argument("--base", default="HEAD", help="realism: git revision of the base tests")
    parser.add_argument("--blind", type=Path, default=None, help="realism: a blind solution")
    parser.add_argument("--out", type=Path, required=True, help="directory for results")
    parser.add_argument("--only", action="append", default=[], help="problem id prefix")
    parser.add_argument("--workers", type=int, default=10)
    parser.add_argument("--limit", type=int, default=MAX_MUTANTS, help="mutants per problem")
    parser.add_argument("--cap", type=int, default=15, help="survivors per problem to triage")
    args = parser.parse_args(argv)
    out: Path = args.out
    out.mkdir(parents=True, exist_ok=True)
    problems = discover(REPO, args.only)
    commands: dict[str, Callable[[], int]] = {
        "baseline": lambda: cmd_baseline(problems, out, args.workers),
        "mutate": lambda: cmd_mutate(problems, out, args.workers, args.limit),
        "precision": lambda: cmd_precision(problems, out, args.workers),
        "coverage": lambda: cmd_coverage(problems, out, args.workers),
        "seeds": lambda: cmd_seeds(problems, out, args.workers),
        "bundle": lambda: cmd_bundle(problems, out),
        "public": lambda: cmd_public(problems, args.args[0], Path(args.args[1])),
        "submit": lambda: cmd_submit(problems, out, args.args[0], Path(args.args[1])),
        "survivors": lambda: cmd_survivors(problems, out, args.limit, args.cap),
        "witness": lambda: cmd_witness(problems, out, args.args[0], Path(args.args[1])),
        "realism": lambda: cmd_realism(problems, out, args.args, args.base, args.blind),
        "report": lambda: cmd_report(out),
        "summary": lambda: cmd_summary(out),
    }
    return commands[args.command]()


if __name__ == "__main__":
    raise SystemExit(main())
