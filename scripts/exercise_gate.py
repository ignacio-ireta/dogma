"""Gate for I-32 test strength: lint hidden tests, list clauses, probe a solution, keep notes.

Usage (the torch group is needed for the py-torch runtime)::

    uv run --frozen --group torch python scripts/exercise_gate.py COMMAND ...

    lint     [--only PREFIX]... --out DIR
    clauses  --out DIR ID
    probe    --out DIR ID FILE
    note     --out DIR ID --kind KIND --text TEXT
    ledger   --out DIR [--all] [--kind KIND]...
    add-reject   --out DIR ID FILE --mistake TEXT --clause LABEL [--killed-by NAME]
                 [--message-has TEXT]
    add-accept   --out DIR ID FILE --origin blind|freedom [--freedom TEXT] [--clause LABEL]
    add-uncovered --out DIR ID --clause LABEL --reason TEXT
    remove-variant --out DIR ID NAME
    adopt-blind  --out DIR [--only PREFIX]...
    variants     --out DIR [--only PREFIX]... [--workers N]
    gate         ID --out DIR [--mode none|add|text|refactor] [--base REV] [--wrong FILE]...
    bump-version ID --out DIR [--base REV]
    add-test     ID TESTFILE --out DIR --wrong FILE --mistake TEXT --clause LABEL [--base REV]
    mark-swept   ID --out DIR --kind reject|mutants --mode none|add [--base REV]
    worklist     --out DIR --kind hygiene|blind|reject|mutants [--today YYYY-MM-DD]
    scorecard    --out DIR

Every sub-command takes ``--out DIR`` (created when missing). Files are written only under DIR,
or under a problem's variants/ directory for the commands that change a manifest.
Exit codes: 0 success, 1 a check failed, 2 wrong usage.
"""

from __future__ import annotations

import argparse
import ast
import difflib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from collections import Counter
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ValidationError

from dogma.content.loader import load_core_path, load_problem
from dogma.content.markdown import parse_outline
from dogma.content.schemas import (
    VARIANT_ACCEPT_PATTERN,
    VARIANT_REJECT_PATTERN,
    AcceptVariant,
    CorePath,
    PathWeek,
    RejectVariant,
    UncoveredClause,
    VariantsManifest,
)
from dogma.content.testlint import id_leaks, lint_hidden_tests
from dogma.content.validate import validate_variants
from dogma.core.config import Settings
from dogma.core.fsio import atomic_write
from dogma.grading.display import learner_message
from dogma.grading.models import GradeReport, RunStatus, TestResult

sys.path.insert(0, str(Path(__file__).resolve().parent))
import exercise_audit as audit

NOTE_KINDS: tuple[str, ...] = (
    "blocked",
    "question",
    "unpromised",
    "no-witness",
    "overreach",
    "reference-doubt",
    "message-defect",
    "inspect",
    "done",
)
_PLAIN = re.compile(r"[A-Za-z0-9 ,.;:()/+=<>_'-]+")
_PLAIN_REFUSAL = "use plain words: no backticks, quotes, dollar signs or backslashes"
_BULLET = re.compile(r"^[-*] ")
_LABEL_PREFIX = {"Contract": "C", "Constraints": "K"}
_FAILED = ("fail", "error")


class GateError(Exception):
    """A check failed: the command exits 1."""


class UsageError(Exception):
    """Wrong arguments, an unknown id or an unknown kind: the command exits 2."""


# --- Text helpers -------------------------------------------------------------------------------


def squash(text: str) -> str:
    return " ".join(text.split())


def _chunks(body: str) -> list[tuple[int, str]]:
    """Split a section body into numbered clauses. Index 0 is the text before the first bullet."""
    pieces: list[str] = []
    buffer: list[str] = []
    for line in body.splitlines():
        if _BULLET.match(line):
            pieces.append("\n".join(buffer))
            buffer = [line[2:]]
        else:
            buffer.append(line)
    pieces.append("\n".join(buffer))
    return [(index, squash(piece)) for index, piece in enumerate(pieces) if squash(piece)]


def contract_clauses(problem_md: str) -> list[tuple[str, str]]:
    """(label, text) for each clause of the Contract (C) and Constraints (K) sections."""
    found: list[tuple[str, str]] = []
    for section in parse_outline(problem_md).sections:
        prefix = _LABEL_PREFIX.get(section.heading)
        if prefix is None:
            continue
        found.extend((f"{prefix}{index}", text) for index, text in _chunks(section.text))
    return found


def clause_text(problem_md: str, label: str) -> str:
    for found, text in contract_clauses(problem_md):
        if found == label:
            return text
    raise GateError(f"no clause {label}: run clauses")


def plain_words(value: str) -> str:
    if not _PLAIN.fullmatch(value):
        raise GateError(_PLAIN_REFUSAL)
    return value


# --- Problems -----------------------------------------------------------------------------------


def find_problem(problem_id: str) -> audit.Problem:
    problems = audit.discover(audit.REPO, [])
    for problem in problems:
        if problem.id == problem_id:
            return problem
    close = difflib.get_close_matches(problem_id, [p.id for p in problems], n=3)
    detail = f"\n  close: {', '.join(close)}" if close else ""
    raise UsageError(f"no problem {problem_id}{detail}")


def _problem_md(problem: audit.Problem) -> str:
    path = problem.path / "problem.md"
    if not path.is_file():
        raise GateError(f"no problem.md for {problem.id}")
    return path.read_text(encoding="utf-8")


# --- lint ---------------------------------------------------------------------------------------


def cmd_lint(args: argparse.Namespace) -> int:
    rows: list[dict[str, Any]] = []
    for problem in audit.discover(audit.REPO, args.only):
        hidden = problem.path / "test_hidden.py"
        if not hidden.is_file():
            continue
        for finding in lint_hidden_tests(hidden.read_text(encoding="utf-8")):
            rows.append(
                {
                    "problem": problem.id,
                    "rule": finding.rule,
                    "line": finding.line,
                    "message": finding.message,
                }
            )
    for row in rows:
        print(f"{row['problem']}\t{row['rule']}\t{row['line']}\t{row['message']}")
    body = "".join(json.dumps(row) + "\n" for row in rows)
    (args.out / "lint.jsonl").write_text(body, encoding="utf-8")
    problems = len({row["problem"] for row in rows})
    print(f"lint: {len(rows)} findings in {problems} problems")
    return 1 if rows else 0


# --- clauses ------------------------------------------------------------------------------------


def cmd_clauses(args: argparse.Namespace) -> int:
    problem = find_problem(args.id)
    print(f"CLAUSES {problem.id}")
    for label, text in contract_clauses(_problem_md(problem)):
        print(f"{label}  {text}")
    print("Give a label (C2, K1) when a command asks for --clause.")
    return 0


# --- probe --------------------------------------------------------------------------------------


def probe_verdict(report: GradeReport) -> str:
    if report.status is RunStatus.REJECTED:
        reasons = "; ".join(finding.message for finding in report.static)
        return f"broken (rejected: {reasons})"
    if report.status in (RunStatus.TIMEOUT, RunStatus.OOM, RunStatus.ERROR):
        return f"infra ({report.status.value})"
    if report.status is RunStatus.PASSED:
        return "survived"
    if any(test.outcome == "pass" for test in report.tests):
        return "killed"
    return "broken (no test passed)"


def _failed_tests(report: GradeReport) -> list[TestResult]:
    return [test for test in report.tests if test.outcome in _FAILED]


def _failure_lines(report: GradeReport) -> list[str]:
    lines: list[str] = []
    for test in _failed_tests(report):
        message = learner_message(test)
        first = message.strip().splitlines()[0] if message.strip() else ""
        kind = audit.classify_message(message)
        lines.append(f"FAILED {test.file} {test.name}")
        lines.append(f"    kind={kind}  message: {first}")
    return lines


def _killed_by(report: GradeReport) -> str | None:
    failing = _failed_tests(report)
    public = [test for test in failing if test.file == "public"]
    chosen = (public or failing)[:1]
    if not chosen:
        return None
    test = chosen[0]
    return f"killed_by: {test.name.partition('[')[0]} ({test.file})"


def _print_probe(problem: audit.Problem, file: Path, report: GradeReport) -> str:
    """Print the probe block for one graded file and return its verdict."""
    public_passed, public_total = report.counts("public")
    hidden_passed, hidden_total = report.counts("hidden")
    print(f"PROBE {problem.id}  file={file}")
    print(
        f"status: {report.status.value}   public {public_passed}/{public_total}"
        f"   hidden {hidden_passed}/{hidden_total}"
    )
    for line in _failure_lines(report):
        print(line)
    verdict = probe_verdict(report)
    print(f"VERDICT: {verdict}")
    killed = _killed_by(report) if verdict == "killed" else None
    if killed is not None:
        print(killed)
    return verdict


def cmd_probe(args: argparse.Namespace) -> int:
    problem = find_problem(args.id)
    if not args.file.is_file():
        raise UsageError(f"no file {args.file}")
    source = args.file.read_text(encoding="utf-8")
    report = audit.grade_source(problem, source, Settings())
    _print_probe(problem, args.file, report)
    return 0


# --- note and ledger ----------------------------------------------------------------------------


def append_note(out: Path, problem_id: str, kind: str, text: str) -> None:
    row = {
        "ts": datetime.now(UTC).isoformat(timespec="seconds"),
        "problem": problem_id,
        "kind": kind,
        "text": text,
    }
    out.mkdir(parents=True, exist_ok=True)
    with (out / "ledger.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row) + "\n")


def _check_kinds(kinds: Sequence[str]) -> None:
    for kind in kinds:
        if kind not in NOTE_KINDS:
            raise UsageError(f"unknown kind {kind}: one of {', '.join(NOTE_KINDS)}")


def cmd_note(args: argparse.Namespace) -> int:
    _check_kinds([args.kind])
    problem = find_problem(args.id)
    text = plain_words(args.text)
    append_note(args.out, problem.id, args.kind, text)
    print(f"noted: {args.kind} {problem.id}")
    return 0


def cmd_ledger(args: argparse.Namespace) -> int:
    _check_kinds(args.kind)
    if args.all:
        paths = sorted(audit.REPO.glob("var/audit-i32/*/ledger.jsonl"))
    else:
        paths = [args.out / "ledger.jsonl"]
    rows = [row for path in paths for row in audit.read_jsonl(path)]
    if args.kind:
        rows = [row for row in rows if row["kind"] in args.kind]
    for kind in NOTE_KINDS:
        matching = [row for row in rows if row["kind"] == kind]
        if not matching:
            continue
        print(f"### {kind}")
        for row in matching:
            print(f"- `{row['problem']}`: {row['text']}")
        print()
    print(f"ledger: {len(rows)} rows")
    return 0


# --- manifest: reading, writing and rollback ----------------------------------------------------

_ACCEPT_BLIND = "accept_blind"
# A variant is evidence, not reference code: a blind solution with too many branches, a zip
# without strict= or a leftover variable is what a learner writes. Findings of every enabled
# rule outside _REAL_ERRORS are marked in the file with noqa, as is an unused variable. What
# stays refused: syntax errors, undefined names and the other pyflakes and import findings.
# The rules are named by exclusion on purpose: selecting families on the command line would
# switch rules back on that the repository's configuration ignores.
_REAL_ERRORS = "F,W,I,E4,E7,E9"
_UNUSED_VARIABLE = "F841"
_NOQA_ROUNDS = 2


def _variants_dir(problem: audit.Problem) -> Path:
    return problem.path / "variants"


def _manifest_path(problem: audit.Problem) -> Path:
    return _variants_dir(problem) / "manifest.yaml"


def _reason(exc: ValidationError) -> str:
    return str(exc.errors()[0]["msg"])


def _build[M: BaseModel](model: type[M], **fields: object) -> M:
    try:
        return model(**fields)
    except ValidationError as exc:
        raise GateError(_reason(exc)) from exc


def read_manifest(problem: audit.Problem) -> VariantsManifest | None:
    path = _manifest_path(problem)
    if not path.is_file():
        return None
    try:
        return VariantsManifest.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))
    except ValidationError as exc:
        raise GateError(f"{path} is not a valid manifest: {_reason(exc)}") from exc
    except yaml.YAMLError as exc:
        raise GateError(f"{path} is not valid YAML: {' '.join(str(exc).split())}") from exc


def write_manifest(problem: audit.Problem, manifest: VariantsManifest) -> None:
    text = yaml.safe_dump(
        manifest.model_dump(mode="json", exclude_none=True),
        sort_keys=False,
        allow_unicode=True,
        width=100,
    )
    atomic_write(_manifest_path(problem), text)


def _problem_version(problem: audit.Problem) -> int:
    meta = load_problem(problem.path).meta
    if meta is None:
        raise GateError(f"no meta.yaml for {problem.id}")
    return meta.version


def _manifest_or_new(problem: audit.Problem) -> VariantsManifest:
    manifest = read_manifest(problem)
    if manifest is None:
        return VariantsManifest(problem_version=_problem_version(problem))
    return manifest


def _listed(manifest: VariantsManifest) -> list[str]:
    return [variant.file for variant in (*manifest.accept, *manifest.reject)]


def _with_entry(manifest: VariantsManifest, field: str, entry: BaseModel) -> VariantsManifest:
    """The manifest with ``entry`` appended to ``field``. The model checks the result."""
    fields = {**dict(manifest), field: (*getattr(manifest, field), entry)}
    try:
        return VariantsManifest(**fields)
    except ValidationError as exc:
        raise GateError(_reason(exc)) from exc


def _without(manifest: VariantsManifest, name: str) -> VariantsManifest:
    fields = {
        **dict(manifest),
        "accept": tuple(v for v in manifest.accept if v.file != name),
        "reject": tuple(v for v in manifest.reject if v.file != name),
    }
    return VariantsManifest(**fields)


def failing_functions(report: GradeReport, file: str) -> list[str]:
    """Names of the test functions in ``file`` that failed, in report order, without repeats."""
    names: list[str] = []
    for test in _failed_tests(report):
        base = test.name.partition("[")[0]
        if test.file == file and base not in names:
            names.append(base)
    return names


def _failing_messages(report: GradeReport, file: str, function: str) -> list[str]:
    return [
        learner_message(test)
        for test in _failed_tests(report)
        if test.file == file and test.name.partition("[")[0] == function
    ]


class _Txn:
    """Every file one command touches, so that a command which fails half-way leaves none."""

    def __init__(self, problem: audit.Problem) -> None:
        self.problem = problem
        self._manifest = _manifest_path(problem)
        self._manifest_before = self._manifest.read_bytes() if self._manifest.is_file() else None
        self._dir_existed = _variants_dir(problem).is_dir()
        self._created: list[Path] = []
        self._removed: dict[Path, bytes] = {}

    def created(self, path: Path) -> None:
        self._created.append(path)

    def remove(self, path: Path) -> None:
        self._removed[path] = path.read_bytes()
        path.unlink()

    def rollback(self) -> None:
        for path in self._created:
            path.unlink(missing_ok=True)
        for path, data in self._removed.items():
            atomic_write(path, data)
        if self._manifest_before is None:
            self._manifest.unlink(missing_ok=True)
        else:
            atomic_write(self._manifest, self._manifest_before)
        directory = _variants_dir(self.problem)
        if not self._dir_existed and directory.is_dir() and not any(directory.iterdir()):
            directory.rmdir()


@contextmanager
def _transaction(problem: audit.Problem) -> Iterator[_Txn]:
    txn = _Txn(problem)
    try:
        yield txn
    except BaseException:
        txn.rollback()
        raise


# --- installing a variant -----------------------------------------------------------------------


def _ruff(step: tuple[str, ...], target: Path) -> tuple[bool, str]:
    """One ruff step on one file, with the repository configuration (see verify.run_ruff)."""
    pyproject = audit.REPO / "pyproject.toml"
    scope = ["--config", str(pyproject)] if pyproject.is_file() else ["--isolated"]
    proc = subprocess.run(
        [sys.executable, "-m", "ruff", *step, *scope, str(target)],
        capture_output=True,
        text=True,
        check=False,
        cwd=audit.REPO,
    )
    return proc.returncode == 0, (proc.stdout + proc.stderr).strip()


def install_variant(problem: audit.Problem, source: Path, name: str) -> Path:
    """Copy ``source`` to variants/``name``, tidied when it is Python. The caller records it."""
    if source.resolve().is_relative_to(problem.path.resolve()):
        raise UsageError("write candidates under <OUT>/candidates, not in variants/")
    target = _variants_dir(problem) / name
    if target.exists():
        raise GateError(f"variants/{name} already exists: remove it or choose another name")
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, target)
    if target.suffix == ".py":
        failure = _tidy(target)
        if failure is not None:
            target.unlink()
            raise GateError(f"ruff {failure[0]} failed on {name}:\n{failure[1]}")
    return target


def _tidy(target: Path) -> tuple[str, str] | None:
    """Format and lint one installed variant. Returns (step, output) for a failure, else None.

    Safe fixes are applied. Style findings that remain are marked with noqa (``_REAL_ERRORS``
    names what is never marked).
    """
    ok, output = _ruff(("format",), target)
    if not ok:
        return "format", output
    _ruff(("check", "--fix"), target)  # its exit code also counts what it could not fix
    for _ in range(_NOQA_ROUNDS):
        ok, output = _ruff(("check",), target)
        if ok:
            return None
        _ruff(("check", "--ignore", _REAL_ERRORS, "--add-noqa"), target)
        _ruff(("check", "--select", _UNUSED_VARIABLE, "--add-noqa"), target)
        _ruff(("format",), target)
    ok, output = _ruff(("check",), target)
    return None if ok else ("check", output)


# --- add-reject, add-accept, add-uncovered, remove-variant --------------------------------------


def _variant_name(problem: audit.Problem, name: str, pattern: str) -> None:
    if not re.fullmatch(pattern, name) or not name.endswith(problem.suffix):
        raise UsageError(f"bad variant file name {name}: match {pattern}, end in {problem.suffix}")


def _candidate(path: Path) -> None:
    if not path.is_file():
        raise UsageError(f"no file {path}")


def _pick_killer(report: GradeReport, requested: str | None) -> tuple[str, str]:
    """(visible, function) of the test that kills a variant. Public tests are preferred."""
    public = failing_functions(report, "public")
    hidden = failing_functions(report, "hidden")
    if requested is not None:
        if requested in public:
            return "public", requested
        if requested in hidden:
            return "hidden", requested
        failing = ", ".join(public + hidden) or "none"
        raise GateError(f"{requested} does not fail for this variant: failing tests are {failing}")
    if public:
        return "public", public[0]
    return "hidden", hidden[0]


def _add_reject(
    problem: audit.Problem,
    *,
    source: Path,
    name: str,
    mistake: str,
    clause_label: str,
    killed_by: str | None,
    message_has: str | None,
) -> tuple[RejectVariant, str]:
    with _transaction(problem) as txn:
        manifest = _manifest_or_new(problem)
        if name in _listed(manifest):
            raise GateError(f"{name} is already in manifest.yaml")
        clause = clause_text(_problem_md(problem), clause_label)
        target = install_variant(problem, source, name)
        txn.created(target)
        report = audit.grade_source(problem, target.read_text(encoding="utf-8"), Settings())
        verdict = probe_verdict(report)
        if verdict != "killed":
            _print_probe(problem, target, report)
            if verdict == "survived":
                raise GateError(
                    "SURVIVED: no test fails for this wrong solution "
                    "(see your runbook: candidate gap)"
                )
            raise GateError(f"NOT KILLED: the variant is {verdict}, not a wrong solution")
        visible, function = _pick_killer(report, killed_by)
        messages = _failing_messages(report, visible, function)
        if message_has is not None and not any(
            message_has.lower() in text.lower() for text in messages
        ):
            raise GateError(f"no failing case of {function} shows {message_has}")
        entry = _build(
            RejectVariant,
            file=name,
            mistake=mistake,
            clause=clause,
            killed_by=function,
            visible=visible,
            message_has=message_has,
        )
        write_manifest(problem, _with_entry(manifest, "reject", entry))
    return entry, audit.classify_message(messages[0])


def cmd_add_reject(args: argparse.Namespace) -> int:
    problem = find_problem(args.id)
    name = args.file.name
    _variant_name(problem, name, VARIANT_REJECT_PATTERN)
    _candidate(args.file)
    mistake = plain_words(args.mistake)
    message_has = plain_words(args.message_has) if args.message_has is not None else None
    entry, kind = _add_reject(
        problem,
        source=args.file,
        name=name,
        mistake=mistake,
        clause_label=args.clause,
        killed_by=args.killed_by,
        message_has=message_has,
    )
    print(
        f"added reject {name}: killed_by {entry.killed_by} ({entry.visible}), message kind {kind}"
    )
    if kind != "named":
        note = f"{name}: {entry.killed_by} shows kind={kind}"
        append_note(args.out, problem.id, "message-defect", note)
    return 0


def _add_accept(
    problem: audit.Problem,
    *,
    source: Path,
    name: str,
    origin: str,
    freedom: str | None,
    clause_label: str | None,
) -> AcceptVariant:
    with _transaction(problem) as txn:
        manifest = _manifest_or_new(problem)
        if name in _listed(manifest):
            raise GateError(f"{name} is already in manifest.yaml")
        clause = None if clause_label is None else clause_text(_problem_md(problem), clause_label)
        entry = _build(AcceptVariant, file=name, origin=origin, freedom=freedom, clause=clause)
        target = install_variant(problem, source, name)
        txn.created(target)
        report = audit.grade_source(problem, target.read_text(encoding="utf-8"), Settings())
        if probe_verdict(report) != "survived":
            _print_probe(problem, target, report)
            raise GateError("NOT ACCEPTED: a test fails for this solution (see your runbook)")
        write_manifest(problem, _with_entry(manifest, "accept", entry))
    return entry


def cmd_add_accept(args: argparse.Namespace) -> int:
    problem = find_problem(args.id)
    name = args.file.name
    if args.origin == "blind":
        if args.freedom is not None:
            raise UsageError("--freedom goes with --origin freedom only")
        if name != f"{_ACCEPT_BLIND}{problem.suffix}":
            raise UsageError(f"a blind accept must be named {_ACCEPT_BLIND}{problem.suffix}")
    elif args.freedom is None:
        raise UsageError("--origin freedom needs --freedom: say what it does differently")
    _variant_name(problem, name, VARIANT_ACCEPT_PATTERN)
    _candidate(args.file)
    freedom = plain_words(args.freedom) if args.freedom is not None else None
    _add_accept(
        problem,
        source=args.file,
        name=name,
        origin=args.origin,
        freedom=freedom,
        clause_label=args.clause,
    )
    print(f"added accept {name} ({args.origin})")
    return 0


def cmd_add_uncovered(args: argparse.Namespace) -> int:
    problem = find_problem(args.id)
    reason = plain_words(args.reason)
    clause = clause_text(_problem_md(problem), args.clause)
    with _transaction(problem):
        manifest = _manifest_or_new(problem)
        entry = _build(UncoveredClause, clause=clause, reason=reason)
        write_manifest(problem, _with_entry(manifest, "uncovered", entry))
    print(f"added uncovered {args.clause}")
    return 0


def cmd_remove_variant(args: argparse.Namespace) -> int:
    problem = find_problem(args.id)
    manifest = read_manifest(problem)
    file = _variants_dir(problem) / args.name
    listed = manifest is not None and args.name in _listed(manifest)
    if not listed and not file.is_file():
        raise UsageError(f"no variant {args.name} in {problem.id}")
    with _transaction(problem) as txn:
        if manifest is not None:
            rest = _without(manifest, args.name)
            if rest.accept or rest.reject or rest.uncovered or rest.sweeps:
                write_manifest(problem, rest)
            else:
                txn.remove(_manifest_path(problem))
        if file.is_file():
            txn.remove(file)
        directory = _variants_dir(problem)
        if directory.is_dir() and not any(directory.iterdir()):
            directory.rmdir()
    print(f"removed {args.name}")
    return 0


# --- adopt-blind and variants -------------------------------------------------------------------


def _code_lines(text: str) -> list[str]:
    stripped = (line.strip() for line in text.splitlines())
    return [line for line in stripped if line and not line.startswith(("#", "--"))]


def blind_similarity(problem: audit.Problem, text: str) -> tuple[float, int]:
    """How close ``text`` is to the reference, and how many lines the reference adds itself.

    Lines the starter already has (signature, docstring) are left out of both sides: every
    solution shares them, so they say nothing about whether the solver saw the reference.
    """
    starter_file = problem.path / f"starter{problem.suffix}"
    given = (
        set(_code_lines(starter_file.read_text(encoding="utf-8")))
        if starter_file.is_file()
        else set()
    )
    reference = [
        line
        for line in _code_lines(problem.solution.read_text(encoding="utf-8"))
        if line not in given
    ]
    mine = [line for line in _code_lines(text) if line not in given]
    return difflib.SequenceMatcher(None, mine, reference).ratio(), len(reference)


def cmd_adopt_blind(args: argparse.Namespace) -> int:
    rows = audit.read_jsonl(args.out / "blind.jsonl")
    adopted = skipped = 0
    for problem in audit.discover(audit.REPO, args.only):
        manifest = read_manifest(problem)
        if manifest is not None and any(v.origin == "blind" for v in manifest.accept):
            continue
        blind = args.out / "blind" / problem.id / f"solution{problem.suffix}"
        if not blind.is_file():
            print(f"{problem.id}  skipped: no file")
            skipped += 1
            continue
        text = blind.read_text(encoding="utf-8")
        sha = audit._text_sha(text)
        passed = [
            row
            for row in rows
            if row["problem"] == problem.id
            and row["verdict"] == "survived"
            and row.get("source_sha") == sha
        ]
        if not passed:
            print(f"{problem.id}  skipped: no passing submit for this file")
            skipped += 1
            continue
        try:
            _add_accept(
                problem,
                source=blind,
                name=f"{_ACCEPT_BLIND}{problem.suffix}",
                origin="blind",
                freedom=None,
                clause_label=None,
            )
        except (GateError, UsageError) as exc:
            print(f"{problem.id}  skipped: {exc}")
            skipped += 1
            continue
        ratio, own_lines = blind_similarity(problem, text)
        if own_lines >= 15 and ratio >= 0.9:
            note = f"accept_blind is {ratio:.2f} similar to the reference"
            append_note(args.out, problem.id, "question", note)
        print(f"{problem.id}  adopted  similarity={ratio:.2f}")
        adopted += 1
    print(f"adopt-blind: {adopted} adopted, {skipped} skipped")
    return 0


def _static_row(problem: audit.Problem, issue: Any) -> dict[str, Any]:
    return {
        "problem": problem.id,
        "file": Path(issue.path).name,
        "expect": "-",
        "got": f"static {issue.code}",
        "ok": False,
        "detail": f"MISMATCH: {issue.message}",
    }


def _reject_mismatch(report: GradeReport, verdict: str, entry: RejectVariant) -> str:
    if verdict == "survived":
        return "MISMATCH: no test fails"
    if verdict != "killed":
        return f"MISMATCH: {verdict}"
    if entry.killed_by not in failing_functions(report, entry.visible):
        return f"MISMATCH: {entry.killed_by} does not fail"
    if entry.message_has is not None:
        messages = _failing_messages(report, entry.visible, entry.killed_by)
        if not any(entry.message_has.lower() in text.lower() for text in messages):
            return f"MISMATCH: no failing message has {entry.message_has}"
    return ""


def _judge(
    problem: audit.Problem, entry: AcceptVariant | RejectVariant, text: str, key: str
) -> dict[str, Any]:
    report = audit.grade_source(problem, text, Settings())
    verdict = probe_verdict(report)
    if isinstance(entry, AcceptVariant):
        detail = "" if verdict == "survived" else "MISMATCH: a test fails for this solution"
        expect = "pass"
    else:
        detail = _reject_mismatch(report, verdict, entry)
        expect = "fail"
    return {
        "key": key,
        "problem": problem.id,
        "file": entry.file,
        "expect": expect,
        "got": verdict,
        "ok": detail == "",
        "detail": detail,
    }


def replay_problem(problem: audit.Problem, out: Path, workers: int = 10) -> list[dict[str, Any]]:
    """Re-check every listed variant of one problem. Empty when it has no manifest."""
    manifest = read_manifest(problem)
    if manifest is None:
        return []
    rows = [_static_row(problem, issue) for issue in validate_variants(load_problem(problem.path))]
    suite = audit._suite_sha(audit.working_suite(problem))
    log_path = out / "variants.jsonl"
    cache = {row["key"]: row for row in audit.read_jsonl(log_path) if "key" in row}
    log = audit.Log(log_path, lambda row: row["key"])
    graded: list[dict[str, Any]] = []
    jobs: list[Callable[[], None]] = []
    for entry in (*manifest.accept, *manifest.reject):
        path = _variants_dir(problem) / entry.file
        if not path.is_file():
            continue  # validate_variants has reported it already
        text = path.read_text(encoding="utf-8")
        entry_sha = audit._text_sha(json.dumps(entry.model_dump(mode="json"), sort_keys=True))
        key = f"{problem.id}:{entry.file}:{audit._text_sha(text)}:{suite}:{entry_sha}"
        if key in cache:
            rows.append(cache[key])
            continue

        def run(entry=entry, text=text, key=key) -> None:
            row = _judge(problem, entry, text, key)
            log.write(row)
            graded.append(row)

        jobs.append(run)
    audit._run_all(jobs, workers)
    rows += graded
    return sorted(rows, key=lambda row: row["file"])


def _variant_line(row: dict[str, Any]) -> str:
    status = row["detail"] or "OK"
    return f"{row['problem']}  {row['file']}  expect={row['expect']}  got={row['got']}  {status}"


def cmd_variants(args: argparse.Namespace) -> int:
    ok = bad = problems = 0
    for problem in audit.discover(audit.REPO, args.only):
        if not (_manifest_path(problem)).is_file():
            continue
        problems += 1
        for row in replay_problem(problem, args.out, workers=args.workers):
            print(_variant_line(row))
            if row["ok"]:
                ok += 1
            else:
                bad += 1
    print(f"variants: {ok} ok, {bad} mismatches in {problems} problems")
    return 1 if bad else 0


# --- command line -------------------------------------------------------------------------------


# --- gate: the checks one edit must pass --------------------------------------------------------

GATE_MODES: tuple[str, ...] = ("none", "add", "text", "refactor")
GATE_SEEDS: tuple[int, ...] = (1, 2, 3, 4, 5)
_INFRA = ("timeout", "oom", "error")
_MESSAGE_LEAK = re.compile(r"array\(|tensor\(|\d+\.\d{2,}|\d[eE][+-]?\d")
_ASSERT_HELPERS = frozenset(
    {
        "assert_allclose",
        "assert_array_equal",
        "assert_array_almost_equal",
        "assert_almost_equal",
        "assert_equal",
        "assert_close",
    }
)
_FROZEN_TEXT = ("problem.md", "hints.md", "test_public.py")
_BLOCKING = frozenset({"frozen", "manifest", "reference"})
_FAST_MS = 10_000
_SLOWER_FACTOR = 1.25
_SLACK_MS = 1_000


@dataclass(frozen=True)
class Check:
    """One line of the gate report."""

    name: str
    status: str  # "PASS" | "FAIL" | "SKIP"
    detail: str = ""


# --- normalising and reading tests --------------------------------------------------------------


def _call_name(node: ast.expr) -> str:
    if isinstance(node, ast.Attribute):
        return node.attr
    return node.id if isinstance(node, ast.Name) else ""


def _is_docstring(stmt: ast.stmt) -> bool:
    return (
        isinstance(stmt, ast.Expr)
        and isinstance(stmt.value, ast.Constant)
        and isinstance(stmt.value.value, str)
    )


def _is_bare_settings(node: ast.expr) -> bool:
    return (
        isinstance(node, ast.Call)
        and _call_name(node.func) == "settings"
        and not node.args
        and not node.keywords
    )


_Definition = ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef


def _without_keywords(node: ast.Call, names: frozenset[str]) -> None:
    node.keywords = [keyword for keyword in node.keywords if keyword.arg not in names]


class _Normaliser(ast.NodeTransformer):
    """Strips what can change without changing a test: messages, docstrings, ids, settings."""

    def generic_visit(self, node: ast.AST) -> ast.AST:
        super().generic_visit(node)
        for field in ("body", "orelse", "finalbody"):
            value = getattr(node, field, None)
            if isinstance(value, list):
                setattr(node, field, [stmt for stmt in value if not _is_docstring(stmt)])
        return node

    def visit_Assert(self, node: ast.Assert) -> ast.AST:
        self.generic_visit(node)
        node.msg = None
        return node

    def visit_Raise(self, node: ast.Raise) -> ast.AST:
        self.generic_visit(node)
        exc = node.exc
        if isinstance(exc, ast.Call) and _call_name(exc.func) == "AssertionError":
            exc.args = []
            exc.keywords = []
        return node

    def visit_Call(self, node: ast.Call) -> ast.AST:
        self.generic_visit(node)
        name = _call_name(node.func)
        if name in _ASSERT_HELPERS:
            _without_keywords(node, frozenset({"err_msg", "msg"}))
        elif ast.unparse(node.func) == "pytest.fail":
            node.args = []
            node.keywords = []
        elif name == "parametrize":
            _without_keywords(node, frozenset({"ids"}))
        elif name == "settings":
            _without_keywords(node, frozenset({"deadline", "report_multiple_bugs"}))
        return node

    def visit_ImportFrom(self, node: ast.ImportFrom) -> ast.AST | None:
        if node.module == "hypothesis":
            node.names = [alias for alias in node.names if alias.name != "settings"]
            if not node.names:
                return None
        return node

    def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.AST:
        return self._definition(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> ast.AST:
        return self._definition(node)

    def visit_ClassDef(self, node: ast.ClassDef) -> ast.AST:
        return self._definition(node)

    def _definition(self, node: _Definition) -> ast.AST:
        self.generic_visit(node)
        node.decorator_list = [d for d in node.decorator_list if not _is_bare_settings(d)]
        return node


def normalise_tests(source: str) -> str:
    """The tree of a test file, with messages, docstrings and reporting options removed."""
    return ast.dump(_Normaliser().visit(ast.parse(source)))


def _top_level_bindings(source: str) -> Counter[str]:
    counts: Counter[str] = Counter()
    for node in ast.parse(source).body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            counts[node.name] += 1
        elif isinstance(node, ast.Assign):
            counts.update(t.id for t in node.targets if isinstance(t, ast.Name))
    return counts


def duplicate_top_level_names(source: str) -> list[str]:
    """Names bound more than once at module level, sorted."""
    return sorted(name for name, count in _top_level_bindings(source).items() if count > 1)


def test_function_names(source: str) -> set[str]:
    """Names of the top-level test functions."""
    return {
        node.name
        for node in ast.parse(source).body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name.startswith("test_")
    }


def _working(path: Path) -> str | None:
    return path.read_text(encoding="utf-8") if path.is_file() else None


def _rel(problem: audit.Problem) -> str:
    return problem.path.relative_to(audit.REPO).as_posix()


def _hidden_names(report: GradeReport) -> set[str]:
    return {test.name for test in report.tests if test.file == "hidden"}


def _meta_key(text: str | None) -> dict[str, Any] | None:
    """meta.yaml as data, without its version: the part an edit must not change."""
    if text is None:
        return None
    data = yaml.safe_load(text) or {}
    data.pop("version", None)
    return data


def expected_version(problem: audit.Problem, base: str) -> int:
    """The version the tests call for: the start version, plus 1 for a new test function.

    The start is the tag ``i32-start`` when it has this problem's meta.yaml, else ``base``.
    """
    rel = _rel(problem)
    start = "i32-start" if audit._git_show("i32-start", f"{rel}/meta.yaml") is not None else base
    meta_text = audit._git_show(start, f"{rel}/meta.yaml")
    if meta_text is None:
        return _problem_version(problem)
    version = int(yaml.safe_load(meta_text)["version"])
    before = audit._git_show(start, f"{rel}/test_hidden.py") or ""
    now = _working(problem.path / "test_hidden.py") or ""
    new_names = test_function_names(now) - test_function_names(before)
    return version + (1 if new_names else 0)


# --- the checks ---------------------------------------------------------------------------------


class _GateRun:
    """What the checks of one gate run share: the suites, and each grade made only once."""

    def __init__(
        self,
        problem: audit.Problem,
        *,
        mode: str,
        base: str,
        wrongs: Sequence[Path],
        out: Path,
        workers: int,
    ) -> None:
        self.problem = problem
        self.mode = mode
        self.base = base
        self.wrongs = list(wrongs)
        self.out = out
        self.workers = workers
        self.rel = _rel(problem)
        self.reference = problem.solution.read_text(encoding="utf-8")
        self.settings = Settings()
        self.new_suite = audit.working_suite(problem)
        self.base_suite = audit.base_suite(problem, base)
        self._new: list[GradeReport] = []
        self._base: GradeReport | None = None
        self._rows: list[dict[str, Any]] | None = None

    def grade(self, source: str, suite: audit.Suite) -> GradeReport:
        return audit.grade_suite(self.problem, source, suite, self.settings)

    def new_reports(self) -> list[GradeReport]:
        """The reference on the new suite, twice (once only when the first run fails)."""
        if not self._new:
            self._new.append(self.grade(self.reference, self.new_suite))
            if self._new[0].status is RunStatus.PASSED:
                self._new.append(self.grade(self.reference, self.new_suite))
        return self._new

    def base_report(self) -> GradeReport:
        if self._base is None:
            self._base = self.grade(self.reference, self.base_suite)
        return self._base

    def replayed(self) -> list[dict[str, Any]]:
        if self._rows is None:
            self._rows = replay_problem(self.problem, self.out, workers=self.workers)
        return self._rows


def _check_frozen(g: _GateRun) -> Check:
    if audit._git_show(g.base, f"{g.rel}/meta.yaml") is None:
        return Check("frozen", "PASS", "new problem")
    names = (*_FROZEN_TEXT, f"solution{g.problem.suffix}", f"starter{g.problem.suffix}")
    changed = [
        n for n in names if _working(g.problem.path / n) != audit._git_show(g.base, f"{g.rel}/{n}")
    ]
    meta_now = _meta_key(_working(g.problem.path / "meta.yaml"))
    if meta_now != _meta_key(audit._git_show(g.base, f"{g.rel}/meta.yaml")):
        changed.append("meta.yaml (other than version)")
    if changed:
        return Check("frozen", "FAIL", "changed: " + ", ".join(changed))
    return Check("frozen", "PASS", "no frozen file changed")


def _check_manifest(g: _GateRun) -> Check:
    if not _variants_dir(g.problem).is_dir():
        return Check("manifest", "PASS", "none")
    issues = validate_variants(load_problem(g.problem.path))
    if issues:
        detail = "; ".join(f"{Path(issue.path).name}: {issue.message}" for issue in issues)
        return Check("manifest", "FAIL", detail)
    return Check("manifest", "PASS", "valid")


def _outcomes(report: GradeReport) -> list[tuple[str, str, str]]:
    return sorted((test.file, test.name, test.outcome) for test in report.tests)


def _counts_text(report: GradeReport) -> str:
    public_passed, public_total = report.counts("public")
    hidden_passed, hidden_total = report.counts("hidden")
    return f"public {public_passed}/{public_total} hidden {hidden_passed}/{hidden_total}"


def _check_reference(g: _GateRun) -> Check:
    reports = g.new_reports()
    for report in reports:
        if report.status is not RunStatus.PASSED:
            failed = ", ".join(f"{t.file} {t.name}" for t in _failed_tests(report))
            detail = f"{report.status.value}: {failed or _counts_text(report)}"
            return Check("reference", "FAIL", detail)
    if len(reports) == 2 and _outcomes(reports[0]) != _outcomes(reports[1]):
        return Check("reference", "FAIL", "the two runs differ")
    return Check("reference", "PASS", f"{_counts_text(reports[0])}, twice")


def _variant_check(g: _GateRun, expect: str, name: str) -> Check:
    rows = [row for row in g.replayed() if row["expect"] == expect]
    if not rows:
        return Check(name, "PASS", "none")
    bad = [row for row in rows if not row["ok"]]
    if bad:
        return Check(name, "FAIL", "; ".join(f"{r['file']}: {r['detail']}" for r in bad))
    return Check(name, "PASS", f"{len(rows) - len(bad)} of {len(rows)}")


def _check_accepts(g: _GateRun) -> Check:
    return _variant_check(g, "pass", "accepts")


def _check_rejects(g: _GateRun) -> Check:
    return _variant_check(g, "fail", "rejects")


def _numstat(text: str) -> tuple[int, int]:
    """(added, deleted) from one ``git diff --numstat`` line; nothing printed means no change."""
    if not text.strip():
        return 0, 0
    added, deleted, _path = text.strip().split("\t", 2)
    return (0 if added == "-" else int(added)), (0 if deleted == "-" else int(deleted))


def _change_add(g: _GateRun, base: str | None, new: str | None) -> Check:
    diff = audit._git("diff", "--numstat", g.base, "--", f"{g.rel}/test_hidden.py")
    added, deleted = _numstat(diff.stdout.decode("utf-8"))
    if deleted:
        return Check("change", "FAIL", f"{deleted} lines deleted: add tests, do not remove them")
    duplicates = duplicate_top_level_names(new or "")
    if duplicates:
        return Check("change", "FAIL", "names bound twice: " + ", ".join(duplicates))
    missing = _hidden_names(g.base_report()) - _hidden_names(g.new_reports()[0])
    if missing:
        return Check("change", "FAIL", "tests gone: " + ", ".join(sorted(missing)))
    new_tests = len(test_function_names(new or "") - test_function_names(base or ""))
    return Check("change", "PASS", f"+{added} -{deleted} lines, {new_tests} new test")


def _change_refactor(g: _GateRun, base: str | None, new: str | None) -> Check:
    lost = test_function_names(base or "") - test_function_names(new or "")
    if lost:
        return Check("change", "FAIL", "test functions removed: " + ", ".join(sorted(lost)))
    results = _mutant_verdicts(g)
    weakened = [
        mutant for mutant, base_v, new_v in results if base_v != "survived" and new_v == "survived"
    ]
    detail = f"{len(results)} mutants, {len(weakened)} weakened"
    if weakened:
        changes = "; ".join(mutant.change for mutant in weakened[:3])
        return Check("change", "FAIL", f"{detail}: {changes}")
    return Check("change", "PASS", detail)


def _mutant_verdicts(g: _GateRun) -> list[tuple[audit.Mutant, str, str]]:
    """Each sampled mutant's verdict on the base suite and on the new suite."""
    mutants = audit.mutants_for(g.problem, audit.MAX_MUTANTS)
    pairs: dict[int, tuple[str, str]] = {}

    def job(index: int, mutant: audit.Mutant) -> Callable[[], None]:
        def run() -> None:
            base_v = audit.verdict(g.grade(mutant.source, g.base_suite))
            new_v = audit.verdict(g.grade(mutant.source, g.new_suite))
            pairs[index] = (base_v, new_v)

        return run

    audit._run_all([job(i, m) for i, m in enumerate(mutants)], g.workers)
    results: list[tuple[audit.Mutant, str, str]] = []
    for index, mutant in enumerate(mutants):
        base_v, new_v = pairs[index]
        if base_v in _INFRA:  # infrastructure noise: grade alone, on both suites, before deciding
            base_v = audit.verdict(g.grade(mutant.source, g.base_suite))
            new_v = audit.verdict(g.grade(mutant.source, g.new_suite))
        results.append((mutant, base_v, new_v))
    return results


def _change_none(base: str | None, new: str | None) -> Check:
    if new == base:
        return Check("change", "PASS", "unchanged")
    return Check("change", "FAIL", "test_hidden.py changed: use --mode add, text or refactor")


def _change_text(base: str | None, new: str | None) -> Check:
    try:
        same = _normal_or_none(base) == _normal_or_none(new)
    except SyntaxError as exc:
        return Check("change", "FAIL", f"test_hidden.py does not parse: {exc.msg}")
    if same:
        return Check("change", "PASS", "same tests, reworded only")
    return Check("change", "FAIL", "a test changed beyond its messages and options")


def _check_change(g: _GateRun) -> Check:
    base = audit._git_show(g.base, f"{g.rel}/test_hidden.py")
    new = _working(g.problem.path / "test_hidden.py")
    if g.mode == "none":
        return _change_none(base, new)
    if g.mode == "add":
        return _change_add(g, base, new)
    if g.mode == "refactor":
        return _change_refactor(g, base, new)
    return _change_text(base, new)


def _normal_or_none(text: str | None) -> str | None:
    return None if text is None else normalise_tests(text)


def _check_wrong(g: _GateRun) -> Check:
    if g.mode != "add":
        return Check("wrong", "SKIP")
    if not g.wrongs:
        return Check("wrong", "PASS", "none given")
    problems: list[str] = []
    details: list[str] = []
    for path in g.wrongs:
        text = path.read_text(encoding="utf-8")
        if audit.verdict(g.grade(text, g.base_suite)) != "survived":
            problems.append(f"{path.name} is already rejected by the base tests")
        report = g.grade(text, g.new_suite)
        failing = [t for t in report.tests if t.file == "hidden" and t.outcome in _FAILED]
        if report.status is not RunStatus.FAILED or not failing:
            problems.append(f"{path.name} is not rejected by the new tests")
            continue
        message = learner_message(failing[0])
        kind = audit.classify_message(message)
        if kind != "named":
            problems.append(f"{path.name}: message kind {kind}")
        elif _MESSAGE_LEAK.search(message):
            problems.append(f"{path.name}: message shows values")
        details.append(f"{path.name}: {failing[0].name}")
    if problems:
        return Check("wrong", "FAIL", "; ".join(problems))
    return Check("wrong", "PASS", "; ".join(details))


def _check_version(g: _GateRun) -> Check:
    current = _problem_version(g.problem)
    expected = expected_version(g.problem, g.base)
    if current == expected:
        return Check("version", "PASS", str(current))
    return Check("version", "FAIL", f"meta.yaml says {current}, expected {expected}: bump-version")


def _check_lint(g: _GateRun) -> Check:
    findings = lint_hidden_tests(_working(g.problem.path / "test_hidden.py") or "")
    if g.mode in ("text", "refactor"):
        if findings:
            return Check("lint", "FAIL", f"{len(findings)} findings, first line {findings[0].line}")
        return Check("lint", "PASS", "0 findings")
    base_text = audit._git_show(g.base, f"{g.rel}/test_hidden.py") or ""
    base_counts = Counter(f.rule for f in lint_hidden_tests(base_text))
    new_counts = Counter(f.rule for f in findings)
    above = [rule for rule in new_counts if new_counts[rule] > base_counts[rule]]
    if above:
        return Check("lint", "FAIL", "more findings than before: " + ", ".join(above))
    return Check("lint", "PASS", f"{len(findings)} findings, none above base")


def _check_names(g: _GateRun) -> Check:
    if g.mode == "none":
        return Check("names", "PASS", "not checked in mode none")
    leaks = id_leaks(_hidden_names(g.new_reports()[0]))
    if g.mode in ("text", "refactor"):
        if leaks:
            return Check("names", "FAIL", "test ids show values: " + ", ".join(leaks))
        return Check("names", "PASS", "no id leaks")
    base_leaks = set(id_leaks(_hidden_names(g.base_report())))
    fresh = [name for name in leaks if name not in base_leaks]
    if fresh:
        return Check("names", "FAIL", "new test ids show values: " + ", ".join(fresh))
    return Check("names", "PASS", f"{len(leaks)} ids, none new")


def _run_seed(
    problem: audit.Problem, source: str, suite: audit.Suite, seed: int
) -> tuple[bool, list[str]]:
    """Plain pytest on one suite under one Hypothesis seed: (passed, failing test names)."""
    with tempfile.TemporaryDirectory(prefix="gate-seed-") as tmp:
        root = Path(tmp)
        (root / f"submission{problem.suffix}").write_text(source, encoding="utf-8")
        for name, text in zip(audit._TEST_FILES, (suite.public, suite.hidden), strict=True):
            if text is not None:
                (root / name).write_text(text, encoding="utf-8")
        if (problem.path / "fixtures").is_dir():
            shutil.copytree(problem.path / "fixtures", root / "fixtures")
        names = [name for name in audit._TEST_FILES if (root / name).is_file()]
        command = [
            sys.executable,
            "-m",
            "pytest",
            "-q",
            "--tb=line",
            "-p",
            "no:cacheprovider",
            f"--hypothesis-seed={seed}",
            *names,
        ]
        env = {**os.environ, "PYTHONHASHSEED": str(seed), "DOGMA_DATA_SALT": str(seed)}
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
        except subprocess.TimeoutExpired:
            return False, ["<timeout>"]
    failed = re.findall(r"^FAILED (\S+)", done.stdout, re.MULTILINE)
    return done.returncode == 0, failed[:6]


def _check_seeds(g: _GateRun) -> Check:
    failing: dict[int, list[str]] = {}
    for seed in GATE_SEEDS:
        passed, failed = _run_seed(g.problem, g.reference, g.new_suite, seed)
        if passed:
            continue
        if g.mode != "none":
            base_passed, _ = _run_seed(g.problem, g.reference, g.base_suite, seed)
            if not base_passed:
                continue  # the base suite fails this seed too: not this edit's doing
        failing[seed] = failed
    detail = f"{len(GATE_SEEDS) - len(failing)} of {len(GATE_SEEDS)}"
    if failing:
        detail += "; failing " + "; ".join(f"seed {s}: {', '.join(n)}" for s, n in failing.items())
    if failing and g.mode != "none":
        return Check("seeds", "FAIL", detail)
    return Check("seeds", "PASS", detail)


def _check_time(g: _GateRun) -> Check:
    new_ms = g.new_reports()[0].duration_ms
    if g.mode == "none":
        return Check("time", "PASS", f"{new_ms / 1000:.1f} s")
    base_ms = g.base_report().duration_ms
    detail = f"{new_ms / 1000:.1f} s (base {base_ms / 1000:.1f} s)"
    limit = max(_FAST_MS, _SLOWER_FACTOR * base_ms + _SLACK_MS)
    if new_ms <= limit:
        return Check("time", "PASS", detail)
    return Check("time", "FAIL", detail)


_STEPS: tuple[tuple[str, Callable[[_GateRun], Check]], ...] = (
    ("frozen", _check_frozen),
    ("manifest", _check_manifest),
    ("reference", _check_reference),
    ("accepts", _check_accepts),
    ("rejects", _check_rejects),
    ("change", _check_change),
    ("wrong", _check_wrong),
    ("version", _check_version),
    ("lint", _check_lint),
    ("names", _check_names),
    ("seeds", _check_seeds),
    ("time", _check_time),
)


def run_gate(
    problem: audit.Problem,
    *,
    mode: str,
    base: str,
    wrongs: Sequence[Path],
    out: Path,
    workers: int,
) -> list[Check]:
    """Every check, in order. After frozen, manifest or reference fails, the rest are SKIP."""
    gate = _GateRun(problem, mode=mode, base=base, wrongs=wrongs, out=out, workers=workers)
    checks: list[Check] = []
    blocked = False
    for name, step in _STEPS:
        if blocked:
            checks.append(Check(name, "SKIP"))
            continue
        check = step(gate)
        checks.append(check)
        blocked = check.status == "FAIL" and name in _BLOCKING
    return checks


def _commit_of(rev: str) -> str:
    done = audit._git("rev-parse", "--verify", "--quiet", f"{rev}^{{commit}}")
    if done.returncode != 0:
        raise UsageError(f"unknown revision {rev}")
    return done.stdout.decode("utf-8").strip()


def _check_line(check: Check) -> str:
    return f"{check.name:<10} {check.status}  {check.detail}".rstrip()


def _print_checks(problem: audit.Problem, mode: str, base: str, checks: list[Check]) -> str:
    """Print the gate report and return its verdict, PASS or FAIL."""
    print(f"GATE {problem.id}  mode={mode}  base={base}")
    for check in checks:
        print(_check_line(check))
    verdict = "FAIL" if any(check.status == "FAIL" for check in checks) else "PASS"
    print(f"GATE: {verdict}")
    return verdict


def _record_gate(
    out: Path,
    *,
    problem: audit.Problem,
    mode: str,
    commit: str,
    verdict: str,
    checks: list[Check],
) -> None:
    texts = {
        "test_hidden.py": _working(problem.path / "test_hidden.py"),
        "test_public.py": _working(problem.path / "test_public.py"),
        "solution": _working(problem.solution),
        "manifest": _working(_manifest_path(problem)),
    }
    row = {
        "time": datetime.now(UTC).isoformat(timespec="seconds"),
        "problem": problem.id,
        "mode": mode,
        "base": commit,
        "result": verdict,
        "checks": [asdict(check) for check in checks],
        "sha": {
            name: None if text is None else audit._text_sha(text) for name, text in texts.items()
        },
    }
    out.mkdir(parents=True, exist_ok=True)
    with (out / "gate.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row) + "\n")


def cmd_gate(args: argparse.Namespace) -> int:
    if args.wrong and args.mode != "add":
        raise UsageError("--wrong goes with --mode add only")
    problem = find_problem(args.id)
    commit = _commit_of(args.base)
    for path in args.wrong:
        if not path.is_file():
            raise UsageError(f"no file {path}")
    checks = run_gate(
        problem,
        mode=args.mode,
        base=commit,
        wrongs=args.wrong,
        out=args.out,
        workers=args.workers,
    )
    verdict = _print_checks(problem, args.mode, args.base, checks)
    _record_gate(
        args.out, problem=problem, mode=args.mode, commit=commit, verdict=verdict, checks=checks
    )
    return 0 if verdict == "PASS" else 1


def _bump_meta(problem: audit.Problem, version: int) -> None:
    meta = problem.path / "meta.yaml"
    text = meta.read_text(encoding="utf-8")
    updated, count = re.subn(
        r"^(version:\s*)\d+",
        lambda match: f"{match.group(1)}{version}",
        text,
        count=1,
        flags=re.MULTILINE,
    )
    if count != 1:
        raise GateError("meta.yaml has no version line")
    atomic_write(meta, updated)


def _apply_version(problem: audit.Problem, commit: str) -> tuple[int, int]:
    """Set meta.yaml and the manifest to the version the tests call for. Returns (old, new)."""
    old = _problem_version(problem)
    new = expected_version(problem, commit)
    if new != old:
        _bump_meta(problem, new)
    manifest = read_manifest(problem)
    if manifest is not None and manifest.problem_version != new:
        write_manifest(
            problem, _build(VariantsManifest, **{**dict(manifest), "problem_version": new})
        )
    return old, new


def cmd_bump_version(args: argparse.Namespace) -> int:
    problem = find_problem(args.id)
    old, new = _apply_version(problem, _commit_of(args.base))
    if new == old:
        print(f"version: unchanged ({old})")
    else:
        print(f"version: {old} -> {new}")
    return 0


# --- add-test, mark-swept, worklist, scorecard --------------------------------------------------

_NO_BLIND = "no blind accept: record the gap with a no-witness note, add no test"
_TEST_MAX_LINES = 40
_TEST_REFUSED = re.compile(r"pytest\.raises|skip|xfail")
_WORKERS = 10
_SWEEPS = ("reject", "mutants")
_WORKLIST_KINDS = ("hygiene", "blind", "reject", "mutants")
_OPEN_NOTES = ("blocked", "question")


def _bytes_or_none(path: Path) -> bytes | None:
    return path.read_bytes() if path.is_file() else None


def _restore(path: Path, data: bytes | None) -> None:
    if data is None:
        path.unlink(missing_ok=True)
    else:
        atomic_write(path, data)


def _new_test(text: str, hidden: str) -> str:
    """The name of the one test TESTFILE adds. Refuses a test the hidden suite must not get."""
    if len(text.splitlines()) > _TEST_MAX_LINES:
        raise GateError(f"a new test has at most {_TEST_MAX_LINES} lines")
    if _TEST_REFUSED.search(text):
        raise GateError("a new test does not use pytest.raises, skip or xfail")
    try:
        body = ast.parse(text).body
    except SyntaxError as exc:
        raise GateError(f"the new test does not parse: {exc.msg}") from exc
    names = [
        node.name
        for node in body
        if isinstance(node, ast.FunctionDef) and node.name.startswith("test_")
    ]
    if len(names) != 1:
        raise GateError(f"a new test defines exactly one test_ function, found {len(names)}")
    clash = sorted(set(_top_level_bindings(text)) & set(_top_level_bindings(hidden)))
    if clash:
        raise GateError("the new test repeats a name in test_hidden.py: " + ", ".join(clash))
    findings = lint_hidden_tests(text)
    if findings:
        lines = "\n".join(f"  line {f.line}: {f.rule} {f.message}" for f in findings)
        raise GateError(f"the new test has {len(findings)} lint findings:\n{lines}")
    return names[0]


def _append_test(path: Path, text: str) -> None:
    before = path.read_text(encoding="utf-8")
    body = before if before.endswith("\n") else before + "\n"
    atomic_write(path, body + "\n" + text.strip("\n") + "\n")


def _format_hidden(path: Path) -> None:
    """Format, then lint, the whole hidden suite. A failing check is an error (no noqa here)."""
    ok, output = _ruff(("format",), path)
    if not ok:
        raise GateError(f"ruff format failed on test_hidden.py:\n{output}")
    ok, output = _ruff(("check",), path)
    if not ok:
        raise GateError(f"ruff check failed on test_hidden.py:\n{output}")


def cmd_add_test(args: argparse.Namespace) -> int:
    problem = find_problem(args.id)
    name = args.wrong.name
    _variant_name(problem, name, VARIANT_REJECT_PATTERN)
    _candidate(args.wrong)
    _candidate(args.testfile)
    manifest = read_manifest(problem)
    if manifest is None or not any(v.origin == "blind" for v in manifest.accept):
        raise GateError(_NO_BLIND)
    mistake = plain_words(args.mistake)
    clause_text(_problem_md(problem), args.clause)
    hidden = problem.path / "test_hidden.py"
    if not hidden.is_file():
        raise GateError(f"no test_hidden.py for {problem.id}")
    new_test_text = args.testfile.read_text(encoding="utf-8")
    new_name = _new_test(new_test_text, hidden.read_text(encoding="utf-8"))
    commit = _commit_of(args.base)

    files = (hidden, problem.path / "meta.yaml", _manifest_path(problem))
    saved = {path: _bytes_or_none(path) for path in files}
    variants = _variants_dir(problem)
    variants_existed = variants.is_dir()
    target = variants / name
    target_existed = target.exists()

    def restore() -> None:
        for path, data in saved.items():
            _restore(path, data)
        if not target_existed:
            target.unlink(missing_ok=True)
        if not variants_existed and variants.is_dir() and not any(variants.iterdir()):
            variants.rmdir()

    try:
        _append_test(hidden, new_test_text)
        _format_hidden(hidden)
        _apply_version(problem, commit)
        checks = run_gate(
            problem,
            mode="add",
            base=commit,
            wrongs=[args.wrong],
            out=args.out,
            workers=_WORKERS,
        )
        if _print_checks(problem, "add", args.base, checks) != "PASS":
            raise GateError("the gate did not pass")
        entry, kind = _add_reject(
            problem,
            source=args.wrong,
            name=name,
            mistake=mistake,
            clause_label=args.clause,
            killed_by=new_name,
            message_has=None,
        )
    except GateError as exc:
        restore()
        print(exc, file=sys.stderr)
        print("ADD-TEST: FAILED, nothing changed")
        return 1
    except BaseException:
        restore()
        raise
    if kind != "named":
        append_note(
            args.out,
            problem.id,
            "message-defect",
            f"{name}: {entry.killed_by} shows kind={kind}",
        )
    print(f"ADD-TEST: added {new_name} and {name}")
    return 0


def cmd_mark_swept(args: argparse.Namespace) -> int:
    problem = find_problem(args.id)
    commit = _commit_of(args.base)
    checks = run_gate(
        problem,
        mode=args.mode,
        base=commit,
        wrongs=[],
        out=args.out,
        workers=_WORKERS,
    )
    if _print_checks(problem, args.mode, args.base, checks) != "PASS":
        return 1
    manifest = _manifest_or_new(problem)
    if args.kind in manifest.sweeps:
        print(f"already marked {args.kind}")
        return 0
    sweeps = (*manifest.sweeps, args.kind)
    write_manifest(problem, _build(VariantsManifest, **{**dict(manifest), "sweeps": sweeps}))
    print(f"marked {args.kind}")
    return 0


def _current_week(weeks: Sequence[PathWeek], today: date) -> int:
    """The last week that has started by ``today``; week 1 before the path starts."""
    started = [week.week for week in weeks if week.starts <= today]
    return started[-1] if started else 1


def week_order(weeks: Sequence[PathWeek], today: date) -> list[int]:
    """Week numbers to work through: the weeks after the current one, then the current one,
    then the weeks before it, nearest first."""
    if not weeks:
        return []
    current = _current_week(weeks, today)
    numbers = [week.week for week in weeks]
    ahead = sorted(number for number in numbers if number > current)
    behind = sorted((number for number in numbers if number < current), reverse=True)
    return [*ahead, current, *behind]


def _core_path() -> CorePath:
    return load_core_path(Settings().curriculum_dir / "paths" / "core.yaml")


def _left(problem: audit.Problem, kind: str) -> bool:
    if kind == "hygiene":
        hidden = problem.path / "test_hidden.py"
        return hidden.is_file() and bool(lint_hidden_tests(hidden.read_text(encoding="utf-8")))
    manifest = read_manifest(problem)
    if kind == "blind":
        return manifest is None or not any(v.origin == "blind" for v in manifest.accept)
    return manifest is None or kind not in manifest.sweeps


def _notes_by_problem() -> dict[str, Counter[str]]:
    found: dict[str, Counter[str]] = {}
    for path in sorted(audit.REPO.glob("var/audit-i32/*/ledger.jsonl")):
        for row in audit.read_jsonl(path):
            found.setdefault(row["problem"], Counter())[row["kind"]] += 1
    return found


def _arrange(
    problems: Sequence[audit.Problem], weeks: Sequence[PathWeek], today: date
) -> tuple[list[tuple[int, list[audit.Problem]]], list[audit.Problem]]:
    """Problems grouped by week, in worklist order, and the problems whose topic is off the path.

    A topic counts for the first week that lists it. Inside a week, topics keep path order and
    problems keep id order.
    """
    slots: dict[str, tuple[int, int]] = {}
    position = 0
    for week in weeks:
        for topic in week.topics:
            slots.setdefault(topic, (week.week, position))
            position += 1
    grouped: dict[int, list[tuple[tuple[int, str], audit.Problem]]] = {}
    off: list[audit.Problem] = []
    for problem in problems:
        slot = slots.get(problem.id.rsplit("/", 1)[0])
        if slot is None:
            off.append(problem)
        else:
            grouped.setdefault(slot[0], []).append(((slot[1], problem.id), problem))
    arranged: list[tuple[int, list[audit.Problem]]] = []
    for number in week_order(weeks, today):
        members = [item for _, item in sorted(grouped.get(number, []), key=lambda p: p[0])]
        if members:
            arranged.append((number, members))
    return arranged, sorted(off, key=lambda item: item.id)


def _annotation(counts: Counter[str]) -> str:
    parts = [f"{kind} {counts[kind]}" for kind in _OPEN_NOTES if counts[kind]]
    return f"  ({', '.join(parts)})" if parts else ""


def _day(text: str | None) -> date:
    if text is None:
        return date.today()
    try:
        return date.fromisoformat(text)
    except ValueError as exc:
        raise UsageError(f"--today wants YYYY-MM-DD, not {text}") from exc


def cmd_worklist(args: argparse.Namespace) -> int:
    today = _day(args.today)
    weeks = _core_path().weeks
    left = [p for p in audit.discover(audit.REPO, []) if _left(p, args.kind)]
    arranged, off = _arrange(left, weeks, today)
    notes = _notes_by_problem()
    current = _current_week(weeks, today)
    print(f"WORKLIST {args.kind}  today={today.isoformat()}  current week={current}")
    for number, members in arranged:
        print(f"week {number}")
        for problem in members:
            print(f"  {problem.id}{_annotation(notes.get(problem.id, Counter()))}")
    if off:
        print("not on the path")
        for problem in off:
            print(f"  {problem.id}{_annotation(notes.get(problem.id, Counter()))}")
    print(f"total: {len(left)}")
    return 0


_SCORE_HEADER = (
    "| Problem | Version | Lint | Blind | Accepts | Rejects | Bullets | Sweeps | Notes |"
)


def bullet_coverage(problem_md: str, manifest: VariantsManifest | None) -> tuple[int, int]:
    """(covered, total) over the Contract bullets C1 and up. A bullet is covered when the
    squashed clause of a reject or an uncovered entry is a substring of it."""
    bullets = [
        text
        for label, text in contract_clauses(problem_md)
        if label.startswith("C") and label != "C0"
    ]
    entries: list[str] = []
    if manifest is not None:
        entries = [squash(entry.clause) for entry in (*manifest.reject, *manifest.uncovered)]
    covered = sum(1 for text in bullets if any(entry in text for entry in entries if entry))
    return covered, len(bullets)


def _score_row(problem: audit.Problem, notes: dict[str, Counter[str]]) -> dict[str, Any]:
    manifest = read_manifest(problem)
    hidden = problem.path / "test_hidden.py"
    lint = len(lint_hidden_tests(hidden.read_text(encoding="utf-8"))) if hidden.is_file() else 0
    covered, total = bullet_coverage(_problem_md(problem), manifest)
    counts = notes.get(problem.id, Counter())
    open_notes = ", ".join(f"{kind} {counts[kind]}" for kind in NOTE_KINDS if counts[kind])
    return {
        "problem": problem.id,
        "version": _problem_version(problem),
        "lint": lint,
        "blind": manifest is not None and any(v.origin == "blind" for v in manifest.accept),
        "accepts": len(manifest.accept) if manifest else 0,
        "rejects": len(manifest.reject) if manifest else 0,
        "covered": covered,
        "total": total,
        "sweeps": tuple(manifest.sweeps) if manifest else (),
        "notes": open_notes or "-",
    }


def cmd_scorecard(args: argparse.Namespace) -> int:
    notes = _notes_by_problem()
    problems = sorted(audit.discover(audit.REPO, []), key=lambda p: p.id)
    rows = [_score_row(problem, notes) for problem in problems]
    print(_SCORE_HEADER)
    print("|" + "---|" * 9)
    for row in rows:
        sweeps = ",".join(row["sweeps"]) or "-"
        print(
            f"| {row['problem']} | {row['version']} | {row['lint']} | "
            f"{'yes' if row['blind'] else 'no'} | {row['accepts']} | {row['rejects']} | "
            f"{row['covered']}/{row['total']} | {sweeps} | {row['notes']} |"
        )
    print()
    print(f"problems: {len(rows)}")
    print(f"with a blind accept: {sum(1 for r in rows if r['blind'])}")
    print(f"accepts: {sum(r['accepts'] for r in rows)}")
    print(f"rejects: {sum(r['rejects'] for r in rows)}")
    print(f"every bullet covered: {sum(1 for r in rows if r['covered'] == r['total'])}")
    print(f"both sweeps: {sum(1 for r in rows if set(_SWEEPS) <= set(r['sweeps']))}")
    print(f"lint findings: {sum(r['lint'] for r in rows)}")
    return 0


def _add_out(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--out", type=Path, required=True, help="directory for results")


def _parser() -> argparse.ArgumentParser:  # noqa: PLR0915 - one add_parser block per command
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)

    lint = commands.add_parser("lint", help="lint the hidden tests")
    _add_out(lint)
    lint.add_argument("--only", action="append", default=[], help="problem id prefix")
    lint.set_defaults(run=cmd_lint)

    clauses = commands.add_parser("clauses", help="list the contract clauses of a problem")
    _add_out(clauses)
    clauses.add_argument("id")
    clauses.set_defaults(run=cmd_clauses)

    probe = commands.add_parser("probe", help="grade a file and say how it fared")
    _add_out(probe)
    probe.add_argument("id")
    probe.add_argument("file", type=Path)
    probe.set_defaults(run=cmd_probe)

    note = commands.add_parser("note", help="append a note to the ledger")
    _add_out(note)
    note.add_argument("id")
    note.add_argument("--kind", required=True)
    note.add_argument("--text", required=True)
    note.set_defaults(run=cmd_note)

    ledger = commands.add_parser("ledger", help="print the ledger as Markdown")
    _add_out(ledger)
    ledger.add_argument("--all", action="store_true", help="every var/audit-i32 ledger")
    ledger.add_argument("--kind", action="append", default=[], help="only this kind")
    ledger.set_defaults(run=cmd_ledger)

    add_reject = commands.add_parser("add-reject", help="add a wrong solution the tests must fail")
    _add_out(add_reject)
    add_reject.add_argument("id")
    add_reject.add_argument("file", type=Path)
    add_reject.add_argument("--mistake", required=True, help="the mistake, in plain words")
    add_reject.add_argument("--clause", required=True, help="the contract label, e.g. C2")
    add_reject.add_argument("--killed-by", dest="killed_by", help="the test that must fail")
    add_reject.add_argument("--message-has", dest="message_has", help="text the message shows")
    add_reject.set_defaults(run=cmd_add_reject)

    add_accept = commands.add_parser("add-accept", help="add a correct solution the tests pass")
    _add_out(add_accept)
    add_accept.add_argument("id")
    add_accept.add_argument("file", type=Path)
    add_accept.add_argument("--origin", required=True, choices=("blind", "freedom"))
    add_accept.add_argument("--freedom", help="what it does differently (freedom only)")
    add_accept.add_argument("--clause", help="the contract label it covers, e.g. C2")
    add_accept.set_defaults(run=cmd_add_accept)

    uncovered = commands.add_parser("add-uncovered", help="record a clause no variant covers")
    _add_out(uncovered)
    uncovered.add_argument("id")
    uncovered.add_argument("--clause", required=True, help="the contract label, e.g. C3")
    uncovered.add_argument("--reason", required=True, help="why, in plain words")
    uncovered.set_defaults(run=cmd_add_uncovered)

    remove = commands.add_parser("remove-variant", help="remove one variant and its entry")
    _add_out(remove)
    remove.add_argument("id")
    remove.add_argument("name", help="the variant file name")
    remove.set_defaults(run=cmd_remove_variant)

    adopt = commands.add_parser("adopt-blind", help="adopt passing blind solutions as accepts")
    _add_out(adopt)
    adopt.add_argument("--only", action="append", default=[], help="problem id prefix")
    adopt.set_defaults(run=cmd_adopt_blind)

    variants = commands.add_parser("variants", help="re-check every listed variant")
    _add_out(variants)
    variants.add_argument("--only", action="append", default=[], help="problem id prefix")
    variants.add_argument("--workers", type=int, default=10, help="grading jobs at once")
    variants.set_defaults(run=cmd_variants)

    gate = commands.add_parser("gate", help="run the checks for one edited problem")
    _add_out(gate)
    gate.add_argument("id")
    gate.add_argument("--mode", choices=GATE_MODES, default="none", help="what the edit did")
    gate.add_argument("--base", default="HEAD", help="git revision the edit started from")
    gate.add_argument(
        "--wrong", type=Path, action="append", default=[], help="a wrong solution (mode add)"
    )
    gate.add_argument("--workers", type=int, default=10, help="grading jobs at once")
    gate.set_defaults(run=cmd_gate)

    bump = commands.add_parser("bump-version", help="set meta.yaml version from the tests")
    _add_out(bump)
    bump.add_argument("id")
    bump.add_argument("--base", default="HEAD", help="git revision the edit started from")
    bump.set_defaults(run=cmd_bump_version)

    add_test = commands.add_parser("add-test", help="add one hidden test as one transaction")
    _add_out(add_test)
    add_test.add_argument("id")
    add_test.add_argument("testfile", type=Path, help="a file holding the one new test")
    add_test.add_argument("--wrong", type=Path, required=True, help="the wrong solution it kills")
    add_test.add_argument("--mistake", required=True, help="the mistake, in plain words")
    add_test.add_argument("--clause", required=True, help="the contract label, e.g. C2")
    add_test.add_argument("--base", default="HEAD", help="git revision the edit started from")
    add_test.set_defaults(run=cmd_add_test)

    mark = commands.add_parser("mark-swept", help="record that a sweep passed for a problem")
    _add_out(mark)
    mark.add_argument("id")
    mark.add_argument("--kind", required=True, choices=_SWEEPS)
    mark.add_argument("--mode", required=True, choices=("none", "add"))
    mark.add_argument("--base", default="HEAD", help="git revision the edit started from")
    mark.set_defaults(run=cmd_mark_swept)

    worklist = commands.add_parser("worklist", help="list the problems one kind still needs")
    _add_out(worklist)
    worklist.add_argument("--kind", required=True, choices=_WORKLIST_KINDS)
    worklist.add_argument("--today", help="the date to plan from (default: today)")
    worklist.set_defaults(run=cmd_worklist)

    score = commands.add_parser("scorecard", help="one row per problem, as Markdown")
    _add_out(score)
    score.set_defaults(run=cmd_scorecard)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    try:
        return int(args.run(args))
    except UsageError as exc:
        print(exc, file=sys.stderr)
        return 2
    except GateError as exc:
        print(exc, file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
