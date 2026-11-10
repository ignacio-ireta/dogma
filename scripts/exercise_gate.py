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

Every sub-command takes ``--out DIR`` (created when missing). Files are written only under DIR,
or under a problem's variants/ directory for the commands that change a manifest.
Exit codes: 0 success, 1 a check failed, 2 wrong usage.
"""

from __future__ import annotations

import argparse
import difflib
import json
import re
import shutil
import subprocess
import sys
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ValidationError

from dogma.content.loader import load_problem
from dogma.content.markdown import parse_outline
from dogma.content.schemas import (
    VARIANT_ACCEPT_PATTERN,
    VARIANT_REJECT_PATTERN,
    AcceptVariant,
    RejectVariant,
    UncoveredClause,
    VariantsManifest,
)
from dogma.content.testlint import lint_hidden_tests
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
_RUFF_STEPS: tuple[tuple[str, ...], ...] = (("format",), ("check", "--fix"), ("check",))


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
        for step in _RUFF_STEPS:
            ok, output = _ruff(step, target)
            if not ok:
                target.unlink()
                raise GateError(f"ruff {' '.join(step)} failed on {name}:\n{output}")
    return target


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
