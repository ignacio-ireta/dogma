"""Gate for I-32 test strength: lint hidden tests, list clauses, probe a solution, keep notes.

Usage (the torch group is needed for the py-torch runtime)::

    uv run --frozen --group torch python scripts/exercise_gate.py COMMAND ...

    lint     [--only PREFIX]... --out DIR
    clauses  --out DIR ID
    probe    --out DIR ID FILE
    note     --out DIR ID --kind KIND --text TEXT
    ledger   --out DIR [--all] [--kind KIND]...

Every sub-command takes ``--out DIR`` (created when missing). Files are written only under DIR.
Exit codes: 0 success, 1 a check failed, 2 wrong usage.
"""

from __future__ import annotations

import argparse
import difflib
import json
import re
import sys
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from dogma.content.markdown import parse_outline
from dogma.content.testlint import lint_hidden_tests
from dogma.core.config import Settings
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


def cmd_probe(args: argparse.Namespace) -> int:
    problem = find_problem(args.id)
    if not args.file.is_file():
        raise UsageError(f"no file {args.file}")
    source = args.file.read_text(encoding="utf-8")
    report = audit.grade_source(problem, source, Settings())
    public_passed, public_total = report.counts("public")
    hidden_passed, hidden_total = report.counts("hidden")
    print(f"PROBE {problem.id}  file={args.file}")
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


# --- command line -------------------------------------------------------------------------------


def _add_out(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--out", type=Path, required=True, help="directory for results")


def _parser() -> argparse.ArgumentParser:
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
