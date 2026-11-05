"""Stopgap practice loop until the ``dogma`` CLI exists (ROADMAP increment I-05).

Commands (all take a problem directory, e.g. content/cml/vector-operations/problems/p01-...):

    open    copy the starter into workspace/ and print its path
    run     grade your workspace file against the public tests
    submit  grade your workspace file against public + hidden tests
    verify  author check: solution passes everything (twice), starter fails, no banned imports

Tests run in a fresh temporary directory exactly as the real runner will run them
(docs/CONTENT_SPEC.md §6.4).
"""

from __future__ import annotations

import argparse
import ast
import shutil
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path

import yaml
from rich.console import Console
from rich.markup import escape

REPO = Path(__file__).resolve().parent.parent
WORKSPACE = REPO / "workspace"
console = Console(highlight=False)


@dataclass
class TestResult:
    name: str
    file: str  # "public" | "hidden"
    outcome: str  # "pass" | "fail" | "error" | "skip"
    message: str = ""


@dataclass
class RunResult:
    tests: list[TestResult] = field(default_factory=list)
    crashed: str = ""  # non-empty when pytest itself could not run (timeout, collection error)

    def counts(self, file: str) -> tuple[int, int]:
        mine = [t for t in self.tests if t.file == file]
        return sum(t.outcome == "pass" for t in mine), len(mine)

    @property
    def all_passed(self) -> bool:
        ok = all(t.outcome == "pass" for t in self.tests)
        return not self.crashed and bool(self.tests) and ok


def load_meta(problem: Path) -> dict[str, object]:
    meta = yaml.safe_load((problem / "meta.yaml").read_text())
    if not isinstance(meta, dict):
        raise SystemExit(f"{problem}/meta.yaml is not a mapping")
    return meta


def suffix(meta: dict[str, object]) -> str:
    return ".sql" if meta.get("language") == "sql" else ".py"


def workspace_file(problem: Path, meta: dict[str, object]) -> Path:
    rel = problem.resolve().relative_to(REPO / "content")
    return WORKSPACE / (str(rel).replace("/problems/", "/") + suffix(meta))


def run_tests(problem: Path, submission: Path, meta: dict[str, object], hidden: bool) -> RunResult:
    files = ["test_public.py"] + (["test_hidden.py"] if hidden else [])
    timeout = int(meta.get("timeout_s", 30))  # type: ignore[call-overload]
    with tempfile.TemporaryDirectory(prefix="dogma-run-") as tmp:
        run_dir = Path(tmp)
        shutil.copy(submission, run_dir / f"submission{suffix(meta)}")
        for name in files:
            shutil.copy(problem / name, run_dir / name)
        if (problem / "fixtures").is_dir():
            shutil.copytree(problem / "fixtures", run_dir / "fixtures")
        cmd = [
            sys.executable,
            "-m",
            "pytest",
            "-q",
            "-p",
            "no:cacheprovider",
            "--hypothesis-seed=0",
            "--junitxml=report.xml",
            *files,
        ]
        try:
            proc = subprocess.run(
                cmd, cwd=run_dir, capture_output=True, text=True, timeout=timeout, check=False
            )
        except subprocess.TimeoutExpired:
            return RunResult(crashed=f"timed out after {timeout}s (infinite loop?)")
        report = run_dir / "report.xml"
        if not report.exists():
            return RunResult(crashed=(proc.stdout + proc.stderr)[-2000:])
        return parse_junit(report, proc.stdout)


def parse_junit(report: Path, stdout: str) -> RunResult:
    result = RunResult()
    # report.xml is written by our own pytest run in a temp dir, not untrusted input.
    for case in ET.parse(report).getroot().iter("testcase"):  # noqa: S314
        file = "hidden" if "test_hidden" in (case.get("classname") or "") else "public"
        outcome, message = "pass", ""
        for tag in ("failure", "error", "skipped"):
            node = case.find(tag)
            if node is not None:
                outcome = {"failure": "fail", "error": "error", "skipped": "skip"}[tag]
                message = (node.get("message") or node.text or "").strip()
                break
        result.tests.append(TestResult(case.get("name", "?"), file, outcome, message))
    if not result.tests:  # collection error: e.g. submission has a syntax error or missing name
        result.crashed = stdout[-2000:]
    return result


def banned_usage(source: Path, banned: list[str]) -> list[str]:
    """Return human-readable violations of ``banned_imports`` (imports and attribute access)."""
    if source.suffix != ".py" or not banned:
        return []
    tree = ast.parse(source.read_text())
    found: list[str] = []
    for node in ast.walk(tree):
        names: list[str] = []
        if isinstance(node, ast.Import):
            names = [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom) and node.module:
            names = [node.module] + [f"{node.module}.{alias.name}" for alias in node.names]
        elif isinstance(node, ast.Attribute):
            names = [node.attr]
        for name in names:
            for mod in banned:
                last = mod.rsplit(".", 1)[-1]
                if isinstance(node, ast.Attribute):  # bare banned names match imports only
                    hit = "." in mod and name == last
                else:
                    hit = name == mod or name.startswith(mod + ".")
                if hit:
                    found.append(f"line {getattr(node, 'lineno', '?')}: uses {mod}")
    return sorted(set(found))


# Lines in numpy.testing / pytest messages that reveal hidden inputs or values, not the mistake.
_VALUE_LINES = (
    "x:",
    "y:",
    "ACTUAL",
    "DESIRED",
    "Max absolute",
    "Max relative",
    "Mismatched",
    "Arrays are not",
    "Mismatch at",
    "+",
    "-",
    "E ",
    "Not equal to tolerance",
    "Falsifying example",
    "assert ",
    "where ",
)


def hidden_summary(message: str) -> str:
    """Keep the lines of a hidden-test failure that name the mistake, drop the ones with values."""
    lines = [ln.strip() for ln in message.splitlines()]
    keep = [
        ln for ln in lines if ln and ln != "AssertionError:" and not ln.startswith(_VALUE_LINES)
    ]
    if not keep:
        return "(the test failed; its name describes what was checked)"
    return keep[0].removeprefix("AssertionError: ").removeprefix("Failed: ")


def show(result: RunResult, hidden: bool) -> None:
    if result.crashed:
        console.print(
            "[bold red]Could not run the tests[/] (syntax error, missing function, or timeout):"
        )
        console.print(result.crashed)
        return
    for t in result.tests:
        mark = {"pass": "[green]✓[/]", "skip": "[yellow]-[/]"}.get(t.outcome, "[red]✗[/]")
        console.print(f"  {mark} [dim]{t.file:6}[/] {t.name}")
        if t.outcome in {"fail", "error"}:
            detail = t.message if t.file == "public" else hidden_summary(t.message)
            console.print(f"      [red]{escape(detail[:1500])}[/]")
    pub = result.counts("public")
    line = f"public {pub[0]}/{pub[1]}"
    if hidden:
        hid = result.counts("hidden")
        line += f" · hidden {hid[0]}/{hid[1]}"
    colour = "green" if result.all_passed else "yellow"
    console.print(f"[bold {colour}]{line}[/]")


def cmd_open(problem: Path) -> int:
    meta = load_meta(problem)
    dest = workspace_file(problem, meta)
    if dest.exists():
        console.print(f"Already open (not overwritten): {dest}")
    else:
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(problem / f"starter{suffix(meta)}", dest)
        console.print(f"Starter copied to: {dest}")
    console.print(f"Read the task: {problem / 'problem.md'}")
    return 0


def cmd_grade(problem: Path, hidden: bool, submission: Path | None) -> int:
    meta = load_meta(problem)
    sub = submission or workspace_file(problem, meta)
    if not sub.exists():
        console.print(f"No submission at {sub}. Run `open` first.")
        return 2
    violations = banned_usage(sub, list(meta.get("banned_imports") or []))  # type: ignore[call-overload]
    if violations:
        console.print("[bold red]Rejected: banned imports[/]\n  " + "\n  ".join(violations))
        return 1
    result = run_tests(problem, sub, meta, hidden=hidden)
    show(result, hidden)
    return 0 if result.all_passed else 1


def cmd_verify(problem: Path) -> int:
    meta = load_meta(problem)
    ext = suffix(meta)
    problems: list[str] = []
    banned_list = list(meta.get("banned_imports") or [])  # type: ignore[call-overload]
    banned = banned_usage(problem / f"solution{ext}", banned_list)
    problems += [f"solution {v}" for v in banned]
    first = run_tests(problem, problem / f"solution{ext}", meta, hidden=True)
    second = run_tests(problem, problem / f"solution{ext}", meta, hidden=True)
    if not first.all_passed:
        show(first, hidden=True)
        problems.append("solution does not pass every test")
    if [(t.name, t.outcome) for t in first.tests] != [(t.name, t.outcome) for t in second.tests]:
        problems.append("outcomes differ between two runs (non-deterministic tests)")
    n_pub, n_hid = first.counts("public")[1], first.counts("hidden")[1]
    if n_pub < 3:
        problems.append(f"needs >= 3 public tests, has {n_pub}")
    if n_hid < 5:
        problems.append(f"needs >= 5 hidden tests, has {n_hid}")
    starter = run_tests(problem, problem / f"starter{ext}", meta, hidden=True)
    if starter.all_passed:
        problems.append("starter passes every test (tests are too weak)")
    if problems:
        console.print(f"[bold red]✗ {problem}[/]\n  " + "\n  ".join(problems))
        return 1
    console.print(
        f"[bold green]✓ {problem}[/]  public {n_pub} · hidden {n_hid}"
        " · deterministic · starter fails"
    )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("command", choices=["open", "run", "submit", "verify"])
    parser.add_argument("problems", nargs="+", type=Path, help="problem directory/directories")
    parser.add_argument(
        "--submission", type=Path, help="grade this file instead of the workspace copy"
    )
    args = parser.parse_args()
    codes = []
    for raw in args.problems:
        problem = raw.resolve()
        if not (problem / "meta.yaml").exists():
            console.print(f"[red]{problem} is not a problem directory (no meta.yaml)[/]")
            codes.append(2)
            continue
        if args.command == "open":
            codes.append(cmd_open(problem))
        elif args.command == "verify":
            codes.append(cmd_verify(problem))
        else:
            codes.append(cmd_grade(problem, args.command == "submit", args.submission))
    return max(codes)


if __name__ == "__main__":
    raise SystemExit(main())
