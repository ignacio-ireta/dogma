"""Reject whitespace mistakes and files over 1 MB without modifying files."""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
MAX_BYTES = 1_000_000
# Add exact tests/fixtures/ paths here only with an explanation of the required size.
LARGE_FILE_EXCEPTIONS: dict[str, str] = {}


def check_files(
    root: Path,
    files: Sequence[Path],
    exceptions: Mapping[str, str] = LARGE_FILE_EXCEPTIONS,
) -> list[str]:
    """Return diagnostics; size exceptions require a fixture path and a reason."""
    issues: list[str] = []
    for path in files:
        rel = path.relative_to(root).as_posix()
        size = path.stat().st_size
        exempt = rel.startswith("tests/fixtures/") and bool(exceptions.get(rel, "").strip())
        if size > MAX_BYTES and not exempt:
            issues.append(f"{rel}: {size} exceeds {MAX_BYTES} bytes")
            continue
        data = path.read_bytes()
        if b"\0" in data:
            continue
        try:
            text = data.decode("utf-8")
        except UnicodeDecodeError:
            continue
        for number, line in enumerate(text.splitlines(), start=1):
            if line.rstrip(" \t") != line:
                issues.append(f"{rel}:{number}: trailing whitespace")
        if text and not text.endswith("\n"):
            issues.append(f"{rel}: missing final newline")
    return issues


def main(argv: list[str] | None = None) -> int:
    """Check filenames supplied by pre-commit, returning a failing status on errors."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=REPO)
    parser.add_argument("files", type=Path, nargs="*")
    args = parser.parse_args(argv)
    root = args.root.resolve()
    try:
        paths = [(root / path).resolve() for path in args.files]
        issues = check_files(root, paths)
    except (OSError, ValueError) as exc:
        print(f"commit-files: {exc}")
        return 1
    for issue in issues:
        print(issue)
    return int(bool(issues))


if __name__ == "__main__":
    raise SystemExit(main())
