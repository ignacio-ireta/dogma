"""Privacy audit (docs/IMPORT_PLAN.md section 2): fail on personal or employer terms in the repo.

Scans the product working tree and generated browser assets for:

- every regex in ``private/denylist.txt`` (one per line; case-insensitive unless the line starts
  with ``(?-i)``, which is then stripped and matched case-sensitively; ``#`` lines and blank lines
  are ignored), and
- built-in patterns: non-example email addresses, phone numbers with a leading ``+`` and
  local account paths. Common Unicode, HTML and URL escapes are checked after decoding.

All tracked/unignored files and explicit build locations are checked; an exported tree without
Git is scanned recursively, excluding private state and dependencies. Filenames are checked too.
Private rules are required and never printed. Private file bodies are never inspected; a
force-tracked private path, a publication symlink or a file over 5 MB blocks certification.
UTF-8 and BOM-marked UTF-16 text are supported; arbitrary binary payloads are not decoded.
Checksum-pinned upstream attribution/parser data exemptions are documented beside the hashes.

``--staged`` also checks actual index blobs, even when disk contents differ; the commit hook
uses it. Git history is outside this audit. Exit codes: 0 clean, 1 hits, 2 incomplete audit.
"""

from __future__ import annotations

import argparse
import hashlib
import html
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import unquote

REPO = Path(__file__).resolve().parent.parent
DENYLIST_RELPATH = Path("private/denylist.txt")
SCAN_DIRS = (
    "sources",
    "content",
    "docs",
    "src",
    "scripts",
    "templates",
    "tests",
    "curriculum",
    "web/src",
    "web/public",
    "web/dist",
)
SCAN_GLOB = "tools/*/VENDORED.md"
SCAN_FILES = (
    "README.md",
    "CLAUDE.md",
    "corpus/README.md",
    "pyproject.toml",
    "uv.lock",
    "Makefile",
    ".env.example",
    ".gitignore",
    ".pre-commit-config.yaml",
)
# The unchanged upstream license must retain its attribution/contact. A modified copy is scanned
# normally; private terms are always checked, including in this license.
LICENSE_HASHES = {
    "web/public/fonts/opendyslexic/OFL.txt": (
        "caafcccfb70fc72458fcbda812ec8f0a06cb300cbabb87eabbb30b946124394b"
    ),
    "web/dist/fonts/opendyslexic/OFL.txt": (
        "caafcccfb70fc72458fcbda812ec8f0a06cb300cbabb87eabbb30b946124394b"
    ),
}
# Required third-party copyright attribution, retained byte-for-byte. This is public upstream
# authorship, not learner context. Changes to the license invalidate the exemption.
ATTRIBUTION_HASHES = {
    "tools/docmd/LICENSE": "835ed7dfbbc61cb1c98488722a51c8ea7e92b3e32b6a5bfd776beb9f5440c062",
}
# A compressed parser table contains coincidental short letter sequences. Exempt only this
# exact, reviewed JS string literals; modified data and all surrounding code remain checked.
VERIFIED_PARSER_LITERALS = frozenset(
    {
        "dceea9ff3ef025abbaf82fa5c42831193f24f2b137e4d2e115e3c3edafb4caba",
        "0cc9040e98fa675ad9968d17a0b7c68c175f2c8b41cee13c753a380574dc0d95",
    }
)
_PARSER_LITERAL = re.compile(r'\b(?:stateData|goto)\s*:\s*("(?:[^"\\]|\\.)*")')
MAX_BYTES = 5 * 1024 * 1024
CASE_SENSITIVE_PREFIX = "(?-i)"
SKIP_DIR_NAMES = frozenset({"__pycache__", "node_modules", ".git"})
ALLOWED_EMAIL_DOMAIN = (
    r"(?!(?:[A-Za-z0-9-]+\.)*(?:example\.(?:com|org|net)|(?:test|invalid|localhost))"
    r"(?!\.?[A-Za-z0-9-]))"
)
BUILTIN_RULES: tuple[tuple[str, str, int], ...] = (
    (
        "builtin: email",
        r"[A-Za-z0-9._%+-]+@" + ALLOWED_EMAIL_DOMAIN + r"(?:[A-Za-z0-9-]+\.)+[A-Za-z]{2,}\b",
        0,
    ),
    ("builtin: phone", r"\+\d{1,3}[\s-]?\d{1,4}[\s-]?\d{3,4}[\s-]?\d{4}", 0),
    (
        "builtin: account path",
        r"/(?:Users|home)/[^/\s\"'<>]+|[A-Za-z]:\\+Users\\+[^\\\s\"'<>]+",
        re.IGNORECASE,
    ),
)


@dataclass(frozen=True)
class Rule:
    label: str
    regex: re.Pattern[str]


@dataclass(frozen=True)
class Hit:
    path: str
    lineno: int
    label: str


def load_denylist(path: Path) -> list[Rule]:
    """Compile the denylist; raise SystemExit(2) on a bad regex (the message names the line)."""
    rules: list[Rule] = []
    for lineno, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        case_sensitive = line.startswith(CASE_SENSITIVE_PREFIX)
        source = line[len(CASE_SENSITIVE_PREFIX) :] if case_sensitive else line
        try:
            regex = re.compile(source, 0 if case_sensitive else re.IGNORECASE)
        except re.error:
            print(f"error: private rule {lineno} is not a valid regex", file=sys.stderr)
            raise SystemExit(2) from None
        rules.append(Rule(label=f"private rule {lineno}", regex=regex))
    return rules


def builtin_rules() -> list[Rule]:
    return [
        Rule(label=label, regex=re.compile(pattern, flags))
        for label, pattern, flags in BUILTIN_RULES
    ]


def _walk_tree(base: Path, *, publication: bool = False) -> list[Path]:
    files: list[Path] = []
    for dirpath, dirnames, filenames in os.walk(base):
        files.extend(
            Path(dirpath) / name for name in dirnames if (Path(dirpath) / name).is_symlink()
        )
        if publication:
            dirnames[:] = [
                name
                for name in dirnames
                if name
                not in {
                    ".superpowers",
                    ".claude",
                    ".pytest_cache",
                    ".mypy_cache",
                    ".ruff_cache",
                    ".hypothesis",
                    ".git",
                }
                and not (Path(dirpath) == base and name in {"private", "var", "workspace"})
                and not (Path(dirpath) / name).is_symlink()
            ]
            if Path(dirpath).relative_to(base).parts[:1] == ("corpus",):
                dirnames.clear()
        dirnames[:] = [
            name
            for name in dirnames
            if name not in SKIP_DIR_NAMES
            and not name.startswith(".venv")
            and not (Path(dirpath) / name).is_symlink()
        ]
        for name in filenames:
            files.append(Path(dirpath) / name)
    return files


def _no_symlinks(root: Path, path: Path) -> bool:
    current = root
    for part in path.relative_to(root).parts:
        current /= part
        if current.is_symlink():
            return False
    return True


def _publication_files(root: Path) -> list[Path]:
    """Include every tracked or unignored publication file, not just the conventional folders."""
    if not (root / ".git").exists():
        return [
            path
            for path in _walk_tree(root, publication=True)
            if not _forbidden(path.relative_to(root).as_posix())
        ]
    try:
        result = subprocess.run(
            [
                "git",
                "-C",
                str(root),
                "ls-files",
                "--cached",
                "--others",
                "--exclude-standard",
                "-z",
            ],
            capture_output=True,
            check=True,
            timeout=15,
        )
    except (OSError, subprocess.SubprocessError):
        print("error: cannot enumerate publication files", file=sys.stderr)
        raise SystemExit(2) from None
    paths = [root / os.fsdecode(name) for name in result.stdout.split(b"\0") if name]
    return [path for path in paths if not _no_symlinks(root, path) or path.is_file()]


def scan_targets(root: Path) -> list[Path]:
    """Publication files plus symlinks to reject, including ignored build locations."""
    targets: list[Path] = []
    for name in SCAN_DIRS:
        base = root
        for part in Path(name).parts:
            base /= part
            if base.is_symlink():
                targets.append(base)
                break
        else:
            if base.is_dir():
                targets.extend(_walk_tree(base))
    targets.extend(sorted(p for p in root.glob(SCAN_GLOB) if p.is_file() and _no_symlinks(root, p)))
    targets.extend(
        root / name
        for name in SCAN_FILES
        if (root / name).is_file() and _no_symlinks(root, root / name)
    )
    targets.extend(_publication_files(root))
    return sorted(set(targets))


def scan_text(rel: str, text: str, rules: list[Rule]) -> list[Hit]:
    if rel.endswith(".js"):
        text = _PARSER_LITERAL.sub(_parser_literal, text)
    hits: list[Hit] = []
    for lineno, line in enumerate(text.splitlines(), start=1):
        normalized = _normalized(line)
        hits.extend(Hit(rel, lineno, rule.label) for rule in rules if rule.regex.search(normalized))
    return hits


def _parser_literal(match: re.Match[str]) -> str:
    literal = match.group(1)
    if hashlib.sha256(literal.encode()).hexdigest() in VERIFIED_PARSER_LITERALS:
        return "[verified parser table]" + "\n" * literal.count("\n")
    return match.group()


_UNICODE_ESCAPE = re.compile(
    r"\\(?:u\{([0-9a-fA-F]{1,6})\}|u([0-9a-fA-F]{4})|U([0-9a-fA-F]{8})|x([0-9a-fA-F]{2}))"
)


def _unicode_character(match: re.Match[str]) -> str:
    value = int(next(group for group in match.groups() if group is not None), 16)
    return chr(value) if value <= 0x10FFFF else match.group()


def _normalized(text: str) -> str:
    return html.unescape(unquote(_UNICODE_ESCAPE.sub(_unicode_character, text).replace(r"\/", "/")))


def _forbidden(rel: str) -> bool:
    first = Path(rel).parts[0]
    return (
        first in {"private", "var", "workspace", ".superpowers", ".claude", ".git"}
        or (first == "corpus" and rel != "corpus/README.md")
        or (Path(rel).name.startswith(".env") and Path(rel).name != ".env.example")
    )


def _safe_relative(rel: str, rules: list[Rule]) -> str:
    result = _normalized(rel)
    for rule in rules:
        result = rule.regex.sub("[redacted]", result)
    return result


def _path_hits(rel: str, rules: list[Rule]) -> list[Hit]:
    safe = _safe_relative(rel, rules)
    hits = [Hit(safe, 0, rule.label) for rule in rules if rule.regex.search(_normalized(rel))]
    if _forbidden(rel):
        hits.append(Hit(safe, 0, "builtin: forbidden publication"))
    return hits


def _scan_content(rel: str, data: bytes, rules: list[Rule]) -> tuple[list[Hit], int]:
    safe = _safe_relative(rel, rules)
    if len(data) > MAX_BYTES:
        return [Hit(safe, 0, "builtin: file size limit")], 0
    try:
        if data.startswith((b"\xff\xfe", b"\xfe\xff")):
            text = data.decode("utf-16")
        elif b"\0" not in data:
            text = data.decode("utf-8-sig")
        else:
            return [], 0
    except UnicodeDecodeError:
        return [], 0
    checksum = hashlib.sha256(data).hexdigest()
    file_rules = rules
    if ATTRIBUTION_HASHES.get(rel) == checksum:
        file_rules = builtin_rules()
    if LICENSE_HASHES.get(rel) == checksum:
        file_rules = [rule for rule in rules if rule.label != "builtin: email"]
    return scan_text(safe, text, file_rules), 1


def _audit_index(root: Path, rules: list[Rule]) -> tuple[list[Hit], int]:
    if not (root / ".git").exists():
        print("error: staged auditing requires a Git checkout", file=sys.stderr)
        raise SystemExit(2)
    entries = _git(root, "ls-files", "--stage", "-z")
    hits: list[Hit] = []
    scanned = 0
    for entry in entries.split(b"\0"):
        if not entry:
            continue
        metadata, name = entry.split(b"\t", 1)
        mode, blob, stage = metadata.split()
        rel = os.fsdecode(name)
        hits.extend(_path_hits(rel, rules))
        if _forbidden(rel):
            continue  # Never read a private blob, even when it was force-added to Git.
        if mode not in {b"100644", b"100755"} or stage != b"0":
            hits.append(
                Hit(_safe_relative(rel, rules), 0, "builtin: unsupported publication entry")
            )
            continue
        more, count = _scan_content(
            rel, _git(root, "cat-file", "blob", blob.decode("ascii")), rules
        )
        hits.extend(more)
        scanned += count
    return hits, scanned


def _git(root: Path, *arguments: str) -> bytes:
    try:
        return subprocess.run(
            ["git", "-C", str(root), *arguments], capture_output=True, check=True, timeout=15
        ).stdout
    except (OSError, subprocess.SubprocessError):
        print("error: cannot inspect publication metadata", file=sys.stderr)
        raise SystemExit(2) from None


def audit(root: Path, denylist: Path, *, staged: bool = False) -> tuple[list[Hit], int]:
    """Check the product tree; optionally also check actual index blobs before a commit."""
    try:
        private_rules = load_denylist(denylist)
    except OSError:
        private_rules = []
    if not private_rules:
        print(
            "error: nonempty local privacy rules are required in private/denylist.txt",
            file=sys.stderr,
        )
        raise SystemExit(2)
    rules = private_rules + builtin_rules()
    hits: list[Hit] = []
    scanned = 0
    for path in scan_targets(root):
        rel = path.relative_to(root).as_posix()
        hits.extend(_path_hits(rel, rules))
        if _forbidden(rel):
            continue
        if not _no_symlinks(root, path):
            hits.append(Hit(_safe_relative(rel, rules), 0, "builtin: publication symlink"))
            continue
        if path.stat().st_size > MAX_BYTES:
            hits.append(Hit(_safe_relative(rel, rules), 0, "builtin: file size limit"))
            continue
        more, count = _scan_content(rel, path.read_bytes(), rules)
        hits.extend(more)
        scanned += count
    if staged:
        more, count = _audit_index(root, rules)
        hits.extend(more)
        scanned += count
    return sorted(set(hits), key=lambda hit: (hit.path, hit.lineno, hit.label)), scanned


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fail on personal or employer terms in the repo.")
    parser.add_argument("--root", type=Path, default=REPO, help=argparse.SUPPRESS)
    parser.add_argument("--denylist", type=Path, default=None, help=argparse.SUPPRESS)
    parser.add_argument(
        "--staged", action="store_true", help="Also audit the Git index before a commit."
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    root: Path = args.root
    denylist: Path = args.denylist or root / DENYLIST_RELPATH
    try:
        hits, scanned = audit(root, denylist, staged=args.staged)
    except SystemExit as exc:
        return int(exc.code or 2)
    except OSError:
        print("error: cannot read a publication file; audit incomplete", file=sys.stderr)
        return 2
    for hit in hits:
        print(f"{hit.path}:{hit.lineno}: {hit.label}")
    files_with_hits = len({hit.path for hit in hits})
    if hits:
        print(f"audit: {len(hits)} hit(s) in {files_with_hits} file(s); scanned {scanned} files")
        return 1
    print(f"audit: clean; scanned {scanned} files")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
