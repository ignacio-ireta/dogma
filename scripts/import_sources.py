"""Import source material into the repo, as described by docs/IMPORT_PLAN.md section 4.

Reads ``sources/manifest.yaml`` and the origin roots in ``private/import_roots.yaml``, then:

- copies ``import`` files to ``sources/<id>/`` (scrub them afterwards, then ``--relock``),
- copies ``corpus`` files to ``corpus/raw/<id>/`` (third-party, gitignored),
- copies ``vendor`` files to ``tools/<id>/`` (license kept, provenance in VENDORED.md),
- records every copy in ``sources/manifest.lock.json``.

Guarantees:

- Idempotent: with unchanged origins and copies, a re-run writes nothing.
- Scrub-preserving: a scrubbed copy is never overwritten by a plain copy. A changed origin is
  reported and skipped unless ``--force`` is given.
- ``--relock`` records the current disk state: it recomputes ``sha256`` for existing copies and
  marks as ``scrubbed`` any copy that differs from its origin hash. Lock entries whose file is
  gone are dropped.
- Nothing written contains an absolute path or an origin root path. ``origin_rel`` is the origin
  path with the manifest ``rename`` rules applied, so personal name suffixes are never recorded.
- Origins are read only for files matched by a manifest rule. Git is queried only for roots that
  contain their own ``.git``, so no parent repository is ever touched.
- ``sources/withdrawn.yaml`` lists destinations that were deliberately removed after review
  (``- {dest: ..., reason: ...}``). They are never re-imported.
- ``--source-id`` (repeatable) and ``--kind`` (snapshot, corpus, vendor; repeatable) select manifest
  entries. Unselected entries are neither resolved nor read, so a selected run does not need every
  origin root to exist on this machine.
- A corpus entry may list ``include:`` exact relative paths instead of a ``glob``. Every listed
  file must exist, or nothing is copied.
- ``--inventory <json>`` checks ``items[].path`` / ``items[].sha256`` before anything is written:
  each inventory file that a selected entry would import must exist under that entry's root and
  match its hash. Failures list relative paths only and exit nonzero with no writes.

Usage::

    uv run --frozen python scripts/import_sources.py            # import / refresh
    uv run --frozen python scripts/import_sources.py --relock   # after scrubbing
    uv run --frozen python scripts/import_sources.py --force    # re-import changed origins
    uv run --frozen python scripts/import_sources.py --source-id library-refresh \\
        --kind snapshot --inventory docs/source-refresh-2026-10-28.json   # notes only
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Literal, TypedDict, cast

import yaml

REPO = Path(__file__).resolve().parent.parent
MANIFEST_PATH = REPO / "sources" / "manifest.yaml"
ROOTS_PATH = REPO / "private" / "import_roots.yaml"
LOCK_PATH = REPO / "sources" / "manifest.lock.json"
WITHDRAWN_PATH = REPO / "sources" / "withdrawn.yaml"
SKIP_DIR_NAMES = frozenset({".git", "node_modules", "__pycache__"})
SKIP_FILE_NAMES = frozenset({".DS_Store"})
HASH_CHUNK_BYTES = 1 << 20
Action = Literal["copied", "kept", "forced", "skipped", "withdrawn"]
ENTRY_KEYS: dict[str, str] = {"snapshot": "snapshots", "corpus": "corpus", "vendor": "vendor"}
Selector = Callable[[str], bool]


class LockEntry(TypedDict):
    dest: str
    source_id: str
    origin_rel: str
    origin_commit: str | None
    sha256_origin: str
    sha256: str
    scrubbed: bool


@dataclass(frozen=True)
class PlannedFile:
    """One origin file that a manifest rule selects, with its destination."""

    source_id: str
    origin_path: Path
    origin_rel: str
    dest_rel: str
    origin_commit: str | None


def glob_to_regex(pattern: str) -> re.Pattern[str]:
    """Compile a manifest glob into a regex to be used with ``fullmatch`` on POSIX paths.

    ``**/`` matches zero or more directories, a bare ``**`` matches anything including slashes,
    and ``*`` / ``?`` never cross a slash. Everything else is literal. (Python 3.12 has no
    ``PurePath.full_match``, and the manifest uses no character classes.)
    """
    parts: list[str] = []
    i = 0
    while i < len(pattern):
        if pattern.startswith("**/", i):
            parts.append("(?:.*/)?")
            i += 3
        elif pattern.startswith("**", i):
            parts.append(".*")
            i += 2
        else:
            char = pattern[i]
            parts.append({"*": "[^/]*", "?": "[^/]"}.get(char, re.escape(char)))
            i += 1
    return re.compile("".join(parts))


def warn(message: str) -> None:
    print(f"warning: {message}", file=sys.stderr)


def fail(message: str) -> SystemExit:
    return SystemExit(f"error: {message}")


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(HASH_CHUNK_BYTES):
            digest.update(chunk)
    return digest.hexdigest()


def load_yaml(path: Path) -> Any:  # Any: YAML documents are untyped
    if not path.exists():
        raise fail(f"{path.relative_to(REPO).as_posix()} is missing")
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def resolve_roots(spec: dict[str, Any]) -> dict[str, Path]:
    return {key: Path(str(value)).expanduser() for key, value in spec.items()}


def root_for(roots: dict[str, Path], key: str) -> Path:
    path = roots.get(key)
    if path is None:
        raise fail(f"no origin root named {key!r} in private/import_roots.yaml")
    if not path.is_dir():
        raise fail(f"origin root {key!r} is not present on this machine")
    return path


def git_commit(root: Path) -> str | None:
    """Short HEAD of ``root`` if it is itself a git checkout; never looks at parent folders."""
    if not (root / ".git").exists():
        return None
    result = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "--short", "HEAD"],
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )
    commit = result.stdout.strip()
    return commit if result.returncode == 0 and commit else None


def _skip_dir(name: str) -> bool:
    return name in SKIP_DIR_NAMES or name.startswith(".venv")


def walk_files(root: Path) -> list[str]:
    """POSIX paths of regular files under ``root``, relative to it, sorted. Symlinks are refused."""
    found: list[str] = []
    for dirpath, dirnames, filenames in os.walk(root):
        current = Path(dirpath)
        for name in dirnames:
            if (current / name).is_symlink() and not _skip_dir(name):
                warn(f"symlinked folder ignored: {(current / name).relative_to(root).as_posix()}")
        dirnames[:] = [name for name in dirnames if not _skip_dir(name)]
        for name in filenames:
            path = current / name
            if name in SKIP_FILE_NAMES:
                continue
            if path.is_symlink():
                warn(f"symlink ignored: {path.relative_to(root).as_posix()}")
                continue
            found.append(path.relative_to(root).as_posix())
    return sorted(found)


def is_selected(rel: str, rules: list[tuple[re.Pattern[str], str]]) -> bool:
    """Last matching rule wins; no match means not selected."""
    action = "exclude"
    for regex, rule_action in rules:
        if regex.fullmatch(rel):
            action = rule_action
    return action == "import"


def apply_renames(rel: str, renames: list[tuple[re.Pattern[str], str]]) -> str:
    parent, sep, name = rel.rpartition("/")
    for regex, replacement in renames:
        name = regex.sub(replacement, name)
    return f"{parent}{sep}{name}"


def is_plain_file(path: Path) -> bool:
    return path.is_file() and not path.is_symlink()


def checked_rel(rel: str, where: str) -> str:
    """Reject absolute or parent-relative paths from the manifest or an inventory."""
    pure = PurePosixPath(rel)
    if not rel or pure.is_absolute() or ".." in pure.parts:
        raise fail(f"{where} contains an unsafe path (absolute or '..'); not recorded here")
    return rel


def snapshot_selector(snap: dict[str, Any]) -> Selector:
    rules = [(glob_to_regex(str(rule["glob"])), str(rule["action"])) for rule in snap["rules"]]
    return lambda rel: is_selected(rel, rules)


def corpus_selector(entry: dict[str, Any]) -> Selector:
    if "include" in entry:
        allowed = frozenset(checked_rel(str(p), str(entry["id"])) for p in entry["include"])
        return lambda rel: rel in allowed
    regex = glob_to_regex(str(entry["glob"]))
    return lambda rel: regex.fullmatch(rel) is not None


def vendor_selector(entry: dict[str, Any]) -> Selector:
    includes = [glob_to_regex(str(p)) for p in entry["include"]]
    excludes = [glob_to_regex(str(p)) for p in entry.get("exclude") or []]
    return lambda rel: (
        any(p.fullmatch(rel) for p in includes) and not any(p.fullmatch(rel) for p in excludes)
    )


def selector_for(kind: str, entry: dict[str, Any]) -> Selector:
    if kind == "snapshot":
        return snapshot_selector(entry)
    if kind == "corpus":
        return corpus_selector(entry)
    return vendor_selector(entry)


def plan_snapshot(snap: dict[str, Any], roots: dict[str, Path]) -> list[PlannedFile]:
    source_id = str(snap["id"])
    root = root_for(roots, str(snap["root"]))
    select = snapshot_selector(snap)
    renames = [
        (re.compile(pattern), str(replacement))
        for pattern, replacement in (snap.get("rename") or {}).items()
    ]
    commit = git_commit(root)
    planned: list[PlannedFile] = []
    for rel in walk_files(root):
        if not select(rel):
            continue
        origin_rel = apply_renames(rel, renames)
        planned.append(
            PlannedFile(
                source_id, root / rel, origin_rel, f"sources/{source_id}/{origin_rel}", commit
            )
        )
    return planned


def plan_corpus(entry: dict[str, Any], roots: dict[str, Path]) -> list[PlannedFile]:
    source_id = str(entry["id"])
    if "include" in entry and "glob" in entry:
        raise fail(f"corpus {source_id} sets both glob and include; use one")
    root = root_for(roots, str(entry["root"]))
    commit = git_commit(root)
    if "include" in entry:
        rels = sorted(checked_rel(str(p), source_id) for p in entry["include"])
        missing = [rel for rel in rels if not is_plain_file(root / rel)]
        if missing:
            raise fail(
                f"corpus {source_id}: {len(missing)} allowlisted file(s) missing under the "
                "origin root; nothing copied:\n  " + "\n  ".join(missing)
            )
        return [
            PlannedFile(source_id, root / rel, rel, f"corpus/raw/{source_id}/{rel}", commit)
            for rel in rels
        ]
    select = corpus_selector(entry)
    return [
        PlannedFile(source_id, root / rel, rel, f"corpus/raw/{source_id}/{rel}", commit)
        for rel in walk_files(root)
        if select(rel)
    ]


def plan_vendor(entry: dict[str, Any], roots: dict[str, Path]) -> list[PlannedFile]:
    source_id = str(entry["id"])
    root = root_for(roots, str(entry["root"]))
    select = vendor_selector(entry)
    commit = str(entry["origin_commit"])
    return [
        PlannedFile(source_id, root / rel, rel, f"tools/{source_id}/{rel}", commit)
        for rel in walk_files(root)
        if select(rel)
    ]


def load_inventory(path: Path) -> dict[str, str]:
    """Map each inventory ``path`` (Library-relative) to its ``sha256``."""
    data = cast(Any, json.loads(path.read_text(encoding="utf-8")))
    items = data.get("items") if isinstance(data, dict) else None
    if not isinstance(items, list):
        raise fail(f"inventory {path.name} has no items list")
    expected: dict[str, str] = {}
    for index, item in enumerate(items):
        rel = str(item.get("path", ""))
        sha = str(item.get("sha256", ""))
        checked_rel(rel, f"inventory {path.name} item {index}")
        if not re.fullmatch(r"[0-9a-f]{64}", sha):
            raise fail(f"inventory {path.name} item {index} has no valid sha256")
        if rel in expected:
            raise fail(f"inventory {path.name} lists item {index} twice")
        expected[rel] = sha
    return expected


def validate_inventory(root: Path, inventory: Path, *, selected: Selector) -> None:
    """Check every inventory item that ``selected`` accepts against ``root``, before any write.

    ``selected`` receives the Library-relative inventory path and decides whether the entry being
    imported would copy it. Each such file must exist under ``root`` as a regular file (not a
    symlink) and match its ``sha256``. Otherwise this raises ``SystemExit`` listing relative paths
    only. Files that ``selected`` rejects are never read.
    """
    expected = {rel: sha for rel, sha in load_inventory(inventory).items() if selected(rel)}
    missing: list[str] = []
    mismatched: list[str] = []
    for rel in sorted(expected):
        path = root / rel
        if not is_plain_file(path):
            missing.append(rel)
        elif sha256_of(path) != expected[rel]:
            mismatched.append(rel)
    if not missing and not mismatched:
        return
    lines = ["inventory check failed; nothing was written"]
    if missing:
        lines.append(f"{len(missing)} selected file(s) missing under the origin root:")
        lines += [f"  {rel}" for rel in missing]
    if mismatched:
        lines.append(f"{len(mismatched)} selected file(s) differ from the inventory sha256:")
        lines += [f"  {rel}" for rel in mismatched]
    raise fail("\n".join(lines))


def select_manifest(
    manifest: dict[str, Any], source_ids: frozenset[str], kinds: frozenset[str]
) -> tuple[dict[str, Any], list[tuple[str, dict[str, Any]]]]:
    """Keep the entries selected by ``--source-id`` and ``--kind``.

    The second value lists the ``(kind, entry)`` pairs that a ``--source-id`` matched but
    ``--kind`` excluded, so the report can say they were not requested. Unselected entries are
    never resolved, so their origin roots need not exist.
    """
    known = {str(entry["id"]) for key in ENTRY_KEYS.values() for entry in manifest.get(key) or []}
    unknown = sorted(source_ids - known)
    if unknown:
        raise fail(f"unknown --source-id: {', '.join(unknown)}")
    selected: dict[str, Any] = {"version": manifest.get("version")}
    excluded: list[tuple[str, dict[str, Any]]] = []
    for kind, key in ENTRY_KEYS.items():
        kept: list[dict[str, Any]] = []
        for entry in manifest.get(key) or []:
            if source_ids and str(entry["id"]) not in source_ids:
                continue
            if kinds and kind not in kinds:
                if source_ids:
                    excluded.append((kind, entry))
                continue
            kept.append(entry)
        selected[key] = kept
    return selected, excluded


def build_plan(manifest: dict[str, Any], roots: dict[str, Path]) -> list[PlannedFile]:
    if manifest.get("version") != 2:
        raise fail("sources/manifest.yaml must declare version: 2")
    planned: list[PlannedFile] = []
    for snap in manifest.get("snapshots") or []:
        planned += plan_snapshot(snap, roots)
    for entry in manifest.get("corpus") or []:
        planned += plan_corpus(entry, roots)
    for entry in manifest.get("vendor") or []:
        planned += plan_vendor(entry, roots)
    seen: set[str] = set()
    for item in planned:
        if item.dest_rel in seen:
            raise fail(f"two origin files map to the same destination: {item.dest_rel}")
        seen.add(item.dest_rel)
    return sorted(planned, key=lambda item: item.dest_rel)


def load_lock() -> dict[str, LockEntry]:
    if not LOCK_PATH.exists():
        return {}
    entries = cast(list[LockEntry], json.loads(LOCK_PATH.read_text(encoding="utf-8")))
    return {entry["dest"]: entry for entry in entries}


def load_withdrawn() -> set[str]:
    if not WITHDRAWN_PATH.exists():
        return set()
    items = cast(list[dict[str, str]], load_yaml(WITHDRAWN_PATH) or [])
    return {item["dest"] for item in items}


def write_lock(entries: dict[str, LockEntry]) -> bool:
    """Write the lock atomically, sorted by dest. Returns False when nothing changed."""
    ordered = [entries[dest] for dest in sorted(entries)]
    text = json.dumps(ordered, indent=2, ensure_ascii=False) + "\n"
    if LOCK_PATH.exists() and LOCK_PATH.read_text(encoding="utf-8") == text:
        return False
    tmp = LOCK_PATH.with_name(f".{LOCK_PATH.name}.tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(LOCK_PATH)
    return True


def copy_atomic(src: Path, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(f".{dest.name}.tmp")
    shutil.copy2(src, tmp)
    tmp.replace(dest)


def new_entry(item: PlannedFile, origin_sha: str, sha: str) -> LockEntry:
    return {
        "dest": item.dest_rel,
        "source_id": item.source_id,
        "origin_rel": item.origin_rel,
        "origin_commit": item.origin_commit,
        "sha256_origin": origin_sha,
        "sha256": sha,
        "scrubbed": False,
    }


def sync_one(
    item: PlannedFile,
    existing: LockEntry | None,
    origin_sha: str,
    force: bool,
) -> tuple[Action, LockEntry | None]:
    """Decide what to do with one planned file and return its action and new lock entry."""
    dest = REPO / item.dest_rel
    if not dest.exists():
        copy_atomic(item.origin_path, dest)
        return "copied", new_entry(item, origin_sha, sha256_of(dest))
    if existing is None:
        warn(f"{item.dest_rel} exists without a lock entry; left alone (run --relock)")
        return "skipped", None
    disk_sha = sha256_of(dest)
    if disk_sha != existing["sha256"]:
        warn(f"{item.dest_rel} differs from its lock entry; left alone (run --relock)")
        return "kept", existing
    if origin_sha == existing["sha256_origin"]:
        return "kept", existing
    if not force:
        warn(f"origin of {item.dest_rel} changed; skipped (use --force to re-import)")
        return "skipped", existing
    if existing["scrubbed"]:
        warn(f"--force discards the scrub of {item.dest_rel}")
    copy_atomic(item.origin_path, dest)
    return "forced", new_entry(item, origin_sha, sha256_of(dest))


def sync(
    plan: list[PlannedFile],
    lock: dict[str, LockEntry],
    withdrawn: set[str],
    origin_hashes: dict[str, str],
    force: bool,
) -> tuple[dict[str, LockEntry], dict[str, dict[Action, int]]]:
    entries = dict(lock)
    counts: dict[str, dict[Action, int]] = {}
    for item in plan:
        counts.setdefault(item.source_id, {})
        if item.dest_rel in withdrawn:
            action: Action = "withdrawn"
            result = lock.get(item.dest_rel) if (REPO / item.dest_rel).exists() else None
        else:
            action, result = sync_one(
                item, lock.get(item.dest_rel), origin_hashes[item.dest_rel], force
            )
        counts[item.source_id][action] = counts[item.source_id].get(action, 0) + 1
        if result is None:
            entries.pop(item.dest_rel, None)
        else:
            entries[item.dest_rel] = result
    return entries, counts


def relock_entry(entry: LockEntry, origin_sha: str | None) -> LockEntry | None:
    dest = REPO / entry["dest"]
    if not dest.exists():
        warn(f"{entry['dest']} is gone; its lock entry is dropped")
        return None
    if origin_sha is not None and origin_sha != entry["sha256_origin"]:
        warn(f"origin of {entry['dest']} changed since import; lock keeps the import-time hash")
    disk_sha = sha256_of(dest)
    return {
        **entry,
        "sha256": disk_sha,
        "scrubbed": disk_sha != entry["sha256_origin"],
    }


def relock(
    plan: list[PlannedFile],
    lock: dict[str, LockEntry],
    origin_hashes: dict[str, str],
    scope: frozenset[str] | None = None,
) -> dict[str, LockEntry]:
    """Recompute lock entries from disk. With ``scope``, entries of other source ids are kept as-is.

    Without this, a selected relock would drop lock entries for corpus files that are not on this
    machine, because their files are "gone".
    """
    planned = {item.dest_rel: item for item in plan}
    result: dict[str, LockEntry] = {}
    for dest, entry in lock.items():
        if scope is not None and entry["source_id"] not in scope:
            result[dest] = entry
            continue
        updated = relock_entry(entry, origin_hashes.get(dest))
        if updated is not None:
            result[dest] = updated
    for dest, item in planned.items():
        if dest in result or not (REPO / dest).exists():
            continue
        disk_sha = sha256_of(REPO / dest)
        origin_sha = origin_hashes[dest]
        result[dest] = {**new_entry(item, origin_sha, disk_sha), "scrubbed": disk_sha != origin_sha}
    return result


def report(
    counts: dict[str, dict[Action, int]], lock_changed: bool, not_requested: list[str]
) -> None:
    for source_id in sorted(counts):
        parts = ", ".join(f"{action} {n}" for action, n in sorted(counts[source_id].items()))
        print(f"{source_id}: {parts}")
    for line in not_requested:
        print(line)
    print(f"lock: {'updated' if lock_changed else 'unchanged'} ({LOCK_PATH.relative_to(REPO)})")


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Import sources per sources/manifest.yaml.")
    parser.add_argument(
        "--force", action="store_true", help="re-import copies whose origin changed"
    )
    parser.add_argument(
        "--relock", action="store_true", help="record the disk state after scrubbing"
    )
    parser.add_argument(
        "--source-id",
        action="append",
        metavar="ID",
        help="process only this manifest entry; repeatable (unselected roots are not read)",
    )
    parser.add_argument(
        "--kind",
        action="append",
        choices=sorted(ENTRY_KEYS),
        help="process only entries of this kind; repeatable",
    )
    parser.add_argument(
        "--inventory",
        type=Path,
        metavar="JSON",
        help="verify inventory paths and sha256 for the selected entries before writing anything",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    manifest = cast(dict[str, Any], load_yaml(MANIFEST_PATH))
    roots = resolve_roots(cast(dict[str, Any], load_yaml(ROOTS_PATH)))
    source_ids = frozenset(args.source_id or [])
    kinds = frozenset(args.kind or [])
    selected, excluded = select_manifest(manifest, source_ids, kinds)
    inventory_items: dict[str, str] | None = None
    if args.inventory is not None:
        inventory_items = load_inventory(args.inventory)
        for kind, key in ENTRY_KEYS.items():
            for entry in selected[key]:
                validate_inventory(
                    root_for(roots, str(entry["root"])),
                    args.inventory,
                    selected=selector_for(kind, entry),
                )
    not_requested: list[str] = []
    for kind, entry in excluded:
        line = f"not requested: {kind} {entry['id']} (excluded by --kind; not read or copied)"
        if inventory_items is not None:
            select = selector_for(kind, entry)
            count = sum(1 for rel in inventory_items if select(rel))
            line += f"; {count} inventory file(s) not imported in this run"
        not_requested.append(line)
    plan = build_plan(selected, roots)
    origin_hashes: dict[str, str] = {}
    for item in plan:
        try:
            origin_hashes[item.dest_rel] = sha256_of(item.origin_path)
        except OSError as exc:
            raise fail(f"cannot read origin file for {item.dest_rel}: {exc.strerror}") from None
    lock = load_lock()
    if args.relock:
        scope = (
            frozenset(str(e["id"]) for key in ENTRY_KEYS.values() for e in selected[key])
            if source_ids or kinds
            else None
        )
        entries = relock(plan, lock, origin_hashes, scope)
        counts: dict[str, dict[Action, int]] = {}
    else:
        entries, counts = sync(plan, lock, load_withdrawn(), origin_hashes, args.force)
    lock_changed = write_lock(entries)
    if args.relock:
        state = "updated" if lock_changed else "unchanged"
        print(f"relock: {len(entries)} entries, lock {state}")
    else:
        report(counts, lock_changed, not_requested)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
