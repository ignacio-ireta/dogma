"""Sync pinned HuggingFace dataset snapshots into the dogma_data test layer (I-31a3).

Default mode (needs network and the ``hf`` group)::

    uv run --frozen --isolated --group hf python -I scripts/sync_hf_datasets.py [--only ID]

reads ``src/dogma_data/hf_manifest.yaml``, checks each licence against the policy and the dataset
card at the pinned revision, selects rows and columns, writes text files under
``src/dogma_data/data/hf/``, privacy-scans them, then rewrites ``hf_manifest.lock.json`` and the
``NOTICE`` file.

``--check`` mode is offline. It imports neither ``datasets`` nor ``huggingface_hub``; it
re-validates the manifest and verifies every lock record and file on disk::

    uv run --frozen python scripts/sync_hf_datasets.py --check

Snapshots are streamed (``streaming=True``) so that a filtered window of a large repo stops
downloading once the window is full. ``trust_remote_code`` is never passed.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import io
import json
import re
import sys
import tempfile
import unicodedata
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any

import yaml

REPO = Path(__file__).resolve().parent.parent
DATA_ROOT = REPO / "src" / "dogma_data"
MANIFEST_NAME = "hf_manifest.yaml"
LOCK_NAME = "hf_manifest.lock.json"
HF_SUBDIR = Path("data") / "hf"
NOTICE_NAME = "NOTICE"
PRIVATE_DENYLIST = REPO / "private" / "denylist.txt"
HARD_MAX_BYTES = 1_000_000
MAX_ROW_INDEX = 10_000_000
SHA_LENGTH = 40
ID_PATTERN = re.compile(r"[a-z0-9_]+")
REVISION_PATTERN = re.compile(rf"[0-9a-f]{{{SHA_LENGTH}}}")
FORMATS = frozenset({"csv", "jsonl", "txt"})
KNOWN_KEYS = frozenset(
    {
        "id",
        "repo",
        "revision",
        "config",
        "split",
        "data_files",
        "licence",
        "notice",
        "columns",
        "rows",
        "filter",
        "text",
        "format",
        "max_bytes",
    }
)
ALLOWED_LICENCES = frozenset({"cc0-1.0", "mit", "apache-2.0", "pddl", "public domain"})
ATTRIBUTION_LICENCES = frozenset({"cc-by-4.0", "cc-by-3.0"})
UNKNOWN_LICENCES = frozenset({"", "unknown", "other"})
RESTRICTIVE_TOKEN = re.compile(r"(?:^|-)(?:nc|nd|sa)(?:-|$)")
ENCODING = "utf-8"


class ManifestError(Exception):
    """The manifest is malformed or breaks the licence policy."""


class SyncError(Exception):
    """A snapshot cannot be synced: licence mismatch, bad selection, size or privacy hit."""


@dataclass(frozen=True)
class Filter:
    column: str
    contains: str | None
    equals: str | None

    def matches(self, row: Mapping[str, Any]) -> bool:
        if self.column not in row:
            raise SyncError(f"filter column {self.column!r} is not in the dataset")
        value = "" if row[self.column] is None else str(row[self.column])
        if self.equals is not None:
            return value == self.equals
        return self.contains is not None and self.contains in value


@dataclass(frozen=True)
class RowWindow:
    start: int
    stop: int


@dataclass(frozen=True)
class TextLimit:
    column: str
    max_chars: int
    start_after: str | None = None


@dataclass(frozen=True)
class Entry:
    id: str
    repo: str
    revision: str
    split: str
    licence: str
    format: str
    max_bytes: int
    config: str | None = None
    data_files: tuple[str, ...] | None = None
    notice: str | None = None
    columns: tuple[str, ...] | None = None
    rows: RowWindow | None = None
    row_filter: Filter | None = None
    text: TextLimit | None = None

    @property
    def filename(self) -> str:
        return f"{self.id}.{self.format}"

    @property
    def relpath(self) -> str:
        return (HF_SUBDIR / self.filename).as_posix()


@dataclass(frozen=True)
class LockRecord:
    repo: str
    revision: str
    licence: str
    file: str
    sha256: str
    bytes: int
    rows: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "repo": self.repo,
            "revision": self.revision,
            "licence": self.licence,
            "file": self.file,
            "sha256": self.sha256,
            "bytes": self.bytes,
            "rows": self.rows,
        }


# ---------------------------------------------------------------------------
# Manifest parsing and validation (pure)
# ---------------------------------------------------------------------------


def licence_problem(licence: str, notice: str | None) -> str | None:
    """Return why a licence is refused, or None when the policy allows it."""
    value = licence.strip().lower()
    if value in UNKNOWN_LICENCES:
        return "licence is missing, unknown or 'other'"
    if value in ALLOWED_LICENCES or value.startswith("bsd"):
        return None
    if value in ATTRIBUTION_LICENCES:
        return None if notice else f"{licence} needs a notice entry in the manifest"
    if RESTRICTIVE_TOKEN.search(value):
        return f"{licence} is refused: non-commercial, no-derivatives or share-alike terms"
    return f"{licence} is not on the allowed licence list"


def card_licence_matches(expected: str, card_value: object) -> bool:
    """Compare the manifest licence with the dataset card value (a string or one-item list)."""
    if isinstance(card_value, list):
        return len(card_value) == 1 and card_value[0] == expected
    return card_value == expected


def normalise_text(text: str) -> str:
    """NFC, LF line endings and no trailing whitespace on any line."""
    text = unicodedata.normalize("NFC", text).replace("\r\n", "\n").replace("\r", "\n")
    return "\n".join(line.rstrip() for line in text.split("\n"))


def normalise_value(value: Any) -> Any:
    return normalise_text(value) if isinstance(value, str) else value


def start_at_marker(text: str, marker: str) -> str:
    """Start at the line holding the first occurrence of marker (both are NFC-normalised)."""
    index = text.find(normalise_text(marker))
    if index < 0:
        raise SyncError(f"start_after marker {marker!r} is not in the text")
    return text[text.rfind("\n", 0, index) + 1 :]


def truncate_at_line(text: str, max_chars: int) -> str:
    """Cut text to at most max_chars, at the last line boundary before the limit."""
    if len(text) <= max_chars:
        return text
    cut = text.rfind("\n", 0, max_chars)
    if cut < 0:
        raise SyncError(f"no line boundary within the first {max_chars} characters")
    return text[: cut + 1]


def select_rows(
    rows: Iterable[Mapping[str, Any]],
    *,
    row_filter: Filter | None,
    window: RowWindow | None,
) -> list[dict[str, Any]]:
    """Filter, then take the window of the filtered rows. Stops reading once the window is full."""
    start = window.start if window else 0
    stop = window.stop if window else None
    selected: list[dict[str, Any]] = []
    kept = 0
    for row in rows:
        if row_filter is not None and not row_filter.matches(row):
            continue
        if kept >= start and (stop is None or kept < stop):
            selected.append(dict(row))
        kept += 1
        if stop is not None and kept >= stop:
            break
    return selected


def _optional_str(raw: Mapping[str, Any], key: str) -> str | None:
    value = raw.get(key)
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise ManifestError(f"{key} must be a non-empty string")
    return value


def _required_str(raw: Mapping[str, Any], key: str) -> str:
    value = _optional_str(raw, key)
    if value is None:
        raise ManifestError(f"missing required key {key!r}")
    return value


def _int_in(value: object, key: str, low: int, high: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ManifestError(f"{key} must be an integer")
    if not low <= value <= high:
        raise ManifestError(f"{key} must be between {low} and {high}")
    return value


def _str_list(raw: Mapping[str, Any], key: str) -> tuple[str, ...] | None:
    value = raw.get(key)
    if value is None:
        return None
    if not isinstance(value, list) or not value or not all(isinstance(v, str) for v in value):
        raise ManifestError(f"{key} must be a non-empty list of strings")
    return tuple(value)


def _mapping(raw: Mapping[str, Any], key: str, allowed: frozenset[str]) -> Mapping[str, Any]:
    value = raw.get(key)
    if not isinstance(value, dict):
        raise ManifestError(f"{key} must be a mapping")
    unknown = sorted(str(k) for k in value if k not in allowed)
    if unknown:
        raise ManifestError(f"{key} has unknown key(s): {', '.join(unknown)}")
    return value


def _row_window(raw: Mapping[str, Any]) -> RowWindow | None:
    if "rows" not in raw:
        return None
    value = _mapping(raw, "rows", frozenset({"start", "stop"}))
    start = _int_in(value.get("start", 0), "rows.start", 0, MAX_ROW_INDEX)
    if "stop" not in value:
        raise ManifestError("rows.stop is required")
    stop = _int_in(value["stop"], "rows.stop", 1, MAX_ROW_INDEX)
    if start >= stop:
        raise ManifestError("rows.start must be below rows.stop")
    return RowWindow(start=start, stop=stop)


def _row_filter(raw: Mapping[str, Any]) -> Filter | None:
    if "filter" not in raw:
        return None
    value = _mapping(raw, "filter", frozenset({"column", "contains", "equals"}))
    column = _required_str(value, "column")
    has_contains = "contains" in value
    has_equals = "equals" in value
    if has_contains == has_equals:
        raise ManifestError("filter needs exactly one of contains or equals")
    contains = _required_str(value, "contains") if has_contains else None
    equals = _required_str(value, "equals") if has_equals else None
    return Filter(column=column, contains=contains, equals=equals)


def _text_limit(raw: Mapping[str, Any]) -> TextLimit | None:
    if "text" not in raw:
        return None
    value = _mapping(raw, "text", frozenset({"column", "max_chars", "start_after"}))
    column = _required_str(value, "column")
    max_chars = _int_in(value.get("max_chars"), "text.max_chars", 1, HARD_MAX_BYTES)
    start_after = _optional_str(value, "start_after")
    return TextLimit(column=column, max_chars=max_chars, start_after=start_after)


def parse_entry(raw: object) -> Entry:
    """Validate one manifest entry strictly. Raises ManifestError with the reason."""
    if not isinstance(raw, dict):
        raise ManifestError("entry must be a mapping")
    unknown = sorted(str(k) for k in raw if k not in KNOWN_KEYS)
    if unknown:
        raise ManifestError(f"unknown key(s): {', '.join(unknown)}")
    entry_id = _required_str(raw, "id")
    if not ID_PATTERN.fullmatch(entry_id):
        raise ManifestError(f"id {entry_id!r} must match [a-z0-9_]+")
    where = entry_id
    revision = _required_str(raw, "revision")
    if not REVISION_PATTERN.fullmatch(revision):
        raise ManifestError(f"{where}: revision must be a full {SHA_LENGTH}-character commit sha")
    fmt = _required_str(raw, "format")
    if fmt not in FORMATS:
        raise ManifestError(f"{where}: format must be one of csv, jsonl, txt")
    licence = _required_str(raw, "licence")
    notice = _optional_str(raw, "notice")
    problem = licence_problem(licence, notice)
    if problem:
        raise ManifestError(f"{where}: {problem}")
    columns = _str_list(raw, "columns")
    if fmt == "txt" and (columns is None or len(columns) != 1):
        raise ManifestError(f"{where}: txt format needs exactly one column")
    text = _text_limit(raw)
    if text is not None and columns is not None and text.column not in columns:
        raise ManifestError(f"{where}: text.column must be one of the selected columns")
    max_bytes = _int_in(raw.get("max_bytes", HARD_MAX_BYTES), "max_bytes", 1, HARD_MAX_BYTES)
    return Entry(
        id=entry_id,
        repo=_required_str(raw, "repo"),
        revision=revision,
        split=_required_str(raw, "split"),
        licence=licence,
        format=fmt,
        max_bytes=max_bytes,
        config=_optional_str(raw, "config"),
        data_files=_str_list(raw, "data_files"),
        notice=notice,
        columns=columns,
        rows=_row_window(raw),
        row_filter=_row_filter(raw),
        text=text,
    )


def parse_manifest(raw: object) -> list[Entry]:
    """Validate the whole manifest list; report every bad entry at once."""
    if not isinstance(raw, list):
        raise ManifestError("manifest must be a list of entries")
    entries: list[Entry] = []
    problems: list[str] = []
    seen: set[str] = set()
    for index, item in enumerate(raw):
        try:
            entry = parse_entry(item)
        except ManifestError as exc:
            problems.append(f"entry {index}: {exc}")
            continue
        if entry.id in seen:
            problems.append(f"entry {index}: duplicate id {entry.id!r}")
        seen.add(entry.id)
        entries.append(entry)
    if problems:
        raise ManifestError("; ".join(problems))
    return entries


def load_manifest(path: Path) -> list[Entry]:
    try:
        raw = yaml.safe_load(path.read_text(encoding=ENCODING))
    except (OSError, yaml.YAMLError) as exc:
        raise ManifestError(f"cannot read {path.name}: {exc}") from exc
    return parse_manifest(raw)


# ---------------------------------------------------------------------------
# Rendering (pure)
# ---------------------------------------------------------------------------


def render_csv(columns: Sequence[str], rows: Sequence[Mapping[str, Any]]) -> str:
    buffer = io.StringIO()
    writer = csv.writer(buffer, quoting=csv.QUOTE_NONNUMERIC, lineterminator="\n")
    writer.writerow(columns)
    for row in rows:
        writer.writerow([row[column] for column in columns])
    return buffer.getvalue()


def render_jsonl(columns: Sequence[str], rows: Sequence[Mapping[str, Any]]) -> str:
    return "".join(
        json.dumps({c: row[c] for c in columns}, ensure_ascii=False, sort_keys=True) + "\n"
        for row in rows
    )


def _ensure_newline(text: str) -> str:
    return text if text.endswith("\n") else text + "\n"


def render_entry(entry: Entry, rows: Sequence[Mapping[str, Any]]) -> bytes:
    """Turn selected rows into the file bytes: columns, normalisation, truncation, format."""
    if not rows:
        raise SyncError(f"{entry.id}: the selection is empty")
    columns = list(entry.columns) if entry.columns else list(rows[0].keys())
    missing = [c for c in columns if c not in rows[0]]
    if missing:
        raise SyncError(f"{entry.id}: columns not in the dataset: {', '.join(missing)}")
    records = [{c: normalise_value(row[c]) for c in columns} for row in rows]
    if entry.text is not None:
        for record in records:
            value = record[entry.text.column]
            if entry.text.start_after is not None:
                value = start_at_marker(value, entry.text.start_after)
            record[entry.text.column] = truncate_at_line(value, entry.text.max_chars)
    if entry.format == "csv":
        content = render_csv(columns, records)
    elif entry.format == "jsonl":
        content = render_jsonl(columns, records)
    else:
        content = "".join(_ensure_newline(record[columns[0]]) for record in records)
    data = content.encode(ENCODING)
    if len(data) > entry.max_bytes:
        raise SyncError(f"{entry.id}: {len(data)} bytes exceeds max_bytes {entry.max_bytes}")
    return data


def describe_selection(entry: Entry) -> str:
    """One line for NOTICE: what was taken from the source."""
    parts = [f"{entry.split} split"]
    if entry.config:
        parts.append(f"config {entry.config}")
    parts.append(f"columns {', '.join(entry.columns)}" if entry.columns else "all columns")
    if entry.row_filter is not None:
        test = entry.row_filter
        clause = (
            f"contains {test.contains!r}" if test.contains is not None else f"= {test.equals!r}"
        )
        parts.append(f"rows where {test.column} {clause}")
    if entry.rows is not None:
        parts.append(f"rows {entry.rows.start} to {entry.rows.stop} of the (filtered) split")
    if entry.text is not None:
        start = (
            f", starting at the line with {entry.text.start_after!r}"
            if entry.text.start_after
            else ""
        )
        parts.append(
            f"{entry.text.column} cut at a line boundary within {entry.text.max_chars}{start}"
        )
    return "; ".join(parts) + "."


def render_notice(entries: Sequence[Entry], records: Mapping[str, LockRecord]) -> str:
    lines = [
        "Vendored HuggingFace dataset snapshots used by the dogma_data test layer.",
        "Each snapshot is pinned to a dataset revision and was checked against the licence",
        "policy in scripts/sync_hf_datasets.py. Regenerate with:",
        "  uv run --frozen --isolated --group hf python -I scripts/sync_hf_datasets.py",
        "",
    ]
    for entry in entries:
        record = records.get(entry.id)
        if record is None:
            continue
        lines.extend(
            [
                f"## {entry.id}",
                f"- Source: https://huggingface.co/datasets/{entry.repo} at {entry.revision}",
                f"- Licence: {entry.licence}",
                f"- File: {HF_SUBDIR.as_posix()}/{entry.filename} ({record.rows} rows)",
                f"- Selection: {describe_selection(entry)}",
                f"- Notice: {entry.notice or 'none required by the licence'}",
                "",
            ]
        )
    return "\n".join(lines).rstrip("\n") + "\n"


def lock_text(records: Mapping[str, LockRecord]) -> str:
    payload = {rid: records[rid].to_dict() for rid in sorted(records)}
    return json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n"


def read_lock(path: Path) -> dict[str, dict[str, Any]]:
    raw = json.loads(path.read_text(encoding=ENCODING))
    if not isinstance(raw, dict):
        raise ValueError("lock file must be a JSON object keyed by id")
    return raw


# ---------------------------------------------------------------------------
# Verification (offline)
# ---------------------------------------------------------------------------


def _record_count(fmt: str, text: str) -> int:
    if fmt == "txt":
        return 1
    if fmt == "jsonl":
        return sum(1 for line in text.split("\n") if line)
    return max(0, sum(1 for _ in csv.reader(io.StringIO(text))) - 1)


def file_problems(path: Path, rel: str, record: Mapping[str, Any], fmt: str) -> list[str]:
    """Verify one vendored file against its lock record and the text rules."""
    if not path.is_file():
        return [f"{rel}: file is missing"]
    data = path.read_bytes()
    problems: list[str] = []
    if len(data) != record.get("bytes"):
        problems.append(f"{rel}: size {len(data)} does not match lock {record.get('bytes')}")
    if hashlib.sha256(data).hexdigest() != record.get("sha256"):
        problems.append(f"{rel}: sha256 does not match the lock")
    if len(data) > HARD_MAX_BYTES:
        problems.append(f"{rel}: {len(data)} bytes exceeds {HARD_MAX_BYTES}")
    try:
        text = data.decode(ENCODING)
    except UnicodeDecodeError:
        return [*problems, f"{rel}: not valid UTF-8"]
    if not text.endswith("\n"):
        problems.append(f"{rel}: missing final newline")
    if "\r" in text:
        problems.append(f"{rel}: contains carriage returns")
    if unicodedata.normalize("NFC", text) != text:
        problems.append(f"{rel}: not NFC-normalised")
    for number, line in enumerate(text.split("\n"), start=1):
        if line.rstrip() != line:
            problems.append(f"{rel}:{number}: trailing whitespace")
    try:
        count = _record_count(fmt, text)
    except csv.Error as exc:
        return [*problems, f"{rel}: not parseable as CSV: {exc}"]
    if count != record.get("rows"):
        problems.append(f"{rel}: {count} records do not match lock rows {record.get('rows')}")
    return problems


def check(data_root: Path = DATA_ROOT) -> list[str]:
    """Return every problem found offline; an empty list means the layer is consistent."""
    manifest_path = data_root / MANIFEST_NAME
    lock_path = data_root / LOCK_NAME
    hf_dir = data_root / HF_SUBDIR
    try:
        entries = load_manifest(manifest_path)
    except ManifestError as exc:
        return [f"manifest: {exc}"]
    try:
        lock = read_lock(lock_path)
    except (OSError, ValueError) as exc:
        return [f"lock: cannot read {LOCK_NAME}: {exc}"]
    problems: list[str] = []
    by_id = {entry.id: entry for entry in entries}
    for stale in sorted(set(lock) - set(by_id)):
        problems.append(f"{stale}: lock record has no manifest entry")
    records: dict[str, LockRecord] = {}
    for entry in entries:
        raw = lock.get(entry.id)
        if raw is None:
            problems.append(f"{entry.id}: no lock record; run the sync")
            continue
        for key in ("repo", "revision", "licence"):
            if raw.get(key) != getattr(entry, key):
                problems.append(f"{entry.id}: lock {key} differs from the manifest")
        if raw.get("file") != entry.relpath:
            problems.append(f"{entry.id}: lock file must be {entry.relpath}")
        problems.extend(file_problems(data_root / entry.relpath, entry.relpath, raw, entry.format))
        if all(
            k in raw for k in ("repo", "revision", "licence", "file", "sha256", "bytes", "rows")
        ):
            records[entry.id] = LockRecord(
                repo=raw["repo"],
                revision=raw["revision"],
                licence=raw["licence"],
                file=raw["file"],
                sha256=raw["sha256"],
                bytes=raw["bytes"],
                rows=raw["rows"],
            )
    problems.extend(_stray_problems(hf_dir, {entry.filename for entry in entries}))
    notice_path = hf_dir / NOTICE_NAME
    if notice_path.is_file():
        expected = render_notice(entries, records)
        if notice_path.read_text(encoding=ENCODING) != expected:
            problems.append(f"{HF_SUBDIR.as_posix()}/{NOTICE_NAME} is out of date; run the sync")
    else:
        problems.append(f"{HF_SUBDIR.as_posix()}/{NOTICE_NAME} is missing")
    return problems


def _stray_problems(hf_dir: Path, expected: set[str]) -> list[str]:
    if not hf_dir.is_dir():
        return [f"{HF_SUBDIR.as_posix()} is missing"]
    present = {p.name for p in hf_dir.iterdir() if p.is_file()} - {NOTICE_NAME}
    return [
        f"{HF_SUBDIR.as_posix()}/{name}: not in the manifest" for name in sorted(present - expected)
    ]


# ---------------------------------------------------------------------------
# Privacy audit (reuses scripts/audit_privacy.py)
# ---------------------------------------------------------------------------


def load_audit_module() -> ModuleType:
    path = REPO / "scripts" / "audit_privacy.py"
    spec = importlib.util.spec_from_file_location("audit_privacy", path)
    if spec is None or spec.loader is None:
        raise SyncError("cannot load scripts/audit_privacy.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def privacy_problems(files: Mapping[str, bytes], denylist: Path | None) -> list[str]:
    """Scan the new files with the built-in rules and, when given, the private denylist."""
    audit = load_audit_module()
    rules = audit.builtin_rules()
    if denylist is not None:
        rules = audit.load_denylist(denylist) + rules
    problems: list[str] = []
    for rel, data in files.items():
        for hit in audit.scan_text(rel, data.decode(ENCODING), rules):
            problems.append(f"{hit.path}:{hit.lineno}: {hit.label}")
    return problems


# ---------------------------------------------------------------------------
# Network sync (lazy imports)
# ---------------------------------------------------------------------------


def _card_licence(repo: str, revision: str) -> object:
    from huggingface_hub import HfApi  # noqa: PLC0415 - lazy: the hf group is optional

    info = HfApi().dataset_info(repo, revision=revision)
    card = getattr(info, "card_data", None)
    if card is None:
        return None
    to_dict = getattr(card, "to_dict", None)
    data: Mapping[str, Any] = to_dict() if callable(to_dict) else {}
    return data.get("license")


def _load_rows(entry: Entry, cache_dir: str) -> Iterable[Mapping[str, Any]]:
    import datasets  # noqa: PLC0415 - lazy: the hf group is optional

    return datasets.load_dataset(
        entry.repo,
        name=entry.config,
        split=entry.split,
        revision=entry.revision,
        data_files=list(entry.data_files) if entry.data_files else None,
        cache_dir=cache_dir,
        streaming=True,
    )


def fetch_entry(entry: Entry) -> tuple[bytes, int]:
    """Check the licence at the pinned revision, select rows and render the file bytes."""
    card = _card_licence(entry.repo, entry.revision)
    if not card_licence_matches(entry.licence, card):
        raise SyncError(f"{entry.id}: card licence {card!r} does not match {entry.licence!r}")
    with tempfile.TemporaryDirectory(prefix="dogma-hf-") as cache_dir:
        rows = _load_rows(entry, cache_dir)
        selected = select_rows(rows, row_filter=entry.row_filter, window=entry.rows)
    return render_entry(entry, selected), len(selected)


def sync(
    entries: Sequence[Entry],
    *,
    data_root: Path = DATA_ROOT,
    denylist: Path | None = None,
) -> dict[str, LockRecord]:
    """Fetch, audit and write the selected entries, then rewrite the lock file and NOTICE."""
    hf_dir = data_root / HF_SUBDIR
    lock_path = data_root / LOCK_NAME
    all_entries = load_manifest(data_root / MANIFEST_NAME)
    existing: dict[str, LockRecord] = {}
    if lock_path.is_file():
        for rid, raw in read_lock(lock_path).items():
            if rid in {e.id for e in all_entries}:
                existing[rid] = LockRecord(**raw)
    fetched: dict[str, tuple[Entry, bytes, int]] = {}
    for entry in entries:
        data, rows = fetch_entry(entry)
        fetched[entry.id] = (entry, data, rows)
    hits = privacy_problems({e.relpath: d for e, d, _ in fetched.values()}, denylist)
    if hits:
        raise SyncError("privacy audit hits: " + "; ".join(hits))
    hf_dir.mkdir(parents=True, exist_ok=True)
    records = dict(existing)
    for entry, data, rows in fetched.values():
        (data_root / entry.relpath).write_bytes(data)
        records[entry.id] = LockRecord(
            repo=entry.repo,
            revision=entry.revision,
            licence=entry.licence,
            file=entry.relpath,
            sha256=hashlib.sha256(data).hexdigest(),
            bytes=len(data),
            rows=rows,
        )
    lock_path.write_text(lock_text(records), encoding=ENCODING)
    (hf_dir / NOTICE_NAME).write_text(render_notice(all_entries, records), encoding=ENCODING)
    return records


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--check", action="store_true", help="verify offline; no network")
    parser.add_argument("--only", action="append", default=[], metavar="ID", help="sync one id")
    parser.add_argument(
        "--denylist",
        type=Path,
        default=None,
        help="private denylist for the privacy audit (default: private/denylist.txt if present)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.check:
        problems = check()
        for problem in problems:
            print(problem)
        if problems:
            print(f"hf sync check: {len(problems)} problem(s)")
            return 1
        print("hf sync check: clean")
        return 0
    default_denylist = PRIVATE_DENYLIST if PRIVATE_DENYLIST.is_file() else None
    denylist: Path | None = args.denylist or default_denylist
    if denylist is None:
        print("warning: no private denylist; privacy audit uses built-in rules only")
    try:
        entries = load_manifest(DATA_ROOT / MANIFEST_NAME)
        wanted = set(args.only)
        unknown = sorted(wanted - {e.id for e in entries})
        if unknown:
            raise SyncError(f"unknown id(s): {', '.join(unknown)}")
        selected = [e for e in entries if not wanted or e.id in wanted]
        records = sync(selected, denylist=denylist)
    except (ManifestError, SyncError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    for entry in selected:
        record = records[entry.id]
        print(f"{entry.id}: {record.rows} rows, {record.bytes} bytes, {record.licence}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
