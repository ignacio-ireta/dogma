"""Offline tests for the pinned HuggingFace snapshot sync (no network, no datasets package)."""

from __future__ import annotations

import csv
import hashlib
import importlib.util
import io
import json
import shutil
import subprocess
import sys
from collections.abc import Iterator, Mapping
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

SCRIPT = Path(__file__).resolve().parents[3] / "scripts" / "sync_hf_datasets.py"
DATA_ROOT = Path(__file__).resolve().parents[3] / "src" / "dogma_data"
SHA = "740312add88f781978c0658806c59bc2815b9866"


def load_sync() -> ModuleType:
    spec = importlib.util.spec_from_file_location("sync_hf_datasets", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


sync = load_sync()


def raw_entry(**overrides: Any) -> dict[str, Any]:
    entry: dict[str, Any] = {
        "id": "sample_set",
        "repo": "example-org/sample",
        "revision": SHA,
        "split": "train",
        "licence": "mit",
        "format": "csv",
    }
    entry.update(overrides)
    return entry


# ---------------------------------------------------------------------------
# Manifest validation
# ---------------------------------------------------------------------------


def test_valid_entry_parses_all_fields() -> None:
    entry = sync.parse_entry(
        raw_entry(
            config="main",
            data_files=["main/test.parquet"],
            columns=["question", "answer"],
            rows={"start": 0, "stop": 300},
            filter={"column": "question", "contains": "egg"},
            text={"column": "question", "max_chars": 40000},
            max_bytes=500000,
        )
    )
    assert entry.id == "sample_set"
    assert entry.rows == sync.RowWindow(start=0, stop=300)
    assert entry.row_filter == sync.Filter(column="question", contains="egg", equals=None)
    assert entry.text == sync.TextLimit(column="question", max_chars=40000)
    assert entry.relpath == "data/hf/sample_set.csv"


def test_unknown_key_is_an_error() -> None:
    with pytest.raises(sync.ManifestError, match="unknown key"):
        sync.parse_entry(raw_entry(trust_remote_code=True))


@pytest.mark.parametrize("revision", ["main", "740312add", SHA.upper(), SHA + "0"])
def test_revision_must_be_a_full_lowercase_sha(revision: str) -> None:
    with pytest.raises(sync.ManifestError, match="revision"):
        sync.parse_entry(raw_entry(revision=revision))


@pytest.mark.parametrize("entry_id", ["Adult", "gsm-8k", "", "a b"])
def test_id_must_match_pattern(entry_id: str) -> None:
    with pytest.raises(sync.ManifestError, match="id"):
        sync.parse_entry(raw_entry(id=entry_id))


def test_unknown_format_is_an_error() -> None:
    with pytest.raises(sync.ManifestError, match="format"):
        sync.parse_entry(raw_entry(format="parquet"))


def test_txt_needs_exactly_one_column() -> None:
    with pytest.raises(sync.ManifestError, match="exactly one column"):
        sync.parse_entry(raw_entry(format="txt", columns=["a", "b"]))


def test_filter_needs_exactly_one_operator() -> None:
    with pytest.raises(sync.ManifestError, match="exactly one"):
        sync.parse_entry(raw_entry(filter={"column": "a", "contains": "x", "equals": "y"}))
    with pytest.raises(sync.ManifestError, match="exactly one"):
        sync.parse_entry(raw_entry(filter={"column": "a"}))


def test_row_window_must_be_ordered() -> None:
    with pytest.raises(sync.ManifestError, match="below"):
        sync.parse_entry(raw_entry(rows={"start": 5, "stop": 5}))


def test_max_bytes_cannot_exceed_hard_cap() -> None:
    with pytest.raises(sync.ManifestError, match="max_bytes"):
        sync.parse_entry(raw_entry(max_bytes=1_000_001))


def test_text_column_must_be_selected() -> None:
    with pytest.raises(sync.ManifestError, match="selected columns"):
        sync.parse_entry(raw_entry(columns=["a"], text={"column": "b", "max_chars": 10}))


def test_parse_manifest_reports_every_bad_entry_and_duplicates() -> None:
    with pytest.raises(sync.ManifestError) as info:
        sync.parse_manifest([raw_entry(revision="x"), raw_entry(id="Bad"), raw_entry()])
    message = str(info.value)
    assert "entry 0" in message and "entry 1" in message
    with pytest.raises(sync.ManifestError, match="duplicate"):
        sync.parse_manifest([raw_entry(), raw_entry()])


def test_parse_manifest_requires_a_list() -> None:
    with pytest.raises(sync.ManifestError, match="list"):
        sync.parse_manifest({"id": "x"})


# ---------------------------------------------------------------------------
# Licence policy
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "licence",
    ["cc0-1.0", "mit", "apache-2.0", "bsd-3-clause", "bsd-2-clause", "pddl", "public domain"],
)
def test_allowed_licences_pass(licence: str) -> None:
    assert sync.licence_problem(licence, None) is None


@pytest.mark.parametrize(
    "licence", ["cc-by-nc-4.0", "cc-by-nd-4.0", "cc-by-sa-4.0", "cc-by-nc-sa-3.0"]
)
def test_restrictive_creative_commons_are_refused(licence: str) -> None:
    problem = sync.licence_problem(licence, "notice")
    assert problem is not None and "refused" in problem


@pytest.mark.parametrize("licence", ["unknown", "other", "", "gpl-3.0", "cc-by-2.0"])
def test_unknown_or_unlisted_licences_are_refused(licence: str) -> None:
    assert sync.licence_problem(licence, None) is not None


@pytest.mark.parametrize("licence", ["cc-by-4.0", "cc-by-3.0"])
def test_cc_by_needs_a_notice(licence: str) -> None:
    assert sync.licence_problem(licence, None) is not None
    assert sync.licence_problem(licence, "Attribution: example") is None


def test_manifest_licence_policy_is_enforced_at_parse_time() -> None:
    with pytest.raises(sync.ManifestError, match="refused"):
        sync.parse_entry(raw_entry(licence="cc-by-nc-4.0"))


@pytest.mark.parametrize(
    ("expected", "card", "matches"),
    [
        ("mit", "mit", True),
        ("mit", ["mit"], True),
        ("mit", "apache-2.0", False),
        ("mit", ["mit", "cc0-1.0"], False),
        ("mit", None, False),
    ],
)
def test_card_licence_comparison(expected: str, card: object, matches: bool) -> None:
    assert sync.card_licence_matches(expected, card) is matches


# ---------------------------------------------------------------------------
# Text normalisation, truncation, selection
# ---------------------------------------------------------------------------


def test_normalise_text_applies_nfc_lf_and_strips_trailing_spaces() -> None:
    decomposed = "café"
    assert sync.normalise_text(decomposed) == "café"
    assert sync.normalise_text("one  \r\ntwo\t\r\nthree") == "one\ntwo\nthree"
    assert sync.normalise_text("a\rb") == "a\nb"
    assert sync.normalise_text("keep\n\n") == "keep\n\n"


def test_normalise_value_leaves_non_text_alone() -> None:
    assert sync.normalise_value(3.5) == 3.5
    assert sync.normalise_value(None) is None


def test_truncate_keeps_short_text() -> None:
    assert sync.truncate_at_line("short\n", 100) == "short\n"


def test_truncate_cuts_at_the_last_line_boundary_before_the_limit() -> None:
    text = "alpha\nbravo\ncharlie\ndelta\n"
    cut = sync.truncate_at_line(text, 14)
    assert cut == "alpha\nbravo\n"
    assert len(cut) <= 14


def test_truncate_refuses_when_there_is_no_line_boundary() -> None:
    with pytest.raises(sync.SyncError, match="line boundary"):
        sync.truncate_at_line("x" * 50, 10)


class _Counting:
    """Iterable that records how many rows were pulled, to prove early stop."""

    def __init__(self, rows: list[dict[str, Any]]) -> None:
        self.rows = rows
        self.pulled = 0

    def __iter__(self) -> Iterator[dict[str, Any]]:
        for row in self.rows:
            self.pulled += 1
            yield row


def test_select_rows_filters_then_takes_the_window() -> None:
    rows = [{"t": "a"}, {"t": "b"}, {"t": "a"}, {"t": "a"}, {"t": "b"}, {"t": "a"}]
    selected = sync.select_rows(
        rows,
        row_filter=sync.Filter(column="t", contains=None, equals="a"),
        window=sync.RowWindow(start=1, stop=3),
    )
    assert selected == [{"t": "a"}, {"t": "a"}]


def test_select_rows_stops_reading_once_the_window_is_full() -> None:
    source = _Counting([{"n": i} for i in range(1000)])
    selected = sync.select_rows(source, row_filter=None, window=sync.RowWindow(start=0, stop=3))
    assert [row["n"] for row in selected] == [0, 1, 2]
    assert source.pulled == 3


def test_select_rows_without_window_keeps_everything() -> None:
    assert sync.select_rows([{"n": 1}, {"n": 2}], row_filter=None, window=None) == [
        {"n": 1},
        {"n": 2},
    ]


def test_filter_contains_and_missing_column() -> None:
    row_filter = sync.Filter(column="body", contains="Pride", equals=None)
    assert row_filter.matches({"body": "Pride and Prejudice"})
    assert not row_filter.matches({"body": None})
    with pytest.raises(sync.SyncError, match="not in the dataset"):
        row_filter.matches({"other": "x"})


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


def test_csv_quotes_text_with_edge_spaces_and_keeps_numbers_bare() -> None:
    text = sync.render_csv(["name", "n"], [{"name": " padded ", "n": 3}, {"name": "?", "n": 2.5}])
    assert text.splitlines()[0] == '"name","n"'
    assert text.splitlines()[1] == '" padded ",3'
    assert text.splitlines()[2] == '"?",2.5'
    rows = list(csv.DictReader(io.StringIO(text)))
    assert rows[0]["name"] == " padded "


def test_jsonl_sorts_keys_and_keeps_unicode() -> None:
    text = sync.render_jsonl(["b", "a"], [{"a": "é", "b": 1}])
    assert text == '{"a": "é", "b": 1}\n'
    assert json.loads(text) == {"a": "é", "b": 1}


def test_txt_writes_one_column_with_final_newline() -> None:
    entry = sync.parse_entry(raw_entry(format="txt", columns=["TEXT"]))
    data = sync.render_entry(entry, [{"TEXT": "first line  \r\nsecond"}])
    assert data == b"first line\nsecond\n"


def test_render_entry_applies_text_truncation_and_normalisation() -> None:
    entry = sync.parse_entry(
        raw_entry(format="jsonl", columns=["q"], text={"column": "q", "max_chars": 12})
    )
    data = sync.render_entry(entry, [{"q": "line one\nline two\nline three"}])
    assert json.loads(data.decode())["q"] == "line one\n"


def test_render_entry_refuses_empty_selection_and_oversize() -> None:
    entry = sync.parse_entry(raw_entry(max_bytes=20))
    with pytest.raises(sync.SyncError, match="empty"):
        sync.render_entry(entry, [])
    with pytest.raises(sync.SyncError, match="max_bytes"):
        sync.render_entry(entry, [{"a": "x" * 100}])


def test_describe_selection_is_one_line() -> None:
    entry = sync.parse_entry(
        raw_entry(
            config="main",
            columns=["question"],
            rows={"start": 0, "stop": 3},
            filter={"column": "question", "contains": "egg"},
        )
    )
    line = sync.describe_selection(entry)
    assert "\n" not in line and line.endswith(".")
    assert "config main" in line and "columns question" in line


# ---------------------------------------------------------------------------
# --check on the committed layer and on tampered copies
# ---------------------------------------------------------------------------


def test_check_passes_on_the_committed_layer() -> None:
    assert sync.check(DATA_ROOT) == []


def test_check_does_not_import_datasets_or_huggingface_hub() -> None:
    code = (
        "import importlib.util, sys\n"
        f"spec = importlib.util.spec_from_file_location('s', {str(SCRIPT)!r})\n"
        "m = importlib.util.module_from_spec(spec)\n"
        "sys.modules['s'] = m\n"
        "spec.loader.exec_module(m)\n"
        "assert m.check() == []\n"
        "assert 'datasets' not in sys.modules\n"
        "assert 'huggingface_hub' not in sys.modules\n"
    )
    result = subprocess.run(
        [sys.executable, "-I", "-c", code], capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, result.stderr


@pytest.fixture
def layer_copy(tmp_path: Path) -> Path:
    root = tmp_path / "dogma_data"
    shutil.copytree(DATA_ROOT / "data", root / "data")
    shutil.copy(DATA_ROOT / "hf_manifest.yaml", root / "hf_manifest.yaml")
    shutil.copy(DATA_ROOT / "hf_manifest.lock.json", root / "hf_manifest.lock.json")
    return root


def _lock(root: Path) -> dict[str, Mapping[str, Any]]:
    return json.loads((root / "hf_manifest.lock.json").read_text(encoding="utf-8"))


def test_check_passes_on_the_copy(layer_copy: Path) -> None:
    assert sync.check(layer_copy) == []


def test_check_fails_on_a_tampered_file(layer_copy: Path) -> None:
    target = layer_copy / "data" / "hf" / "gsm8k.jsonl"
    target.write_text(target.read_text(encoding="utf-8").replace("18", "19", 1), encoding="utf-8")
    problems = sync.check(layer_copy)
    assert any("gsm8k" in p and "sha256" in p for p in problems)


def test_check_fails_on_a_missing_file(layer_copy: Path) -> None:
    (layer_copy / "data" / "hf" / "adult.csv").unlink()
    assert any("adult.csv: file is missing" in p for p in sync.check(layer_copy))


def test_check_flags_trailing_whitespace_even_when_the_lock_is_updated(layer_copy: Path) -> None:
    target = layer_copy / "data" / "hf" / "california_housing.csv"
    data = target.read_bytes().replace(b"\n", b"  \n", 1)
    target.write_bytes(data)
    lock = _lock(layer_copy)
    record = dict(lock["california_housing"])
    record["sha256"] = hashlib.sha256(data).hexdigest()
    record["bytes"] = len(data)
    lock["california_housing"] = record
    (layer_copy / "hf_manifest.lock.json").write_text(json.dumps(lock), encoding="utf-8")
    assert any("trailing whitespace" in p for p in sync.check(layer_copy))


def test_check_flags_a_stray_file_and_a_missing_lock_record(layer_copy: Path) -> None:
    (layer_copy / "data" / "hf" / "stray.txt").write_text("x\n", encoding="utf-8")
    lock = _lock(layer_copy)
    del lock["adult"]
    (layer_copy / "hf_manifest.lock.json").write_text(json.dumps(lock), encoding="utf-8")
    problems = sync.check(layer_copy)
    assert any("stray.txt" in p for p in problems)
    assert any(p.startswith("adult: no lock record") for p in problems)


def test_check_reports_a_bad_manifest(layer_copy: Path) -> None:
    manifest = layer_copy / "hf_manifest.yaml"
    manifest.write_text(manifest.read_text(encoding="utf-8") + "  trust_remote_code: true\n")
    problems = sync.check(layer_copy)
    assert problems and problems[0].startswith("manifest:")


def test_notice_is_regenerated_and_compared(layer_copy: Path) -> None:
    notice = layer_copy / "data" / "hf" / "NOTICE"
    notice.write_text(notice.read_text(encoding="utf-8") + "edited\n", encoding="utf-8")
    assert any("NOTICE is out of date" in p for p in sync.check(layer_copy))


def test_check_flags_carriage_returns(layer_copy: Path) -> None:
    lock = _lock(layer_copy)
    target = layer_copy / "data" / "hf" / "gsm8k.jsonl"
    data = b'{"answer": "x", "question": "y"}\r\n'
    target.write_bytes(data)
    record = dict(lock["gsm8k"])
    record.update(sha256=hashlib.sha256(data).hexdigest(), bytes=len(data), rows=1)
    lock["gsm8k"] = record
    (layer_copy / "hf_manifest.lock.json").write_text(json.dumps(lock), encoding="utf-8")
    problems = " ".join(sync.check(layer_copy))
    assert "carriage returns" in problems


def test_license_refusal_has_a_clear_message() -> None:
    with pytest.raises(sync.ManifestError, match="share-alike"):
        sync.parse_entry(raw_entry(licence="cc-by-sa-4.0"))


def test_builtin_privacy_scan_is_clean_for_plain_data() -> None:
    assert sync.privacy_problems({"data/hf/x.csv": b"a,b\n1,2\n"}, None) == []


def test_builtin_privacy_rules_flag_account_paths() -> None:
    # Built from parts so this committed file does not contain an account path itself.
    account_path = "/" + "Users" + "/someone/notes"
    hits = sync.privacy_problems({"data/hf/x.txt": f"see {account_path}\n".encode()}, None)
    assert hits and "account path" in hits[0]


def test_iterable_rows_from_generator_work_with_select_rows() -> None:
    def gen() -> Iterator[Mapping[str, Any]]:
        yield {"k": "a"}
        yield {"k": "b"}

    assert sync.select_rows(gen(), row_filter=None, window=None) == [{"k": "a"}, {"k": "b"}]


def test_notice_lists_each_snapshot_with_its_licence() -> None:
    entry = sync.parse_entry(raw_entry(notice="Attribution: example authors"))
    record = sync.LockRecord(
        repo=entry.repo,
        revision=entry.revision,
        licence=entry.licence,
        file=entry.relpath,
        sha256="0" * 64,
        bytes=1,
        rows=1,
    )
    text = sync.render_notice([entry], {entry.id: record})
    assert "## sample_set" in text and "mit" in text
    assert "Attribution: example authors" in text
    assert text.endswith("\n") and not text.endswith("\n\n")
