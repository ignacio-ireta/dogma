"""Registry: load/save roundtrip, corruption handling, reserved repo names."""

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from docmd.errors import RegistryCorruptError, ReservedRepoNameError
from docmd.tracking.registry import (
    Registry,
    RepoEntry,
    load_registry,
    registry_path,
    save_registry,
    validate_repo_name,
)
from docmd.tracking.report import SyncCounts


def make_entry(name: str = "notes", run_id: str = "20260701T090000Z") -> RepoEntry:
    return RepoEntry(
        name=name,
        source_root=f"/data/{name}",
        first_synced=datetime(2026, 7, 1, 9, tzinfo=UTC),
        last_synced=datetime(2026, 7, 1, 9, tzinfo=UTC),
        last_run_id=run_id,
        run_count=1,
        file_count=3,
        last_counts=SyncCounts(added=3),
    )


def test_load_registry_missing_file_returns_empty(tmp_path: Path) -> None:
    registry = load_registry(tmp_path)
    assert registry == Registry()
    assert registry.repos == {}
    assert not registry_path(tmp_path).exists()  # loading must not create the file


def test_save_registry_roundtrip_reloads_from_disk(tmp_path: Path) -> None:
    registry = Registry()
    registry.repos["notes"] = make_entry("notes")
    path = save_registry(tmp_path, registry)

    assert path == registry_path(tmp_path)
    reloaded = Registry.model_validate_json(path.read_text(encoding="utf-8"))
    assert reloaded == registry
    entry = reloaded.repos["notes"]
    assert entry.last_run_id == "20260701T090000Z"
    assert entry.last_counts is not None and entry.last_counts.added == 3


def test_load_registry_garbage_text_raises_corrupt(tmp_path: Path) -> None:
    path = registry_path(tmp_path)
    path.write_text("{this is not json", encoding="utf-8")
    with pytest.raises(RegistryCorruptError) as excinfo:
        load_registry(tmp_path)
    assert excinfo.value.error_code == "REGISTRY_CORRUPT"
    assert excinfo.value.path == path
    assert excinfo.value.detail  # underlying parse error is preserved


def test_load_registry_valid_json_failing_validation_raises_corrupt(tmp_path: Path) -> None:
    # Parses as JSON but violates the schema: RepoEntry is missing required fields.
    bad = {"schema_version": 1, "repos": {"notes": {"name": "notes"}}}
    registry_path(tmp_path).write_text(json.dumps(bad), encoding="utf-8")
    with pytest.raises(RegistryCorruptError):
        load_registry(tmp_path)


def test_validate_repo_name_reserved_names_raise() -> None:
    for name in ["runs", "RUNS", "Runs", ".docmd", "registry.json", "Registry.JSON"]:
        with pytest.raises(ReservedRepoNameError):
            validate_repo_name(name)


def test_validate_repo_name_hidden_names_raise() -> None:
    for name in [".hidden", ".git", ".anything-dotted"]:
        with pytest.raises(ReservedRepoNameError):
            validate_repo_name(name)


def test_validate_repo_name_empty_raises() -> None:
    with pytest.raises(ReservedRepoNameError) as excinfo:
        validate_repo_name("")
    assert excinfo.value.error_code == "REPO_NAME_RESERVED"
    assert excinfo.value.exit_code == 2


def test_validate_repo_name_normal_names_accepted() -> None:
    for name in ["notes", "My Project", "runs-archive", "docmd", "registry", "a.b"]:
        validate_repo_name(name)  # must not raise


def test_registry_upsert_keeps_one_entry_per_name(tmp_path: Path) -> None:
    registry = Registry()
    registry.repos["notes"] = make_entry("notes", run_id="20260701T090000Z")
    save_registry(tmp_path, registry)

    updated = make_entry("notes", run_id="20260702T100000Z")
    updated.run_count = 2
    registry.repos["notes"] = updated
    path = save_registry(tmp_path, registry)

    reloaded = Registry.model_validate_json(path.read_text(encoding="utf-8"))
    assert list(reloaded.repos) == ["notes"]
    assert reloaded.repos["notes"].last_run_id == "20260702T100000Z"
    assert reloaded.repos["notes"].run_count == 2
    # the raw JSON also holds exactly one repo object
    assert list(json.loads(path.read_text(encoding="utf-8"))["repos"]) == ["notes"]
