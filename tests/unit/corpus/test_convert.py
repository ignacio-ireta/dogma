"""Conversion boundary uses docmd's real argv contract and failures-as-data."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from dogma.core.config import Settings
from dogma.corpus.convert import convert_sources, sync_corpus
from dogma.corpus.index import search


@pytest.fixture
def conversion_repo(tmp_path: Path) -> Settings:
    root = tmp_path / "repo with spaces"
    (root / "tools/docmd").mkdir(parents=True)
    (root / "tools/docmd/pyproject.toml").write_text('[project]\nname="docmd"\n')
    (root / "corpus/raw/library papers").mkdir(parents=True)
    (root / "corpus/raw/library papers/paper.pdf").write_bytes(b"%PDF-stub")
    return Settings(repo_root=root)


def test_exact_converter_arguments(
    conversion_repo: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[tuple[list[str], dict[str, object]]] = []

    def run(argv: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        calls.append((argv, kwargs))
        return subprocess.CompletedProcess(argv, 0, "", "")

    monkeypatch.setattr(subprocess, "run", run)
    report = convert_sources(conversion_repo)
    root = conversion_repo.repo_root
    assert report.converted == 1
    assert report.issues == ()
    assert calls[0][0] == [
        "uv",
        "run",
        "--frozen",
        "--project",
        str(root / "tools/docmd"),
        "docmd",
        "sync",
        str(root / "corpus/raw/library papers"),
        "--out",
        str(root / "corpus/md"),
        "--deterministic",
        "--ocr",
        "off",
        "--workers",
        "1",
        "--no-progress",
    ]
    assert calls[0][1] == {
        "cwd": root,
        "capture_output": True,
        "text": True,
        "timeout": 3600,
        "check": False,
    }


@pytest.mark.parametrize("policy", ["auto", "force"])
def test_ocr_opt_in(
    conversion_repo: Settings, monkeypatch: pytest.MonkeyPatch, policy: str
) -> None:
    calls: list[list[str]] = []

    def run(argv: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        calls.append(argv)
        return subprocess.CompletedProcess(argv, 0, "", "")

    monkeypatch.setattr(subprocess, "run", run)
    monkeypatch.setattr("dogma.corpus.convert.sys.platform", "darwin")
    assert convert_sources(conversion_repo, ocr=policy).issues == ()
    assert calls[0][calls[0].index("--ocr") + 1] == policy
    assert calls[0][5:7] == ["--extra", "ocr"]


@pytest.mark.parametrize(
    "failure,code",
    [
        ("missing", "CORPUS_CONVERTER_MISSING"),
        ("timeout", "CORPUS_CONVERSION_TIMEOUT"),
        ("nonzero", "CORPUS_CONVERSION_FAILED"),
    ],
)
def test_converter_failures_are_typed(
    conversion_repo: Settings, monkeypatch: pytest.MonkeyPatch, failure: str, code: str
) -> None:
    def run(argv: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        if failure == "missing":
            raise FileNotFoundError("uv")
        if failure == "timeout":
            raise subprocess.TimeoutExpired(argv, 3600)
        return subprocess.CompletedProcess(argv, 2, "SCANNED_PDF_OCR_DISABLED: paper.pdf", "")

    monkeypatch.setattr(subprocess, "run", run)
    report = convert_sources(conversion_repo)
    assert report.converted == 0
    assert report.issues[0].code == code
    assert report.issues[0].path == "corpus/raw/library papers"
    if failure == "nonzero":
        assert "OCR" in report.issues[0].message
        assert "--ocr auto" in report.issues[0].message


def test_missing_vendor_is_reported(conversion_repo: Settings) -> None:
    (conversion_repo.repo_root / "tools/docmd/pyproject.toml").unlink()
    report = convert_sources(conversion_repo)
    assert report.issues[0].code == "CORPUS_CONVERTER_MISSING"


def test_partial_conversion_indexes_success_and_keeps_good_old_source(
    conversion_repo: Settings,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = conversion_repo.repo_root
    old = root / "corpus/md/library papers/old.pdf.md"
    old.parent.mkdir(parents=True)
    old.write_text("### Page 2\nPreviously readable theorem.")
    sync_corpus(conversion_repo, convert=False)

    def run(argv: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        (old.parent / "new.pdf.md").write_text("### Page 1\nNewly readable lemma.")
        report = root / "corpus/md/runs/new-run/changes.json"
        report.parent.mkdir(parents=True)
        report.write_text(
            json.dumps(
                {
                    "repo": "library papers",
                    "counts": {"failed": 1},
                    "changes": [
                        {
                            "source": "old.pdf",
                            "status": "failed",
                            "error": {"code": "CORRUPT_FILE", "message": "Document is corrupt"},
                        }
                    ],
                }
            )
        )
        return subprocess.CompletedProcess(argv, 2, "", "")

    monkeypatch.setattr(subprocess, "run", run)
    report = sync_corpus(conversion_repo)
    assert report.issues
    assert report.conversion.issues[0].path == "corpus/raw/library papers/old.pdf"
    assert len(search(conversion_repo, "theorem")) == 1
    assert len(search(conversion_repo, "lemma")) == 1


def test_raw_symlinks_are_not_passed_to_vendor(
    conversion_repo: Settings, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    outside = tmp_path / "outside.pdf"
    outside.write_text("private bytes")
    (conversion_repo.repo_root / "corpus/raw/library papers/escape.pdf").symlink_to(outside)

    def run(*args: object, **kwargs: object) -> None:
        pytest.fail("Unsafe input must not reach docmd")

    monkeypatch.setattr(subprocess, "run", run)
    report = convert_sources(conversion_repo)
    assert report.issues[0].code == "CORPUS_PATH_INVALID"


def test_docmd_empty_tree_warning_is_an_issue(
    conversion_repo: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    def run(argv: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(
            argv, 0, "Known repo with an empty source tree: deletions were NOT processed.", ""
        )

    monkeypatch.setattr(subprocess, "run", run)
    report = convert_sources(conversion_repo)
    assert report.issues[0].code == "CORPUS_CONVERSION_FAILED"
    assert "deletions" in report.issues[0].message
