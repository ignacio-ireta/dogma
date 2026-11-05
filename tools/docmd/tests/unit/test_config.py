"""Config layering: defaults < DOCMD_* env vars < CLI overrides."""

import pytest
from pydantic import ValidationError

from docmd.config import AppConfig, OcrMode


def test_defaults() -> None:
    cfg = AppConfig()
    assert cfg.recursive is False
    assert cfg.log_level == "INFO"
    assert 1 <= cfg.workers <= 4
    assert cfg.max_rows_per_sheet == 10_000
    assert cfg.log_document_content is False
    assert cfg.ocr is OcrMode.OFF
    assert cfg.ocr_lang is None
    assert cfg.ocr_dpi == 300


def test_env_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DOCMD_WORKERS", "8")
    monkeypatch.setenv("DOCMD_LOG_LEVEL", "debug")
    cfg = AppConfig()
    assert cfg.workers == 8
    assert cfg.log_level == "DEBUG"


def test_cli_beats_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DOCMD_WORKERS", "8")
    cfg = AppConfig.from_cli(workers=2, recursive=True)
    assert cfg.workers == 2
    assert cfg.recursive is True


def test_cli_none_falls_back_to_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DOCMD_WORKERS", "8")
    cfg = AppConfig.from_cli(workers=None)
    assert cfg.workers == 8


def test_invalid_workers_rejected() -> None:
    with pytest.raises(ValidationError):
        AppConfig(workers=0)
    with pytest.raises(ValidationError):
        AppConfig(workers=33)


def test_invalid_log_level_rejected() -> None:
    with pytest.raises(ValidationError):
        AppConfig(log_level="verbose")


def test_ocr_env_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DOCMD_OCR", "auto")
    monkeypatch.setenv("DOCMD_OCR_DPI", "150")
    cfg = AppConfig()
    assert cfg.ocr is OcrMode.AUTO
    assert cfg.ocr_dpi == 150


def test_ocr_languages_parsing() -> None:
    assert AppConfig().ocr_languages is None
    assert AppConfig(ocr_lang="en-US").ocr_languages == ["en-US"]
    assert AppConfig(ocr_lang=" en-US, de-DE ,").ocr_languages == ["en-US", "de-DE"]


def test_invalid_ocr_settings_rejected() -> None:
    with pytest.raises(ValidationError):
        AppConfig(ocr="sometimes")
    with pytest.raises(ValidationError):
        AppConfig(ocr_dpi=71)
    with pytest.raises(ValidationError):
        AppConfig(ocr_dpi=601)
