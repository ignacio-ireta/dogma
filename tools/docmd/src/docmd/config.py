"""Application configuration: defaults < environment (DOCMD_*) < CLI flags."""

from __future__ import annotations

import os
from enum import StrEnum
from pathlib import Path
from typing import Any

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

VALID_LOG_LEVELS = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}


class OcrMode(StrEnum):
    OFF = "off"  # scanned PDFs fail with a structured error (v0.1 behavior)
    AUTO = "auto"  # OCR only pages that have no text layer
    FORCE = "force"  # OCR every page, ignoring any existing text layer


def _default_workers() -> int:
    return min(4, os.cpu_count() or 1)


class AppConfig(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="DOCMD_", env_file=".env", extra="ignore")

    out_dir: Path | None = None
    recursive: bool = False
    resume: bool = False
    fail_fast: bool = False
    force: bool = False
    log_level: str = "INFO"
    workers: int = Field(default_factory=_default_workers, ge=1, le=32)
    deterministic: bool = False
    max_rows_per_sheet: int = Field(default=10_000, ge=1)
    max_cols_per_sheet: int = Field(default=256, ge=1)
    log_document_content: bool = False
    confirm_deletes: bool = False
    archive_keep_runs: int = Field(default=0, ge=0)  # 0 = keep archived conversions forever
    ocr: OcrMode = OcrMode.OFF
    ocr_lang: str | None = None
    ocr_dpi: int = Field(default=300, ge=72, le=600)

    @property
    def ocr_languages(self) -> list[str] | None:
        """``ocr_lang`` split into Vision language codes; None = auto-detect."""
        if not self.ocr_lang:
            return None
        return [code.strip() for code in self.ocr_lang.split(",") if code.strip()]

    @field_validator("log_level")
    @classmethod
    def _normalize_log_level(cls, value: str) -> str:
        upper = value.upper()
        if upper not in VALID_LOG_LEVELS:
            raise ValueError(f"invalid log level {value!r}; expected one of {VALID_LOG_LEVELS}")
        return upper

    @classmethod
    def from_cli(cls, **overrides: Any) -> AppConfig:
        """CLI flags beat env vars: None means 'flag not given', so drop it."""
        return cls(**{k: v for k, v in overrides.items() if v is not None})
