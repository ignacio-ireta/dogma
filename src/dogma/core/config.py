"""Single source of settings (ENGINEERING_STANDARDS §4.6). Env prefix ``DOGMA_``."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

_REPO_ROOT = Path(__file__).resolve().parents[3]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="DOGMA_", env_file=".env", extra="ignore")

    repo_root: Path = _REPO_ROOT

    # Local model gateway (ADR-0008). Must stay loopback; validated by the tutor gateway.
    llm_base_url: str = "http://127.0.0.1:8080/v1"
    llm_model: str = "qwen3.8-27b-q6"
    allow_external: bool = False

    # Runner defaults (ARCHITECTURE §4.2); problems may override timeout in meta.yaml.
    run_timeout_s: int = 30
    run_memory_mb: int = 2048  # torch problems fit; runaway code is stopped well before swap

    # Global switch for attempt recording (DOGMA_RECORD_ATTEMPTS=false for scratch/agent use).
    record_attempts: bool = True

    @property
    def content_dir(self) -> Path:
        return self.repo_root / "content"

    @property
    def curriculum_dir(self) -> Path:
        return self.repo_root / "curriculum"

    @property
    def syllabus_path(self) -> Path:
        return self.curriculum_dir / "syllabus.yaml"

    @property
    def var_dir(self) -> Path:
        return self.repo_root / "var"

    @property
    def runs_dir(self) -> Path:
        return self.var_dir / "runs"

    @property
    def workspace_dir(self) -> Path:
        return self.repo_root / "workspace"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
