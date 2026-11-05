"""Open the learner database: create it if needed, migrate to head, then hand out a session.

Every failure leaves this module as a StorageError with a stable code. Callers never see a raw
SQLAlchemy or Alembic exception.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from alembic.util import CommandError
from sqlalchemy import Engine
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from dogma.core.config import Settings
from dogma.db.engine import DB_FILENAME, make_engine
from dogma.db.errors import StorageError, StorageErrorCode
from dogma.db.migrate import UpgradeResult, upgrade_to_head


def database_path(settings: Settings) -> Path:
    return settings.var_dir / DB_FILENAME


def upgrade_database(settings: Settings) -> UpgradeResult:
    """Create or migrate the learner database. Re-running it changes nothing."""
    engine = _engine(settings)
    try:
        return ensure_schema(engine)
    finally:
        engine.dispose()


@contextmanager
def open_session(settings: Settings, *, on_failure: StorageErrorCode) -> Iterator[Session]:
    """A session in one transaction: committed when the block ends, rolled back on an error.

    ``on_failure`` is the code for a database error raised inside the block (a read or a write).
    Opening and migrating the file have their own codes.
    """
    engine = _engine(settings)
    try:
        ensure_schema(engine)
        with Session(engine) as session, session.begin():
            yield session
    except (SQLAlchemyError, OSError) as exc:
        raise StorageError(on_failure, _reason(exc)) from exc
    finally:
        engine.dispose()


def ensure_schema(engine: Engine) -> UpgradeResult:
    try:
        return upgrade_to_head(engine)
    except (SQLAlchemyError, CommandError) as exc:
        raise StorageError(StorageErrorCode.DB_MIGRATION_FAILED, _reason(exc)) from exc


def _engine(settings: Settings) -> Engine:
    path = database_path(settings)
    try:
        return make_engine(path)
    except OSError as exc:
        raise StorageError(
            StorageErrorCode.DB_UNAVAILABLE, f"cannot use {path}: {_reason(exc)}"
        ) from exc


def _reason(exc: BaseException) -> str:
    """The first line of an error message: the part a learner can act on."""
    lines = str(exc).strip().splitlines()
    return lines[0] if lines else type(exc).__name__
