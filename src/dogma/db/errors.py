"""Storage failures with stable codes (ENGINEERING_STANDARDS §4.4).

The codes mirror ``dogma.core.errors.ErrorCode``, but they live here because the increment that
added persistence does not own ``dogma.core``. Once ErrorCode gains these members, StorageError
can become a DogmaError subclass.
"""

from __future__ import annotations

from enum import StrEnum


class StorageErrorCode(StrEnum):
    DB_UNAVAILABLE = "DB_UNAVAILABLE"
    DB_MIGRATION_FAILED = "DB_MIGRATION_FAILED"
    DB_READ_FAILED = "DB_READ_FAILED"
    DB_WRITE_FAILED = "DB_WRITE_FAILED"
    BACKUP_WRITE_FAILED = "BACKUP_WRITE_FAILED"


class StorageError(Exception):
    """The learner database (or a backup of it) could not be opened, migrated, read or written."""

    def __init__(self, code: StorageErrorCode, message: str) -> None:
        super().__init__(f"[{code}] {message}")
        self.code = code
        self.message = message
