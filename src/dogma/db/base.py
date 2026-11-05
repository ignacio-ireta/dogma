"""Declarative base and the one column type for timestamps (ENGINEERING_STANDARDS §4.7, §9).

Timestamps are stored as fixed-width UTC ISO-8601 text, so string order is time order. Naive
datetimes are refused rather than guessed at.
"""

from __future__ import annotations

from datetime import UTC, datetime

from sqlalchemy import MetaData, String
from sqlalchemy.engine import Dialect
from sqlalchemy.orm import DeclarativeBase
from sqlalchemy.types import TypeDecorator

# Every constraint and index gets a predictable name, so migrations and autogenerate agree.
NAMING_CONVENTION = {
    "ix": "ix_%(column_0_label)s",
    "uq": "uq_%(table_name)s_%(column_0_name)s",
    "ck": "ck_%(table_name)s_%(constraint_name)s",
    "pk": "pk_%(table_name)s",
}


class Base(DeclarativeBase):
    metadata = MetaData(naming_convention=NAMING_CONVENTION)


class UtcDateTime(TypeDecorator[datetime]):
    """A timezone-aware instant, stored as ``YYYY-MM-DDTHH:MM:SS.ffffff+00:00`` text."""

    impl = String(32)
    cache_ok = True

    def process_bind_param(self, value: datetime | None, dialect: Dialect) -> str | None:
        if value is None:
            return None
        if value.tzinfo is None:
            raise ValueError("timestamps must be timezone-aware; got a naive datetime")
        return value.astimezone(UTC).isoformat(timespec="microseconds")

    def process_result_value(self, value: str | None, dialect: Dialect) -> datetime | None:
        if value is None:
            return None
        return datetime.fromisoformat(value).astimezone(UTC)
