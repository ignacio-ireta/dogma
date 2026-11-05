"""The SQLite engine: WAL, foreign keys on, and transactions that migrations can roll back.

pysqlite's implicit transaction handling would run DDL outside a transaction, so a failed
migration could leave half a schema behind. The recipe below turns that off and issues BEGIN
ourselves (SQLAlchemy's documented pysqlite workaround).
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

from sqlalchemy import URL, Connection, Engine, create_engine, event

DB_FILENAME = "dogma.db"


def make_engine(path: Path) -> Engine:
    """An engine for the SQLite file at ``path``. Creates the parent directory, not the file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    engine = create_engine(URL.create("sqlite", database=str(path)))

    @event.listens_for(engine, "connect")
    def _configure(dbapi_connection: sqlite3.Connection, _record: object) -> None:
        dbapi_connection.isolation_level = None
        cursor = dbapi_connection.cursor()
        try:
            cursor.execute("PRAGMA foreign_keys = ON")
            cursor.execute("PRAGMA journal_mode = WAL")
            cursor.execute("PRAGMA busy_timeout = 5000")
        finally:
            cursor.close()

    @event.listens_for(engine, "begin")
    def _begin(connection: Connection) -> None:
        # IMMEDIATE takes the write lock up front, so an upgrade never fails halfway
        # through because another writer got in first.
        connection.exec_driver_sql("BEGIN IMMEDIATE")

    return engine
