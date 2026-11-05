"""SQLite persistence: the learner's attempts and topic states (ARCHITECTURE §4.9, ADR-0004).

Everything in ``var/dogma.db`` is written through this package. Schema changes go through
Alembic migrations only, and the app upgrades the schema on first use.
"""
