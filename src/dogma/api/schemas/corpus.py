"""Corpus search: cited passages from the local index (ARCHITECTURE §4.6)."""

from __future__ import annotations

from pydantic import BaseModel


class HitOut(BaseModel):
    source_id: str
    path: str
    page: int | None
    headings: list[str]
    ordinal: int
    text: str
    topic_ids: list[str]
    citation: str
    score: float


class SearchOut(BaseModel):
    query: str
    hits: list[HitOut]
