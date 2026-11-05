"""Corpus search over the local FTS index (ARCHITECTURE §4.6). Read-only; never stores state."""

from __future__ import annotations

from dogma.api.ids import is_topic_id
from dogma.api.schemas.corpus import HitOut, SearchOut
from dogma.core.config import Settings
from dogma.core.errors import DogmaError, ErrorCode
from dogma.corpus.index import search


def find(settings: Settings, query: str, topics: list[str], limit: int) -> SearchOut:
    bad = [topic for topic in topics if not is_topic_id(topic)]
    if bad:
        raise DogmaError(ErrorCode.CORPUS_QUERY_INVALID, f"not a topic id: {bad[0]!r}")
    hits = search(settings, query, topics=tuple(topics), k=limit)
    return SearchOut(
        query=query,
        hits=[
            HitOut(
                source_id=hit.chunk.source_id,
                path=hit.chunk.path,
                page=hit.chunk.page,
                headings=list(hit.chunk.headings),
                ordinal=hit.chunk.ordinal,
                text=hit.chunk.text,
                topic_ids=list(hit.chunk.topic_ids),
                citation=hit.citation,
                score=hit.score,
            )
            for hit in hits
        ],
    )
