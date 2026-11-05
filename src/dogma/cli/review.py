"""A bounded retrieval loop: question, reveal, then an explicit persisted rating."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Annotated

import typer
from rich.markup import escape

from dogma.cli.context import out, settings_of
from dogma.learning.cards import ReviewCard, due_cards, sync_cards
from dogma.learning.review import rate_card


def review_command(
    ctx: typer.Context,
    limit: Annotated[int, typer.Option(min=1, help="Maximum cards for this review.")] = 20,
) -> None:
    """Review due cards with Again, Hard, Good and Easy ratings."""
    settings = settings_of(ctx)
    now = datetime.now(UTC)
    sync_cards(settings, now=now)
    cards = due_cards(settings, now=now, limit=limit)
    console = out()
    if not cards:
        console.print("No cards due. Pick up where you left off when you're ready.")
        return
    reviewed = 0
    for card in cards:
        rating = _present(card)
        if rating is None:
            break
        result = rate_card(
            settings,
            card.card_id,
            rating,
            now=datetime.now(UTC),
            expected_due=card.due,
            expected_content_hash=card.content_hash,
        )
        reviewed += 1
        console.print(f"Next review: {result.due.astimezone():%d %b %Y %H:%M}")
    console.print(f"Reviewed {reviewed} {'card' if reviewed == 1 else 'cards'}.")


def _present(card: ReviewCard) -> int | None:
    console = out()
    console.print(f"[dim]{escape(card.topic_id)}[/]\n{escape(card.question)}")
    if _prompt("Enter to reveal, q to quit") is None:
        return None
    console.print(escape(card.answer))
    while True:
        answer = _prompt("1 Again / 2 Hard / 3 Good / 4 Easy (q to quit)")
        if answer is None:
            return None
        if answer in {"1", "2", "3", "4"}:
            return int(answer)
        console.print("Choose a rating from 1 to 4, or q to quit.")


def _prompt(message: str) -> str | None:
    try:
        answer = out().input(f"{message}: ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        return None
    return None if answer in {"q", "quit"} else answer
