"""The review interaction reveals answers and persists only completed ratings."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest
from sqlalchemy import func, select

from dogma.cli import app, review
from dogma.db.errors import StorageErrorCode
from dogma.db.models import Mastery, TopicState
from dogma.db.review_models import CardReview
from dogma.db.session import open_session
from dogma.learning.cards import sync_cards


def _enroll(settings):
    with open_session(settings, on_failure=StorageErrorCode.DB_WRITE_FAILED) as session:
        session.add(
            TopicState(
                topic_id="cml/vector-operations",
                mastery=Mastery.SOLVED,
                solved_at=datetime.now(UTC),
            )
        )


def _events(settings):
    with open_session(settings, on_failure=StorageErrorCode.DB_READ_FAILED) as session:
        return session.scalar(select(func.count()).select_from(CardReview))


def test_review_has_a_calm_empty_state(cli, settings):
    outcome = cli.invoke(app, ["review"], obj=settings)
    assert outcome.exit_code == 0, outcome.output
    assert "No cards due" in outcome.output


def test_review_reveals_and_records_a_rating(cli, settings):
    _enroll(settings)
    outcome = cli.invoke(app, ["review", "--limit", "1"], obj=settings, input="\n3\n")
    assert outcome.exit_code == 0, outcome.output
    assert "orthogonal projection" in outcome.output
    assert "scalar multiple" in outcome.output
    assert "Again" in outcome.output and "Easy" in outcome.output
    assert _events(settings) == 1


def test_card_changed_after_reveal_rejects_rating_without_recording_it(cli, settings, monkeypatch):
    _enroll(settings)
    present = review._present

    def edit_after_reveal(card):
        rating = present(card)
        path = settings.content_dir / "cml/vector-operations/cards.yaml"
        path.write_text(path.read_text().replace("orthogonal projection", "changed projection"))
        sync_cards(settings, now=datetime.now(UTC))
        return rating

    monkeypatch.setattr(review, "_present", edit_after_reveal)
    outcome = cli.invoke(app, ["review", "--limit", "1"], obj=settings, input="\n3\n")
    assert outcome.exit_code == 1, outcome.output
    assert "REVIEW_STALE" in outcome.output
    assert "orthogonal projection" in outcome.output
    assert "changed projection" not in outcome.output
    assert "Traceback" not in outcome.output
    assert _events(settings) == 0


@pytest.mark.parametrize("interaction", ["q\n", "", "\nq\n", "\n"])
def test_quit_or_eof_before_rating_keeps_history_unchanged(cli, settings, interaction):
    _enroll(settings)
    outcome = cli.invoke(app, ["review", "--limit", "1"], obj=settings, input=interaction)
    assert outcome.exit_code == 0, outcome.output
    assert _events(settings) == 0
    if interaction in {"q\n", ""}:
        assert "scalar multiple" not in outcome.output


def test_invalid_rating_can_be_corrected_without_an_extra_event(cli, settings):
    _enroll(settings)
    outcome = cli.invoke(app, ["review", "--limit", "1"], obj=settings, input="\n9\n2\n")
    assert outcome.exit_code == 0, outcome.output
    assert _events(settings) == 1


def test_review_limit_has_usage_validation(cli, settings):
    outcome = cli.invoke(app, ["review", "--limit", "0"], obj=settings)
    assert outcome.exit_code == 2


def test_review_help_has_an_example(cli, settings):
    outcome = cli.invoke(app, ["review", "--help"], obj=settings)
    assert outcome.exit_code == 0
    assert "Example:" in outcome.output
