"""Live smoke test against the local model server. Marked ``llm``: deselected by default.

Run with ``uv run --frozen pytest -m llm``. It skips unless the server answers its model list.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import httpx
import pytest

from dogma.content.rubrics import load_rubric_version
from dogma.core.config import Settings
from dogma.tutor.gateway import Gateway
from dogma.tutor.interviewer import pick_reveal
from dogma.tutor.models import CompletionRequest, Message
from dogma.tutor.providers.local import LocalProvider
from dogma.tutor.score import score_answer


def _server_answers(base_url: str) -> bool:
    try:
        return httpx.get(f"{base_url}/models", timeout=3.0).status_code == 200
    except httpx.HTTPError:
        return False


@pytest.mark.llm
def test_the_local_server_completes_a_tiny_prompt() -> None:
    settings = Settings()
    if not _server_answers(settings.llm_base_url):
        pytest.skip(f"no local model server answers at {settings.llm_base_url}")

    provider = LocalProvider(settings.llm_base_url)
    result = provider.complete(
        CompletionRequest(
            model=settings.llm_model,
            messages=(Message("user", "Reply with the single word: ready."),),
            max_tokens=16,
        )
    )

    assert result.text


@pytest.fixture
def live_gateway(tmp_path: Path) -> Gateway:
    settings = Settings()
    if not _server_answers(settings.llm_base_url):
        pytest.skip(f"no local model server answers at {settings.llm_base_url}")
    return Gateway(settings.model_copy(update={"repo_root": tmp_path}))


@pytest.mark.llm
def test_live_score_keeps_valid_evidence_and_reports_partial_compliance(
    live_gateway: Gateway, record_property: Callable[[str, object], None]
) -> None:
    curriculum = Path(__file__).resolve().parents[3] / "curriculum"
    rubric = load_rubric_version(curriculum, "topic-answer", 1)
    text = (
        "First I define a churn label as cancellation within thirty days, then split by time to "
        "avoid leakage. I compare logistic regression with a boosted tree baseline using "
        "precision-recall AUC, because churn is rare. I choose the contact threshold using "
        "retention value minus contact cost. The trade-off is recall versus wasted contacts; "
        "a randomized holdout measures incremental retention rather than predictive accuracy. "
        "Finally I monitor calibration and drift and explain the decision to stakeholders."
    )
    result = score_answer(live_gateway, rubric, text)
    # The contract permits partial results: malformed items are counted and never accepted.
    # A live model is not guaranteed to supply every dimension or copy every quote correctly.
    record_property("dropped_items", result.dropped)
    record_property("accepted_dimensions", len(result.scores))
    assert result.scores, "the model must provide at least one usable advisory score"
    assert {item.criterion for item in result.scores} <= set(rubric.criteria_ids)
    assert all(item.quote in text and 1 <= item.score <= 4 for item in result.scores)


@pytest.mark.llm
@pytest.mark.parametrize(
    ("question", "expected"),
    [("How does the current retention process work?", 1), ("What is the weather tomorrow?", None)],
)
def test_live_interviewer_releases_only_a_matching_fact(
    live_gateway: Gateway, question: str, expected: int | None
) -> None:
    result = pick_reveal(
        live_gateway,
        brief="Design a churn prevention service for a subscription business.",
        when_asked=("What does the current retention process do?", "What is the cost of contact?"),
        question=question,
    )
    assert result.number == expected
