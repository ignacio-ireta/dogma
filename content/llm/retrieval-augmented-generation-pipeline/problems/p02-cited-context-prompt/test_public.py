# Public tests: the examples from problem.md plus at least one edge case (CONTENT_SPEC §6.4).
# The learner's code is imported from submission.py; tests never import solution.py.
import numpy as np
import pytest
from submission import build_cited_prompt


def test_top_two_by_cosine_are_cited_in_rank_order():
    got = build_cited_prompt(
        "How do I roll back a deploy?",
        np.array([1.0, 0.0]),
        ["b", "a", "c"],
        [
            "Roll back by redeploying the previous tag.",
            "Deploys run in stages.",
            "Cats are unrelated.",
        ],
        np.array([[0.9, 0.1], [0.1, 0.9], [0.0, 1.0]]),
        k=2,
        word_budget=100,
    )
    want = (
        "Context:\n[b] Roll back by redeploying the previous tag.\n"
        "[a] Deploys run in stages.\n\nQuestion: How do I roll back a deploy?"
    )
    assert got == want, "wrong chunks or order: rank by cosine, keep k, cite in rank order"


def test_chunk_that_does_not_fit_is_skipped_and_later_chunks_still_fit():
    got = build_cited_prompt(
        "q",
        np.array([1.0, 0.0]),
        ["x", "y", "z"],
        ["one two three four five six seven eight nine ten", "alpha beta gamma", "delta"],
        np.array([[1.0, 0.0], [1.0, 0.1], [1.0, 0.2]]),
        k=3,
        word_budget=5,
    )
    if got != "Context:\n[y] alpha beta gamma\n[z] delta\n\nQuestion: q":
        pytest.fail(
            "wrong chunks under the budget: cost a chunk by its text words only (not the id "
            "label), and skip a chunk that does not fit instead of stopping",
            pytrace=False,
        )


def test_zero_budget_gives_an_empty_context_block():
    got = build_cited_prompt("q", np.array([1.0]), ["a"], ["word"], np.array([[1.0]]), 1, 0)
    assert got == "Context:\n\nQuestion: q", (
        "an empty selection must keep the Context and Question lines"
    )


def test_zero_query_vector_is_rejected():
    with pytest.raises(ValueError):
        build_cited_prompt("q", np.zeros(2), ["a"], ["word"], np.array([[1.0, 0.0]]), 1, 5)
