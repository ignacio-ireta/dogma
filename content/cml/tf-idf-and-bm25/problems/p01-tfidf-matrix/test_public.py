"""Public tests: the examples from problem.md and one degenerate case.

The learner's code is imported from submission.py. Tests never import solution.py.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from submission import tfidf_matrix


def _solve(docs: Any, **kwargs: Any) -> tuple[np.ndarray, list[str]]:
    """Call the learner on input that must be accepted. A crash is a contract mistake."""
    try:
        matrix, vocab = tfidf_matrix(docs, **kwargs)
    except Exception as exc:
        pytest.fail(
            f"valid input raised {type(exc).__name__}: check the tokens, the idf and the vocabulary"
        )
    return np.asarray(matrix), vocab


def test_two_documents_with_repeated_term() -> None:
    matrix, vocab = _solve(["cat sat", "cat cat"])
    assert vocab == ["cat", "sat"], "with no vocabulary given, the columns are the tokens, sorted"
    expected = np.array([[0.5797, 0.8148], [1.0, 0.0]])
    assert np.allclose(matrix, expected, atol=1e-4), (
        "weight = raw count * (ln((1 + n) / (1 + df)) + 1), then each row divided by its l2 norm"
    )


def test_single_character_tokens_are_dropped() -> None:
    _, vocab = _solve(["a cat", "the cat"])
    assert vocab == ["cat", "the"], (
        "tokens are runs of two or more word characters: the single letter 'a' is not a term"
    )


def test_fixed_vocabulary_fixes_the_columns_and_ignores_other_tokens() -> None:
    matrix, vocab = _solve(["red apple", "green apple"], vocabulary=["apple", "pear", "red"])
    assert vocab == ["apple", "pear", "red"], "a given vocabulary keeps its own order"
    expected = np.array([[0.5797, 0.0, 0.8148], [1.0, 0.0, 0.0]])
    assert np.allclose(matrix, expected, atol=1e-4), (
        "'green' is not in the vocabulary and must be ignored; 'pear' is a zero column"
    )


def test_empty_corpus_raises_value_error() -> None:
    try:
        tfidf_matrix([])
    except ValueError:
        return
    except Exception as exc:
        pytest.fail(f"an empty corpus must raise ValueError (raised {type(exc).__name__})")
    pytest.fail("an empty corpus was accepted: raise ValueError when there are no documents")
