from __future__ import annotations

from collections import Counter
from string import punctuation

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from submission import build_cited_prompt

from dogma_data import load_text, rng

VOCAB = ["alpha", "beta", "gamma", "delta", "eps", "zeta"]


def checked(call):
    try:
        result = call()
    except Exception:
        pytest.fail("valid input raised: honor the documented contract", pytrace=False)
    if not isinstance(result, str):
        pytest.fail("return one prompt string", pytrace=False)
    return result


def rejects(call, exc, mistake=None):
    # ``mistake`` names the contract rule that was broken; it is the first line the learner sees.
    message = mistake or f"invalid input must raise {exc.__name__}"
    try:
        call()
    except exc:
        return
    except Exception:
        pytest.fail(message, pytrace=False)
    pytest.fail(message, pytrace=False)


def reference(*, query, qv, ids, texts, vectors, k, budget):
    # Independent path: unit-normalise every vector, then take dot products with the unit query.
    qn = qv / np.linalg.norm(qv)
    unit = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
    scores = [round(float(s), 9) for s in unit @ qn]
    order = sorted(range(len(ids)), key=lambda i: (-scores[i], ids[i]))[:k]
    chosen, used = [], 0
    for i in order:
        cost = len(texts[i].split())
        if used + cost <= budget:
            chosen.append(i)
            used += cost
    lines = ["Context:"] + [f"[{ids[i]}] {texts[i]}" for i in chosen] + ["", f"Question: {query}"]
    return "\n".join(lines)


def random_case(seed: int):
    rng = np.random.default_rng(seed)
    m = int(rng.integers(1, 8))
    d = int(rng.integers(1, 5))
    names = [f"c{j}" for j in range(m)]
    rng.shuffle(names)
    texts = [" ".join(rng.choice(VOCAB, size=int(rng.integers(0, 6)))) for _ in range(m)]
    vectors = rng.normal(size=(m, d))
    qv = rng.normal(size=d)
    k = int(rng.integers(1, m + 2))
    budget = int(rng.integers(0, 16))
    return "what is it?", qv, names, texts, vectors, k, budget


@settings(max_examples=100, deadline=None, derandomize=True, report_multiple_bugs=False)
@given(st.integers(0, 10**6))
def test_random_prompts_match_the_unit_normalised_oracle(seed):
    query, qv, ids, texts, vectors, k, budget = random_case(seed)
    got = checked(lambda: build_cited_prompt(query, qv, ids, texts, vectors, k, budget))
    want = reference(query=query, qv=qv, ids=ids, texts=texts, vectors=vectors, k=k, budget=budget)
    if got != want:
        pytest.fail(
            "selected chunks or their order differ: check cosine (divide by both norms), "
            "the k candidates, and the greedy packing rule",
            pytrace=False,
        )


def test_equal_scores_break_by_chunk_id_not_insertion_order():
    got = checked(
        lambda: build_cited_prompt(
            "q",
            np.array([1.0, 0.0]),
            ["c", "a", "b"],
            ["x", "y", "z"],
            np.array([[2.0, 0.0], [1.0, 0.0], [3.0, 0.0]]),
            k=2,
            word_budget=10,
        )
    )
    assert got == "Context:\n[a] y\n[b] z\n\nQuestion: q", (
        "tied scores must be ordered by chunk id ascending, not by list position"
    )


def test_cosine_not_dot_product_decides_the_ranking():
    got = checked(
        lambda: build_cited_prompt(
            "q",
            np.array([1.0, 0.0]),
            ["long", "short"],
            ["many words here", "one"],
            np.array([[5.0, 5.0], [1.0, 0.0]]),
            k=1,
            word_budget=10,
        )
    )
    assert got == "Context:\n[short] one\n\nQuestion: q", (
        "a long vector pointing off the query direction won: rank by cosine, not raw dot product"
    )


def test_k_counts_candidates_before_the_budget_is_applied():
    got = checked(
        lambda: build_cited_prompt(
            "q",
            np.array([1.0, 0.0]),
            ["a", "b", "c"],
            ["one two three four five six seven eight nine ten", "x y z", "w"],
            np.array([[1.0, 0.0], [0.99, 0.1], [0.9, 0.4]]),
            k=2,
            word_budget=5,
        )
    )
    assert got == "Context:\n[b] x y z\n\nQuestion: q", (
        "k must select the top-k candidates first; a chunk outside the top k must not fill the gap"
    )


def test_no_chunks_gives_the_empty_context_form():
    got = checked(
        lambda: build_cited_prompt("q", np.array([1.0, 0.0]), [], [], np.zeros((0, 2)), 3, 10)
    )
    assert got == "Context:\n\nQuestion: q", (
        "zero chunks must still produce the Context and Question lines"
    )


def test_integer_arrays_match_float_arrays():
    args = (["a", "b"], ["one", "two"], [[2, 0], [1, 1]])
    as_int = checked(
        lambda: build_cited_prompt("q", np.array([1, 0]), *args[:2], np.array(args[2]), 2, 9)
    )
    as_float = checked(
        lambda: build_cited_prompt(
            "q", np.array([1.0, 0.0]), *args[:2], np.array(args[2], float), 2, 9
        )
    )
    assert as_int == as_float, "integer vectors must give the same prompt as their float versions"


@pytest.mark.parametrize("scale", [1e-8, 1e8], ids=["tiny-scale", "huge-scale"])
def test_scale_of_vectors_does_not_change_the_ranking(scale):
    vectors = np.array([[0.2, 0.9], [0.8, 0.1], [0.5, 0.5]])
    base = checked(
        lambda: build_cited_prompt(
            "q", np.array([1.0, 0.3]), ["a", "b", "c"], ["u", "v", "w"], vectors, 3, 9
        )
    )
    scaled = checked(
        lambda: build_cited_prompt(
            "q",
            np.array([1.0, 0.3]) * scale,
            ["a", "b", "c"],
            ["u", "v", "w"],
            vectors * scale,
            3,
            9,
        )
    )
    assert base == scaled, "cosine is scale-invariant: scaling a vector must not change its rank"


def test_inputs_are_not_mutated():
    # Non-unit rows: a normalise-in-place mutant must change these values to be caught.
    vectors = np.array([[3.0, 0.0], [0.0, 4.0]])
    query_vector = np.array([0.6, 0.8])
    ids, texts = ["a", "b"], ["one", "two"]
    before = (vectors.copy(), query_vector.copy(), list(ids), list(texts))
    checked(lambda: build_cited_prompt("q", query_vector, ids, texts, vectors, 2, 9))
    assert np.array_equal(vectors, before[0]) and np.array_equal(query_vector, before[1]), (
        "NumPy inputs must not be changed"
    )
    assert ids == before[2] and texts == before[3], "list inputs must not be changed"


def test_blank_query_is_rejected():
    rejects(
        lambda: build_cited_prompt("  ", np.ones(2), ["a"], ["w"], np.ones((1, 2)), 1, 5),
        ValueError,
    )


def test_non_str_query_or_chunk_text_raises_type_error():
    mistake = "a non-str query or chunk text was accepted; the contract requires TypeError"
    rejects(
        lambda: build_cited_prompt(7, np.ones(2), ["a"], ["w"], np.ones((1, 2)), 1, 5),
        TypeError,
        mistake,
    )  # type: ignore[arg-type]
    rejects(
        lambda: build_cited_prompt("q", np.ones(2), ["a"], [3], np.ones((1, 2)), 1, 5),
        TypeError,
        mistake,
    )  # type: ignore[list-item]


def test_bool_or_non_int_k_and_budget_raise_type_error():
    rejects(
        lambda: build_cited_prompt("q", np.ones(2), ["a"], ["w"], np.ones((1, 2)), True, 5),
        TypeError,
    )
    rejects(
        lambda: build_cited_prompt("q", np.ones(2), ["a"], ["w"], np.ones((1, 2)), 1, 2.0),
        TypeError,
    )  # type: ignore[arg-type]


def test_out_of_range_k_and_budget_raise_value_error():
    rejects(
        lambda: build_cited_prompt("q", np.ones(2), ["a"], ["w"], np.ones((1, 2)), 0, 5), ValueError
    )
    rejects(
        lambda: build_cited_prompt("q", np.ones(2), ["a"], ["w"], np.ones((1, 2)), 1, -1),
        ValueError,
    )


def test_complex_or_nonfinite_vectors_are_rejected():
    mistake = "a complex or non-finite (NaN/inf) vector was accepted; contract needs ValueError"
    rejects(
        lambda: build_cited_prompt("q", np.ones(2, complex), ["a"], ["w"], np.ones((1, 2)), 1, 5),
        ValueError,
        mistake,
    )
    rejects(
        lambda: build_cited_prompt("q", np.ones(2), ["a"], ["w"], np.array([[np.nan, 1.0]]), 1, 5),
        ValueError,
        mistake,
    )


def test_zero_norm_chunk_vector_is_rejected():
    rejects(
        lambda: build_cited_prompt(
            "q", np.ones(2), ["a", "b"], ["w", "v"], np.array([[1.0, 1.0], [0.0, 0.0]]), 2, 5
        ),
        ValueError,
    )


def test_duplicate_or_empty_ids_and_length_mismatch_are_rejected():
    rejects(
        lambda: build_cited_prompt("q", np.ones(2), ["a", "a"], ["w", "v"], np.ones((2, 2)), 2, 5),
        ValueError,
        "duplicate chunk ids were accepted; ids must be unique",
    )
    rejects(
        lambda: build_cited_prompt("q", np.ones(2), [""], ["w"], np.ones((1, 2)), 1, 5),
        ValueError,
        "an empty chunk id was accepted; ids must be non-empty",
    )
    rejects(
        lambda: build_cited_prompt("q", np.ones(2), ["a", "b"], ["w"], np.ones((2, 2)), 1, 5),
        ValueError,
        "ids and texts differ in length, and the mismatch was accepted",
    )


def test_dimension_mismatch_is_rejected():
    rejects(
        lambda: build_cited_prompt("q", np.ones(3), ["a"], ["w"], np.ones((1, 2)), 1, 5),
        ValueError,
        "a query whose length differs from the chunk dimension was accepted",
    )


def test_chunk_vectors_with_a_wrong_row_count_are_rejected():
    rejects(
        lambda: build_cited_prompt("q", np.ones(2), ["a"], ["w"], np.ones((2, 2)), 1, 5),
        ValueError,
        "chunk_vectors must have one row per chunk id: a row count that differs was accepted",
    )


def test_query_vector_must_be_one_dimensional():
    rejects(
        lambda: build_cited_prompt("q", np.ones((1, 1)), ["a"], ["w"], np.ones((1, 1)), 1, 5),
        ValueError,
        "query_vector must have shape (d,): a two-dimensional query was accepted",
    )


def test_scores_equal_to_nine_decimal_places_tie_and_go_to_the_smaller_id():
    # Both cosines round to 0.5 at nine decimal places, so the tie goes to the smaller id
    # even though chunk "b" scores a little higher in full precision.
    high, low = 0.5000000004, 0.5000000001
    vectors = np.array([[high, np.sqrt(1 - high**2)], [low, np.sqrt(1 - low**2)]])
    got = checked(
        lambda: build_cited_prompt(
            "q", np.array([1.0, 0.0]), ["b", "a"], ["x", "y"], vectors, k=1, word_budget=10
        )
    )
    assert got == "Context:\n[a] y\n\nQuestion: q", (
        "scores that agree to nine decimal places are ties: the smaller chunk id must win"
    )


def test_scores_that_differ_in_the_ninth_decimal_are_not_ties():
    # 0.500000002 and 0.500000004 agree to eight decimal places but not to nine, so the
    # higher score wins even though its chunk id sorts later.
    low, high = 0.5000000020, 0.5000000040
    vectors = np.array([[low, np.sqrt(1 - low**2)], [high, np.sqrt(1 - high**2)]])
    got = checked(
        lambda: build_cited_prompt(
            "q", np.array([1.0, 0.0]), ["a", "b"], ["x", "y"], vectors, k=1, word_budget=10
        )
    )
    assert got == "Context:\n[b] y\n\nQuestion: q", (
        "scores are rounded to nine decimal places, so a difference at the ninth place must rank"
    )


@pytest.fixture(scope="module")
def passages() -> list[str]:
    """Paragraphs of a public-domain novel. Many hold line breaks inside them, as real text does."""
    return [part for part in load_text("gutenberg_pride_prejudice").split("\n\n") if part.strip()]


def test_real_passages_print_exactly_and_cost_their_words_across_line_breaks(passages):
    # Bag-of-words count vectors over the 150 most frequent words, computed here in the test.
    tally = Counter(
        word.strip(punctuation + "\u201c\u201d\u2018\u2019\u2014").lower()
        for part in passages
        for word in part.split()
    )
    vocabulary = [word for word, _ in tally.most_common(150) if word]

    def count_vector(text: str) -> np.ndarray:
        counts = Counter(
            word.strip(punctuation + "\u201c\u201d\u2018\u2019\u2014").lower()
            for word in text.split()
        )
        return np.array([float(counts[word]) for word in vocabulary])

    pool = [part for part in passages if "\n" in part.strip() and count_vector(part).any()]
    draw = rng("llm/retrieval-augmented-generation-pipeline/p02-cited-context-prompt:prose:0")
    picks = sorted(int(i) for i in draw.choice(len(pool), size=min(30, len(pool)), replace=False))
    texts = [pool[i] for i in picks]
    vectors = np.stack([count_vector(text) for text in texts])
    ids = [f"p{i:02d}" for i in range(len(texts))]
    query_vector = count_vector("the man and his wife and a good fortune")
    if not query_vector.any():
        pytest.fail("the query vector is empty: the test cannot run", pytrace=False)
    for k, budget in ((6, 90), (12, 60), (12, 250)):
        want = reference(
            query="what does the wife want?",
            qv=query_vector,
            ids=ids,
            texts=texts,
            vectors=vectors,
            k=k,
            budget=budget,
        )
        got = checked(
            lambda k=k, budget=budget: build_cited_prompt(
                "what does the wife want?", query_vector, ids, texts, vectors, k, budget
            )
        )
        if got != want:
            pytest.fail(
                "the prompt differs on real prose: print each chunk text exactly as given, "
                "including its line breaks, and cost a chunk by the words str.split() finds",
                pytrace=False,
            )


def test_more_texts_than_ids_is_rejected():
    rejects(
        lambda: build_cited_prompt("q", np.ones(2), ["a"], ["w", "v"], np.ones((1, 2)), 1, 5),
        ValueError,
        "chunk texts outnumber chunk ids, and the length mismatch was accepted",
    )
