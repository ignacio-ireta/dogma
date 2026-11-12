# BM25 scores for a query

Implement `bm25_scores(query_tokens, docs_tokens, *, k1=1.2, b=0.75)`. It scores every document in a
collection against one query with Okapi BM25, the ranking function behind most keyword search engines.
The input is already tokenized: you do not lowercase or split anything. Return one score per document,
in input order.

## Contract

- `query_tokens`: a sequence of `str`. It may be empty. Each occurrence counts: a token that appears
  twice in the query contributes twice.
- `docs_tokens`: a non-empty sequence of documents. Each document is a sequence of `str` tokens and may
  be empty. Let `N` be the number of documents.
- For a document `D`, let `|D|` be its token count, and `f(q, D)` the number of times token `q` occurs
  in `D`. Let `avgdl` be the mean of `|D|` over **all** `N` documents, empty ones included.
- Let `n(q)` be the number of documents that contain `q` at least once. The inverse document frequency
  is the Lucene form, which is always positive: `idf(q) = ln(1 + (N - n(q) + 0.5) / (n(q) + 0.5))`.
- Score of `D`: the sum over the query tokens `q` of
  `idf(q) * f(q, D) * (k1 + 1) / (f(q, D) + k1 * (1 - b + b * |D| / avgdl))`.
  A term with `f(q, D) = 0` contributes `0`. A query token that is in no document contributes `0`.
- If every document is empty, `avgdl` is 0 and every score is `0.0`. Do not divide by zero.
- An empty query gives `0.0` for every document.
- Returns: a list of `N` Python floats, in input order.
- Raises `ValueError` for: an empty `docs_tokens`, a negative `k1`, and a `b` outside `[0, 1]`.
- Must not mutate its inputs.

## Examples

```python
>>> docs = [["cat", "sat"], ["dog"], ["cat", "cat", "dog"]]
>>> [round(s, 4) for s in bm25_scores(["cat"], docs)]
[0.47, 0.0, 0.5666]
# N = 3, n(cat) = 2, so idf = ln(1.6) = 0.47. avgdl = 2. The third document has |D| = 3 and f = 2.
```

```python
>>> [round(s, 4) for s in bm25_scores(["cat", "zebra"], docs)]
[0.47, 0.0, 0.5666]
# "zebra" is in no document, so it adds nothing
```

```python
>>> [round(s, 4) for s in bm25_scores(["cat"], docs, k1=0.0)]
[0.47, 0.0, 0.47]
# with k1 = 0 the term frequency is ignored: each matching document gets exactly idf
```

## Constraints

- Allowed: the Python standard library (`math`, `collections`) and NumPy.
- Not allowed: `sklearn` (listed in `banned_imports`). Write the formula yourself.

## Done looks like

All public tests pass, then **Submit** runs the hidden tests. They compare the scores with an independent
NumPy implementation on random collections with empty documents, repeated tokens and random `k1` and `b`.
They check the properties of the formula: term frequency saturates, so the score rises with the count but
never past `idf * (k1 + 1)`; with `b = 0` the length of a document does not matter, and with `b = 1` a
longer document with the same count scores lower; with `k1 = 0` each matching document scores exactly
its idf; an unseen query token adds nothing; a repeated query token counts twice; and a term found in
most documents still adds a positive score. They check that `avgdl` includes empty documents, that a
collection of empty documents gives zeros without an error, that an empty query gives zeros, that the
argument errors raise `ValueError`, that reordering the documents reorders the scores, that the inputs
are unchanged, and that a 5,000-document collection finishes quickly.

The hidden tests draw their bulk inputs from the real datasets `gutenberg_frankenstein`,
`gutenberg_pride_prejudice` and `gutenberg_walden` (see `docs/DATASETS.md`).
