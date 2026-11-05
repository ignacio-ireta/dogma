# TF-IDF matrix with l2 normalisation

Implement `tfidf_matrix(docs, *, vocabulary=None)`. It turns a list of short texts into a bag-of-words
matrix: one row per document, one column per term, each cell a term's raw count times its inverse
document frequency, with every row scaled to unit length. The weighting is the smoothed form that
scikit-learn's `TfidfVectorizer` uses by default. Return the matrix and the list of terms that name its
columns.

## Contract

- `docs`: a non-empty sequence of `str`. `n` is its length.
- **Tokens.** Lowercase each document with `str.lower`, then take every match of the regular expression
  `(?u)\b\w\w+\b`. A token is a run of two or more word characters, so single characters such as `a`,
  `I` and `7` are dropped, and `42` is kept. Punctuation separates tokens.
- **Term frequency** `tf(d, t)`: the raw count of token `t` in document `d`. No logarithm.
- **Document frequency** `df(t)`: the number of documents that contain `t`, counted over all of `docs`,
  even when a `vocabulary` is given.
- **Inverse document frequency:** `idf(t) = ln((1 + n) / (1 + df(t))) + 1`. A term present in every
  document has `idf = 1`.
- **Weight:** `tf(d, t) * idf(t)`. Each row is then divided by its l2 norm. A row whose weights are all
  zero (an empty document, or only dropped tokens) stays all zeros.
- **Vocabulary.** If `vocabulary` is `None`, the terms are every token that occurs in `docs`, sorted in
  ascending string order. Otherwise the terms are the given strings in the given order, which fixes the
  column order. Tokens that are not in the given vocabulary are ignored. A vocabulary term that no
  document contains is still a column (its weight is 0).
- Returns `(matrix, vocabulary)`: `matrix` is a NumPy `float64` array of shape `(n, number of terms)`, and
  `vocabulary` is a new `list[str]` naming the columns.
- Raises `ValueError` for: an empty `docs`; a `vocabulary` that repeats a term; and an empty vocabulary,
  which happens when no token survives (no document has a two-character token and no vocabulary is
  given, or the given vocabulary is empty).
- Must not mutate its inputs.

## Examples

```python
>>> matrix, vocab = tfidf_matrix(["cat sat", "cat cat"])
>>> vocab
['cat', 'sat']
>>> matrix.round(4).tolist()
[[0.5797, 0.8148], [1.0, 0.0]]
# df(cat) = 2, so idf = 1. df(sat) = 1, so idf = ln(3/2) + 1 = 1.4055.
# Row 1 is (1, 1.4055) / sqrt(1 + 1.4055^2). Row 2 holds only "cat", so it normalises to (1, 0).
```

```python
>>> tfidf_matrix(["a cat", "the cat"])[1]
['cat', 'the']
# "a" is a single character, so it is not a token. The vocabulary is sorted: 'cat' before 'the'.
```

```python
>>> matrix, vocab = tfidf_matrix(["red apple", "green apple"], vocabulary=["apple", "pear", "red"])
>>> matrix.round(4).tolist()
[[0.5797, 0.0, 0.8148], [1.0, 0.0, 0.0]]
# "green" is not in the vocabulary, so it is ignored. "pear" is a column of zeros.
```

## Constraints

- Allowed: the Python standard library (`re`, `math`, `collections`) and NumPy.
- Not allowed: `sklearn` (listed in `banned_imports`). Do not use `TfidfVectorizer` or any other
  vectoriser or transformer. Build the counts and the weights yourself.

## Done looks like

All public tests pass, then **Submit** runs the hidden tests. They compare the matrix and the vocabulary
with scikit-learn's `TfidfVectorizer` on random corpora, with and without a fixed vocabulary. They check
the tokenizer (case folding, single characters dropped, digits kept, accented letters kept), the sorted
default vocabulary, the idf of a term present in every document, that each non-empty row has unit length,
that an empty document gives a zero row, the order of a fixed vocabulary, and the errors for an empty
corpus, a corpus with no tokens, a repeated vocabulary term and an empty vocabulary. They check that the
inputs are unchanged, that the matrix is `float64`, and that a 3,000-document corpus finishes quickly.
