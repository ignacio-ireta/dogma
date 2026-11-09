# Retrieve top-k chunks by cosine similarity and pack a cited prompt

Dense retrieval embeds every chunk and the query as vectors, ranks chunks by cosine similarity, and
keeps the best few. The model can only read a limited number of words, so the selected chunks must
fit a word budget. Implement `build_cited_prompt(query, query_vector, chunk_ids, chunk_texts,
chunk_vectors, k, word_budget)`. It returns one prompt string whose context lines carry the chunk
ids, so the answer can cite them.

## Contract

- `query`: `str`, not blank after `strip()`.
- `query_vector`: real NumPy array, shape `(d,)`, `d >= 1`. Must not be the zero vector.
- `chunk_ids`: `list[str]` of length `m >= 0`. Ids are unique and non-empty.
- `chunk_texts`: `list[str]` of length `m`.
- `chunk_vectors`: real NumPy array, shape `(m, d)`. No row may be the zero vector.
- `k`: `int >= 1`. `word_budget`: `int >= 0`.
- All vector entries must be finite. Integer and float arrays are accepted; complex, boolean and
  object arrays are not.

**Score.** `s_i = (q . v_i) / (|q| * |v_i|)`, the cosine similarity. Round each score to 9 decimal
places before comparing, so that numerically equal scores are ties.

**Ranking.** Sort chunks by the key `(-s_i, chunk_id)`. Equal scores are broken by the chunk id in
ascending string order. Insertion order never decides a tie.

**Candidates.** The first `min(k, m)` chunks in the ranking are the candidates. `k` counts
candidates, not selected chunks.

**Packing.** Walk the candidates in rank order. A chunk costs `len(text.split())` words. Add it if
the running total plus its cost is at most `word_budget`. Otherwise skip it and keep going, so a
shorter lower-ranked candidate can still fit. The budget counts chunk-text words only; ids, labels
and the question are not counted. A chunk text may hold line breaks or runs of spaces: it is printed
unchanged, and its cost is the number of words `str.split()` finds (`"a\nb"` costs 2).

**Output.** A string built with `"\n".join` from these lines, in order:
1. `Context:`
2. one line per selected chunk, in rank order: `[<chunk id>] <chunk text exactly as given>`
3. an empty line
4. `Question: <query exactly as given>`

With no selected chunks, the context block is empty and the output is `Context:`, an empty line,
then the question line.

**Errors.** Raise `TypeError` when `query` or any id or text is not a `str`, or when `k` or
`word_budget` is not an `int` (bool is not accepted). Raise `ValueError` for a blank query, a
non-numeric or non-finite array, a wrong shape or length mismatch, a zero-norm query or chunk
vector, a duplicate or empty chunk id, `k < 1`, or `word_budget < 0`.

Must not mutate inputs.

## Examples

```python
build_cited_prompt(
    "How do I roll back a deploy?",
    np.array([1.0, 0.0]),
    ["b", "a", "c"],
    ["Roll back by redeploying the previous tag.", "Deploys run in stages.", "Cats are unrelated."],
    np.array([[0.9, 0.1], [0.1, 0.9], [0.0, 1.0]]),
    k=2,
    word_budget=100,
)
# -> "Context:\n[b] Roll back by redeploying the previous tag.\n[a] Deploys run in stages.\n\nQuestion: How do I roll back a deploy?"
```

```python
# ranking: x (cos 1.0), y (cos ~0.995), z (cos ~0.98); k=3; budget 5 words
build_cited_prompt(
    "q",
    np.array([1.0, 0.0]),
    ["x", "y", "z"],
    ["one two three four five six seven eight nine ten", "alpha beta gamma", "delta"],
    np.array([[1.0, 0.0], [1.0, 0.1], [1.0, 0.2]]),
    k=3,
    word_budget=5,
)
# -> "Context:\n[y] alpha beta gamma\n[z] delta\n\nQuestion: q"
# x costs 10 words, does not fit, and is skipped; y then z fit in the 5-word budget.
```

Real text: a paragraph of `gutenberg_pride_prejudice` (public domain) can hold a line break, as in
`"It is a truth\nuniversally acknowledged"`. The chunk is printed with its line break and
costs 6 words (splitting on spaces alone would count 5):

```python
build_cited_prompt(
    "q",
    np.array([1.0]),
    ["p"],
    ["It is a truth\nuniversally acknowledged"],
    np.array([[1.0]]),
    k=1,
    word_budget=6,
)
# -> "Context:\n[p] It is a truth\nuniversally acknowledged\n\nQuestion: q"
```

## Constraints

Allowed: Python standard library and NumPy. Not allowed: `sklearn`, `scipy.spatial` (including its
cosine distance), or any vector-database or retrieval library (`faiss`, `rank_bm25`, `langchain`,
`llama_index`). Compute the cosine yourself from dot products and norms.

## Done looks like

The public examples pass, plus a degenerate case with no chunks or a zero-norm vector. Hidden tests
compare the ranking against an independent unit-normalised computation on seeded random inputs and on
bag-of-words vectors of real paragraphs (with line breaks), checking the prompt text and the word budget,
and check ties broken by chunk id, a long vector that a dot product would wrongly favour, the
skip-and-continue packing rule, the exact output format including the empty-context form, the
k-candidate rule, and every error family.
