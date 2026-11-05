## Hint 1 — concept

A bag of words throws away order and keeps counts. Each term's count is scaled by how rare the term
is across the corpus: a word in every document says little, a word in one document says a lot. Dividing
each row by its length makes long and short documents comparable.

## Hint 2 — formula / approach

Tokenize each document with the regular expression `(?u)\b\w\w+\b` after lowercasing. Count the tokens
per document. Count, for each term, how many documents contain it. The idf of a term is
`ln((1 + n) / (1 + df)) + 1`. The weight is the count times the idf. Divide each row by its l2 norm, and
leave a row of zeros as zeros.

## Hint 3 — code shape

Build `counts`, a list of `Counter` objects, one per document. Build the column list: the given
vocabulary, or `sorted` of every key that occurs. Build a `column` dictionary from term to index. Fill
`np.zeros((n, m))` by looping over each document's counter, and skip terms that are not columns. Finish
with `np.sqrt((matrix ** 2).sum(axis=1, keepdims=True))` and a `np.divide` that uses `where=norms > 0`.
