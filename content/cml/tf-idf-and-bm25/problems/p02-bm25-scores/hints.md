## Hint 1 — concept

BM25 is a weighted count of query terms. Three ideas do the work. A term's weight grows with how rare it
is across the collection (the idf). Its count saturates: the tenth occurrence adds less than the second.
A longer document is discounted, because a long text contains more words by chance. The constants k1 and
b set how strong the saturation and the length discount are.

## Hint 2 — formula / approach

Count, for each query token, the documents that contain it: that is n(q). The idf is
`ln(1 + (N - n + 0.5) / (n + 0.5))`. For each document with f > 0 occurrences, add
`idf * f * (k1 + 1) / (f + k1 * (1 - b + b * length / avgdl))`. The average length includes empty
documents. A query token that matches no document adds nothing, and a document with no matching token
scores 0.

## Hint 3 — code shape

Validate `k1` and `b` first, and reject an empty collection. Build one `Counter` per document and a list
of lengths, and compute `avgdl` as their mean. Keep a list of zeros for the scores. Loop over the query
tokens, one pass per token: find the documents that contain it, compute its idf once, then add the term
part to each of those documents. Return the list.
