# Interview notes: TF-IDF and BM25

## Q1. Build a keyword search over a collection of documents. Which score would you use?

**60-second answer.** I would tokenize consistently, lowercase and keep words, then score with BM25. Its idf
favours rare terms, its term frequency saturates so repeated words stop dominating, and its length term stops long
documents winning on size alone. It is cheap, it can be indexed, and its scores can be explained term by term. I would
check it on labelled queries before tuning $k_1$ and $b$, because the defaults are a strong start.

**Follow-ups.**
- *Why does the saturation matter?* Without it, a document that repeats a word ten times would dominate one that uses it once, although the second is just as relevant.
- *What does the length term fix?* A long document contains more words by chance. Without $b$, long documents rank high for common reasons.

## Q2. BM25 or embeddings for a RAG retriever?

**60-second answer.** I would not pick one. BM25 matches exact terms, such as identifiers, names and rare words, which
embeddings can blur. Embeddings match meaning across paraphrase, which BM25 cannot see. A hybrid retriever runs both and
fuses their rankings, for example with reciprocal rank fusion, and I would measure recall at $k$ for each, and for the
fused list, on labelled queries before I decide.

**Follow-ups.**
- *What is the failure mode of BM25 in RAG?* A paraphrased question shares no term with its passage, so the passage gets a score of zero.
- *Why measure recall rather than precision?* The generator can only use retrieved text. A missed passage cannot be recovered later.

## Common mistakes

- Taking a raw dot product of TF-IDF rows without l2 normalisation, so long documents score higher for length alone.
- Assuming every BM25 variant uses positive idf: the Robertson form can be negative, while the Lucene form used here is positive.
- Comparing BM25 scores across queries, although they depend on the query terms.
- Assuming the search engine's BM25 matches the textbook formula. Check its idf and its clamps.
