---
title: TF-IDF and BM25
minutes: 12
min_viable: "Read The system and the Worked example, then solve p01."
---

## The system

Both scores treat a document as a bag of words: a count of each term, with word order dropped. Let $N$ be the number of documents and $n(t)$ the number of documents that contain term $t$.

| Part | TF-IDF (as in scikit-learn's default) | BM25 (Okapi) |
|---|---|---|
| Term weight | $\mathrm{tf}\cdot\mathrm{idf}$, with $\mathrm{idf}=\ln\frac{1+N}{1+n}+1$ | $\mathrm{idf}\cdot\frac{f(k_1+1)}{f+k_1\left(1-b+b\,\lvert D\rvert/\mathrm{avgdl}\right)}$ |
| Length | row divided by its $\ell_2$ norm | $b$ scales the term part by document length |
| Saturation | none: a count of 10 weighs ten times a count of 1 | $k_1$ caps the term part near $k_1+1$ |
| Scoring | cosine similarity between rows | sum over the query terms |

With BM25's idf written in the Lucene form, $\mathrm{idf}(t)=\ln\!\left(1+\frac{N-n+0.5}{n+0.5}\right)$, which is always positive.

## Worked example

Two documents: $D_1=$ "cat cat dog", $D_2=$ "dog". The tokenizer keeps runs of two or more word characters, so the corpus vocabulary is cat, dog; $D_2$ contains only dog.

- TF-IDF: $n(\text{cat})=1$, $n(\text{dog})=2$. The $\ell_2$-normalised rows are $(0.942,\ 0.335)$ for $D_1$ and $(0,\ 1)$ for $D_2$, with the columns in the order cat, dog.
- BM25, query "cat", $k_1=1.2$, $b=0.75$: $N=2$, $n=1$, so $\mathrm{idf}=\ln 2=0.693$. The average length is 2 and $|D_1|=3$, so $D_1$ scores $0.693\cdot\frac{2(2.2)}{2+1.2(0.25+1.125)}=0.836$. $D_2$ contains no "cat", so it scores 0.

Saturation with a document at average length: a count of 1, 2, 4 and 10 gives term parts of $1.00$, $1.38$, $1.69$ and $1.96$ times idf. The ceiling is $k_1+1=2.2$.

## Why it works

**Rare terms carry information.** The idf idea goes back to Spärck Jones (1972): a term found in few documents says more about a match. The `+1` in TF-IDF keeps a term found everywhere from weighing zero.

**The Robertson idf goes negative.** The original BM25 idf is $\ln\frac{N-n+0.5}{n+0.5}$. It is negative once a term appears in more than half the documents. With $N=10$ and $n=7$ it is $-0.76$, so a common term would lower the score of a matching document. The Lucene form adds 1 inside the log and gives $+0.38$ for the same case.

**Saturation and length.** $k_1$ sets how quickly the count stops mattering. $k_1=0$ gives a binary match weighted by idf. $b=0$ ignores length, and $b=1$ scales fully by $|D|/\mathrm{avgdl}$, so a long document needs more occurrences to score the same. Robertson and Zaragoza (2009) derive the form from a probabilistic model.

## Signal vs. noise

- **Signal:** BM25 with its usual defaults, $k_1\approx1.2$ and $b\approx0.75$, is a strong baseline for keyword-heavy queries. Tune them only with a labelled set.
- **Signal:** the tokenizer, case folding and stemming often change results more than the exact formula.
- **Noise:** a small gap between two scores is not a probability. Only the order of the scores is meaningful.

## Where it breaks

- **Vocabulary mismatch.** A paraphrase that shares no term scores zero. Dense embeddings handle that case, which is why hybrid systems exist.
- **Exact identifiers are strong.** Product codes and names are matched exactly, which an embedding may blur. This is a reason to keep BM25 in the mix.
- **Repeated query terms.** Whether a repeated query word counts twice depends on the engine. Check it before comparing numbers.
- **Chunk size.** In RAG, changing the chunk length changes the average length, and so the length term.

**Punctuation that is not ASCII.** Real prose has curly apostrophes and em dashes, and the token rule treats both as separators. On 45 paragraphs of *Pride and Prejudice* that contain one, a tokenizer that first removes the curly apostrophe gets 349 terms where scikit-learn gets 342. It gains `dont` and `bingleys`, and it loses `don` and `brother`, so every row that holds those words changes. Test a tokenizer on real text with its punctuation, not on lowercase ASCII words.

**The system you already use.** `dogma corpus search` ranks chunks with SQLite FTS5's `bm25()`, ordered ascending, since lower values are better. The idf in FTS5 is the Robertson form, and in a check a term found in more than half the rows scored about $-10^{-6}$, that is, nearly nothing. A common word in your corpus therefore barely affects the ranking.

## Interview angle

"Why not just use embeddings?" Answer shape: BM25 is cheap, explainable and exact on terms. Its saturation and length normalisation are the two properties to name. A RAG system usually runs BM25 and a dense retriever, fuses the two rankings, and measures recall at $k$ on labelled queries. Follow-ups are in `interview.md`.

## Connections

- **Prerequisites:** `cml/cosine-similarity` (the TF-IDF score is a cosine on normalised rows).
- **Unlocks:** `cml/ranking-metrics` (how to measure the ranked lists these scores produce).
- **Lineage:** Spärck Jones (1972) for idf; Robertson and Zaragoza (2009) for BM25; Manning, Raghavan and Schütze (2008) for the vector space model.
- **References:** [The TF-IDF guide](https://scikit-learn.org/stable/modules/feature_extraction.html#tfidf-term-weighting) states the smoothed formula; [Lucene BM25Similarity](https://lucene.apache.org/core/9_12_1/core/org/apache/lucene/search/similarities/BM25Similarity.html) documents its positive idf.
