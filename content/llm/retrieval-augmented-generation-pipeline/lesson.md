---
title: "Retrieval-augmented generation pipeline"
minutes: 12
min_viable: "Read The system and Where it breaks, then implement p01 using Hint 1 if needed."
---

## The system

A RAG pipeline answers a question from documents the model did not memorise. It runs in two phases.

**Offline (index build).** Split each document into chunks, then embed every chunk.

- A chunk is a window of $W$ words. Consecutive windows start $S = W - O$ words apart, so they share $O$ words. $O$ is the overlap, with $0 \le O < W$.
- An embedding is a vector $v_i \in \mathbb{R}^d$ for chunk $i$, with a stable id.

**Online (per question).**

| Stage | Input | Output |
|---|---|---|
| Embed | question $q \in \mathbb{R}^d$ | query vector |
| Score | $s_i = \dfrac{q \cdot v_i}{\lVert q\rVert\,\lVert v_i\rVert}$ | cosine per chunk |
| Retrieve | scores | top-$k$ candidates. Each score is rounded to 9 decimals; equal scores are ordered by ascending chunk id |
| Pack | the $k$ candidates, word budget $B$ | selected chunks, in rank order. $k$ is applied before packing |
| Prompt | selected chunks with ids | text the model answers, citing ids |

**Hybrid scoring** adds a lexical score (BM25, from [cml/tf-idf-and-bm25](../../cml/tf-idf-and-bm25/)) that matches exact terms such as error codes. With z-scores $z(\cdot)$ taken over the candidates, the combined score is $h_i = \alpha\, z(s_i) + (1-\alpha)\, z(b_i)$, where $\alpha \in [0,1]$ and $b_i$ is the BM25 score. Tune $\alpha$ on questions held out from the ones you report, or the recall number is optimistic.

## Worked example

Document: ten words $a\ b\ c\ d\ e\ f\ g\ h\ i\ j$, with $W = 4$ and $O = 2$, so $S = 2$.
Windows start at words 0, 2, 4 and 6, and the last one reaches the end:

- $c_0 = a\,b\,c\,d$, $c_1 = c\,d\,e\,f$, $c_2 = e\,f\,g\,h$, $c_3 = g\,h\,i\,j$.

A fourth window would cover only $i\,j$, which is already inside $c_3$, so the loop stops.

Edge cases, hand-checked:

- The stop rule: a window that reaches the last word ends the loop. For $a\ b\ c\ d\ e$ with $W = 4$, $O = 2$: $c_0 = a\,b\,c\,d$ does not reach the end, and $c_1 = c\,d\,e$ does, so the result is two chunks.
- Empty or whitespace-only text gives no chunks, not one empty chunk.

Retrieval with $q = (1, 0)$, $v_1 = (0.9, 0.1)$ for $c_1$, and $v_2 = (0.1, 0.9)$ for $c_2$. Only these two chunks are scored in this example:

- $s_1 = 0.9 / \sqrt{0.82} \approx 0.994$ and $s_2 = 0.1 / \sqrt{0.82} \approx 0.110$.
- With $k = 2$ and budget $B = 6$, $c_1$ costs 4 words and fits. $c_2$ would bring the total to 8, so it is skipped. Packing counts chunk-text words only, not ids, labels or the question. A skipped chunk does not stop the loop, so a shorter candidate ranked lower can still fit.

## Why it works

Cosine compares direction, not length, which suits embeddings whose norm tracks text length more than meaning.

Overlap protects against boundary splits. Any span of at most $O+1$ words lies wholly inside at least one window. Let $p$ be the start of the span and $j\,S$ the largest multiple of $S$ that is at most $p$. Then $p - jS \le S - 1$, so the span ends by $jS + (S-1) + (O+1) = jS + W$, which is the end of window $j$.

Recall@k measures retrieval alone: the fraction of labelled questions whose gold chunk is in the top $k$. If it is low, no prompt can rescue the answer.

## Signal vs. noise

Signal: chunk size and overlap, the scoring function, the top-$k$ and budget decisions, and the stable chunk ids.

Noise for this session: the choice of vector database, approximate-nearest-neighbour indexes, and the specific embedding model.

## Where it breaks

- **Chunk boundaries.** A fact longer than $O+1$ words can be cut across two windows, and neither chunk holds all of it. Check recall@k as $W$ and $O$ change.
- **Stale index.** The source text changed but the embeddings did not. Store a content hash per chunk id, and rebuild a chunk when its hash changes.
- **Position effects.** Whether prompt position changes answers is not cited here. Measure it on your own questions.
- **Citation hallucination.** The model cites an id that was never in the prompt. Check every cited id against the ids you packed, and reject any answer that cites an unknown id. Validation checks that the id exists, not that the claim is supported: an
  existing id with an unsupported claim is a grounding failure, and it needs a separate check.
- **Budget loss.** Greedy packing can drop the gold chunk to make room for a shorter, weaker one. Measure recall@k before packing and again after it.

## Interview angle

Sixty-second answer: "I chunk with overlap so a boundary fact survives in one window, embed each chunk, and rank by cosine with a fixed tie rule. I keep the top $k$ and pack under a word budget with stable ids. I measure recall@k first, then check that every citation maps to a packed chunk."

Likely follow-ups:
- *How do you choose $W$ and $O$?* Sweep them on labelled questions and plot recall@k against the number of words the model reads.
- *When is lexical search needed?* When the questions contain exact identifiers or rare terms that embeddings blur. Hybrid scoring combines both.

## Connections

This topic uses [cml/cosine-similarity](../../cml/cosine-similarity/) for scoring and [cml/tf-idf-and-bm25](../../cml/tf-idf-and-bm25/) for the lexical half. Top-$k$ cosine search is exact nearest-neighbour search ([cml/nearest-neighbour-search-and-knn](../../cml/nearest-neighbour-search-and-knn/)).
