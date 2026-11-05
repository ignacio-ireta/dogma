# Interview notes — Retrieval-augmented generation pipeline

## Q1. Walk me through a RAG pipeline and where you would measure it.

**60-second answer.** Offline, I chunk each document into word windows with overlap, so a fact near a boundary survives in one chunk. Each chunk gets an embedding and a stable id. Online, I embed the question, rank chunks by cosine with a fixed tie rule, keep the top $k$, optionally rerank, and pack the survivors under a word budget. The prompt labels each chunk with its id. I measure recall@k on labelled questions first, because a missing gold chunk cannot be fixed downstream. Then I measure answer grounding.

**Follow-ups.** How do you pick chunk size and overlap? Sweep them and plot recall@k against the words the model reads. Why keep rank order in the prompt? It puts the best evidence first. Whether position changes answers is something to measure on your own questions, not assume.

## Q2. When would you add lexical search to a dense retriever?

**60-second answer.** When questions depend on exact tokens such as error codes, product names or rare words. Embeddings can blur those, while BM25 matches them literally. I combine the two with a weight $\alpha$ on normalised dense and lexical scores, and I tune $\alpha$ on recall@k for the questions that fail. I keep the cost in view: a second index, and a second score to explain.

**Follow-ups.** How do you normalise the two scores? Use z-scores over the candidate set so the scales are comparable. What if $\alpha$ is tuned on the test set? Then the recall number is optimistic; hold out a split.

## Q3. How do you stop an answer from citing a chunk the model invented?

**60-second answer.** I validate after generation. Every cited id must be one of the chunk ids that went into the prompt, and a citation to an unknown id fails the answer. I also keep chunk ids stable across re-indexing, so an old citation still resolves or fails loudly. Validation checks that a citation exists, not that the claim is supported. Support needs a separate check.

**Follow-ups.** Why not trust the model's citations? It can generate a plausible id from pattern alone. What about a correct id with an unsupported claim? That is a grounding failure, and recall@k cannot catch it.

## Common mistakes

- Judging answers before measuring recall@k, so a retrieval miss looks like a model error.
- Re-embedding without re-chunking, or the reverse, so chunk ids and text drift apart.
- Breaking ties by insertion order, which makes results change when the index is rebuilt.
- Counting the id labels against the word budget, which shrinks the context for no reason.
