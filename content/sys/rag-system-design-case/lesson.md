---
title: RAG system design
minutes: 12
min_viable: "Read The system and Where it breaks, then the brief and the twist."
---

## The system

An assistant that answers from internal documents is a pipeline with two separate failure points. Name them
separately in every answer.

| Stage | Job | Share of the 4 s p95 budget |
|---|---|---|
| Ingest (offline) | extract text and tables; chunk by heading; attach version, status, effective date and access groups | none |
| Retrieve | hybrid search (keyword plus dense), filtered by the asker's live access groups; keep 50 | 0.3 s |
| Rerank | order the 50 candidates; keep 5 | 0.5 s |
| Generate | answer from the 5 chunks, citing their ids, or abstain | 2.5 s |
| Check | every citation is a packed chunk id; every claim is supported by it | 0.1 s |
| Overhead | query embedding, access lookup, network | 0.6 s |

The two failures:

- **Retrieval failure.** The gold passage is not among the packed chunks. No prompt can fix it.
- **Generation failure.** The gold passage is present, and the answer is wrong, unsupported or uncited.

Access rights and freshness belong in the retrieval filter, not in a review step after generation.

## Worked example

Two hundred labelled questions. The gold passage is among the packed chunks for 170 of them, so recall@5 is
170/200 = 0.85.

- Gold present (170): 153 answers are correct. Generation accuracy is 153/170 = 0.90.
- Gold absent (30): the grounding check makes the system abstain on all 30. Abstaining is safe, but it still counts
  as a miss, because the answer was in the corpus and retrieval never packed it.
- Answered correctly: 153/200 = 0.765.
- Misses: 47 in total. 30 are retrieval failures and 17 are generation failures, so fix retrieval first.

## Why it works

Generation can only be right when the gold passage is packed, so recall@k sets the ceiling. Measure it on
passages, not whole documents. A document hit can hide a passage that never reached the prompt.

Chunk by heading so a passage keeps its section title. Keep each table whole, because PDF extraction often
splits table rows across lines. Keyword scores find error codes and product names that embeddings blur, and
dense scores find paraphrases. Hybrid scoring combines them, and a reranker then turns a wide, cheap net into a
short, precise list. The retrieval-augmented design comes from Lewis et al. (2020). The access control and
versioning in this case are not in that paper.

Evaluation set: 400 questions from the support log, 100 per question type, with gold passages labelled by two
reviewers. Double-judge 20 percent. Freeze the set and refresh a fifth of it each quarter. At 400 questions and
accuracy 0.80, the 95 percent interval is about plus or minus 4 points: $1.96\sqrt{0.8 \times 0.2 / 400} \approx 0.039$.
Compare two systems scored on the same questions with a paired test, which removes question difficulty from the
comparison.

## Signal vs. noise

**Signal:** gold-passage recall, the split of misses into retrieval and generation, the permission test results,
and cost and latency per answer.

**Noise:** the choice of vector store, and an embedding model's leaderboard rank, before the evaluation set exists.

## Where it breaks

- **Permissions copied into the index.** A copy is correct only until the source changes. A revocation that waits
  for a nightly sync can leak for up to 24 hours. Check the live access groups on every query, and filter before
  generation.
- **Stale or superseded documents.** A retrieved runbook can be accurate text from the wrong version. Store status
  and effective date, and exclude superseded versions at retrieval.
- **Broken tables.** A split row gives confident, wrong numbers. Test extraction on the most-asked documents.
- **Invented citations.** An answer that cites an id never packed is wrong, even when its sentence is true. Reject it.
- **Judging by fluency.** Reviewers reward fluent answers. Judge correctness against the gold passage, and grounding
  claim by claim.

## Interview angle

Sixty-second shape: "I split the system into retrieval and generation and measure each. Retrieval is hybrid and
filtered by the asker's live permissions. Generation answers from five cited chunks or abstains. The evaluation
set has 400 stratified, double-judged questions. Launch needs zero permission leaks, and cost and latency inside
budget." Expect the follow-ups "how do you know retrieval is good enough?" and "what happens when a document is
revoked?"

## Connections

Builds on [cml/ranking-metrics](../../cml/ranking-metrics/) for recall@k, and on
[llm/retrieval-augmented-generation-pipeline](../../llm/retrieval-augmented-generation-pipeline/) for chunking,
packing and citation checks. The [recommendation case](../recommendation-system-case/) has the same two-stage
shape, with a ranker in place of generation.
