# Interview notes — RAG system design

## Q1. Design an internal question-answering assistant over documents that have access controls.
**60-second answer.** Split the system into ingest, retrieval, reranking, generation and a check. Ingest keeps
each document's version, status and access groups. At query time, retrieval filters by the asker's live groups,
combines keyword and dense search, and reranks to five chunks. Generation answers from those chunks with
citations, or abstains. Launch needs a frozen evaluation set and zero permission leaks.
**Follow-ups.** Why not check permissions after generation? Restricted text in the prompt can be quoted, so a
post-check is too late. How do you handle the four-second budget? Name the share for each stage, and cut the
reranker's candidate count first if retrieval runs long.

## Q2. An answer was wrong. How do you tell whether retrieval or generation failed?
**60-second answer.** Look at the gold passage. If it was not among the packed chunks, retrieval failed, and no
prompt change will help. If it was packed and the answer is still wrong or unsupported, generation failed. Keep
both counts for every evaluation run, so the next fix goes to the larger bucket.
**Follow-ups.** How do you find the gold passage? Label it during evaluation, with two reviewers, and keep the
label set frozen. What about answers that are right with no gold passage? Count them as failures of the grounding
rule, because the documents did not support them.

## Q3. How big should the evaluation set be, and how would you judge it?
**60-second answer.** About 400 questions, stratified by question type with 100 per type, so rare types are not
lost. Two reviewers judge 20 percent, and I report their agreement. At 400 questions and accuracy around 0.80 the
95 percent interval is about plus or minus 4 points, so I compare systems with a paired test on the same questions.
**Follow-ups.** Why freeze the set? Tuning on it makes the reported number optimistic, so refresh a fifth each
quarter and keep a separate tuning set. What if reviewers disagree? Treat disagreement as a rubric problem, and
fix the rubric before the next run.

## Q4. A document was revoked yesterday, and an answer still quoted it. What do you do?
**60-second answer.** Treat it as an incident. Stop serving the affected sources, find every answer that quoted
the document since the revocation, and notify the owners. Then change the design: check live access on every
query, and make revocations push to the index within minutes, not at the next nightly sync.
**Follow-ups.** How do you prove no other leak exists? Run the permission probes against the restricted set
until they pass with zero leaks, and keep them in the launch gate. Why not just shorten the sync interval? It
narrows the window but keeps a stale copy of the rule, so the check must still run at query time.

## Common mistakes
- Naming a vector store before naming the decision, the access rule and the metric.
- Measuring answer fluency and calling it accuracy.
- Reporting one accuracy number with no retrieval split and no interval.
- Treating permissions as a filter applied after the answer is written.
