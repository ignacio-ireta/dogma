# Interview notes: ranking metrics

## Q1. How would you evaluate the retriever in a RAG system?

**60-second answer.** I start with recall at $k$ over a labelled set of queries, because the generator can only use
what the retriever returns. If the answer is not in the top $k$, nothing downstream can recover it. Then I measure the
order with NDCG at $k$ if the labels are graded, or with MRR if one item answers each query. I report the $k$, the number
of queries and an interval from a bootstrap, and I say that the offline gain still has to be confirmed by an online test.

**Follow-ups.**
- *Why not just use NDCG?* It is a good ranking score, but it hides the recall ceiling. A retriever that never returns the answer can still have a decent NDCG on the items it returns.
- *What if the labels are incomplete?* Unjudged relevant items count as irrelevant, so the scores are pessimistic for new retrievers. Pool judgements from several systems, or judge a sample of new results.

## Q2. NDCG or MRR: which one, and why?

**60-second answer.** It depends on what the user does with the list. MRR fits one right answer, such as a question
with a single correct passage, because it asks how early that one answer appears. NDCG fits a list the user scans, with
graded relevance, because the gain counts more for highly relevant items and the log discount counts early positions
more. I would not compare the two numbers directly, because their scales and their meanings differ.

**Follow-ups.**
- *How are ties handled?* Equal scores have no defined order. The sklearn default averages over the orders, so the value does not depend on the input order of the items.
- *What is the denominator of average precision at $k$?* This course uses min(k, number of relevant items); other conventions divide by the total relevant count. State the convention when comparing systems.

## Common mistakes

- Reporting one metric without its $k$, its query count or its variance.
- Dividing precision at $k$ by the length of a short list, which inflates it.
- Breaking score ties by input position, so the result depends on the order the items were returned.
- Treating offline NDCG as proof of a better product, without an online test.
