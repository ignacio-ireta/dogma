# Interview notes — Recommendation system

## Q1. How would you build a feed for a marketplace where many buyers and listings are new?
**60-second answer.** Use several candidate sources, each with a cold-start path. New buyers get category and
session context. New listings get attribute similarity to established items, plus a small exposure quota so
they can collect labels. Generation returns about 500 candidates per buyer, and a ranker orders them. I judge
generation by recall, the ranker by exposure-corrected NDCG, and the feed by an A/B test with guardrails.
**Follow-ups.** How do you rank a listing with no history? Use a content-based score from its attributes, plus
a short exploration quota that retires once the listing has labels. How large is the quota? Small enough to
protect the purchase rate, and sized using the randomised slice.

## Q2. Offline NDCG rose 6 percent, and the A/B test is flat. What happened?
**60-second answer.** The offline labels come from logs the old feed wrote. Items the old feed never showed have
no labels, so a candidate that favours them is marked down, and one that copies the old feed is marked up. I
would check the randomised slice for an unbiased comparison, then ask whether the test had the power to see a
lift of the size that matters.
**Follow-ups.** How would you estimate the lift before the test? Replay the candidate ranker on randomised-slice
logs and count only the slots where it agrees with what was shown. Is a flat result a no? It is no evidence of a
gain at this sample size. Read the interval, not the point estimate.

## Q3. Two rankers each win on different metrics. How do you choose?
**60-second answer.** Name the guardrails first: new-seller share, p95 latency and return rate. Then compare the
candidates on the same tuning budget, with five seeds each, and plot the non-dominated set on purchase lift,
catalogue coverage and latency. The choice is the front point that clears the guardrails with the best purchase
value per serving dollar.
**Follow-ups.** What if the front has ten points? Use the guardrails and decision value to cut it to two or
three, then test those. Why not a weighted sum? It hides the trade-off and depends on weights nobody can defend.

## Q4. A new ranker lifts purchases but concentrates impressions on a few sellers. What do you do?
**60-second answer.** Treat it as a feedback loop, not a success. Hold the rollout, measure the impression share of
the top listings and the number of listings that sell, and compare both with the control arm. Retrain on
exposure-corrected labels with a diversity constraint, then re-test against the guardrail before widening.
**Follow-ups.** Is a lift with a breached guardrail a win? Not for this decision, because the guardrail is part of
the launch rule. How do you know the loop is the cause? The control arm runs the old feed, so a difference in
top-listing impression share between the arms points at the new ranker.

## Common mistakes
- Naming a model before naming the decision, the metric and the guardrails.
- Treating logged clicks as unbiased relevance labels.
- Comparing a tuned candidate with an untuned baseline, or with a different budget.
- Reporting NDCG as the business result.
