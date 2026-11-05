---
title: Recommendation system
minutes: 12
min_viable: "Read The system and Where it breaks, then the brief and the twist."
---

## The system

A recommender decides which items fill the feed slots for one buyer, to maximise expected purchase value
while respecting guardrails. Serving splits the work into stages, and each stage has its own metric:

| Stage | Job | Output | Judged by |
|---|---|---|---|
| Candidate generation | cut 900,000 listings to about 500 per buyer, from several sources | candidate set | recall@500 of items the buyer later buys |
| Ranking | score each candidate: predicted value from buyer, item and context features | ordered list | NDCG@40 on exposure-corrected labels |
| Re-ranking | enforce constraints: seller share, diversity, freshness | the 40 slots | guardrail metrics |

Labels come from implicit feedback, graded: purchase above cart, cart above save, save above click, and
impression with no click at the bottom. Star ratings are too rare to use.

Two properties of the logs decide most designs:

1. **Exposure.** A buyer can only click what was shown, and top slots are seen far more often. A logged
   click is the product of an item's appeal and the attention its slot receives.
2. **Cold start.** New buyers and new listings have no history, so the candidate sources must work without it.

## Worked example

Weekly value of the feed, from the brief's numbers:

- Purchases: 1,600,000 sessions $\times$ 0.021 = 33,600 per week.
- Revenue per purchase: 38 dollars of sales $\times$ 0.12 commission = 4.56 dollars.
- Weekly revenue: 33,600 $\times$ 4.56 = 153,216 dollars.
- A 3 percent relative lift adds about 1,008 purchases, or about 4,600 dollars a week, or about 240,000 dollars a year.

Exposure correction, from the randomised slice. At slot 1 the shuffled click rate is 4.2 percent, and at
slot 4 it is 2.6 percent. Their ratio, 0.62, is the examination factor for slot 4. The organic slot-4 click
rate is 3.1 percent, so the corrected appeal is $3.1 / 0.62 = 5.0$ percent, against 6.8 percent at slot 1.
The raw gap is 2.2 times; the corrected gap is 1.4 times. About half of the apparent drop is position.

## Why it works

Shuffling the top ten items breaks the link between appeal and slot. Clicks by slot in that slice therefore
measure attention, scaled by the average appeal of the set. Dividing organic clicks by that attention curve
estimates appeal. This is inverse propensity weighting applied to position (Joachims et al. 2017).

The offline metrics answer different questions. Recall@500 asks whether generation can see the item. NDCG
asks whether the ranker orders what it sees (Järvelin and Kekäläinen 2002). Both are computed on logs
written by the old feed, so an item the old feed never showed has no label and counts as irrelevant. A model
that copies the old feed scores well offline and can still be flat online.

Compare models fairly. Give each the same tuning budget (for example 30 trials), the same training window and
features, and five seeds, then report the mean and an interval rather than the best run. A model that wins
at 30 trials and loses at 10 has won on budget, not on design.

## Signal vs. noise

**Signal:** slot exposure, the cold-start path, the guardrails (new-seller share, p95 latency) and purchase
value per session.

**Noise:** the ranker's architecture before the candidate sources and labels are settled, and tenths of a
point of NDCG on one week of logs.

## Where it breaks

- **Popularity feedback loop.** A model trained on its own exposures sends more impressions to items that
  already sell, which gives those items more labels. Watch the impression share of the top 1 percent of
  listings, and the number of listings that sell at all.
- **Offline win, flat online.** Gains on logs can vanish in an A/B test when they come from items the old feed
  already showed.
- **Latency.** Scoring 500 candidates with a heavy model misses a 150-millisecond feed budget. Move the cost
  into generation, or precompute it.
- **No single winner.** Engagement, catalogue coverage and latency seldom peak together. Keep the
  non-dominated set (Deb et al. 2002) and choose by guardrail, not by one blended score.

## Interview angle

Sixty-second shape: "Two stages. Generation gives about 500 candidates from several sources, and a ranker
orders them. I judge generation on recall, the ranker on exposure-corrected NDCG, and the launch on an A/B
test with seller-share and latency guardrails. The question is whether the lift pays for serving." Expect the
follow-ups "how do you rank a new listing with no history?" and "why was the offline win flat?"

## Connections

Builds on [cml/ranking-metrics](../../cml/ranking-metrics/) and
[cml/a-b-test-design-and-analysis](../../cml/a-b-test-design-and-analysis/). The exposure problem is a
selection-bias problem, like the transactions the [fraud case](../fraud-detection-case/) approves without review. The
[RAG case](../rag-system-design-case/) has the same two-stage shape, with retrieval in place of the generator.
