# Marketplace home feed, debrief

## Reframe

Which listings should each buyer see in the 40 feed slots, so that more sessions end in a purchase, the
marketplace keeps its seller mix, and the feed renders within 150 milliseconds? Success is purchases per
session against a holdout, with guardrails on new-seller share and latency. NDCG and recall measure model
quality. They do not measure whether the feed is worth launching.

## Ask

Ask what a purchase earns and what the ranker costs to run; how many buyers are new, and what the feed shows
them; whether any traffic is randomised, so exposure can be estimated; which sellers the business must
protect; and what a holdout can measure within a week. The answers set the value of a lift, the cold-start
path and the guardrails. Without randomised traffic, the candidate has to say how offline numbers will be
trusted, and what evidence would change that.

## Structure

Two stages. Candidate generation draws on several sources (category popularity, item-to-item co-views,
attribute similarity for new listings, and a seller quota) to produce about 500 candidates per buyer. Judge it
by recall@500 of later purchases. The ranking stage scores each candidate with buyer, item and context
features, trained on graded implicit labels and corrected for exposure with the randomised slice. Judge it by
NDCG@40 on those corrected labels, against the popularity baseline, with equal tuning budgets and five seeds.
Re-ranking enforces seller share and diversity. Launch by A/B test, in weekly blocks, with guardrails on
new-seller share, p95 latency and return rate (the share of purchases returned within 30 days). Plot the candidates
on a Pareto view of purchase lift, catalogue coverage (the share of active listings that sell at least once in a
week) and latency. No single model wins all three, so choose the front point that clears the guardrails.

## Synthesize

Model quality and decision value are different numbers, and a model can rank better and still not pay. Here a
3 percent relative lift adds about 1,000 purchases a week, roughly 240,000 dollars a year in commission. Set
against a run cost of about 350,000 dollars a year, the lift does not pay on its own: the break-even lift is about
4.4 percent. The case then needs a larger lift, a cheaper serving path such as precomputed candidates, or retention
evidence the logs do not show. The twist changes the call: a lift with a breached new-seller guardrail is not a
launch. New-seller share is 7 percent against the 9 percent floor, and the top 1 percent of listings take 41 percent
of impressions. Hold the ranker at 10 percent of traffic, retrain on exposure-corrected labels with a diversity constraint, and repeat the
equal-budget comparison before widening.
