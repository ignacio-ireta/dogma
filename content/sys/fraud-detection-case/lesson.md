---
title: Fraud detection system
minutes: 10
min_viable: "Read The system and Where it breaks, then the brief and the twist."
---

## The system

Fraud detection is a two-stage decision under a latency budget and a review budget:

1. **Fast filter.** Rules and cheap features decide the obvious cases in well under the latency limit.
2. **Scoring.** A model scores the borderline cases, and only the top of its ranking reaches a human.

The threshold is not free. It is set so the number of alerts fits the review team's daily capacity.

## Worked example

With 50 million transactions a month and a fraud rate of 0.05 percent, there are about 25,000 frauds
a month. If the review team can handle 30,000 cases a month, the threshold must produce no more than
that many alerts, whatever the model's score distribution looks like.

## Why it works

With a rare positive class, accuracy is useless: a model that approves everything is 99.95 percent
accurate. Precision at the review budget and recall at a fixed false-alarm rate measure what the bank
cares about. Class weights or cost-sensitive loss push the model toward the rare class, and the threshold
then fixes the alert volume.

## Signal vs. noise

**Signal:** velocity (several transactions in five minutes), a spend far from the account's pattern,
a new device, a risky merchant category, and unusual timing. These are fast to compute and explainable.

**Noise:** exotic features that add latency without moving recall, and resampling tricks that inflate
training metrics without changing the alert volume.

## Where it breaks

- **Latency.** A model that needs a slow join misses the 200-millisecond decision window. Keep the
  first stage cheap.
- **Explanations.** A score with no reason cannot be shown to a customer or a regulator. Per-decision
  attributions, or an interpretable model, are a requirement, not a nicety.
- **Feedback loops.** Investigators only see flagged cases, so the training labels for unflagged
  transactions are mostly unknown. Say how you would sample to learn about them.

## Interview angle

Sixty-second shape: "Two stages: a fast rule filter, then a scored model for borderline cases. The
threshold is set for the review queue, and the metric is recall at that queue size. Every decline
carries its top reasons." Expect the follow-ups "how do you handle the imbalance?" and "what about
transactions you never review?"

## Connections

Builds on precision and recall, thresholds, and imbalanced classification. Compare the churn case,
where the cost of each error type is much closer.
