---
title: Churn prediction system
minutes: 10
min_viable: "Read The system and Where it breaks, then the brief and the twist."
---

## The system

Churn is a decision problem with a budget. The model ranks subscribers by the chance they leave
next month; the business spends a limited retention budget on the top of that ranking. Three
things follow from that:

1. **Segment first.** Prepaid and postpaid subscribers behave differently, so each gets its own
   signal and often its own threshold.
2. **Target by value times risk.** A high-risk, low-value subscriber is not worth a costly offer.
3. **Measure saves, not accuracy.** The result is customers kept per dollar of offers.

## Worked example

A segment has 12 million subscribers and a monthly churn rate of 3.4 percent. That is about 408,000
departures a month. Reducing the rate to 2.5 percent avoids about 108,000 of them. At an acquisition
cost of about 120 dollars each, that is roughly 13 million dollars of avoided replacement cost per month.

## Why it works

Tabular subscriber data with a rare outcome suits gradient-boosted trees: they handle mixed features
and missing values with little preprocessing. Logistic regression is the honest baseline, and it should
be compared before the boosted model is defended. Recall is tuned against the offer budget, because a
threshold that flags too many subscribers exhausts the retention team.

## Signal vs. noise

**Signal:** usage trend over three months (a steady decline is a strong warning), changes in payment
behaviour, and spikes in support contacts. These are the features the business can act on.

**Noise:** demographic detail that the retention team cannot use, and a long feature list added to
move the metric by a hundredth of a point.

## Where it breaks

- **Blanket discounts.** A discount sent to everyone spends money on loyal subscribers and can teach
  price-sensitive customers to wait for one. An untargeted campaign can raise churn, as it did here.
- **Leakage from the retention process.** Subscribers who already called to cancel are not a fair
  training population for a predictor of early warning.
- **Imbalance.** At a 3.4 percent rate, accuracy is about 96 percent for a model that never flags anyone.

## Interview angle

Sixty-second shape: "Churn is a targeting problem. I segment by plan, rank by risk with a boosted
model against a logistic baseline, tune the threshold to the offer budget, and tier by lifetime
value. Success is saves per dollar, measured against a holdout." Expect the follow-ups "why did the
blanket discount fail?" and "how do you set the threshold?"

## Connections

Uses the confusion matrix, precision and recall, and thresholds. It shares the decision framing
with the fraud case, where the cost of each error type is far more lopsided.
