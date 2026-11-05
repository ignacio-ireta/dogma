---
title: RASS case framework
minutes: 12
min_viable: "Read The system and Synthesize in Where it breaks, then do the mini-case brief."
---

## The system

RASS is four moves, said out loud in order: **Reframe**, **Ask**, **Structure**, **Synthesize**.
Each move has a job and a time budget for a 25-minute case.

| Move | Job | Budget |
|---|---|---|
| Reframe | Restate the problem, quantify it, turn it into one question with a success metric | 1–2 min |
| Ask | Get the facts that change the design: the decision, the costs, the data | 3–5 min |
| Structure | Tell the modelling story: problem type, features, model with reasons, evaluation, deployment | 10–15 min |
| Synthesize | Land on business impact and a pilot that tests it | 2–3 min |

Define the decision before the model. The Reframe names the decision, the metric follows from the
cost of each error, and the model is the last choice, not the first.

## Worked example

A bike-share operator wants to know which stations will be empty in the next two hours.

- **Reframe:** "One station in six is empty at any time in the evening peak. The question is which
  stations a truck should visit next to avoid empty hours. Success is lost trips avoided per truck-hour."
- **Ask:** the truck count and reach, the cost of an empty station-hour, and the forecast horizon.
- **Structure:** forecast each station's empty probability for two hours ahead, rank by expected
  lost trips, assign the top stations to trucks.
- **Synthesize:** avoiding a quarter of the empty station-hours is worth about 1,500 trips an hour.

## Why it works

The four moves follow the order in which a senior person makes decisions: frame, check the
constraints, choose a method, then justify the value. The order also protects you under pressure.
If you lose the thread, the next move tells you what to say next.

A metric comes from a cost. If a false alarm costs far less than a miss, you tune for recall; if
the review queue is the bottleneck, you tune for precision at the queue size. Asking about cost is
how the metric is found, not a formality.

## Signal vs. noise

**Signal:** the decision, the cost of each error type, the data that exists today, and who acts on
the output. These four facts decide most designs.

**Noise:** a long list of model names, and the full tour of hyperparameters. Name one baseline, one
candidate, and the reason you would choose between them.

## Where it breaks

- **Reframe that is only a restatement.** "So you want a churn model" adds nothing. Add the number
  and the success metric.
- **Ask that never stops.** Ask five questions, state the uncertainty, and move on.
- **Structure as a list of steps.** Tell a story with a reason at each step.
- **Synthesize that quotes accuracy.** Convert the model's effect into trips, customers or money.

## Interview angle

Sixty-second shape: "The decision is X. The metric is Y, because the Z error costs more. I would
start with a baseline W, add the model V, and pilot it on a holdout before scaling." Expect two
follow-ups: "why not a simpler model?" and "how do you know the forecast is good enough to act on?"

## Connections

Builds on the confusion matrix and count-based scores, and on thresholds. It is the frame for the
churn, fraud and other `sys` cases, where the same four moves run on a full-length brief.
