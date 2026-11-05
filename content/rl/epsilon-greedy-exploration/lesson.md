---
title: "Epsilon-greedy exploration"
minutes: 12
min_viable: "Read The system and Worked example, then do p01 with Hint 1 if needed."
---

## The system

An agent with action-value estimates $Q(a)$, one per action $a\in\{0,\dots,A-1\}$, must choose an
action each step. Exploitation means taking an action with the largest estimate. Exploration means
taking a different one to learn more about it. Epsilon-greedy resolves the trade-off with one
parameter $\varepsilon\in[0,1]$:

- With probability $\varepsilon$, pick one of the $A$ actions uniformly at random. This includes
  the greedy action.
- With probability $1-\varepsilon$, pick a greedy action, one with the largest $Q(a)$.

If the greedy action is unique, its selection probability is $1-\varepsilon+\varepsilon/A$. Every
other action has probability $\varepsilon/A$. Ties among greedy actions need a rule. This course
uses the lowest index, which is deterministic and needs no random draw. Under that rule the lowest
tied index gets the whole exploit share, so every other tied action gets only $\varepsilon/A$.

For reproducible code the draws must be ordered. Here one uniform $u\sim\mathrm{U}[0,1)$ decides
explore or exploit. Explore if $u<\varepsilon$. Then one integer, uniform on $\{0,\dots,A-1\}$,
gives the action. Exploitation uses no further draw.

## Worked example

Take $Q=(0.2,\,0.9,\,0.9,\,0.4)$ with $A=4$, and $\varepsilon=0.2$. The greedy set is $\{1,2\}$.
Exploitation returns action 1, the lowest index among ties. The probabilities over the four
actions are:

| Action | Probability | Why |
|---|---|---|
| 0 | $0.2/4=0.05$ | explored, not greedy |
| 1 | $0.8+0.2/4=0.85$ | lowest-index greedy action: exploit share plus exploration |
| 2 | $0.2/4=0.05$ | tied greedy action, but not the lowest index: exploration only |
| 3 | $0.05$ | explored, not greedy |

The four numbers sum to 1. Under the lowest-index rule, action 1 is returned whenever the call
exploits, so action 2 has the same tie value as action 1 but gets no exploit share.

## Why it works

Each action is sampled with positive probability while $\varepsilon>0$, so every estimate keeps
receiving data. With sample-average estimates and enough samples, the greedy choice converges to
the best action, and the exploring draws keep the others in the running
[book: Sutton and Barto, ch. 2.2]. The cost is that a fixed $\varepsilon$ spends a fraction of
every step on random actions. If the action gaps are $\Delta_a=q_*(a^*)-q_*(a)$, the expected
loss per step is $\varepsilon\,\bar\Delta$ once the greedy choice is correct. Here $\bar\Delta$ is
the average gap over the $A$ actions, because the explore draw is uniform. The cumulative loss, or
regret, is therefore linear in the number of steps.

A decaying schedule, for example $\varepsilon_t=\min(\varepsilon_0,\,c/t)$, makes the exploration
share fall over time, so the per-step cost shrinks as the estimates improve. Whether a $1/t$
schedule gives sub-linear regret is a result from the bandit literature that this course does not
prove.

## Signal vs. noise

The signal is the probability table above, and the draw order that makes the table reproducible.
Ignore the choice of a decay schedule until you have a fixed-epsilon baseline. The distribution is
simple enough to check by frequency, so do that rather than trusting one run. The lowest-index
tie-break is a convention, not a discovery. Stating it removes a source of confusion when two
implementations disagree.

## Where it breaks

Taking the uniform draw as "explore among the non-greedy actions" is the most common error. That
policy has probabilities that do not match the table above, and it never picks the greedy action
while exploring. Drawing on every call, even when exploiting, changes the random sequence whenever
the branch changes. Then two runs with the same seed diverge as soon as their decisions differ.
Also check the boundaries: epsilon $0$ never explores, epsilon $1$ always explores, and both must
still consume exactly the draws in the contract.

## Interview angle

Start with the rule: with probability $\varepsilon$ take a uniform action, otherwise take a
greedy one. Then say what it costs. A fixed $\varepsilon$ gives linear regret, so you decay it or
replace it with an upper-confidence rule. Expect two follow-ups. How do you make it reproducible?
Fix the draw order and the tie-break. How do you test it? Compare the selection frequencies with
the table, using a tolerance that matches the sampling error. The answer is in `interview.md`.

## Connections

This topic uses the action-value estimates from the previous one. It is the exploration rule the
Q-learning topic uses in its training loop. The upper-confidence-bound and Thompson-sampling
topics replace the random draw with an uncertainty-aware choice, and the gradient-bandit topic
changes the estimates themselves [book: Sutton and Barto, ch. 2].
