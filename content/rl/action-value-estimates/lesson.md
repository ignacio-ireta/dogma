---
title: "Action-value estimates"
minutes: 12
min_viable: "Read The system and Worked example, then do p01 with Hint 1 if needed."
---

## The system

In a bandit problem there is one action set and no state transitions. Each action $a$ has an
unknown mean reward $q_*(a)$. The agent keeps an estimate $Q_t(a)$ of that mean, and it updates
the estimate only for the action it took at step $t$. Let $N_t(a)$ be the number of times $a$ has
been taken up to and including step $t$. Let $R_t$ be the reward observed for the action taken at
step $t$.

The sample-average estimate is the mean of the rewards seen so far for $a$. It can be written as
an incremental update with step size $\alpha_n = 1/n$:

$$Q_{n+1} = Q_n + \frac{1}{n}\,(r_n - Q_n).$$

Here $n$ counts the current reward, $r_n$ is the $n$th reward for this action, and $Q_n$ is the
estimate before that reward arrived. The general form for any step size is

$$\text{NewEstimate} \leftarrow \text{OldEstimate} + \alpha\,(\text{Target} - \text{OldEstimate}).$$

With a constant step size $\alpha\in(0,1]$ the update becomes
$Q\leftarrow Q+\alpha(r-Q)$. The estimate then tracks recent rewards instead of averaging all
of them.

## Worked example

Take one action with rewards $1$, $0$, $3$ in that order, and initial value $Q_0=0$. With $1/n$:

| Step | Reward $r$ | Count $n$ | Update | $Q$ after |
|---|---|---|---|---|
| 1 | 1 | 1 | $0+\tfrac11(1-0)$ | 1.000 |
| 2 | 0 | 2 | $1+\tfrac12(0-1)$ | 0.500 |
| 3 | 3 | 3 | $0.5+\tfrac13(3-0.5)$ | 1.333 |

The final value $4/3$ equals the batch mean $(1+0+3)/3$. Now run the same rewards with a
constant $\alpha=0.5$ and $Q_0=0$. The estimate is $0.5$, then $0.25$, then $1.625$. The last
reward carries weight $0.5$, the one before it $0.25$, and so on. The weights are exponential,
and they sum to less than one, which is why the initial value keeps some influence.

## Why it works

Expand the batch mean of $n$ rewards. Suppose $\bar r_{n}=\frac1n\sum_{i=1}^n r_i$. Then
$\bar r_{n} = \bar r_{n-1} + \frac1n(r_n-\bar r_{n-1})$ by direct algebra. So the step size
$1/n$ gives the sample mean exactly, with no stored history [book: Sutton and Barto, ch. 2].

Expand the constant-step update the same way. After $n$ updates from $Q_0$,

$$Q_n = (1-\alpha)^n Q_0 + \sum_{i=1}^{n}\alpha(1-\alpha)^{n-i}\,r_i.$$

The reward $r_i$ receives weight $\alpha(1-\alpha)^{n-i}$, which decays geometrically with its age.
The weights on the rewards plus the weight on $Q_0$ sum to one, so each estimate is a weighted
average. The step-size conditions for stochastic approximation require the step sizes to sum to
infinity and their squares to sum to a finite value
[library: Mathematics/11-Optimization-Theory/Robbins-Monro - Stochastic Approximation 1951.md].
A constant step size breaks the second condition on purpose. It buys tracking at the cost of
never settling.

## Signal vs. noise

The signal is the update form and the step size. The arithmetic is bookkeeping and easy to check
against a batch mean. Ignore the choice of optimistic values for now, unless the problem sets
them explicitly. The most common mistake is updating every action on every step. Only the chosen
action's estimate should change.

## Where it breaks

If the count includes the wrong reward, the estimate is off by a factor that depends on $n$.
Pick one convention, $n$ includes the current reward, and check it on a two-sample example.
A constant step size of $0$ never learns, and a step size above $1$ overshoots. An optimistic
initial value that sits below the true means slows exploration instead of helping it. Finally,
a sample average over a drifting problem averages across regimes that no longer exist.

## Interview angle

State the update in one line: $Q\leftarrow Q+\alpha(r-Q)$. Then say what $\alpha$ is. For the
sample average $\alpha=1/n$ and the estimate equals the running mean. For a constant $\alpha$ the
estimate is an exponential moving average and it tracks change. A likely follow-up asks why use a
constant step size. The answer is that the reward distribution drifts. The second follow-up
asks what optimistic initial values do. They encourage early exploration, and the bias fades once
each action has been tried. The full answers are in `interview.md`.

## Connections

This topic builds on the expectation and variance of samples and on the MDP setting from the
previous topic. It is the estimator the next topic uses to choose actions. Epsilon-greedy exploration
uses these estimates, and the upper-confidence and gradient-bandit methods replace them with other
rules. The same incremental form, with a bootstrapped target, appears in temporal-difference
learning [book: Sutton and Barto, ch. 6].
