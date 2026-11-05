---
title: "Q-learning"
minutes: 14
min_viable: "Read The system and Worked example, then do p01 with Hint 1 if needed."
---

## The system

Tabular Q-learning estimates the optimal action values $Q^*(s,a)$ for a finite MDP. Store one number
per state-action pair in a table $Q$ of shape $|\mathcal{S}|\times|\mathcal{A}|$. After each
transition $(s,a,r,s')$ with a flag $d$ for terminal, update one entry:

$$Q(s,a)\leftarrow Q(s,a)+\alpha\Big(\underbrace{r+\gamma\,(1-d)\max_{a'}Q(s',a')}_{\text{target}}-Q(s,a)\Big).$$

Here $\alpha\in(0,1]$ is the step size, $\gamma\in[0,1]$ the discount, and $d\in\{0,1\}$ is one
when $s'$ is terminal. The target uses the greedy value of the next state, whatever action comes
next. So the update does not depend on the behaviour policy that chose $a$.

The behaviour can be $\varepsilon$-greedy, as in the previous topic, which keeps every action
under exploration. The target policy is greedy with respect to $Q$.

## Worked example

Take $\alpha=0.5$, $\gamma=0.9$. The table has $Q(s,a)=0$ except $Q(s',\cdot)=(0.2,\,1.0,\,0.4)$
for three actions at $s'$. Two transitions:

| Case | Target | New $Q(s,a)$ |
|---|---|---|
| Non-terminal, $r=0$ | $0+0.9\cdot\max(0.2,1.0,0.4)=0.9$ | $0+0.5(0.9-0)=0.45$ |
| Terminal, $r=1$ | $1$ (no bootstrap) | $0+0.5(1-0)=0.5$ |

The second row ignores the table at $s'$ entirely. Even if $Q(s',\cdot)$ were $10^6$, a terminal
update would not change. This is the bootstrap rule to remember.

## Why it works

Q-learning is temporal-difference control with the Bellman optimality equation as its target
[book: Sutton and Barto, ch. 6]. Its fixed point satisfies $Q^*(s,a)=\mathbb{E}[r+\gamma\max_{a'}Q^*(s',a')]$.
Each update moves $Q$ a fraction $\alpha$ of the way toward a sample of that right-hand side. When
the step sizes meet the Robbins-Monro conditions, the averaging washes out the noise
[library: Mathematics/11-Optimization-Theory/Robbins-Monro - Stochastic Approximation 1951.md].
Those conditions require the step sizes to sum to infinity and their squares to sum to a finite
value. The problem uses a constant $\alpha=0.5$, which breaks the second condition. In this
deterministic gridworld it still reaches the optimum, because each visit halves the remaining error
in that entry. With noisy rewards a constant step size would keep fluctuating. Tabular convergence additionally needs every state-action pair to be visited infinitely often.
The conditions are sufficient results from the stochastic-approximation literature. Read them as
guarantees about the limit, not as a prediction about a finite run.

In a deterministic MDP the optimum has a closed form. Take a gridworld where moves are certain and
reward is $1$ on entering a goal, $0$ otherwise. Let $d(s)$ be the number of steps from $s$ to the goal.
Then $Q^*(s,a)=\gamma^{d(s')}$, where $s'$ is the state the action leads to and $d(\text{goal})=0$.

## Signal vs. noise

The signal is the one-line target and the terminal flag. The max is the single change that
distinguishes Q-learning from SARSA, and the test suite depends on it. The rest is bookkeeping: the
index arithmetic and the episode loop. Ignore function approximation and experience replay for
now. They change the stability story, and the tabular case is the one interviewers expect you to
derive.

## Where it breaks

Three bugs appear again and again. The first bootstraps through a terminal state, which overvalues
the goal's neighbours. The second uses the next action the behaviour policy took, which is SARSA
under a different name. The third never visits some pairs, so their values stay wrong even when
the aggregate curve looks smooth. Check the visitation counts. Also remember that the update is
only as good as the values at $s'$. A too-large step size makes the table oscillate.

## Interview angle

Give the update first, then say what it is off-policy: the max evaluates the greedy policy, while
the behaviour can explore. Expect two follow-ups. How does SARSA differ? SARSA uses the action the
behaviour takes next. What does convergence need? Robbins-Monro step sizes and infinite visits to
every pair. The 60-second version and the two follow-ups are in `interview.md`.

## Connections

Q-learning builds on the epsilon-greedy exploration rule, and on the discounted-return definitions
of the first topic. It is the tabular core of the deep Q-network line, where a neural network
replaces the table and a replay buffer and target network stabilise the target. The next topic,
`rl/sarsa-and-expected-sarsa`, is the on-policy counterpart [book: Sutton and Barto, ch. 6].
