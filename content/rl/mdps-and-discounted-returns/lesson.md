---
title: "MDPs and discounted returns"
minutes: 12
min_viable: "Read The system and Worked example, then do p01."
---

## The system

A finite Markov decision process has five parts. $\mathcal{S}$ is the finite set of states,
$\mathcal{A}$ the finite set of actions, $P(s'\mid s,a)$ the probability of moving to state
$s'$ after action $a$ in state $s$, $R(s,a)$ the expected reward for taking $a$ in $s$, and
$\gamma\in[0,1]$ the discount factor. A policy $\pi(a\mid s)$ is the probability of choosing
action $a$ in state $s$. Its shape is $|\mathcal{S}|\times|\mathcal{A}|$ and every row sums to one.

The reward received after the action at step $t$ is $r_{t+1}$. The return from step $t$ is

$$G_t = \sum_{k=0}^{\infty} \gamma^k r_{t+k+1}.$$

The state value under policy $\pi$ is $V^\pi(s)=\mathbb{E}_\pi[G_t\mid S_t=s]$, and the
Bellman expectation equation expresses it through one step of lookahead:

$$V^\pi(s)=\sum_a \pi(a\mid s)\Big[R(s,a)+\gamma\sum_{s'}P(s'\mid s,a)\,V^\pi(s')\Big].$$

Episodic tasks stop at a terminal state, so $G_t$ is a finite sum and $\gamma=1$ is allowed.
Continuing tasks have no terminal state. There $\gamma<1$ keeps the sum finite.

## Worked example

Take three rewards from one episode, $r_1=1$, $r_2=0$, $r_3=2$, and $\gamma=0.5$. Compute the
returns from the end backwards, using $G_t=r_{t+1}+\gamma G_{t+1}$ with $G_3=0$:

| Step $t$ | Reward $r_{t+1}$ | Recurrence | $G_t$ |
|---|---|---|---|
| 2 | 2 | $2+0.5\cdot 0$ | 2.0 |
| 1 | 0 | $0+0.5\cdot 2$ | 1.0 |
| 0 | 1 | $1+0.5\cdot 1$ | 1.5 |

Now a two-state policy evaluation. State 0 moves to state 1 with reward 1. State 1 moves to
state 0 with reward 0. The policy has one action, so $\pi$ is all ones. Write the two equations:
$V_0=1+0.5V_1$ and $V_1=0.5V_0$. Substituting gives $V_0=4/3$ and $V_1=2/3$. Iterating from
zero gives the same numbers in the limit.

## Why it works

Unrolling the return once gives the recursion $G_t = r_{t+1} + \gamma G_{t+1}$. Taking the
expectation under the policy and the transition kernel gives the Bellman equation above
[book: Sutton and Barto, ch. 3]. Write $P_\pi(s,s')=\sum_a\pi(a\mid s)P(s'\mid s,a)$ and
$R_\pi(s)=\sum_a\pi(a\mid s)R(s,a)$. Then the equation is the linear system
$V=R_\pi+\gamma P_\pi V$, with $V\in\mathbb{R}^{|\mathcal{S}|}$. Its unique solution is
$V=(I-\gamma P_\pi)^{-1}R_\pi$ when $\gamma<1$.

The update $V\leftarrow R_\pi+\gamma P_\pi V$ contracts in the max norm by a factor $\gamma$
[book: Sutton and Barto, ch. 4.1]. Starting from any $V_0$, the error shrinks by at least
$\gamma$ per sweep. A stopping rule on the change between sweeps bounds the error:
if the max-norm change is below $\varepsilon$, the error to the fixed point is below
$\gamma\varepsilon/(1-\gamma)$.

## Signal vs. noise

The signal is the recursion and the averaging over the policy. Most errors come from the axes.
The action average must use $\pi(a\mid s)$ with the action axis, and the next-state average must
sum over $s'$. Ignore the particular algorithm you will use later. Evaluation is the step that
every later method reuses. Discount conventions vary between books, so state yours and check
it. Here the reward $r_{t+1}$ follows the action at step $t$.

## Where it breaks

Gamma equal to one with a continuing task has no finite value when rewards are nonzero on
average: the sweeps keep growing and never meet the stopping rule. Gamma above one makes the
returns blow up, so reject it. A policy row that does not sum to one changes the
scale of every value, and the error is silent. A transition row that does not sum to one breaks
the same way. Validate both. Inputs must not be modified during evaluation. Finally, the
iteration can look converged while the solution is still far off if the tolerance is too
loose for the discount. With $\gamma=0.9$, a tolerance of $10^{-3}$ can leave an error of order
$10^{-2}$.

## Interview angle

Start with the five-part tuple, then the return, then the Bellman equation. A calm answer
says: the value of a policy is the fixed point of a $\gamma$-contraction, so it is unique and
can be computed by solving a linear system or by iterating. Expect two follow-ups. Why is
$\gamma<1$ needed? Continuing returns are infinite sums without it. What changes for an
episodic task? The sum is finite, so $\gamma=1$ is allowed. The full answer and both
follow-ups are in `interview.md`.

## Connections

This topic builds on expectations of random sums and on linear systems, the two tools behind
the Bellman equation. Next come the Bellman expectation equations in more depth, policy
iteration and value iteration, which both use this evaluation step. The lineage starts with
dynamic programming and runs to temporal-difference learning, which estimates the same values from
samples instead of a known model [book: Sutton and Barto, ch. 4 and ch. 6].
