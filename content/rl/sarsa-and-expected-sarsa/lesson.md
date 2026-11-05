---
title: "Sarsa and expected Sarsa"
minutes: 12
min_viable: "Read The system and Worked example, then do p01 with Hint 1 if needed."
---

## The system

Sarsa and expected Sarsa are tabular TD control methods. Both store $Q(s,a)$ in a table of shape
$|\mathcal S|\times|\mathcal A|$ and change one entry per transition $(s,a,r,s')$. Let $a'$ be the
action the agent takes next in $s'$. The update is

$$Q(s,a)\leftarrow Q(s,a)+\alpha\,(y-Q(s,a)),$$

where $\alpha\in[0,1]$ is the step size, $\gamma\in[0,1]$ the discount, and the target $y$ depends on
the method:

| Method | Target $y$ | Next-state term |
|---|---|---|
| Sarsa | $r+\gamma\,Q(s',a')$ | the action actually taken |
| Expected Sarsa | $r+\gamma\sum_{b}\pi(b\mid s')\,Q(s',b)$ | the average under $\pi$ |
| Q-learning | $r+\gamma\max_{b}Q(s',b)$ | the greedy action |

If $s'$ is terminal, $y=r$ for all three. In code, $s'$ is `None`.

The policy $\pi$ is $\varepsilon$-greedy over $Q(s',\cdot)$. Let $G$ be the set of actions with the
largest value and $k=|G|$. Each action gets $\varepsilon/|\mathcal A|$, and the exploit share
$1-\varepsilon$ is split evenly over $G$:

$$\pi(b\mid s')=\frac{\varepsilon}{|\mathcal A|}+\frac{(1-\varepsilon)\,[\,b\in G\,]}{k}.$$

Ties are exact floating-point equality. The epsilon-greedy topic breaks ties by lowest index when it
picks an action. The expected target does not depend on that choice, as shown below.

## Worked example

Take $\alpha=0.5$, $\gamma=0.5$, $r=1$, $\varepsilon=0.2$, and $|\mathcal A|=3$. The updated entry starts
at $Q(s,a)=0$. The next state has $Q(s',\cdot)=(2,\,4,\,1)$.

- The greedy set is $G=\{1\}$, so $\pi=(\tfrac{1}{15},\,\tfrac{13}{15},\,\tfrac{1}{15})$, which sums to 1.
- The expected next value is $\tfrac{1}{15}\cdot2+\tfrac{13}{15}\cdot4+\tfrac{1}{15}\cdot1=\tfrac{11}{3}$.

The new $Q(s,a)$ is $0.5\,y$:

| Method | Next action | Target $y$ | New $Q(s,a)$ |
|---|---|---|---|
| Sarsa | $a'=0$ | $1+0.5\cdot2=2$ | $1$ |
| Sarsa | $a'=1$ | $1+0.5\cdot4=3$ | $1.5$ |
| Sarsa | $a'=2$ | $1+0.5\cdot1=1.5$ | $0.75$ |
| Expected Sarsa | — | $1+0.5\cdot\tfrac{11}{3}=\tfrac{17}{6}\approx2.83$ | $\approx1.42$ |
| Q-learning | — | $1+0.5\cdot4=3$ | $1.5$ |

The Sarsa targets change with the action drawn. Their average under $\pi$ is
$\tfrac{1}{15}\cdot2+\tfrac{13}{15}\cdot3+\tfrac{1}{15}\cdot1.5=\tfrac{17}{6}$, which is the expected
target. Q-learning takes the max, so its target is the largest of the three.

## Why it works

Sarsa is a sampled version of the Bellman expectation equation for the policy it follows. For a fixed
policy $\pi$, $q_\pi(s,a)=\mathbb E[r+\gamma\,q_\pi(s',A')]$ with $A'\sim\pi(\cdot\mid s')$
[book: Sutton & Barto, temporal-difference chapter]. Sarsa replaces the expectation with one draw of
$A'$. Expected Sarsa computes the expectation. Both are on-policy, because the values refer to the
policy that chooses $a'$. Q-learning's max refers to the greedy policy whatever the agent does.

**Variance.** Condition on $\mathcal F=\sigma(s,a,r,s')$. The Sarsa target $Y_S=r+\gamma Q(s',A')$ has
conditional mean $Y_E$, the expected target, and conditional variance $\gamma^2\,\mathrm{Var}_{A'}(Q(s',A'))$.
The law of total variance gives

$$\mathrm{Var}(Y_S)=\mathrm{Var}(Y_E)+\gamma^2\,\mathbb E\!\left[\mathrm{Var}_{A'}\big(Q(s',A')\big)\right]\ \ge\ \mathrm{Var}(Y_E).$$

Expected Sarsa removes the draw of $A'$, and with it that variance. The cost is one sum over the actions
per update.

**Max bound.** A weighted average is at most the maximum, so for the same table
$\sum_b\pi(b\mid s')\,Q(s',b)\le\max_b Q(s',b)$. Expected Sarsa's target never exceeds Q-learning's.

**Tie invariance.** Tied actions have equal values. The expected target then depends only on the total
mass on $G$, which is $(1-\varepsilon)+\varepsilon k/|\mathcal A|$ for every split. The split changes
which action the sampler favours, not the target.

**Convergence.** On-policy control needs a policy that becomes greedy in the limit while every pair
keeps being visited, plus Robbins-Monro step sizes [library:Mathematics/11-Optimization-Theory/Robbins-Monro - Stochastic Approximation 1951.md]. The proof is not in this
repository, so treat these as conditions, not a derivation.

## Signal vs. noise

**Signal:** the three targets, the next-state term, the $\varepsilon$-greedy weights, and the terminal
rule. Two reductions are worth remembering. At $\varepsilon=0$, expected Sarsa is Q-learning. At
$\varepsilon=1$, it averages $Q(s',\cdot)$ uniformly.

**Noise:** the tie split (no effect on the target) and the shared schedules for $\alpha$ and $\varepsilon$.

## Where it breaks

- **Resampling $a'$.** Sarsa's $a'$ must be the action the agent executes at $s'$ in the next step.
  Drawing a second action makes the target describe a policy the agent does not follow.
- **Terminal next state.** With $s'$ terminal there is no $a'$. Indexing `Q[None, a']` is a bug.
  The target is $r$.
- **Unnormalised weights.** The weights must sum to 1. Dropping the $\varepsilon/|\mathcal A|$ share on
  greedy actions gives weights summing to $14/15$ in the worked example.
- **Near ties.** Ties are exact equality, so rounding noise can change the greedy set.
- **Wrong policy in the expectation.** Expected Sarsa must average under the policy that generates the
  actions. If the behaviour is softmax, the $\varepsilon$-greedy weights above are wrong.

## Interview angle

Sixty-second answer: Sarsa bootstraps from the next action it actually takes, so it learns the value of
its own behaviour, exploration included. Q-learning bootstraps from the max. Expected Sarsa averages
the next action over the policy, which removes the variance of the next-action draw, and it reduces to
Q-learning at $\varepsilon=0$. Follow-up: when are Sarsa and Q-learning equal? When the next action is
greedy, which at $\varepsilon=0$ holds in every step up to ties. Second follow-up: which is safer near risky
states? Sarsa, by the usual argument that its values include the cost of exploratory actions.

## Connections

- Prerequisites: [Q-learning](../q-learning/) gives the update shape and the max target.
  [Epsilon-greedy exploration](../epsilon-greedy-exploration/) gives the policy whose weights the
  expectation uses.
- Unlocks: [Expected vs sample updates](../expected-vs-sample-updates/),
  [n-step TD and n-step Sarsa](../n-step-td-and-n-step-sarsa/), and
  [Semi-gradient Sarsa control](../semi-gradient-sarsa-control/).
- Sources: the repository has no Sutton & Barto text. The update rules here were checked numerically
  against hand-computed fixtures and exact-rational arithmetic, not against the book's wording. The
  convergence statement uses the Robbins-Monro conditions in the library note.
