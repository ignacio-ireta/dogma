---
title: "Speculative decoding speedup math"
minutes: 12
min_viable: "Read The system and Worked example, then solve p01 using Hint 1 if needed."
---

## The system

A small draft model proposes $\gamma$ tokens. The large target model then scores all $\gamma$ positions in one forward pass (the verification pass) and keeps a prefix of the draft. One round therefore emits between $1$ and $\gamma+1$ tokens: the accepted draft prefix, plus one token that the target supplies itself. That token is a correction at the first rejected position, or a bonus token after a full acceptance.

Symbols:

- $\gamma\ge 0$: draft length per round (an integer).
- $\alpha\in[0,1]$: per-token acceptance probability. We assume it is the same at every position and independent across positions. Call this the i.i.d. assumption.
- $c\ge 0$: cost of one draft token, relative to one target pass.
- $A\in\{0,\dots,\gamma\}$: number of accepted draft tokens in a round.
- $N=A+1$: tokens emitted in a round.

A round costs $\gamma c+1$ units: $\gamma$ draft steps plus one target pass. Plain decoding costs $1$ unit per token. The verification pass is assumed to cost the same as one ordinary target decode step, which holds when decode is memory-bound (see [inf/prefill-vs-decode-bottlenecks]).

$$\mathbb E[N]=\sum_{k=0}^{\gamma}\alpha^k=\frac{1-\alpha^{\gamma+1}}{1-\alpha}\quad(\alpha<1),\qquad \mathbb E[N]=\gamma+1\quad(\alpha=1)$$

$$S=\frac{\mathbb E[N]}{\gamma c+1}$$

Here $S$ is the walltime speedup over plain decoding. Edge cases:

| Case | $\mathbb E[N]$ | Speedup $S$ |
|---|---|---|
| $\alpha=1$ (every draft accepted) | $\gamma+1$ | $(\gamma+1)/(\gamma c+1)$ |
| $\alpha=0$ (every draft rejected) | $1$ | $1/(\gamma c+1)$, below 1 when $\gamma c>0$ |
| $\gamma=0$ (no drafting) | $1$ | $1$ for any $c$ |

## Worked example

Take $\alpha=0.8$, $\gamma=4$, $c=0.1$.

1. The accepted prefix is at least $k$ tokens long with probability $\alpha^k$: $0.8,\ 0.64,\ 0.512,\ 0.4096$ for $k=1..4$.
2. $\mathbb E[N]=1+0.8+0.64+0.512+0.4096=3.3616$. The closed form agrees: $(1-0.8^5)/0.2=3.3616$.
3. Round cost: $4\cdot 0.1+1=1.4$ units.
4. Speedup: $S=3.3616/1.4\approx 2.40$.

Same draft, other acceptance rates: $\alpha=1$ gives $\mathbb E[N]=5$ and $S\approx 3.57$. $\alpha=0$ gives $\mathbb E[N]=1$ and $S\approx 0.71$, which is slower than plain decoding. The draft is then pure overhead.

## Why it works

Let $P(A\ge k)$ be the probability that the first $k$ draft tokens are all accepted. Under the i.i.d. assumption $P(A\ge k)=\alpha^k$. Since $N=A+1$,

$$\mathbb E[N]=1+\sum_{k=1}^{\gamma}P(A\ge k)=1+\sum_{k=1}^{\gamma}\alpha^k.$$

Gains saturate: as $\gamma\to\infty$ with $\alpha<1$, $\mathbb E[N]\to 1/(1-\alpha)$, while the round cost keeps growing with $\gamma c$.

**Why the output is unchanged.** Let $p$ be the target distribution and $q$ the draft distribution at one position, both vectors over the vocabulary. The draft token $x\sim q$ is accepted with probability $\min(1,p(x)/q(x))$. If it is rejected, the emitted token is drawn from the residual $r=\operatorname{norm}(\max(0,p-q))$. The probability of emitting $y$ is then

$$P(y)=q(y)\min\!\Big(1,\tfrac{p(y)}{q(y)}\Big)+\beta\, r(y),\qquad \beta=\sum_x\min(p(x),q(x)).$$

Here $\beta$ is the acceptance probability at this position. Because $\sum_x(p-q)^+=1-\beta$, we get $\beta\,r(y)=(p(y)-q(y))^+$. Using $\min(a,b)+(a-b)^+=a$, we have $P(y)=p(y)$ exactly.

Micro-example with $p=[0.6,0.4,0]$ and $q=[0.2,0.3,0.5]$. Acceptance ratios are $[1,\,1,\,0]$, so $\beta=0.5$. The residual is $[0.4,0.1,0]$, normalized to $r=[0.8,0.2,0]$. Then $P(0)=0.2+0.5\cdot0.8=0.6$ and $P(1)=0.3+0.5\cdot0.2=0.4$, as required.

Per position, $\beta=\sum_x\min(p,q)=1-\mathrm{TV}(p,q)$. Acceptance is therefore set by how far the draft is from the target. The i.i.d. $\alpha$ is an idealization of this.

Measured values are reported in the literature. A 2026 technical report gives an average acceptance length of about 3.45 tokens per verification step at draft length 7, and acceptance falls with draft depth [corpus: library-papers/2604.12374.pdf p.6]. Check whether a paper's "acceptance length" counts the bonus token before comparing numbers.

## Signal vs. noise

Signal: the two formulas, the round cost $\gamma c+1$, and the rule that the rejection branch resamples from the residual. Those four pieces decide every speedup question in an interview.

Noise: each paper's notation, tree-shaped drafts (see inf/tree-and-variable-length-verification), and the greedy special case.

## Where it breaks

- **The i.i.d. assumption.** Acceptance drops with depth, so a single constant $\alpha$ misstates $\mathbb E[N]$, and the error can go either way. With per-depth rates $(1.0, 0.0)$ the true value is $2.0$, while the mean-matched constant $0.5$ gives $1.75$. With $(0.1, 0.9)$ the true value is $1.19$, while the constant $0.5$ gives $1.75$. Use the product form $1+\sum_k\prod_{j\le k}\alpha_j$ when you have per-depth rates.
- **The draft cost is not constant.** Take $\alpha=0.5$, $\gamma=8$, $c=0.3$. Then $\mathbb E[N]\approx 1.996$ and the round costs $3.4$, so $S\approx 0.59$. A large draft at moderate acceptance is a loss.
- **Verification is not free for large $\gamma$ or large batches.** A verification pass over $\gamma+1$ positions can become compute-bound, and its cost then rises above 1.
- **Division by zero.** A draft token with $q(x)=0$ cannot have been proposed, so the implementation must reject it rather than divide.

## Interview angle

Tokens per round are $\mathbb E[N]=(1-\alpha^{\gamma+1})/(1-\alpha)$, which saturate at $1/(1-\alpha)$. Round cost is $\gamma c+1$. Speedup is the ratio, so a cheap draft with low acceptance loses, and an expensive draft with high acceptance can still win. The rejection rule keeps the output distribution equal to the target, so speculation changes speed and not the answer.

Likely follow-ups: why acceptance matters more than $\gamma$ (the gain is capped at $1/(1-\alpha)$ in the limit), and why the output is unchanged (the residual makes the mixture exactly $p$).

## Connections

Requires [inf/prefill-vs-decode-bottlenecks] for the memory-bound verification cost, and [llm/temperature-top-k-and-top-p-sampling] for the distributions $p$ and $q$ after temperature. This topic unlocks [inf/speculative-draft-and-verify] (the implementation side) and [inf/tree-and-variable-length-verification]. The original scheme is due to Leviathan et al. 2023 and Chen et al. 2023 (not in the corpus).
