---
title: "Temperature, top-k and top-p sampling"
minutes: 12
min_viable: "Read The system and Worked example, then implement p01 with Hint 1 open."
---

## The system

A model gives logits $z\in\mathbb{R}^{V}$ for a vocabulary of size $V$. Sampling turns them into one token in four stages, always in this order:

1. **Temperature.** $u=z/T$ for a temperature $T>0$, then $p=\operatorname{softmax}(u)$.
2. **Top-k.** Keep the $k$ tokens with the largest $p$, breaking ties toward the lower index, set the rest to $0$, and renormalise.
3. **Top-p (nucleus).** On the renormalised survivors, sort by $p$, descending. Keep the smallest prefix whose cumulative sum $c_j$ satisfies $c_j\ge p_{\text{top}}$, including the token that reaches it. Set the rest to $0$ and renormalise.
4. **Draw.** Sample one index from the final distribution with a seeded generator.

The knobs act as follows:

| Setting | Effect | Limit |
|---|---|---|
| $T\to 0$ | mass concentrates on the argmax (greedy) | $T=0$ is rejected, since $z/0$ is undefined; greedy is a separate branch |
| $T=1$ | plain softmax | none |
| $T\to\infty$ | nearly uniform | none |
| $k=1$ | argmax | ties go to the lower index |
| $k\ge V$ | no change | none |
| $p=1$ | no change | none |
| $p\to 0$ | only the best token survives | at least one token is always kept |

## Worked example

Take $z=[0,\,1]$ and $T=0.5$. Then $u=[0,\,2]$ and $p=\operatorname{softmax}(u)=[0.1192,\,0.8808]$.

Top-p with $p_{\text{top}}=0.9$: sorted order is $[1,0]$, cumulative sums are $c=[0.8808,\,1.0]$. The first entry is below $0.9$, so the second token is added, and both are kept. The result is $[0.1192,\,0.8808]$.

Top-p with $p_{\text{top}}=0.85$: $c_0=0.8808\ge 0.85$, so only token 1 is kept. Renormalising a single token gives $[0,\,1]$. The threshold decides everything, and the boundary is inclusive: $c_j \ge p_{\text{top}}$.

Order matters. Take $p=[0.4,\,0.3,\,0.2,\,0.1]$ with $k=2$ and $p_{\text{top}}=0.5$. Top-k keeps $0.4$ and $0.3$ and renormalises them to $0.571$ and $0.429$. The first alone reaches $0.5$, so the result is $[1,0,0,0]$. If top-p ran on the unrenormalised $0.4$ and $0.7$, it would keep two tokens.

## Why it works

Temperature scales every log-odds: $p_i/p_j=\exp\big((z_i-z_j)/T\big)$. Small $T$ sharpens the distribution, and large $T$ flattens it.

Top-k uses a fixed count $k$, so it keeps $k$ tokens whether the model is sure or unsure. Top-p uses a fixed mass. When the model is confident, one or two tokens reach mass $p$. When it is unsure, many tokens are needed. That is why top-p is a common choice over top-k. The threshold keeps the tail out of the draw, and renormalising makes the result a valid distribution again.

Filters must run before the draw. A draw from the raw distribution would still give tail tokens their full mass. A seeded `torch.Generator` makes the draw reproducible: the same seed gives the same tokens. A global random state would couple the output to every other random call.

## Signal vs. noise

Signal: the order (temperature, top-k, top-p, renormalise, draw), the inclusive boundary $c_j\ge p_{\text{top}}$, the tie-break by lower index, and the rule that at least one token survives. Noise: the folk defaults such as $T=0.7$ or $p=0.9$. They are conventions, not laws, and the exact random number algorithm does not matter as long as the draw uses the passed generator.

## Where it breaks

- **Division by zero at $T=0$.** Treat it as greedy in a separate branch, or reject it. Do not divide.
- **Strict $>$ at the boundary.** It keeps one token too many when the cumulative sum lands exactly on $p_{\text{top}}$. Check with an exact case such as four equal probabilities and $p_{\text{top}}=0.5$.
- **No renormalisation.** The masses no longer sum to 1, and the draw is biased or errors.
- **Overflow.** $\exp(u)$ overflows for large $u$ at low temperature. Subtract the maximum first.
- **Ties at the k-th place.** A rule of "keep all tokens at least as large as the k-th" keeps more than $k$. Break ties by index, so exactly $k$ survive.
- **Global random state.** Two components that share it produce different tokens in different runs.

## Interview angle

Sixty-second answer: temperature divides the logits before the softmax, which sharpens or flattens the distribution. Top-k keeps a fixed number of candidates. Top-p keeps the smallest set whose mass reaches $p$, so it adapts to the model's confidence. Apply the filters in order, renormalise, and draw with a seeded generator. Likely follow-ups: why top-p is preferred over top-k in practice (it adapts to the shape of the distribution), and what temperature zero means (greedy decoding, a separate rule).

## Connections

Prerequisites: [softmax](../../cml/softmax/), which produces the probabilities, and the [autoregressive generation loop](../autoregressive-generation-loop/), which calls the sampler once per step. Next: repetition penalties and constrained decoding, which also edit the logits before this step. The softmax output layer supplies the probabilities. Top-p adds the mass-based filter on top of temperature sampling.
