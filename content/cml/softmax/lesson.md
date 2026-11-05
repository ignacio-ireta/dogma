---
title: Softmax
minutes: 12
min_viable: "Read The system and the Worked example, then solve p01."
---

## The system

Given a vector of real scores (logits) $\mathbf{z} \in \mathbb{R}^K$, for $K\ge2$, the **softmax** is the probability vector

$$\operatorname{softmax}(\mathbf{z})_j = \frac{e^{z_j}}{\sum_{k=1}^{K} e^{z_k}}, \qquad \operatorname{softmax}(\mathbf{z})_j \in (0, 1), \qquad \sum_j \operatorname{softmax}(\mathbf{z})_j = 1.$$

It is the normalised exponential: exponentiate so every score is positive, then divide by the total. Four rules:

| Rule | Statement | Consequence |
|---|---|---|
| Shift invariance | $\operatorname{softmax}(\mathbf{z} + c\mathbf{1}) = \operatorname{softmax}(\mathbf{z})$ | subtract the maximum freely |
| Monotone | larger $z_j$ gives larger probability | the argmax is preserved |
| Two classes | $\operatorname{softmax}([z, 0])_1 = \sigma(z)$ | softmax generalises the sigmoid |
| Temperature | $\operatorname{softmax}(\mathbf{z}/T)$ | $T \to 0^+$ splits mass uniformly among maxima; $T \to \infty$ gives uniform |

Here $T>0$ is scalar temperature and $c\in\mathbb R$ is a common offset. The **log-softmax** is $\log\operatorname{softmax}(\mathbf{z})_j = z_j - \operatorname{logsumexp}(\mathbf{z})$, where $\operatorname{logsumexp}(\mathbf{z}) = \log\sum_k e^{z_k}$.

## Worked example

$\mathbf{z} = (1, 2, 3)$. The maximum is $3$. Subtract it to get $(-2, -1, 0)$, then exponentiate: $(0.1353, 0.3679, 1)$. The sum is $1.5032$, so

$$\operatorname{softmax}(\mathbf{z}) = (0.0900,\ 0.2447,\ 0.6652).$$

Shift invariance gives the same answer from the shifted vector. Temperature: $\operatorname{softmax}((2, 0, 0)/100) = (0.3378, 0.3311, 0.3311)$, almost uniform. At $T = 0.001$ the logits $(2, 0, 0)$ give $(1, 0, 0)$.

Overflow: $\operatorname{logsumexp}(1000, 1000) = 1000 + \log 2 = 1000.693$. The naive $\log\sum_k e^{z_k}$ overflows at $z = 1000$, because $e^{1000}$ exceeds the largest float.

## Why it works

Subtracting the maximum $m$ changes nothing, because the constant cancels between numerator and denominator. After the shift every exponent is at most zero, so no exponential can overflow. The largest term is exactly $e^0 = 1$, so the normaliser is at least $1$ and cannot underflow to zero.

For losses, take the logarithm of the normalised exponential directly:

$$\log\operatorname{softmax}(\mathbf{z})_j = (z_j - m) - \log\sum_k e^{z_k - m}.$$

Keep the calculation in these shifted coordinates: adding a huge $m$ back can erase the small log normalizer. This never forms the probability. A probability near $e^{-1000}$ underflows to $0$, and $\log 0 = -\infty$. The log-form returns $-1000$.

## Signal vs. noise

- **Signal:** softmax is shift-invariant, so subtract the maximum of each slice, along the axis you normalise over.
- **Signal:** with two classes, the softmax is the sigmoid of the difference of the two logits.
- **Signal:** losses use log-softmax, not the log of a softmax.
- **Noise for now:** the choice of base of the logarithm. Nats and bits differ by a constant.

## Where it breaks

- **Maximum over the wrong axis.** A global maximum over a matrix shifts rows that have much smaller logits to exponents that underflow. Those rows become $0/0 = \text{NaN}$.
- **Missing `keepdims`.** For a square matrix, `z - z.max(axis=-1)` broadcasts the maxima along the wrong dimension. It does not raise an error, and the answer is wrong.
- **Probability floors.** Clipping probabilities at $10^{-12}$ makes them no longer sum to 1, and it changes the answer for the tail.
- **Temperature near zero.** Dividing finite logits by a tiny $T$ can itself overflow before softmax runs. Shift away a common offset before temperature scaling and check the scaled inputs.
- **Float64 limits.** Opposite extreme finite logits can have a difference outside float64 range. Its probability rounds to zero; its log-probability is necessarily $-\infty$. Representable log-probabilities remain finite.

## Interview angle

*60-second answer:* Softmax turns scores into a distribution by exponentiating and normalising. Subtract the maximum along the axis first. That leaves the answer unchanged, makes every exponent non-positive, and prevents overflow. For losses, compute log-softmax as the logit minus the log-sum-exp, which stays finite for extreme logits.

*Likely follow-ups:* What does temperature do? (It scales the logits. Small $T$ concentrates the mass on the largest logit, large $T$ flattens it.) Why does softmax with two classes give the sigmoid? (Divide numerator and denominator by $e^{z_2}$.)

## Connections

- **Prerequisites:** `cml/sigmoid-and-log-odds` (the two-class case and the stable exponential).
- **Unlocks:** `cml/cross-entropy` (the loss on softmax outputs), `cml/softmax-regression-and-one-vs-rest` (the multi-class linear model).
- **Lineage:** softmax is the categorical analogue of the sigmoid, and the log-sum-exp reduction is the standard route to stable implementations. Source: Goodfellow et al., chapter on numerical computation.
