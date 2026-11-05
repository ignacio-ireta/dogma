---
title: "pass@k and majority voting"
minutes: 12
min_viable: "Read The system and Why it works, then implement p01 using Hint 1 if needed."
---

## The system

For one problem, draw $n$ samples from a model and check each one. Let $c$ be the number that pass. Let $k$ be the number of attempts a user would inspect, with $1 \le k \le n$.

**pass@k** is the probability that at least one of $k$ attempts passes. The unbiased estimator draws the $k$ attempts *without replacement* from the $n$ samples you already have:

$$
\text{pass@}k = 1 - \frac{\binom{n-c}{k}}{\binom{n}{k}}
$$

Here $\binom{n-c}{k}$ counts the $k$-subsets made only of failing samples, and $\binom{n}{k}$ counts all $k$-subsets. Two rules follow directly:

| Case | Value | Reason |
|---|---|---|
| $c = 0$ | $0$ | no subset can contain a pass |
| $n - c < k$ | $1$ | every $k$-subset includes a pass |
| $k = 1$ | $c/n$ | one draw passes with frequency $c/n$ |

**Majority voting (self-consistency)** samples several answers to one question, normalises each final answer, and returns the most common one. The vote needs a tie rule written down in advance.

Normalise with `" ".join(a.split()).lower()`. This collapses internal whitespace, trims both ends and lowercases. It keeps punctuation, so `"3."` and `"3"` are different answers. Blank answers cast no vote. A tie goes to the answer whose first occurrence is earliest in the list, and not to alphabetical order or the last occurrence:

- `["Paris", "London", "paris", "london"]` gives `"paris"`. Both have 2 votes; Paris appears first.
- `["london", "x", "Paris", "paris", "london"]` gives `"london"`. Both London and Paris have 2 votes; London appears at position 0.

## Worked example

Take $n = 5$, $c = 2$, $k = 2$. There are $\binom{5}{2} = 10$ two-sample subsets. Three of them contain only the three failing samples, since $\binom{3}{2} = 3$. So

$$
\text{pass@2} = 1 - \frac{3}{10} = 0.7.
$$

The naive formula gives $1 - (1 - 0.4)^2 = 0.64$, which is lower. The product form gives the same answer without building the binomials: $\prod_{i=0}^{1}(1 - \tfrac{c}{n-i}) = (1 - \tfrac{2}{5})(1 - \tfrac{2}{4}) = 0.6 \times 0.5 = 0.3$, so $1 - 0.3 = 0.7$.

## Why it works

Assume each sample passes independently with probability $p$, so $c \sim \text{Binomial}(n, p)$. Let $X = n - c$ be the number of failing samples, so $X \sim \text{Binomial}(n, 1-p)$. For a binomial variable, $\mathbb{E}\binom{X}{k} = \binom{n}{k}(1-p)^k$. Dividing by $\binom{n}{k}$ gives

$$
\mathbb{E}\left[\frac{\binom{n-c}{k}}{\binom{n}{k}}\right] = (1-p)^k,
$$

so the estimator has expectation $1 - (1-p)^k$, the true pass@k. The naive formula $1 - (1 - c/n)^k$ replaces $\mathbb{E}$ by a plug-in $\hat p = c/n$ inside a power, which is biased.

The product form avoids overflow. The ratio is $\prod_{i=0}^{k-1}\left(1 - \frac{c}{n-i}\right)$, and $\binom{n}{k}$ itself can exceed the range of a float for large $n$.

Accuracy needs one more step. Computing $1 - \prod$ directly loses relative accuracy when the answer is small. A tiny pass@k means the product is close to 1, and subtracting it from 1 keeps only the digits left after cancellation, with absolute error near $10^{-16}$. For $n = 10^{12}$, $c = k = 1$ the answer is $10^{-12}$, and the direct form has a relative error of about $2\times10^{-5}$. The contract asks for $10^{-9}$. The fix is to sum logarithms with `log1p` and return $-\text{expm1}\left(\sum_{i} \log(1 - \tfrac{c}{n-i})\right)$. `expm1` is accurate near 0, so the small answer keeps its digits. The contract's tiny-value tests check exactly this.

## Signal vs. noise

Signal: what you counted ($c$), what you drew from ($n$), and the rule that subsets are drawn without replacement. Also the voting tie rule.

Noise: which sampler produced the samples. Temperature and top-p choices change $p$, not the estimator, and they are covered in [temperature, top-k and top-p sampling](../temperature-top-k-and-top-p-sampling/).

## Where it breaks

- **Correlated samples.** The estimator assumes independent samples. Repeated near-identical samples make $c$ move in blocks, and the estimate becomes too confident. Check the diversity of your samples.
- **Small $n$.** With $n = 5$, one sample changes $c/n$ by $0.2$. Report $n$ next to every pass@k number.
- **Voting needs extraction.** Majority voting only works when answers can be normalised to one string. Free-form prose has no vote.
- **Normalisation mistakes.** Case and whitespace should merge. Punctuation should not, unless you decide it does.
- **Cost.** Each sample costs one full generation, so cost is linear in $n$.

## Interview angle

Sixty-second answer: "I draw $n$ samples, count the $c$ that pass, and report pass@$k$ with the unbiased estimator. It draws $k$ attempts without replacement from my $n$ samples, so it avoids the bias of $1-(1-c/n)^k$. For a final answer without a test, I vote over normalised answers with a fixed tie rule, and I pay for $n$ generations to do it."

Likely follow-ups:
- *Why not $1-(1-c/n)^k$?* It assumes replacement and plugs an estimate into a power, so it is biased.
- *How large should $n$ be?* At least $k$, and more when the estimate must be precise. Each extra sample costs a full generation.

## Connections

This uses [probability basics](../../cml/probability-basics/) and [Bernoulli, binomial and Poisson](../../cml/bernoulli-binomial-and-poisson/), since the estimator is a binomial factorial moment. The Wei et al. (2022) note in the library, [library:AI-ML/12-Frontiers/Wei Chain-of-Thought 2022.md], reports gains on GSM8K from voting over sampled chains.
