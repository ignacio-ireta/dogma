---
title: Negative log-likelihood and perplexity
minutes: 12
min_viable: "Read The system and Worked example, then implement p01 with Hint 1 open."
---

## The system

A causal model returns logits $z \in \mathbb{R}^{B \times T \times V}$: $B$ sequences, $T$ positions, and $V$ vocabulary entries per position. The targets are $y \in \{0, \dots, V-1\} \cup \{\text{ignore}\}^{B \times T}$, with the ignore id $-100$ by default.

For each position $(n, t)$ whose target is not ignored:

$$\ell_{n,t} = -\log p(y_{n,t} \mid z_{n,t}) = \operatorname{LSE}(z_{n,t}) - z_{n,t,\,y_{n,t}}, \qquad \operatorname{LSE}(z) = \log \sum_{v=1}^{V} e^{z_v}.$$

The mask $m_{n,t} = 1$ if the target is not ignored, else $0$. Then

$$\bar\ell = \frac{\sum_{n,t} m_{n,t}\,\ell_{n,t}}{\sum_{n,t} m_{n,t}}, \qquad \mathrm{PPL} = \exp(\bar\ell).$$

Three rules:

1. **The denominator counts non-ignored tokens.** It is not $B \cdot T$, and it is not $B$.
2. **Use the stable log-sum-exp.** Subtract the row maximum $m = \max_v z_v$ before exponentiating: $\operatorname{LSE}(z) = m + \log \sum_v e^{z_v - m}$.
3. **Perplexity is a function of the mean NLL in nats.** Natural logarithms, so $\mathrm{PPL} = e^{\bar\ell}$. In bits, the mean is $\bar b = \bar\ell / \ln 2$ and $\mathrm{PPL} = 2^{\bar b}$. The two are the same number.

## Worked example

**Uniform guess.** $V = 4$ and every logit is $0$. Then $\operatorname{LSE}(z) = \log 4$ and $\ell = \log 4 - 0 = 1.3863$ nats for each token. The mean is $1.3863$ and $\mathrm{PPL} = 4$. A model that is uniform over four choices has perplexity $4$: the number of effective choices per token.

**Two positions, one peaked.** $B = 2$, $T = 1$, $V = 2$. Row 0 has logits $(2, 0)$ and target $0$. Row 1 has logits $(0, 0)$ and target $1$.

- $\ell_0 = \log(1 + e^{-2}) = 0.1269$
- $\ell_1 = \log 2 = 0.6931$
- $\bar\ell = (0.1269 + 0.6931)/2 = 0.4100$, and $\mathrm{PPL} = e^{0.4100} = 1.5069$

**Ignored positions.** Targets $(1, -100, 2)$ with uniform logits over $V = 4$ give two counted tokens, each $\log 4$. The mean is $\log 4$, not $\tfrac{2}{3}\log 4$.

**Large logits.** Logits $(1000, 999)$ with target $1$. Naive $e^{1000}$ overflows float64. Stable: $\operatorname{LSE} = 1000 + \log(1 + e^{-1})$, so $\ell = 1 + \log(1 + e^{-1}) = 1.3133$ and $\mathrm{PPL} = 1 + e = 3.7183$.

## Why it works

Perplexity is the exponential of the average negative log-likelihood per predicted token. The GPT-2 paper defines the reported language-model quantity as "a scaled or exponentiated version of the average negative log probability per canonical prediction unit - usually a character, a byte, or a word" (Radford et al., 2019, p. 4). The exponential turns a sum of log-probabilities into a number of effective choices. Averaging over tokens, not rows, matches that definition: a row with one real token must not weigh the same as a row with fifty.

The same quantity is what cross-entropy computes. Cross-entropy with `ignore_index` and `reduction="mean"` divides by the count of non-ignored targets, so $\mathrm{PPL} = \exp(\text{that loss})$.

## Signal vs. noise

Signal: the masked denominator, the stable log-sum-exp, and $\mathrm{PPL} = \exp(\bar\ell)$. Noise: the log base, as long as the same base is used throughout (nats and bits give the same perplexity). The sign convention for targets is noise too, once the ignore id is named.

## Where it breaks

- **Mean over the batch with padding.** Padding positions are counted in the denominator and give zero loss, so the average looks better. The reported perplexity falls for no real reason.
- **Naive softmax.** $e^{z}$ overflows for logits near $710$ in float64. The result is `inf` or `NaN`, which then spreads into the mean.
- **Comparing across tokenizers.** Perplexity is per token. A tokenizer that uses more tokens per string gets a lower per-token NLL for the same text. The GPT-2 paper reports perplexity per canonical unit, and its invertible de-tokenizers give "gains of 2.5 to 5 perplexity" (Radford et al., 2019, p. 5). Compare models on bits per byte or bits per character when the tokenizers differ.
- **Evaluating on training data.** A low perplexity on text the model has seen says little about generalisation. Use held-out text.

## Interview angle

Sixty-second answer: the NLL of a token is the log-sum-exp of its logits minus the logit of its target. I average it over the non-ignored tokens only, then exponentiate, and I compute the log-sum-exp with the row maximum subtracted so large logits do not overflow. Perplexity is the number of effective choices per token. It depends on the tokenizer, so across tokenizers I compare bits per byte. Likely follow-up: why not average over the batch? Because padding and short rows then carry the wrong weight. Second follow-up: is a lower perplexity always better? Only on the same held-out text with the same tokenizer.

## Connections

- Prerequisites: [Cross-entropy](../../cml/cross-entropy/), [Softmax](../../cml/softmax/), and [Next-token targets and batches](../next-token-targets-and-batches/), which provides the shifted targets and the ignore id.
- Next: model evaluation on held-out text, and the training loop, where this loss is minimized.
- Lineage: the perplexity definition used for n-gram and neural language models, as reported in the GPT-2 paper (Radford et al., 2019).
