---
title: "Direct preference optimization (DPO)"
minutes: 12
min_viable: "Read The system and Worked example, then do p01 with Hint 1 if stuck."
---

## The system

Three models and one dataset.

- **Policy** $\pi_\theta$: the model being trained.
- **Reference** $\pi_{\text{ref}}$: a frozen copy, usually the supervised fine-tuned model. Its
  parameters get no gradient.
- **Data**: triples $(x, y_c, y_r)$ of a prompt, a chosen response and a rejected response, as in the
  [Bradley-Terry lesson](../bradley-terry-reward-models/lesson.md).

The **sequence log-probability** of a response is the sum of its token log-probabilities, over the
response positions $\mathcal{R}$ only. Prompt tokens are masked out, exactly as in SFT:

$$\log \pi(y \mid x) = \sum_{t \in \mathcal{R}(y)} \log \pi\big(y_t \mid x, y_{<t}\big)$$

Token $t$ is scored by the logits at position $t-1$ (the one-token shift), so the mask applies to
tokens $t \ge 1$.

Define the **implicit reward** with inverse temperature $\beta > 0$:

$$\hat r(x, y) = \beta \big[\log \pi_\theta(y \mid x) - \log \pi_{\text{ref}}(y \mid x)\big]$$

The DPO loss is the Bradley-Terry loss on these implicit rewards:

$$\mathcal{L}_{\text{DPO}} = \frac{1}{B} \sum_{i=1}^{B} -\log \sigma\big(\hat r(x_i, y_{c,i}) - \hat r(x_i, y_{r,i})\big)$$

Shapes: every log-probability is a vector of shape $(B,)$, one per pair.

## Worked example

Take $\beta = 0.1$, $B = 1$. Sequence log-probabilities:

| | policy $\log \pi_\theta$ | reference $\log \pi_{\text{ref}}$ | $\hat r$ |
|---|---|---|---|
| chosen $y_c$ | -10.0 | -11.0 | $0.1 \cdot (-10 + 11) = 0.1$ |
| rejected $y_r$ | -12.0 | -11.0 | $0.1 \cdot (-12 + 11) = -0.1$ |

Margin $\hat r_c - \hat r_r = 0.2$. Loss $= -\log\sigma(0.2) = \operatorname{softplus}(-0.2) = 0.5981$.
A positive margin puts the loss below $\log 2$. If the policy equalled the reference, both implicit rewards would be 0,
the margin 0, and the loss $\log 2 = 0.6931$.

**The shift, with numbers.** One row, $T = 3$ tokens, vocabulary $V = 3$, uniform logits (every
token has probability $1/3$). Ids $[4, 1, 2]$, where token 0 is the prompt. Mask
$[\text{F}, \text{T}, \text{T}]$ marks tokens 1 and 2 as response. Token 1 is scored by the logits
at position 0, and token 2 by the logits at position 1, so the sum is $2 \log(1/3) = -2.1972$.
An unshifted mask $[\text{F}, \text{T}]$ on the logit positions would score only token 2, giving
$-1.0986$. That is the off-by-one to watch for.

## Why it works

**The derivation follows Rafailov et al. (2023), Eqs. 3–5 and 7** [corpus:library-refresh/02-AI-ML/13-Alignment-and-Reasoning/2023 -- Rafailov et al -- Direct Preference Optimization - Your Language Model is Secretly a Reward Model -- arXiv.pdf.md pp.3-4]. The RLHF objective maximises $\mathbb{E}[r(x,y)]$ minus $\beta$ times the KL
to the reference. Its optimum is $\pi^\star(y \mid x) \propto \pi_{\text{ref}}(y \mid x)\,
e^{r(x,y)/\beta}$. Solve for the reward:

$$r(x, y) = \beta \log \frac{\pi^\star(y \mid x)}{\pi_{\text{ref}}(y \mid x)} + \beta \log Z(x)$$

$Z(x)$ depends on the prompt only. It is the same for $y_c$ and $y_r$, so it cancels in the
difference. That is why the Bradley-Terry loss on $\hat r$ needs no separate reward model and no
RL loop. The preference pairs are enough, and the gradient comes from log-probabilities of given
responses. The [DPO note](../../../sources/library/AI-ML/13-Scaling-and-Alignment/Rafailov%20et%20al.%20%E2%80%94%20Direct%20Preference%20Optimization%202023.md)
states the idea that the optimal policy is a function of the reward, without this equation.

## Signal vs. noise

**Signal.** The masked sum of log-probs; the implicit reward and its $\beta$; the loss as
Bradley-Terry on $\hat r$; and the rule that the policy and the reference must score the same
tokens under the same mask.

**Noise.** Optimiser, schedule and batch size.

## Where it breaks

- **$\beta$ sets how far the policy may move.** In the KL-regularised objective, $\beta$ weights the
  anchor: a large $\beta$ keeps the policy near the reference, and a small $\beta$ lets it drift.
  In the loss, $\beta$ also scales the margin, so a large $\beta$ saturates the sigmoid on smaller
  log-ratio gaps. Tune it, and watch the reward margin and the KL.
- **Sums depend on length.** A sum of log-probs shrinks as the response gets longer. Margins between
  responses of different length therefore mix quality with length. Some variants change the
  normalisation, for example SimPO's length-normalised reward (from memory). Check each variant's paper, which is
  not in the corpus, before relying on its definition.
- **Mismatched masks or tokenisation.** If the reference scores different tokens (another chat
  template, a different prompt mask), the offset in $\hat r$ is wrong and the loss trains on noise.
- **Both log-probs can fall together.** The loss only sees the difference, so both responses can
  lose probability while the margin grows. Check the chosen log-probability separately.

## Interview angle

**60-second answer.** DPO optimises the policy directly on preference pairs. The implicit reward is
$\beta(\log\pi_\theta - \log\pi_{\text{ref}})$ on each response, and the loss is the Bradley-Terry
loss on the difference of implicit rewards. The partition function cancels, so there is no reward
model and no sampling during training. $\beta$ controls how far the policy can move from the
reference. Sequence log-probs are sums over response tokens only.

**Follow-ups.** See `interview.md`: how $\beta$ trades off reward against drift, and why the reference
log-probs must use the same mask.

## Connections

Prerequisites: [Bradley-Terry reward models](../bradley-terry-reward-models/lesson.md) (the loss),
[negative log-likelihood and perplexity](../negative-log-likelihood-and-perplexity/) (masked means) and
[instruction-tuning batches and loss masks](../instruction-tuning-batches-and-loss-masks/) (the response
mask). This unlocks the length-normalised variants and [GRPO](../grpo-and-its-variants/), which keeps a
KL term to the same reference. The lineage runs from InstructGPT (RLHF with a KL penalty to the SFT
model) to DPO, which drops the RL loop.
