---
title: "Bradley-Terry reward models"
minutes: 12
min_viable: "Read The system and Worked example, then do p01 using Hint 1 if stuck."
---

## The system

A reward model maps one (prompt $x$, response $y$) pair to one real number, $r_\theta(x, y)$.
Raters are better at comparing two responses than at giving each one an absolute score, so the
training data is a set of pairs: a **chosen** response $y_c$ and a **rejected** response $y_r$
to the same prompt. Write $r_c = r_\theta(x, y_c)$ and $r_r = r_\theta(x, y_r)$, both scalars.

The Bradley-Terry model says the chance that the chosen response wins is a sigmoid of the
difference:

$$P(y_c \succ y_r \mid x) = \sigma(r_c - r_r), \qquad \sigma(z) = \frac{1}{1 + e^{-z}}$$

Training minimises the negative log-likelihood of the observed choices over $B$ pairs. Shapes:
$r_c, r_r \in \mathbb{R}^{B}$, one entry per pair.

$$\mathcal{L} = \frac{1}{B} \sum_{i=1}^{B} -\log \sigma\big(d_i\big), \qquad d_i = r_{c,i} - r_{r,i}$$

Five rules follow:

1. **Only differences matter.** Adding the same constant $c$ to $r_c$ and $r_r$ gives
   $(r_c + c) - (r_r + c) = r_c - r_r$, so $d$, $P$ and $\mathcal{L}$ are unchanged. The offset of
   a reward model is not identified, but its scale is: doubling every reward changes $P$ and
   $\mathcal{L}$. On separable data the fit still pushes this scale up without limit (see Where it breaks).
2. **The loss is the logistic loss of the margin $d$.** $d = 0$ gives $\log 2 \approx 0.693$. A
   confident correct pair ($d \gg 0$) gives a loss near 0. A confident wrong pair ($d \ll 0$) gives
   a loss close to $-d$.
3. **The gradient on each pair is bounded.** $\partial \mathcal{L}_i / \partial d_i = -(1 - \sigma(d_i))$
   has magnitude at most 1, so one mislabelled pair cannot dominate the update.
4. **Pair accuracy** is the fraction of pairs with $r_c > r_r$. A tie ($r_c = r_r$) counts as wrong,
   because the model has no tie outcome.
5. **Compute it stably.** Write $-\log\sigma(d)$ as
   $\operatorname{softplus}(-d) = \max(-d, 0) + \log(1 + e^{-|d|})$. The naive
   $-\log(\sigma(d))$ fails for large negative margins: near $d \approx -710$ the term $e^{-d}$
   overflows float64, so $\sigma$ becomes exactly 0. Its log is $-\infty$, so the naive loss is $+\infty$.
   In code, write $\max(x, 0)$ as $(x + |x|) / 2$. A `clamp` has a kink at 0 and autograd gives it
   slope 0 there, so a tied pair gets no gradient, and a reward head that starts at zero ties every
   pair and never trains.

## Worked example

Two pairs, with margins $d = [2.0 - 0.0, \; 0.0 - 1.0] = [2, -1]$.

| pair | $r_c$ | $r_r$ | $d$ | $-\log\sigma(d)$ | correct? |
|---|---|---|---|---|---|
| 1 | 2.0 | 0.0 | 2 | $\log(1 + e^{-2}) = 0.1269$ | yes |
| 2 | 0.0 | 1.0 | -1 | $\max(1, 0) + \log(1 + e^{-1}) = 1.3133$ | no |

Mean loss $= (0.1269 + 1.3133) / 2 = 0.7201$. Pair accuracy $= 1 / 2 = 0.5$.


## Why it works

The one derivation that matters: a pair is a two-way softmax. With scores $r_c$ and $r_r$,

$$\frac{e^{r_c}}{e^{r_c} + e^{r_r}} = \frac{1}{1 + e^{-(r_c - r_r)}} = \sigma(r_c - r_r).$$

So the Bradley-Terry loss is the cross-entropy of a two-class softmax whose target is "chosen". The
model comes from paired-comparison ranking [Bradley and Terry 1952, cited from memory]. Christiano
et al. (2017) used human comparisons to fit a reward predictor for reinforcement learning
[library: Christiano et al. — Deep RL from Human Preferences 2017]. InstructGPT trained a reward
model to predict human preferences over ranked model outputs before running RL against it
[library: Ouyang InstructGPT RLHF 2022].

## Signal vs. noise

**Signal:** the loss on the margin; offset invariance; the stable softplus form; accuracy as a
quick diagnostic. You must be able to say why $\sigma(r_c - r_r)$ rather than $\sigma(r_c)$: the
second depends on the arbitrary offset.

**Noise:** the reward head's architecture, ensembles and batch size. They matter in practice,
not for the loss.

## Where it breaks

- **Separable data grows the margins without bound.** If every pair is ordered correctly, the loss
  keeps shrinking as $d$ grows, so rewards drift to large magnitudes. Watch the reward scale over
  training. Regularise it, or add a margin term, if it grows too fast.
- **Reward hacking.** A policy optimised against the reward model finds outputs the model scores
  high but the raters would not prefer (for example longer or more confident text). The loss never
  sees those outputs, because it was fit on comparisons from a different distribution. Pair accuracy
  on the training distribution can stay high while the policy exploits the model. Track held-out pairs and a second judge's scores on samples from the current policy.
- **Ties and label noise.** The model has no tie outcome, so drop tied pairs. Noisy labels get
  bounded gradients, but clustered noise still biases the fit.

## Interview angle

**60-second answer.** A reward model is trained on pairwise human comparisons. Bradley-Terry says
$P(\text{chosen} \succ \text{rejected}) = \sigma(r_c - r_r)$, and training minimises the negative
log-likelihood over pairs. Only the difference matters, so the offset is arbitrary, but the scale is not. I compute the
loss with softplus so large margins do not overflow. The main risk is that a policy trained on this
reward finds outputs that score high but are not better, which is reward hacking.

**Follow-ups.** Why comparisons instead of absolute scores, and what happens to the reward scale on
separable data. Full answers are in `interview.md`.

## Connections

Prerequisites: the stable sigmoid and logit (sigmoid and log-odds), and logistic regression, because
the pair loss is logistic regression on the margin. This topic unlocks DPO, which replaces the
separate reward model with an implicit reward from policy and reference log-probabilities but
keeps the same pairwise loss. The lineage runs from Christiano et al. (2017) to InstructGPT (2022)
to DPO (2023).
