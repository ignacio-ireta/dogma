---
title: Cross-entropy
minutes: 12
min_viable: "Read The system and the Worked example, then solve p01."
---

## The system

For $n$ samples and $K\ge2$ classes, each logit row $\mathbf z_i\in\mathbb R^K$ gives a probability vector $\mathbf{p} = \operatorname{softmax}(\mathbf{z})$ over $K$ classes. If the true class is $t$, the likelihood of that example is $p_t$. The **cross-entropy** of one example is the negative log-likelihood, and the loss averages it over $n$ examples:

$$L = -\frac{1}{n}\sum_{i=1}^{n}\log p_{i,t_i}.$$

Its gradient with respect to the logits has a simple form. Let $\mathbf{e}_t$ be the one-hot vector of class $t$:

$$\frac{\partial L}{\partial \mathbf{z}_i} = \frac{1}{n}\big(\mathbf{p}_i - \mathbf{e}_{t_i}\big).$$

Each row sums to zero. The true class is pushed up by $p_t - 1 \le 0$, and every other class is pushed down by its own probability.

For a single logit $z$ with a sigmoid output and binary target $y \in \{0,1\}$, the binary cross-entropy is

$$\ell = \max(z, 0) - z\,y + \log\!\big(1 + e^{-|z|}\big),$$

and its gradient with respect to $z$ is $\sigma(z) - y$.

## Worked example

Logits $(2, 1, 0.1)$, true class $0$. The softmax is $(0.6590, 0.2424, 0.0986)$. The loss is $-\log 0.6590 = 0.4170$. Using the log-sum-exp, $\operatorname{logsumexp}(2, 1, 0.1) - 2 = 2.4170 - 2 = 0.4170$, the same number. The gradient is $(0.6590 - 1, 0.2424, 0.0986) = (-0.3410, 0.2424, 0.0986)$. It sums to zero, and it points to increasing the first logit.

Uniform logits over $K$ classes give loss $\log K$. For $K = 3$ that is $1.0986$.

## Why it works

**Entropy plus divergence.** For a true distribution $\mathbf{p}$ and model $\mathbf{q}$,

$$H(\mathbf{p}, \mathbf{q}) = -\sum_j p_j\log q_j = H(\mathbf{p}) + D_{\mathrm{KL}}(\mathbf{p}\,\|\,\mathbf{q}).$$

With $\mathbf{p} = (0.7, 0.3)$ and $\mathbf{q} = (0.5, 0.5)$: $H(\mathbf{p}) = 0.6109$, $D_{\mathrm{KL}} = 0.0823$, and their sum is $0.6931 = \log 2$. The entropy is the floor no model can beat. The KL term is the excess that the model adds. Minimising cross-entropy over $\mathbf{q}$ is therefore minimising the KL divergence, because $H(\mathbf{p})$ does not depend on the model.

**Why from logits.** The loss needs $\log p_t$. Computing $\log$ of the rounded probability gives $-\infty$ for a confident wrong prediction. The loss is $\operatorname{logsumexp}(\mathbf{z})-z_t$. Evaluate it in shifted coordinates to avoid cancellation from a huge common offset. Representable losses stay finite; a true loss beyond float64 range must be $+\infty$. See Goodfellow, Bengio & Courville, Deep Learning, chapter on numerical computation.

**Label smoothing.** Replace the one-hot target by $(1 - \varepsilon)\mathbf{e}_t + \varepsilon/K$. With $\varepsilon = 0.1$ and $K = 3$ the target is $(0.9333, 0.0333, 0.0333)$. For logits $(2, 0, 0)$ the hard cross-entropy is $0.2395$ and the smoothed one is $0.3729$. For $0<\varepsilon<1$, all target probabilities are positive. At the optimum the logit differences are finite, although adding a common offset still leaves the probabilities unchanged.

## Signal vs. noise

- **Signal:** the gradient is prediction minus target. Divide by $n$ because the loss is a mean.
- **Signal:** compute from logits. The loss depends on the logits only through $\log p_t$.
- **Signal:** the binary gradient $\sigma(z) - y$ has no $\sigma'$ factor, so a confidently wrong sigmoid unit still gets a useful gradient.
- **Noise for now:** the base of the logarithm.

## Where it breaks

- **Log of a rounded probability.** `np.log(softmax(z))` returns $-\infty$ for a logit far below the others, and emits a divide warning.
- **Naive binary form.** At $z = 800$ with $y = 0$, the loss is $800$. A probability-based form has $\log(1 - \sigma(800)) = \log 0 = -\infty$. Clipping the probability at $10^{-12}$ gives $27.6$ instead, which is wrong by a large amount.
- **Sum instead of mean.** A batch twice as large doubles the loss and the gradient scale, which changes the effective learning rate.
- **Float labels.** Class indices stored as floats are silently truncated when the code casts them. Require integer targets for the categorical loss.

## Interview angle

*60-second answer:* Cross-entropy is the negative log-likelihood of a categorical model, averaged over examples. Compute it from the logits with a log-sum-exp. Its gradient with respect to the logits is the predicted probabilities minus the one-hot target, divided by the batch size, so each row sums to zero. Cross-entropy equals the entropy of the target plus the KL divergence, so minimising it fits the model to the target distribution.

*Likely follow-ups:* Why does a sigmoid output with binary cross-entropy avoid the vanishing gradient that squared error has? (The $\sigma'$ factor cancels, leaving $\sigma(z) - y$.) What does label smoothing do? (It replaces the target with a softened distribution, which favours finite logit differences and limits overconfidence.)

## Connections

- **Prerequisites:** `cml/softmax` (the probabilities and log-sum-exp), `cml/entropy-and-mutual-information` (the entropy term), `cml/maximum-likelihood-and-map` (the loss as a likelihood).
- **Unlocks:** `cml/softmax-regression-and-one-vs-rest` (the multi-class model with this loss), and every neural classifier, which is trained on the same loss.
- **Lineage:** cross-entropy as a likelihood, and the logits form, are standard. Source: Goodfellow et al., chapter on numerical computation; Murphy, chapter on logistic regression.
