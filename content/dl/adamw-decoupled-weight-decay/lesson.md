---
title: "AdamW decoupled weight decay"
minutes: 12
min_viable: "Read The system and the zero-gradient example, then do p01 using Hint 1 if needed."
---

## The system

Weight decay shrinks parameters toward zero. Let $\theta$ be a parameter tensor, $g$ its loss gradient,
$\lambda\ge 0$ the decay coefficient, and $\eta$ the learning rate. Adam with L2 folds the penalty into
the gradient, before the adaptive step:

$$
g' = g + \lambda\,\theta, \qquad \text{then the Adam update uses } g'.
$$

AdamW keeps the Adam update on the raw $g$, and decays the weights directly:

$$
\theta \leftarrow (1-\eta\lambda)\,\theta \quad\text{then}\quad
\theta \leftarrow \theta - \eta\,\frac{\hat m}{\sqrt{\hat v}+\epsilon}, \qquad \hat m,\hat v \text{ from } g.
$$

PyTorch's `AdamW` does the first line as `param.mul_(1 - lr * weight_decay)`, then runs Adam on the raw
gradient. This is the `decoupled_weight_decay` branch of `torch/optim/adam.py` in torch 2.14.1, checked by
running it: a zero-gradient step returns $\theta(1-\eta\lambda)$ exactly, and the decay-first order matches
`torch.optim.AdamW` to rounding while the Adam-then-decay order does not. `torch.optim.Adam(weight_decay=...)`
is the L2 form.

For plain SGD the two are identical once $\lambda$ is rescaled by $\eta$. For adaptive methods they
differ, because the $\hat v$ scale divides the L2 term as well as the loss gradient.

## Worked example

Two scalar weights, $\theta = 2$ and $\theta = 0.01$, both with zero loss gradient. Take $\eta = 0.1$ and
$\lambda = 0.5$, and make the first step (so $t=1$).

- AdamW: each weight is multiplied by $1-0.05 = 0.95$. The results are $1.9$ and $0.0095$.
- Adam with L2: $g' = 0.5\,\theta$, so $g'=1$ and $g'=0.005$. Adam normalises each to a step of
  about $\eta = 0.1$ toward zero. The results are $1.9$ and about $-0.09$.

The second weight crosses zero. Adam with L2 pulls every weight by about the same amount, whatever its
size, and small weights overshoot. AdamW shrinks every weight by the same fraction.

## Why it works

With L2, the effective decay on a weight is $\eta\lambda\theta/\sqrt{\hat v}$. A weight with a large typical
loss gradient has large $\sqrt{\hat v}$, so it is decayed less. A weight with small gradients is decayed
more. The paper states this: with L2 both gradients are normalised by their typical size, so weights with
large gradient magnitude are regularised less, while decoupled decay regularises all weights at the same
rate [corpus:library-papers/1711.05101.pdf.md p.2]. Decoupling removes that dependence on $\hat v$.

The paper decouples the two terms on line 12 of its Algorithm 2: the decay term $\lambda\theta_{t-1}$ is
kept outside the adaptive quotient $\hat m/(\sqrt{\hat v}+\epsilon)$, and both are scaled by the schedule
multiplier $\eta_t$ [corpus:library-papers/1711.05101.pdf.md p.3]. PyTorch's form is that line with
$\eta_t=1$ and the learning rate as the step size.

## Signal vs. noise

The signal is where the decay sits: outside the adaptive scale, and multiplied by $\eta$. The
paper's default-setting details and the learning-rate schedule are secondary for this topic. For bias and
norm parameters, the rule is a convention, not a theorem: the paper does not decide which parameters
to exclude. The common practice in transformer training is to exclude biases and 1-D norm gains.

## Where it breaks

- **Mental model.** Writing `weight_decay=0.1` in Adam does not give the decay you would get from SGD with
  L2 $0.1$. The effective decay depends on the gradient history.
- **Decay with the learning rate.** The decay per step is $\eta\lambda$. Halving $\eta$ halves the
  decay. A schedule that changes $\eta$ also changes the decay.
- **Excluded parameters.** Decaying biases or norm gains rarely helps and can hurt. Exclusion is done
  with parameter groups, so check the groups before tuning $\lambda$.
- **Checkpoints.** The group layout is state. A resumed run with different groups does not match.

## Interview angle

Start with the two formulas: L2 adds $\lambda\theta$ to the gradient, and decoupled decay multiplies by
$1-\eta\lambda$. Say that the first is divided by $\sqrt{\hat v}$, so the regularisation is uneven across
weights. For a transformer, exclude biases and 1-D norm gains, and say that this is a convention. Full
versions are in `interview.md`.

## Connections

Ridge regression adds $\frac{\lambda}{2}\lVert\theta\rVert^2$ to the loss, and its gradient is $\lambda\theta$. That
is the L2 term in this topic. Adam was the previous topic, and AdamW is its corrected decay. Parameter groups
are the practical part: they connect to the optimizer built in p02.
