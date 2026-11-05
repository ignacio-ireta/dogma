---
title: "SGD with momentum"
minutes: 12
min_viable: "Read The system and Worked example, then do p01 using Hint 1 if needed."
---

## The system

Momentum keeps a buffer $v$ with the same shape as each parameter tensor $\theta$. The buffer
is an exponential moving sum of past gradients, so the parameters move along a running direction
rather than only the latest gradient. Let $g_t$ be the stochastic gradient at step $t$, $\mu\in[0,1)$
the momentum coefficient, and $\eta\ge 0$ the learning rate.

The PyTorch rule, with dampening $0$, is

$$
v_t = \mu\, v_{t-1} + g_t, \qquad \theta_t = \theta_{t-1} - \eta\, v_t .
$$

Three rules belong to this system:

| Rule | Behaviour |
|---|---|
| First step | $v_1 = g_1$. The buffer is seeded with the raw gradient, not damped by $1-\mu$. |
| Learning rate | $\eta$ multiplies the buffer only in the parameter update. |
| Coupled decay | $g_t \leftarrow g_t + \lambda\,\theta_{t-1}$ before the buffer update. |

A second form, often attributed to Sutskever et al. (2013) and not verified against that paper here, stores the learning rate inside the buffer:
$u_t = \mu u_{t-1} + \eta g_t$ and $\theta_t = \theta_{t-1} - u_t$. With constant $\eta$,
$u_t = \eta v_t$, so both forms give the same parameters. They differ when $\eta$ changes.

## Worked example

Take one scalar parameter $\theta_0 = 1$ with constant gradient $g = 0.5$, $\eta = 0.1$, and
$\mu = 0.9$. Step 1 seeds $v_1 = 0.5$ and moves to $\theta_1 = 0.95$. Step 2 gives
$v_2 = 0.9 \cdot 0.5 + 0.5 = 0.95$ and $\theta_2 = 0.95 - 0.095 = 0.855$.

The buffer grows toward $g/(1-\mu) = 5$ for a constant gradient. At $\mu = 0.9$, the effective step
size is about ten times $\eta$ in that steady state. This is the main reason a learning rate that
worked without momentum can diverge with it.

For Nesterov with coupled decay ($\lambda = 0.1$, $\theta_0 = 1$, $g = 0.5$), the decayed gradient is
$d = 0.5 + 0.1 = 0.6$. The buffer is $v_1 = 0.6$, the look-ahead is $d + \mu v_1 = 1.14$, and
$\theta_1 = 1 - 0.1 \cdot 1.14 = 0.886$.

## Why it works

Write the buffer unrolled: $v_t = \sum_{k\le t}\mu^{\,t-k} g_k$. Each old gradient is discounted
by $\mu^{\,t-k}$. Components of the gradient that flip sign from step to step cancel inside the sum.
Components that point the same way accumulate. In a narrow, curved valley this damps the zig-zag and
speeds progress along the valley floor. Polyak's heavy-ball method
[library:AI-ML/03-Optimization/Polyak Heavy-Ball Momentum 1964.md] adds the previous displacement as an
inertial term, which is the same idea.

Nesterov momentum evaluates the gradient after the momentum move. Hinton's lecture slides put it as
"First make a big jump in the direction of the previous accumulated gradient. Then measure the gradient
where you end up and make a correction"
[corpus:library-papers/hinton-rmsprop-2012.pdf.md p.20]. The PyTorch form
$\theta_t = \theta_{t-1} - \eta(g_t + \mu v_t)$ is that look-ahead written with the current gradient.

Coupled weight decay comes from the penalty $\frac{\lambda}{2}\lVert\theta\rVert_2^2$, whose gradient is
$\lambda\theta$. Adding it to $g$ before the buffer is formed means momentum also smooths the penalty.

## Signal vs. noise

The signal is the buffer recurrence, the first-step rule and where $\eta$ enters. The rest is
bookkeeping. Dampening is a PyTorch option set to $0$ here, and it does not change the first buffer.
Choosing $\mu$ is a tuning question, not a rule to memorise. A common default is $0.9$.

## Where it breaks

- **Learning-rate changes.** Under the PyTorch form, a schedule change applies to the whole buffer
  at once. Under the lr-in-buffer form, old buffer contents keep their old scale. Check which convention
  a checkpoint or library uses before you resume a run.
- **Resumed runs.** The buffers are optimizer state. Restarting with zero buffers changes the first
  steps even when parameters are restored.
- **Too much effective step.** With $\mu = 0.9$ the steady-state step is about ten times $\eta$. If
  training diverges only after momentum is added, halve $\eta$ before blaming the buffer.
- **Momentum is not a convergence guarantee.** On non-convex losses it can overshoot. It is a
  change of trajectory, not a new stopping rule.

## Interview angle

Start with the buffer: "$v$ is the decayed sum of gradients, and the parameters move by $\eta v$." Then
state the first-step rule and the Nesterov look-ahead. The most likely follow-up is why the two
conventions differ. The answer is that they match only for a constant learning rate, and the PyTorch form
applies the current learning rate to the full buffer. A second follow-up asks where weight decay enters:
before the buffer, so momentum sees the penalty gradient. Full versions are in `interview.md`.

## Connections

This extends gradient descent and minibatch SGD by adding state. It is the first buffer-based optimizer
in the sequence. The next topic, Adam, keeps two buffers, one of gradients and one of squared gradients.
The coupled weight decay here is the same penalty as in ridge regression, and AdamW later separates the
two. Lineage: Polyak 1964, then Nesterov's look-ahead, then momentum in deep learning.
