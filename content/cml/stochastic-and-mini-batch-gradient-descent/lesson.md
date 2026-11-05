---
title: Stochastic and mini-batch gradient descent
minutes: 12
min_viable: "Read The system and the Worked example, then solve p01."
---

## The system

For least squares, $L(\mathbf{w}) = \tfrac12\operatorname{mean}_i (\mathbf{x}_i^\top\mathbf{w} - y_i)^2$,
with $\mathbf{x}_i\in\mathbb{R}^d$ the rows of $X\in\mathbb{R}^{n\times d}$. A **batch** $\mathcal{B}$ is a set
of $m=|\mathcal{B}|$ rows, and its gradient is the average of the per-row gradients:

$$\mathbf{g}_{\mathcal{B}} = \frac{1}{m}\sum_{i\in\mathcal{B}} \mathbf{x}_i\,(\mathbf{x}_i^\top\mathbf{w} - y_i) = \frac{1}{m}X_{\mathcal{B}}^\top(X_{\mathcal{B}}\mathbf{w} - \mathbf{y}_{\mathcal{B}}), \qquad \mathbf{w} \leftarrow \mathbf{w} - \eta\,\mathbf{g}_{\mathcal{B}}.$$

An **epoch** is one pass over all $n$ rows. With batch size $B$ it makes $\lceil n/B\rceil$ updates.
The three regimes differ only in $B$:

| Regime | Batch size | Cost per update | Updates per epoch | Gradient noise |
|---|---|---|---|---|
| Full batch | $B=n$ | $O(nd)$ | 1 | none |
| Mini-batch | $1<B<n$ | $O(Bd)$ | $\lceil n/B\rceil$ | falls like $1/B$ |
| Stochastic | $B=1$ | $O(d)$ | $n$ | largest |

Compute per epoch is about $O(nd)$ in every regime. $B$ only decides how that work is split into updates.

## Worked example

Take $X = (1, 2, 3, 4)^\top$, $\mathbf{y} = 2X$, $w_0 = 0$, $\eta = 0.1$, and batch size $B = 2$. Suppose
the epoch's shuffle gives the row order $[2, 0, 3, 1]$ (rows indexed from 0).

1. Batch $\{2, 0\}$: rows $(3, 6)$ and $(1, 2)$. Gradient $= \tfrac12[3(0-6) + 1(0-2)] = -10$, so
   $w_1 = 0 - 0.1(-10) = 1.0$.
2. Batch $\{3, 1\}$: rows $(4, 8)$ and $(2, 4)$. Gradient $= \tfrac12[4(4-8) + 2(2-4)] = -10$, so
   $w_2 = 1.0 + 1.0 = 2.0$.

One epoch of mini-batches reaches the true weight $2$. A full-batch step on the same data costs the
same compute, $4$ row-gradients, but makes one update: gradient $=\tfrac14\sum_i x_i(0-y_i) = -15$, so
$w = 1.5$.

## Why it works

**Unbiased.** If each row is equally likely to be in the batch, $\mathbb{E}[\mathbf{g}_{\mathcal{B}}] =
\nabla L(\mathbf{w})$. Shuffling makes every epoch use each row exactly once, so the batch gradients of one
epoch, evaluated at the same $\mathbf{w}$, add up to the full gradient. Because $\mathbf{w}$ moves between
batches, an epoch is only approximately one full-gradient step.

**Variance.** The per-row gradient has variance $\sigma^2$ around $\nabla L$. Sampling a batch without
replacement gives

$$\operatorname{Var}(\mathbf{g}_{\mathcal{B}}) = \frac{\sigma^2}{B}\cdot\frac{n-B}{n-1},$$

the finite-population correction, which is $0$ at $B = n$. For $0,\dots,19$ ($\sigma^2 = 33.25$, $n=20$, $B=5$),
enumerating all $\binom{20}{5}$ batches gives $5.25$, as the formula predicts; the naive $\sigma^2/B = 6.65$ overstates it.

**Steps.** A batch's own curvature $\lambda_{\max}(X_{\mathcal{B}}^\top X_{\mathcal{B}}/m)$ can exceed the
full-data value: the sharpest size-2 batch (values 3 and 4) has $12.5$, against $7.5$ for all rows. Requiring $\eta<2/\lambda$ on every batch is sufficient, not necessary, for bounded iterates.

**Noise floor.** [source: library-refresh/01-Mathematics/08-Optimization/2018 -- Bottou, Curtis and Nocedal -- Stochastic Gradients, Noise and Computation -- Study Note.md] Assume $F$ is $L$-smooth and bounded below, batch gradients are unbiased with variance at most $\sigma^2/b$, and $\alpha\le 1/L$. Telescoping the expected descent inequality (per the Library note, Assumption 4.1 and Lemma 4.2) gives

$$\frac1T\sum_{k<T}\mathbb E\lVert\nabla F(\mathbf w_k)\rVert^2\le\frac{2(F(\mathbf w_0)-F_{\inf})}{\alpha T}+\frac{L\alpha\sigma^2}{b}.$$

The second term does not shrink with $T$; halving $\alpha$ halves it but doubles the steps needed for the first. This is a stationarity statement, not a global-minimum guarantee.

**Diminishing steps.** The Robbins–Monro conditions, $\sum_k\alpha_k=\infty$ and $\sum_k\alpha_k^2<\infty$ (per the Library note, §4.2), let the floor shrink to zero. Take $F(w)=w^2/2$ with noise variance $\sigma^2=1$ and $w_0=0$. The second moment obeys $v_{k+1}=(1-\alpha_k)^2v_k+\alpha_k^2\sigma^2$ exactly. A constant $\alpha=0.1$ settles at $\alpha\sigma^2/(2-\alpha)=0.0526$; a seeded simulation (`default_rng(0)`, 20,000 paths, 400 steps) gives $0.0528$, within Monte Carlo error. The step $\alpha_k=1/(k+1)$ gives exactly $v_k=\sigma^2/k$, which goes to zero only like $1/k$.

**Work, not iterations.** Count row-gradient evaluations: $n$ per full gradient, $B$ per batch (Library note §4.4 compares methods at equal computation). Stochastic steps are cheap early on large data, but their floor limits final accuracy unless the steps decay.

## Signal vs. noise

- **Signal:** updates per epoch are $\lceil n/B\rceil$; reshuffle every epoch; divide by the batch's own size.
- **Signal:** scaling $\eta$ with $B$ is a heuristic, not a bound. A non-finite or oscillating loss says $\eta$ is too large.
- **Signal:** a constant learning rate with noisy data does not converge to the exact minimiser. It
  hovers in a floor that scales like $\alpha/b$ (exact for the quadratic example). Noise-free data is the exception: its floor is zero.
- **Noise for now:** momentum, Adam, schedules, parallel training. They change the update, not the trade-off.

## Where it breaks

- **Constant step, noisy data.** Toy problem: $n=40$, $d=2$, batch 4, 200 epochs, true weights $(1,-0.5)$, noise scale 0.5.
  Standard-normal rows and noise come from `default_rng(0)`; shuffle $s$ uses `default_rng(s)`, $s=0,\dots,4$.
  Distance from the final weights to the least-squares fit, averaged over the 5 shuffles: $0.142$ at $\eta=0.4$,
  $0.024$ at $\eta=0.1$ and $0.0022$ at $\eta=0.02$. A smaller step lowers the floor at the cost of more updates; decay the rate.
- **Sorted or grouped data.** A column of ones, $n=20$, batch 5, $\eta=0.5$, targets 1 for the
  first half of the rows and 3 for the second. One sorted epoch from $w_0=0$ ends at $2.44$ (mean target 2). Over 200 random orders
  (one `rng.permutation(20)` each, `default_rng(0)`), the end-of-epoch weight is $1.87 \pm 0.15$.
- **Shuffle once.** Reusing one order repeats the same batch sequence, and any order bias with it. Shuffling does not matter for full-batch GD, whose gradient ignores row order.

## Interview angle

"Why mini-batch rather than full batch?" Answer shape: an update costs $O(Bd)$ instead of $O(nd)$,
the gradient is unbiased with variance falling like $1/B$, and an epoch costs the same compute either
way. Tune batch size and learning rate together, checking the sharpest batch. Follow-ups: the linear scaling heuristic, and why reshuffling matters. Full answers are
in `interview.md`.

## Connections

- **Prerequisites:** `cml/gradient-descent` (the update and its stability bound).
- **Unlocks:** `cml/shuffling-splitting-and-batching-data`, `cml/over-and-underfitting-early-stopping`
  (the noise floor is why validation is needed).
- **Lineage:** Robbins & Monro (1951) introduced stochastic approximation.
