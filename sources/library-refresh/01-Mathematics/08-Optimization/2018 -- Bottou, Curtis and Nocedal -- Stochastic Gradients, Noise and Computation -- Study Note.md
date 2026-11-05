# Stochastic Gradients, Noise and Computation

**Work:** Léon Bottou, Frank E. Curtis, and Jorge Nocedal, *Optimization Methods for Large-Scale Machine Learning* (2018). The copy used is their June 23, 2018 manuscript; its page numbers differ from the SIAM Review article.

A full gradient can be accurate and still be a poor use of computation. If a training set is large, evaluating every example before making one move delays learning. Stochastic optimization trades the accuracy of each direction for cheaper, more frequent updates. The central question is how curvature, noise, and computational cost interact.

Write
$$
F(w)=\mathbb E_\xi[f(w;\xi)],\qquad
w_{k+1}=w_k-\alpha_k g_k.
$$
For the basic analysis, assume $F$ is bounded below, its gradient is $L$-Lipschitz, and, conditional on the current history,
$$
\mathbb E[g_k]=\nabla F(w_k),\qquad
\mathbb E\|g_k-\nabla F(w_k)\|^2\le\sigma^2/b.
$$
The $1/b$ bound follows when a minibatch averages $b$ conditionally independent, unbiased estimates with bounded variance. Correlation or a different sampling scheme can change this bound. Unbiasedness itself requires a justified exchange of differentiation and expectation; [dominated convergence](../04-Analysis-and-Measure/2026%20--%20Library%20--%20Dominated%20Convergence%20and%20Gradients%20of%20Expectations%20--%20Concept%20Note.md) supplies one route.

Smoothness gives a quadratic upper bound. Substitution of the update and conditional expectation yield, for $\alpha_k\le1/L$,
$$
\mathbb E[F(w_{k+1})\mid w_k]
\le F(w_k)-\frac{\alpha_k}{2}\|\nabla F(w_k)\|^2
+\frac{L\alpha_k^2\sigma^2}{2b}.
$$
The first extra term rewards a descent signal. The second charges for noise. A sampled update need not reduce the actual loss, even when the expected bound is useful.

For constant $\alpha$ over $T$ iterations, telescope this inequality:
$$
\frac1T\sum_{k=0}^{T-1}\mathbb E\|\nabla F(w_k)\|^2
\le\frac{2(F(w_0)-F_{\inf})}{\alpha T}
+\frac{L\alpha\sigma^2}{b}.
$$
This elementary specialization explains a familiar tradeoff. A smaller step suppresses the noise term but slows the transient. Choosing $\alpha$ proportional to $T^{-1/2}$ gives an $O(T^{-1/2})$ bound on the average squared gradient under these assumptions. This is a stationarity guarantee, not a theorem that a nonconvex network reaches its global minimum.

For a quadratic example $F(w)=w^2/2$, take $g_k=w_k+\varepsilon_k$ with zero-mean independent noise. Then
$$
w_{k+1}=(1-\alpha)w_k-\alpha\varepsilon_k.
$$
For $0<\alpha<2$, the mean contracts, but persistent noise leaves a nonzero stationary variance. Diminishing steps can reduce that floor; the Robbins–Monro conditions $\sum\alpha_k=\infty$, $\sum\alpha_k^2<\infty$ are part of convergence arguments, alongside the problem's other regularity conditions.

Large batches reduce variance per update while increasing work per update. Variance-reduction methods exploit finite-sum structure; curvature methods change the geometry of a direction. Neither improvement is captured by counting iterations alone. Sampling uniformly with replacement from a fixed dataset and reshuffling once per epoch are also distinct processes; one should not transfer an independence-based proof silently between them.

The conceptual route is noisy root finding → sampled gradients → finite-computation training. [Deterministic acceleration](2026%20--%20Library%20--%20Proximal%20Steps%20and%20Accelerated%20Convex%20Optimization%20--%20Concept%20Note.md) addresses another limitation and carries different hypotheses. This note supplies the missing bridge from an ideal gradient to an update computed from a minibatch.

## Sources and locators

- Bottou, Curtis, and Nocedal, Optimization Methods for Large-Scale Machine Learning, §3 (risk and computational setting); Algorithm 4.1, manuscript p.22; Assumption 4.1 and Lemma 4.2, p.23; Assumption 4.3, p.24; §4.2 fixed/diminishing steps; §4.4 computation; chapter 5 variance reduction. [Inspected author PDF](https://leon.bottou.org/publications/pdf/tr-optml-2016.pdf); [author publication metadata](https://coral.ise.lehigh.edu/frankecurtis/publications/).
- Bubeck, Convex Optimization - Algorithms and Complexity, §§6.1–6.3, printed pp.330–338, treats stochastic convex optimization and minibatches. [Versioned source](https://arxiv.org/abs/1405.4980v2). Convex objective-gap results there are distinct from the nonconvex stationarity calculation above.
- Axler, Measure, Integration & Real Analysis, theorem 3.31, printed p.92, for the analytic interchange prerequisite. [Author edition](https://measure.axler.net/).

