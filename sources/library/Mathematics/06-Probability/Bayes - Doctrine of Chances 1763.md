# Thomas Bayes — Doctrine of Chances (1763)

#person #paper #milestone

**Person:** Thomas Bayes (c. 1701–1761)
**Work:** *An Essay towards solving a Problem in the Doctrine of Chances* (published 1763, posthumously by Richard Price)
**Primary source:** Philosophical Transactions of the Royal Society, 53, 370-418
**Link:** https://royalsocietypublishing.org/rstl/article/doi/10.1098/rstl.1763.0053/119736/LII-An-essay-towards-solving-a-problem-in-the

## The Core Question

Bayes asked a deceptively simple question:

> If we observe an event happening some number of times, how should we update our belief about the underlying probability that generates it?

That is the birth of **inverse probability**. Not just, “if I know the probability, what outcomes should I expect?” but the harder and more human question: **given outcomes, what should I now believe about the hidden cause?**

This is the intellectual move that makes modern statistics, Bayesian inference, and a huge portion of machine learning possible.

## Key Ideas

### 1. Probability becomes belief under uncertainty

Classical probability before Bayes mostly ran forward: assume a known chance setup, then compute outcome probabilities.

Bayes turned the arrow around. He treated the unknown probability itself as something we can reason about. In modern language:
- start with a **prior** belief about a parameter,
- observe data,
- update to a **posterior** belief.

That sounds ordinary now because Bayes won. In 1763, it was a conceptual rupture.

### 2. Evidence should change belief proportionally

What we now call **Bayes' theorem** says:

**posterior ∝ likelihood × prior**

In words:
- how much you should believe a hypothesis *after* seeing evidence
- equals how compatible that evidence is with the hypothesis
- times how plausible the hypothesis already was.

This is the grammar of rational updating.

### 3. The hidden parameter becomes a distribution

Bayes considered repeated success/failure trials with an unknown success probability `p`.
After observing data, the uncertainty about `p` is no longer vague. It takes a specific mathematical form.

In modern terms, with a uniform prior and `k` successes in `n` trials, the posterior is:
- **Beta(k + 1, n - k + 1)**

That is one of the cleanest examples in all of statistics. It is also the seed of a deep idea: **learning is distribution update**.

### 4. Probability is no longer only about gambling

The title says “doctrine of chances,” which sounds like lotteries and dice. But the actual importance is broader.
Bayes gave mathematics a way to reason from observations to causes, from data to belief, from noisy evidence to action.

That move reaches everywhere:
- medical diagnosis
- spam filtering
- A/B testing
- fraud detection
- forecasting
- probabilistic ML
- scientific inference itself

## The Lineage

Bayes opens the probability thread that AI desperately needs:

```text
Bayes 1763 (inverse probability)
  └── Laplace 1774-1812 (general Bayesian inference, celestial mechanics)
      └── Gauss 1809 (normal errors, least squares)
          └── Kolmogorov 1933 (axiomatized probability)
              └── Bayesian statistics (20th century)
                  ├── Naive Bayes classifiers
                  ├── Bayesian networks
                  ├── Kalman filters
                  ├── Gaussian processes
                  └── probabilistic machine learning
```

Bayes is not the whole probability story. Kolmogorov later gave the field its formal foundations. But Bayes gave it one of its most important philosophical engines: **belief should update when reality speaks**.

## Why It Matters for AI/ML

This note plugs a real hole in the Library. The AI thread already points toward probability, but without Bayes the connection was hanging in midair.

Bayes matters because modern AI is not only about function approximation. It is also about **uncertainty**.

### Direct descendants in ML
- **Naive Bayes** — a simple baseline that still wins in text classification surprisingly often
- **Bayesian optimization** — how we tune expensive models intelligently
- **Probabilistic graphical models** — reasoning under uncertainty with structured dependencies
- **Gaussian processes** — prediction with calibrated uncertainty
- **Modern LLM evaluation and agent systems** — confidence, calibration, and decision under uncertainty are all Bayesian-shaped problems, even when the implementation is not explicitly Bayesian

## Why It Matters for Mathematics

This is the moment mathematics stops only describing idealized structures and starts formalizing **learning from evidence**.

Archimedes gave us rigorous approximation.
Descartes gave us coordinates.
Cauchy and Weierstrass gave us rigor about limits.
Bayes gives us rigor about **uncertainty**.

If calculus is the mathematics of change, Bayes is one of the foundations of the mathematics of belief.

## Subtle but Important Detail

Bayes did not present the polished modern theorem exactly as textbooks do. The essay was edited and communicated after his death by Richard Price, and later Laplace generalized and operationalized the framework much more aggressively.

So the historical truth is a little messier than the myth.
That is fine. Lineage usually is.

## Leads To
- [[../_MOC|Mathematics Lineage]]
- [[../../AI-ML/_MOC|AI/ML Lineage]]
- [[04-Analysis/Cauchy Limits 1821|Cauchy]] — rigor for convergence and approximation, which probability theory later depends on
- **Laplace** — the great expansion of Bayesian inference
- **Kolmogorov** — axiomatic probability
- **Gauss** — error models, least squares, the normal distribution

## References
- Bayes, T. (1763). *An Essay towards solving a Problem in the Doctrine of Chances.* Philosophical Transactions of the Royal Society, 53, 370-418.
- Stigler, S. M. (1982). Thomas Bayes's Bayesian Inference. *Journal of the Royal Statistical Society. Series A*.
- Wikipedia summary used only for quick orientation on publication history and the modern Beta-distribution formulation.

## Status
- [ ] Add Laplace note, probability thread proper begins there
- [ ] Add Kolmogorov 1933 note, formal foundations of probability
- [ ] Add Gauss note on least squares + normal error model
- [ ] Add a cross-link to any future Naive Bayes or Bayesian networks note in AI/ML

---

*Bayes gave mathematics a disciplined way to say: I have seen something, therefore I should believe differently now.*
