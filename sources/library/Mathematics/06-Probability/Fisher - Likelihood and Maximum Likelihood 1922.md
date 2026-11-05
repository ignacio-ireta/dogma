# R. A. Fisher — Likelihood and Maximum Likelihood (1922)

#person #paper #milestone

**Person:** Ronald Aylmer Fisher (1890–1962)  
**Work:** *On the Mathematical Foundations of Theoretical Statistics* (1922)  
**Primary source:** *Philosophical Transactions of the Royal Society of London, Series A*, 222, 309–368  
**DOI:** [10.1098/rsta.1922.0009](https://doi.org/10.1098/rsta.1922.0009)  
**Repository record:** [University of Adelaide, handle 2440/15172](https://hdl.handle.net/2440/15172)  
**Paper:** `corpus/raw/library-papers/fisher-theoretical-statistics-1922.pdf`  
**Source note:** The repository supplies the paper as two scanned PDF parts. Both were downloaded directly, verified, and joined without reordering into the canonical 61-page local PDF above; the two download artifacts remain beside it as `-part1.pdf` and `-part2.pdf` for provenance.

## Title
Probability becomes an objective that a model can optimize against data.

Gauss had shown that one particular error model and one particular fitting rule fit together beautifully: Gaussian errors lead to least squares. But a general statistical theory still lacked an answer to a broader problem:

> Given a family of possible probability models, which parameter value should the observed sample make us choose?

Consistency alone could not answer it. Many statistics approach the right population quantity while throwing away different amounts of information. Fisher's 1922 paper turns the probability of the observed data, viewed as a function of the unknown parameter, into a general criterion for estimation. That function is **likelihood**; choosing its maximum is **maximum likelihood estimation**.

This is the missing bridge from probability distributions to trainable probabilistic models.

## Paper Reference
**Fisher, R. A. (1922).** “On the Mathematical Foundations of Theoretical Statistics.” *Philosophical Transactions of the Royal Society of London, Series A*, **222**, 309–368.

Local source:
- Canonical paper: `corpus/raw/library-papers/fisher-theoretical-statistics-1922.pdf`
- Downloaded source parts retained for provenance: `fisher-theoretical-statistics-1922-part1.pdf` and `fisher-theoretical-statistics-1922-part2.pdf`

The canonical PDF has 61 pages. The page references below use the original journal pagination printed on the scans.

## What Problem Made This Necessary?
By the early twentieth century, statisticians had many estimators: means, moments, medians, least-squares fits, correlation coefficients, and fitted frequency curves. What they did not have was a single principle that could decide which summary extracted the relevant evidence in a sample most effectively.

Fisher states the problem as **data reduction**: replace a mass of observations by a few quantities while retaining, ideally, all information relevant to the population parameters (p. 311). He then separates statistical work into three problems (pp. 313–314):

1. **Specification:** choose the family of probability distributions.
2. **Estimation:** infer its unknown parameters from a sample.
3. **Distribution:** determine how the resulting statistics vary across repeated samples.

This separation matters. A model family is not an estimate; a parameter is not the statistic used to estimate it; and a rule that is correct in the infinite-population limit need not be the best use of finite data.

## Key Ideas

### 1. Consistency is necessary, but not enough
Fisher first asks that an estimator give the population parameter when applied to the whole population—his 1922 **criterion of consistency** (p. 316). Modern theory states consistency more precisely as convergence of the estimator to the true parameter as sample size grows, but the motivating defect is the same.

Two estimators may both become correct and still have very different uncertainty at any finite sample size. Fisher's heavy-tailed example is deliberately brutal: the sample mean can satisfy an appealing moment rule yet fail to become more informative as more observations are added (pp. 321–323). Correct destination is not enough; the route must use the evidence well.

### 2. Efficiency compares how much sampling precision estimators extract
For large samples, Fisher calls an estimator more **efficient** when it has smaller sampling variation. If one method needs four times as many observations to match another method's precision, it is using only one quarter of the available information in Fisher's operational sense (p. 316).

This turns estimator choice into a quantitative comparison rather than custom or algebraic convenience.

### 3. Sufficiency asks whether a summary discards parameter-relevant evidence
Fisher's strongest criterion is **sufficiency**: a statistic should summarize the whole of the sample's information relevant to the parameter (pp. 316–317).

In the paper's formulation, once a sufficient statistic is known, the conditional distribution of any other statistic no longer depends on the unknown parameter. Looking back at the raw data cannot teach us anything further about that parameter.

```text
raw sample x  ->  statistic T(x)

T is sufficient for theta when the discarded detail carries
no additional information about theta.
```

Sufficiency is therefore a precise version of lossless, task-specific compression.

### 4. Likelihood reverses the role of data and parameter without making the parameter random
For independent observations with model density or mass function $f(x_i; θ)$, the likelihood is

```text
L(theta; x) = product_i f(x_i; theta).
```

The data are fixed at what was observed; $L$ is compared across candidate parameter values. Fisher insists that likelihood is not a probability distribution over $θ$: it need not integrate to one and only relative values matter (pp. 326–327).

That distinction repaired an ambiguity in “inverse probability.” It survives intact in modern modeling: $p_θ(x)$ is a probability over possible data for fixed $θ$, while $L(θ;x)$ is the same expression treated as a score over parameters for fixed data.

Fisher's criticism of uniform priors under reparameterization belongs to the frequentist–Bayesian dispute of his time; it is not a refutation of modern Bayesian inference. The durable contribution is the clean separation of probability, prior assumptions, and likelihood.

### 5. Maximum likelihood gives a general estimation rule
Fisher's formal rule is simple: choose the parameter value that makes the observed sample most likely (pp. 323–324).

```text
theta_hat = argmax_theta L(theta; x)
          = argmax_theta sum_i log f(x_i; theta).
```

The logarithm preserves the maximizer while turning products into sums. That makes independent observations additive, derivatives manageable, and large computations numerically possible.

For $x$ successes and $y$ failures in Bernoulli trials, Fisher obtains

```text
log L(p) = x log p + y log(1-p),
```

whose maximizer is $p̂=x/(x+y)$ (p. 324). This is already the algebra of binary cross-entropy, read as maximization rather than minimization.

### 6. The optimum survives a change of parameter coordinates
Fisher notes that the maximum-likelihood point is unchanged by a one-to-one reparameterization (p. 327). A coordinate label may change, but the selected model does not.

This matters whenever modern models parameterize the same probability in different ways—for example, probability $p$, log-odds, or a neural-network logit.

### 7. Likelihood curvature measures local estimability
Near the maximum, a sharply curved log-likelihood distinguishes nearby parameter values strongly; a flat one does not. Fisher derives the large-sample precision of an optimum statistic from the second derivative of log likelihood (pp. 328–329). This develops into **Fisher information**:

```text
I(theta) = E[(d/dtheta log p_theta(X))^2]
         = -E[d²/dtheta² log p_theta(X)]
```

under suitable regularity conditions.

The score and its curvature later support asymptotic standard errors, information geometry, natural-gradient methods, and curvature-aware optimization.

## Historical Context
Fisher was writing against the biometric tradition associated with Karl Pearson, especially routine fitting by moments. He did not invent every ingredient of maximum likelihood from nothing: inverse-probability arguments, least squares, and likelihood-like comparisons had substantial prehistory, and Fisher himself had used the method earlier. The 1922 paper's decisive role is synthesis and foundation.

It gives estimation a vocabulary—parameter, statistic, likelihood, consistency, efficiency, sufficiency—and a general optimizing principle. Fisher is candid that his proof that maximum likelihood must always yield a sufficient statistic is incomplete (p. 323). Later mathematical statistics would sharpen the regularity conditions, separate finite-sample from asymptotic claims, and show where MLE can fail.

That limitation strengthens the lineage: the paper does not end estimation theory. It makes the next theorems necessary.

## Lineage

```text
Bayes 1763 / Laplace 1812
(probability used for inference)
  └── Gauss 1809
      (Gaussian error + least-squares fitting)
      └── Fisher 1922
          (likelihood, MLE, sufficiency, information)
          ├── likelihood theory and asymptotic statistics
          ├── negative log-likelihood as an optimizable loss
          ├── Gaussian NLL -> mean squared error
          ├── Bernoulli NLL -> binary cross-entropy
          └── categorical NLL -> softmax cross-entropy
              └── probabilistic neural networks and language models

Lebesgue 1902 -> Kolmogorov 1933 -> rigorous probability/expectation
Fisher 1922 + Shannon 1948 -> log loss, cross-entropy, information geometry
```

- **Built on:** [[Gauss - Least Squares and Normal Errors 1809]]—one model-specific fit becomes a general likelihood principle—and [[Laplace - Théorie Analytique des Probabilités 1812]], whose inverse-probability program Fisher explicitly confronts.
- **Strengthened by:** [[Kolmogorov - Foundations of the Theory of Probability 1933]], which later gives random variables, distributions, and expectations their measure-theoretic foundation.
- **Joins:** [[../09-Information-Theory/Shannon - Information Theory 1948|Shannon 1948]], where expected log loss becomes cross-entropy and excess expected log loss becomes relative entropy.
- **Feeds:** [[../11-Optimization-Theory/Cauchy - Steepest Descent 1847|Cauchy 1847]] and [[../11-Optimization-Theory/Robbins-Monro - Stochastic Approximation 1951|Robbins–Monro 1951]], because realistic likelihood objectives often require iterative and then sample-based optimization.
- **Crosses into AI/ML:** probabilistic classifiers, neural language models, softmax training, variational inference, natural gradients, and nearly every model trained by negative log-likelihood.

## Why It Matters for AI
Fisher supplies the move that turns a probabilistic model into a trainable system:

```text
choose p_theta
-> observe data
-> score theta by log p_theta(data)
-> differentiate the score
-> optimize theta
```

Modern supervised learning repeatedly instantiates this template:

- Gaussian output model → minimize squared error.
- Bernoulli output model → minimize binary cross-entropy.
- Categorical output model → minimize softmax cross-entropy.
- Autoregressive language model → minimize summed next-token negative log-likelihood.

For a language model with tokens $t_1, …, t_n$, the training objective is

```text
-log L(theta) = -sum_i log p_theta(t_i | t_<i).
```

That is Fisher's additive log-likelihood, Shannon's measure of surprise, Leibniz's chain rule, and stochastic optimization living in one training loop.

Sufficiency poses a recognizably modern representation-learning question: can a compressed representation preserve everything relevant to a target while discarding nuisance detail? Neural embeddings rarely satisfy classical sufficiency exactly, but the mathematical ideal remains a north star for learned representations and information bottlenecks.

Finally, Fisher information treats nearby models as distinguishable to different degrees. That geometry later motivates natural gradient: move through parameter space according to how much the model distribution changes, not merely according to Euclidean distance between parameter vectors.

## What Fisher Did Not Yet Settle
MLE is powerful, not magical.

- A misspecified model can fit the wrong family very confidently.
- Finite samples can yield bias, multiple maxima, boundary solutions, or no attained maximum.
- Fisher's large-sample arguments require regularity conditions.
- Sufficiency is model-relative: change the assumed family and the “irrelevant” data may become relevant.
- Likelihood alone does not encode prior knowledge or decision costs.

These are the problems that force later robust statistics, Bayesian computation, regularization, generalization theory, and model criticism.

## References
- Fisher, R. A. (1922). “On the Mathematical Foundations of Theoretical Statistics.” *Philosophical Transactions of the Royal Society of London, Series A*, 222, 309–368. DOI: 10.1098/rsta.1922.0009. Local paper: `corpus/raw/library-papers/fisher-theoretical-statistics-1922.pdf`.
- University of Adelaide Research & Scholarship, repository record 2440/15172. The record states that the paper is reproduced with permission of the Royal Society.
- [[Gauss - Least Squares and Normal Errors 1809]]
- [[Kolmogorov - Foundations of the Theory of Probability 1933]]
- [[../09-Information-Theory/Shannon - Information Theory 1948]]

## Status
- [x] Note created (September 9, 2026)
- [x] Source parts downloaded and verified; canonical 61-page PDF assembled and verified
- [x] Closes the probability-to-trainable-likelihood gap
- [x] Next link supplied: [[../09-Information-Theory/Kullback-Leibler - Information Divergence 1951|Kullback–Leibler divergence]] turns expected log-likelihood regret into distribution mismatch and cross-entropy

---

*Gauss showed that one noise model implies one fitting rule. Fisher made the rule general: let the observed data score the model, then optimize the score.*
