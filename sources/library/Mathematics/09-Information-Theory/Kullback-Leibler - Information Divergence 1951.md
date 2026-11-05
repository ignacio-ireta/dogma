# Solomon Kullback & Richard Leibler — Information Divergence (1951)

#person #paper #milestone

**People:** Solomon Kullback (1907–1994) and Richard A. Leibler (1914–2003)  
**Work:** *On Information and Sufficiency* (1951)  
**Primary source:** *The Annals of Mathematical Statistics*, 22(1), 79–86  
**DOI:** [10.1214/aoms/1177729694](https://doi.org/10.1214/aoms/1177729694)  
**Publisher record:** [Project Euclid](https://projecteuclid.org/journals/annals-of-mathematical-statistics/volume-22/issue-1/On-Information-and-Sufficiency/10.1214/aoms/1177729694.full)  
**Paper:** `corpus/raw/library-papers/kullback-leibler-information-sufficiency-1951.pdf`  
**Source note:** The local eight-page article was extracted without reordering from the Internet Archive scan of *The Annals of Mathematical Statistics*, volume 22, issue 1 (journal pp. 79–86; scan PDF pp. 81–88). The canonical Project Euclid PDF endpoint was also checked but rejected automated retrieval through its anti-bot gateway. The local file is a verified PDF document.

## Title
The cost of mistaking one probability distribution for another.

Shannon had quantified the uncertainty inside one source. Fisher had turned the probability of observed data into a likelihood over candidate parameters. But neither idea, by itself, answered the comparison question that learning makes unavoidable:

> If data really follow distribution $P$, how much information is lost when we represent, compress, or predict them as though they followed $Q$?

Kullback and Leibler's answer is the expected log-likelihood ratio. It measures how much evidence observations drawn from $P$ provide against $Q$. Just as importantly, they prove that processing data cannot increase this discriminatory information, and that no information is lost exactly when the retained statistic is sufficient.

This is the missing hinge between Shannon entropy, Fisher likelihood, and the cross-entropy objectives used to train modern probabilistic models.

## Paper Reference
**Kullback, S., & Leibler, R. A. (1951).** “On Information and Sufficiency.” *The Annals of Mathematical Statistics*, **22**(1), 79–86.

Local source:
- `corpus/raw/library-papers/kullback-leibler-information-sufficiency-1951.pdf`

The article's printed page numbers are used below. Its notation is measure-theoretic: two probability measures $\mu_1$ and $\mu_2$ have densities $f_1$ and $f_2$ relative to a common dominating measure. Modern notation writes the same objects as distributions $P$ and $Q$ with densities or masses $p$ and $q$.

## What Problem Made This Necessary?
Shannon's 1948 entropy

$$
H(P)=-\sum_x p(x)\log p(x)
$$

measures uncertainty or optimal average code length for a source considered on its own. It does not directly measure the penalty for using the wrong source model.

Fisher's likelihood compares parameter values after one sample has been observed: $L(\theta;x)=p_\theta(x)$. For learning, we need a population-level statement: average the model's log score over the distribution that actually generates the data. We also need to know whether a statistic, grouping, embedding, or other transformation has discarded evidence that could distinguish competing models.

Kullback and Leibler join those problems. Their quantity compares distributions through expected evidence, then makes Fisher's informal phrase “the whole of the relevant information” mathematically testable.

## Key Ideas

### 1. A single observation carries a log-likelihood ratio
For hypotheses $H_1:P$ and $H_2:Q$, the paper defines the information in observation $x$ for discriminating $H_1$ from $H_2$ as (p. 80)

$$
\log\frac{p(x)}{q(x)}.
$$

Positive values favor $P$; negative values favor $Q$. Independent observations multiply likelihood ratios, so their logarithms add. The logarithm's base only changes units: base 2 gives bits; natural logarithms give nats.

### 2. Mean information is the directional divergence now called KL
Average the log-likelihood ratio under the distribution that supplies the observations:

$$
D_{\mathrm{KL}}(P\|Q)
=\mathbb E_{x\sim P}\!\left[\log\frac{p(x)}{q(x)}\right]
=\int p(x)\log\frac{p(x)}{q(x)}\,dx.
$$

Kullback and Leibler denote this $I(1:2)$ (equation 2.4, p. 80). Direction matters: $D_{\mathrm{KL}}(P\|Q)\neq D_{\mathrm{KL}}(Q\|P)$ in general. It is therefore not a metric. The paper separately defines the symmetric sum

$$
J(P,Q)=D_{\mathrm{KL}}(P\|Q)+D_{\mathrm{KL}}(Q\|P),
$$

building on Jeffreys's earlier divergence (equations 2.5–2.6, p. 81). Calling the directional quantity a “distance” would erase one of its most useful features: choosing which distribution supplies the expectation changes the question.

### 3. Divergence is nonnegative and vanishes only at agreement
Lemma 3.1 establishes what the paper calls “almost positive definite” (p. 82):

$$
D_{\mathrm{KL}}(P\|Q)\ge 0,
$$

with equality only when the densities agree almost everywhere. Individual log ratios can be negative; the guarantee appears after expectation under $P$.

### 4. Independent evidence adds
For independent components, Theorem 3.1 gives (p. 82)

$$
D_{\mathrm{KL}}(P_XP_Y\|Q_XQ_Y)
=D_{\mathrm{KL}}(P_X\|Q_X)+D_{\mathrm{KL}}(P_Y\|Q_Y).
$$

This is the population counterpart of summed log likelihood. It is why sample, token, and minibatch losses naturally accumulate.

### 5. Processing cannot create discriminatory information
Let $T(X)$ be a measurable transformation: a statistic, grouping, quantizer, feature map, or representation. Theorem 4.1 proves (pp. 83–84)

$$
D_{\mathrm{KL}}(P_X\|Q_X)
\ge D_{\mathrm{KL}}(P_{T(X)}\|Q_{T(X)}).
$$

This is an early form of the **data-processing inequality**. Once distinctions have been merged by a transformation, later computation cannot reconstruct evidence that is no longer present.

### 6. Equality characterizes sufficiency
The same theorem states that equality holds exactly when $T$ is a sufficient statistic for the two distributions. A sufficient statistic can compress an observation while retaining all information needed to discriminate $P$ from $Q$.

```text
raw observation X -> representation T(X)

KL after <= KL before
equality <=> T is sufficient
```

This repairs an ambiguity left by Fisher. “Contains all relevant information” becomes invariance of discriminative information under transformation, not merely an intuition about estimator variance.

### 7. Locally, KL recovers Fisher information geometry
For nearby members $p_\theta$ and $p_{\theta+\Delta\theta}$ of a smooth family, the paper expands divergence to second order (equations 2.7–2.9, p. 81). In modern notation,

$$
D_{\mathrm{KL}}(p_\theta\|p_{\theta+\Delta\theta})
\approx \frac12\Delta\theta^\top F(\theta)\Delta\theta,
$$

where $F(\theta)$ is the Fisher information matrix. Fisher information is therefore the local curvature of distributional distinguishability—the geometry later used by natural-gradient methods.

### 8. Cross-entropy is entropy plus mismatch cost
The term **cross-entropy** is not the paper's main notation, but the modern identity follows directly:

$$
H(P,Q)=-\mathbb E_P[\log q(X)]
=H(P)+D_{\mathrm{KL}}(P\|Q).
$$

For fixed data-generating $P$, $H(P)$ does not depend on model $Q$. Thus minimizing expected negative log likelihood minimizes $D_{\mathrm{KL}}(P\|Q)$. Fisher supplies the likelihood, Shannon supplies the code-length interpretation, and Kullback–Leibler supplies the excess cost of believing the wrong model.

## Historical Context
The paper explicitly presents itself as an abstract generalization of Shannon's information and distinguishes that concept from Fisher's “intrinsic accuracy” (p. 79). The Radon–Nikodym theorem lets it cover discrete and continuous probability spaces in one measure-theoretic language.

Kullback and Leibler were not the first to compare distributions. They cite Bhattacharyya, Jeffreys, Mahalanobis, and work on statistical discrimination. Their decisive contribution is to unite expected log evidence, nonnegativity, additivity, contraction under processing, equality under sufficiency, and local Fisher geometry in one object.

The original article uses $I(1:2)$ for the directional quantity and $J(1,2)$ for its symmetric sum. Modern usage often calls $I(1:2)$ itself “KL divergence.” Keeping that distinction prevents a common historical error: they did not present the directional quantity as a symmetric geometric distance.

## Lineage

```text
Lebesgue 1902 -> Kolmogorov 1933
(integration and probability measures)
                |
Shannon 1948 ---------------- Fisher 1922
(entropy, bits, codes)         (likelihood, sufficiency)
                \             /
             Kullback-Leibler 1951
             (expected log ratio;
              data processing;
              sufficiency as equality)
                      |
                      +-> cross-entropy / negative log likelihood
                      +-> information geometry / natural gradient
                      +-> variational inference and ELBOs
                      +-> information bottleneck / representation learning
```

- **Built on:** [[../06-Probability/Fisher - Likelihood and Maximum Likelihood 1922|Fisher 1922]], because a log-likelihood ratio compares how two models score one observation.
- **Built on:** [[Shannon - Information Theory 1948]], because logarithmic information turns probability into additive code length and surprise.
- **Made rigorous by:** [[../06-Probability/Kolmogorov - Foundations of the Theory of Probability 1933|Kolmogorov 1933]] and [[../08-Set-Theory-Infinity/Lebesgue - Measure and Integration 1902|Lebesgue 1902]].
- **Feeds:** [[../11-Optimization-Theory/Robbins-Monro - Stochastic Approximation 1951|Robbins–Monro 1951]], because expected log loss must be estimated and optimized from samples.
- **Crosses into AI/ML:** VAEs, probabilistic classifiers, language-model cross-entropy, policy optimization, model distillation, information bottlenecks, and natural gradients.

## Why It Matters for AI
Most probabilistic learning can be read as distribution matching:

```text
unknown data distribution P
  -> samples x
  -> model Q_theta
  -> loss -log q_theta(x)
  -> sample average estimates H(P,Q_theta)
  -> optimization reduces KL(P || Q_theta)
```

### Language modeling
For tokens $t_1,\ldots,t_n$,

$$
-\log q_\theta(t_{1:n})=-\sum_i\log q_\theta(t_i\mid t_{<i}).
$$

Training minimizes empirical cross-entropy. Perplexity is its exponential. Under the population idealization, lower loss means smaller $D_{\mathrm{KL}}(P\|Q_\theta)$.

### Variational inference
When a posterior $p(z\mid x)$ is intractable, variational methods choose a tractable $q_\phi(z\mid x)$ and optimize a KL divergence. In a VAE, the evidence lower bound contains a reconstruction term and a KL regularizer. The direction matters: the two directions penalize missing modes and extra mass differently.

### Representation learning
Data processing says a deterministic representation cannot create distinctions absent from its input. Sufficiency asks for the ideal: compress heavily while preserving everything relevant to the task. Information bottlenecks, embeddings, and feature selection inherit this tension.

### Natural gradient
Because local KL curvature is Fisher information, natural gradient scales an update by the geometry of the model distribution rather than raw coordinate distance. The aim is a controlled change in predictions, not merely a small change in weights.

### The finite-data warning
Training cannot access the true $P$; it minimizes sample cross-entropy. Empirical loss can fall while population KL rises through overfitting or distribution shift. KL gives the population target. Generalization theory is still needed to justify the leap from observed samples to unseen data.

## What This Paper Did Not Yet Settle
- KL can be infinite when $Q$ assigns zero probability where $P$ assigns positive probability.
- It is not symmetric and not a metric; some tasks need Jensen–Shannon, total variation, Wasserstein, or another discrepancy.
- Data processing does not decide which task-relevant information a learned representation ought to preserve.
- Population KL does not solve finite-sample generalization, optimization, or model misspecification.
- Cross-entropy selects a divergence and a direction with characteristic behavior; it is not a neutral notion of “closeness.”

These limitations explain why later AI needs variational bounds, contrastive objectives, calibration, robust statistics, alternative divergences, and generalization theory.

## References
- Kullback, S., & Leibler, R. A. (1951). “On Information and Sufficiency.” *The Annals of Mathematical Statistics*, 22(1), 79–86. DOI: 10.1214/aoms/1177729694. Local paper: `corpus/raw/library-papers/kullback-leibler-information-sufficiency-1951.pdf`.
- Internet Archive issue scan: `sim_annals-of-mathematical-statistics_1951-03_22_1`.
- [[Shannon - Information Theory 1948]]
- [[../06-Probability/Fisher - Likelihood and Maximum Likelihood 1922|Fisher - Likelihood and Maximum Likelihood 1922]]
- [[../06-Probability/Kolmogorov - Foundations of the Theory of Probability 1933|Kolmogorov - Foundations of Probability 1933]]

## Status
- [x] Note created (September 11, 2026)
- [x] Primary-source article downloaded, extracted from the archival issue scan, and verified as an eight-page PDF
- [x] Closes the Shannon/Fisher-to-cross-entropy gap
- [ ] Natural next link: Akaike 1973/1974 — expected KL risk becomes a practical model-selection penalty

---

*Shannon priced uncertainty. Fisher made probability trainable. Kullback and Leibler measured the bill for training the wrong distribution.*
