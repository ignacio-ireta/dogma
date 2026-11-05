# Probability spaces, conditioning and limit laws

A frequency table answers questions about a finite experiment. An infinite sequence of observations, a continuously distributed parameter, or an expectation conditioned on a measurement requires a framework that also controls limits. This note connects three distinct contributions: Bayes's inverse probability problem, Kolmogorov's axiomatization, and the modern limit theory presented by Durrett. It complements the existing historical concept notes.

## Sources and reading locators

- An Essay towards Solving a Problem in the Doctrine of Chances: modern transcription; Part I, Propositions 5 and 9, and the scholium following Proposition 9. [Primary publication record](https://doi.org/10.1098/rstl.1763.0053).
- Foundations of the Theory of Probability: Second English edition (Chelsea, 1956), translation edited by Nathan Morrison, hosted by [Royal Holloway](https://cml.rhul.ac.uk/resources/fop/index.htm); Chapter I §1 (p. 2) and Chapter II §1 (p. 14) for axioms and continuity; Chapter V §§1–4 (pp. 47–52) for conditioning. The 47-page PDF contains facing-page scans through printed p. 84.
- Probability: Theory and Examples: author Version 5, January 11, 2019; §§1.1, 1.6, 2.2, 2.4, 3.4 and 4.1. [Author contents](https://sites.math.duke.edu/~rtd/PTE/pte.html).

## Mathematical idea and assumptions

In modern notation a probability space is \((\Omega,\mathcal F,P)\), with measurable events, total mass one, and countable additivity. A random variable is a measurable function; expectation is integration against \(P\). These definitions connect discrete sums and continuous densities without assuming every distribution has a density.

For integrable \(X\), conditioning on a sigma algebra \(\mathcal G\) produces a \(\mathcal G\)-measurable variable satisfying

\[
\int_G E[X\mid\mathcal G]\,dP=\int_G X\,dP
\quad\text{for every }G\in\mathcal G.
\]

This formulation avoids dividing by the probability of a single continuous observation, which is usually zero. Conditional expectations are determined only up to events of probability zero.

For independent, identically distributed observations with finite absolute mean, the strong law gives \(\bar X_n\to E[X_1]\) almost surely. With finite positive variance, the classical central limit theorem describes \(\sqrt n(\bar X_n-\mu)/\sigma\) in distribution. These are different claims: eventual convergence and a limiting fluctuation scale.

## Worked bridge: a coin and a latent parameter

The following calculation uses modern beta notation, rather than attributing that notation to Bayes. Let \(p\) be uniform on \([0,1]\), and let four flips be independent conditional on \(p\). Observing three heads and one tail yields

\[
\pi(p\mid D)=20p^3(1-p),\qquad
E[p\mid D]=\frac46=\frac23.
\]

The likelihood maximizer is \(3/4\). The posterior mean differs because it averages over parameter uncertainty under the stated prior. Direct integration gives

\[
P(p>1/2\mid D)=
\left[5p^4-4p^5\right]_{1/2}^{1}
=\frac{13}{16}.
\]

This is a probability statement under a model, not a prior-free guarantee about every possible coin.

A useful subtlety appears before observing data. Conditional on \(p\), two flips are independent. After averaging over the uniform prior, \(P(H_1,H_2)=E[p^2]=1/3\), whereas \(P(H_1)P(H_2)=1/4\). Shared uncertainty creates dependence. Consequently one must state whether an independence assumption is conditional or unconditional before applying an iid theorem.

## Lineage and AI relevance

Lebesgue on measure and integration supplies the measure and integral background. Bayes supplies a particular inverse problem; Kolmogorov supplies a common mathematical setting; Durrett supplies explicit hypotheses for convergence. This is a conceptual reading path, not a claim that each work historically caused the next.

The same distinctions govern hierarchical models, posterior prediction, Monte Carlo averages and uncertainty estimates. More samples reduce simulation error only when the sampling scheme supports the relevant convergence theorem; they do not repair an incorrect likelihood, prior or dependence assumption. Continue with [concentration and uniform convergence](../09-Learning-Theory/1963-2014%20--%20Hoeffding%20et%20al%20--%20Concentration%2C%20Uniform%20Convergence%20and%20PAC%20Learning%20--%20Study%20Note.md) for finite sample control after selecting a model from data.
