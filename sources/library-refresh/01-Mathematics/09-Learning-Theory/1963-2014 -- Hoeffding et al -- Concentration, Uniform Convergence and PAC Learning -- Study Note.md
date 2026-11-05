# Concentration, uniform convergence and PAC learning

An average loss may estimate one fixed model's population loss accurately. Training searches across models using that same average, so a guarantee for one model does not automatically cover the selected model. Concentration controls random fluctuations; uniform convergence pays for selection.

## Sources and reading locators

- Probability Inequalities for Sums of Bounded Random Variables: 1963 JASA article, Theorem 2, printed pp. 16–17; university-hosted journal scan. [Publisher record](https://doi.org/10.1080/01621459.1963.10500830).
- On the Uniform Convergence of Relative Frequencies of Events to Their Probabilities: 1971 English translation by B. Seckler, pp. 264–280; §1.1 growth function and §1.2 Theorem 1. [Publisher record](https://doi.org/10.1137/1116025).
- [Understanding Machine Learning: From Theory to Algorithms](https://www.cs.huji.ac.il/~shais/UnderstandingMachineLearning): remote author copy only; no PDF is held in this repository. Chapter 4, especially §§4.1–4.2, Lemma 4.5 and Corollary 4.6; Chapter 6 for VC dimension and growth functions. [Author book page](https://www.cs.huji.ac.il/~shais/UnderstandingMachineLearning). Section numbers agree with print; pagination differs.

## A finite class calculation

Let \(Z_1,\dots,Z_n\) be iid, let a loss lie in \([0,1]\), and define
\(R(h)=E\ell(h,Z)\) and \(\widehat R_n(h)=n^{-1}\sum_i\ell(h,Z_i)\).
For a fixed \(h\), Hoeffding's inequality gives

\[
P(|\widehat R_n(h)-R(h)|>\epsilon)
\leq 2e^{-2n\epsilon^2}.
\]

If a class \(\mathcal H\) of \(M\) candidates is fixed before seeing the sample, the union bound yields

\[
P\left(\sup_{h\in\mathcal H}|\widehat R_n(h)-R(h)|>\epsilon\right)
\leq 2M e^{-2n\epsilon^2}.
\]

The uniform event covers whichever candidate empirical risk minimization selects. If \(\hat h\) minimizes empirical risk and \(h^\star\) minimizes population risk within the class, adding and subtracting the two empirical risks gives

\[
R(\hat h)\leq R(h^\star)+2\epsilon.
\]

An approximate empirical optimizer adds its optimization tolerance to this bound.

## Worked example: what selection costs

Consider \(M=100\) classifiers and ask that every empirical error be within \(0.05\) of its population error with probability at least \(0.95\). Solving
\(200e^{-0.005n}\leq0.05\) gives

\[
n\geq\frac{\log 4000}{0.005},
\]

so \(n=1659\) is sufficient. For one prespecified classifier the analogous sufficient count is \(738\). These are conservative upper bounds, not minimum sample sizes.

With the first calculation, the selected classifier's excess risk is bounded by \(0.10\), because comparison uses two estimation errors. Requesting excess risk \(0.05\) would require using \(\epsilon=0.025\), increasing the sufficient sample size by a factor of four.

This arithmetic clarifies why a large model search can produce a misleadingly low training error. The selected model exploits both real patterns and sample fluctuations. It also clarifies why a fresh test sample has a different role: conditional on a model chosen independently of that sample, a fixed-model concentration bound applies.

## Infinite classes and the AI connection

For an infinite class, simply replacing \(M\) by infinity is useless. VC theory instead controls the number of distinct binary labelings a class can realize on finite samples. Thresholds on the real line, for example, have infinitely many parameter values but only \(n+1\) labelings on \(n\) ordered distinct inputs and VC dimension one.

The reading path is [probability spaces and limit laws](../06-Probability-and-Statistics/1933%20--%20Kolmogorov%20--%20Probability%20Spaces%2C%20Conditioning%20and%20Limit%20Laws%20--%20Study%20Note.md) → bounded-sum concentration → uniform control → learning guarantees. PAC learning makes the confidence, accuracy, comparator class and sampling assumptions explicit. These results explain one route to generalization; they do not imply every useful learning algorithm requires uniform convergence, or that a worst case VC bound tightly predicts a modern neural network's behavior. Distribution shift and adaptive reuse of the test set require additional assumptions or methods.
