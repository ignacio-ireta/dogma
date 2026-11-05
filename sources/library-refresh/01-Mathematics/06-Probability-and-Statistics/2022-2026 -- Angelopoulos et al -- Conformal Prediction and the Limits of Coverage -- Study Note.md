# Conformal prediction and the limits of coverage

A predictor outputs a point; a prediction set asks which outcomes remain plausible at a stated coverage level. Conformal methods use ranks to obtain finite sample guarantees while allowing a flexible prediction model. The central question is which probability the guarantee averages over.

## Sources and reading locators

- A Gentle Introduction to Conformal Prediction and Distribution-Free Uncertainty Quantification: arXiv v6, December 2022; §1.1 for the construction, §2.2 for conformalized quantile regression. [Version record](https://arxiv.org/abs/2107.07511v6).
- The Limits of Distribution-Free Conditional Predictive Inference: 2021 journal version; §§2.1–2.2 distinguish marginal coverage and conditional impossibility, with further relaxations in §3. [Publisher DOI](https://doi.org/10.1093/imaiai/iaaa017).
- Theoretical Foundations of Conformal Prediction: arXiv v5, March 6, 2026, a pre-publication book manuscript; Chapter 3 for rank-based constructions, Chapter 4 for conditional guarantees and their limits. [Version record](https://arxiv.org/abs/2411.11824v5).

## A rank guarantee

Train a predictor on data separate from the calibration sample. For \(m\) calibration pairs, compute scores
\(S_i=|Y_i-\hat f(X_i)|\), and let \(k=\lceil(m+1)(1-\alpha)\rceil\).
Take \(\hat q=S_{(k)}\), treating \(\hat q=+\infty\) if \(k>m\), and output

\[
C(x)=[\hat f(x)-\hat q,\hat f(x)+\hat q].
\]

Conditional on the separate training data, exchangeability of calibration and test pairs makes the test score's rank exchangeable with the others. This gives
\(P(Y_{\rm new}\in C(X_{\rm new}))\geq1-\alpha\).
Ties make the usual nonrandomized construction conservative.

Exchangeability is a substantive symmetry assumption. Arbitrary time trends, adaptive data collection or deployment shift do not automatically satisfy it.

## Worked example and the missing conditioning

Let \(m=9\), \(\alpha=0.2\), and ordered residual scores be
\(0.1,0.2,0.4,0.5,0.7,0.8,1.0,1.4,2.0\).
Then \(k=8\) and \(\hat q=1.4\). If the model predicts ten, the interval is \([8.6,11.4]\). Using an uncorrected empirical quantile can pick the wrong rank in such a small sample.

Now consider two equally likely groups. A procedure with coverage one in group A and \(0.6\) in group B has marginal coverage \(0.8\). The overall guarantee holds while one group is underserved. This example is an arithmetic demonstration of the difference between
\(P(Y\in C(X))\geq0.8\) and
\(P(Y\in C(X)\mid X=x)\geq0.8\).

A guarantee conditional on the realized calibration set is another distinct object. Repeating deployment with one fixed calibration set need not realize the nominal average exactly; variation across calibration samples remains.

## Lineage and current research questions

[Neyman–Pearson testing](1933%20--%20Neyman%20and%20Pearson%20--%20Hypothesis%20Tests%20as%20Controlled%20Decisions%20--%20Study%20Note.md) supplies a testing perspective: compare candidate outcomes with a reference distribution of scores and retain those not rejected. [uniform convergence](../09-Learning-Theory/1963-2014%20--%20Hoeffding%20et%20al%20--%20Concentration%2C%20Uniform%20Convergence%20and%20PAC%20Learning%20--%20Study%20Note.md) explains the separate role of concentration when stronger guarantees conditional on training data are sought.

The conditional inference paper and the 2026 text show why unrestricted, useful distribution-free coverage at every continuously distributed feature value is generally unattainable without further structure. Finite predefined groups, restricted families of subsets, smoothness assumptions or known forms of shift change the problem and must be stated.

For AI, the predictor determines how informative the sets can be, while calibration supplies a validity argument under the sampling assumptions. Poor prediction can yield broad but valid sets. A high confidence softmax score alone supplies neither the rank construction nor its guarantee. Coverage is also different from accuracy, interval width, subgroup reliability and causal correctness. Record the exact coverage target alongside any deployed prediction set.
