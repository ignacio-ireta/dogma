# Bootstrap and sampling uncertainty

A point estimate does not describe how it would vary if the experiment were repeated. Exact sampling distributions are often unavailable, especially for nonlinear statistics. The nonparametric bootstrap replaces the unknown sampling distribution by the empirical distribution and repeats the entire estimation procedure on simulated samples.

## Sources and reading locators

Bootstrap Methods: Another Look at the Jackknife is Efron's 1979 *Annals of Statistics* article, pp. 1–26, with an archive cover. The [publisher DOI](https://doi.org/10.1214/aos/1176344552) identifies the journal version. Read the opening formulation and examples on the sample median and discriminant analysis; §4 studies the classification error example. The earlier [Stanford BIO 32 report](https://statistics.stanford.edu/technical-reports/bootstrap-methods-another-look-jackknife) is dated July 1977 and is a distinct version. Probability: Theory and Examples, §§2.2–2.4 and 3.4, supplies the probability limit background.

## The plug-in step

For iid observations \(X_1,\ldots,X_n\), the empirical distribution is
\(\widehat F_n=n^{-1}\sum_i\delta_{X_i}\). Draw \(X_1^\star,\ldots,X_n^\star\) independently from \(\widehat F_n\), conditional on the observed sample, and recompute the statistic \(T_n^\star\). Repeating this operation approximates its conditional distribution.

Two approximations must be separated. The statistical approximation replaces \(F\) by \(\widehat F_n\). The Monte Carlo approximation uses finitely many simulated samples to evaluate the resulting bootstrap distribution. Increasing the number of simulations reduces only the latter error.

Validity requires that the empirical distribution approximate the relevant functional sufficiently well. It is not a consequence of resampling alone. Dependence, nonsmooth targets, heavy tails and boundary parameters can change the required method.

## Worked example: enumerate instead of simulate

Take data \(0,3,6\). The mean is \(3\), and the empirical variance, with denominator three, is

\[
\frac{(0-3)^2+(3-3)^2+(6-3)^2}{3}=6.
\]

Every bootstrap sample consists of three draws with replacement from these values. There are \(3^3=27\) equally likely ordered samples. Without enumerating them individually, conditional independence yields

\[
E_\star[\bar X^\star]=3,\qquad
\operatorname{Var}_\star(\bar X^\star)=6/3=2.
\]

So the exact bootstrap standard deviation is \(\sqrt2\). A simulation with many replicates should approach that value.

The usual unbiased sample variance is \(9\), whose standard error estimate for the mean is \(\sqrt{9/3}=\sqrt3\). The finite sample difference is not a programming error: the empirical distribution uses denominator \(n\), while the unbiased variance estimator uses \(n-1\). Their ratio approaches one as \(n\) grows.

Now change the statistic to the maximum. Bootstrap maxima can never exceed six, and the largest observation is absent from a resample with probability \((2/3)^3=8/27\). If the inferential target is an unknown population endpoint, this support restriction is a warning that a naive bootstrap need not reproduce the desired sampling behavior.

## Lineage and AI relevance

Probability limit theory justifies averages and smooth approximations; Efron's method makes the empirical distribution an operational substitute for repeated access to the population. The jackknife is closely connected through local approximations, while the bootstrap can explore a fuller nonlinear response.

In AI, bootstrap estimates can describe variability of metrics, coefficients or fitted predictors under a stated sampling unit. Resampling individual rows in clustered or temporal data may destroy the dependence that governs uncertainty. To study the full training procedure, preprocessing, feature selection and fitting belong inside each resample; to study only a fixed model's metric, the target is different. Bootstrap variation is not automatically Bayesian posterior uncertainty, nor a guarantee against deployment shift. Continue to [conformal prediction](2022-2026%20--%20Angelopoulos%20et%20al%20--%20Conformal%20Prediction%20and%20the%20Limits%20of%20Coverage%20--%20Study%20Note.md) for a different route to predictive coverage.
