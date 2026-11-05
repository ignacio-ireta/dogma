# Hypothesis tests as controlled decisions

A fitted model may have high likelihood without supplying a rule for when to reject a competing explanation. Neyman and Pearson make the decision rule itself the mathematical object: allocate a fixed allowance of false rejections to the observations most informative against a specified null.

## Source and reading locators

On the Problem of the Most Efficient Tests of Statistical Hypotheses is the scanned 1933 journal article, *Philosophical Transactions A* 231, 289–337, with an archive cover. Read §II, beginning at printed p. 293, for the general setup, and §III(a), beginning at p. 298, for simple hypotheses. The [publisher DOI](https://doi.org/10.1098/rsta.1933.0009) identifies the work.

## The optimization problem

For simple hypotheses \(H_0:P=P_0\) and \(H_1:P=P_1\), write a possibly randomized rejection rule as \(\phi(x)\in[0,1]\). Its size is \(E_0\phi\), and its power is \(E_1\phi\). The problem is

\[
\max_\phi E_1\phi
\quad\text{subject to }E_0\phi\leq\alpha.
\]

When \(P_0,P_1\) have densities relative to a common measure, a likelihood ratio rule rejects where \(p_1(x)>k p_0(x)\). Boundary randomization can attain an exact size in discrete experiments.

The essential comparison is pointwise. If \(\phi^\star\) is the threshold rule, then
\((\phi^\star-\phi)(p_1-kp_0)\geq0\). Integrating and using the size constraint gives \(E_1\phi^\star\geq E_1\phi\). Thus optimality is relative to the nominated alternative and error budget.

## Worked example: detecting a shifted mean

Suppose one measurement has known variance one. Compare \(H_0:X\sim N(0,1)\) with \(H_1:X\sim N(1,1)\). Cancelling common density factors gives

\[
\log\frac{p_1(X)}{p_0(X)}=X-\frac12.
\]

Larger measurements therefore carry more evidence for this positive shift. At size \(0.05\), reject if \(X>1.645\). The power is approximately

\[
P_1(X>1.645)=1-\Phi(0.645)\approx0.259.
\]

The weak power is informative: a five percent false alarm rate does not imply strong ability to detect the specified change.

For \(n\) independent observations from the same normal model, the size remains \(0.05\) if the rule is \(\sqrt n\,\bar X>1.645\). Under mean one, power becomes
\(1-\Phi(1.645-\sqrt n)\). With \(n=9\), this is approximately \(0.912\). Replication improves distinguishability because the hypotheses induce increasingly separated distributions of the sample mean.

These numbers depend on known variance, normal sampling, the planned direction and independent observations. Selecting the direction after viewing the data changes the procedure. So does repeatedly testing until a desired result appears.

## Interpretation and descendants

A size guarantee bounds the probability of rejection when the null generates the experiment. It does not give \(P(H_0\mid X)\), the probability a particular discovery is false, or the practical cost of an error. A composite alternative may favor different rejection regions for different parameter values; the simple hypothesis lemma alone does not resolve that conflict.

The connection to Fisher on likelihood is direct: the same likelihood function supports estimation and comparison, but the objectives differ. In AI, threshold selection, anomaly detection and evaluation of model improvements all require specifying the target distribution and tolerated errors. ROC curves display the power versus size tradeoff across thresholds. Multiple comparisons and adaptive evaluation require additional theory beyond this single, planned test. Continue to [conformal prediction](2022-2026%20--%20Angelopoulos%20et%20al%20--%20Conformal%20Prediction%20and%20the%20Limits%20of%20Coverage%20--%20Study%20Note.md) to see how testing and rank arguments support prediction sets.
