# Interview notes — Monte Carlo simulation

## Q1. How would you estimate an expectation by simulation, and how sure are you of the answer?

**60-second answer.** Write the quantity as $\mathbb{E}[f(X)]$ and draw $n$ independent samples
from the distribution of $X$. Report the mean of $f$ over the samples, the standard error
$s/\sqrt{n}$, and a normal interval $\hat\mu \pm z\,\mathrm{SE}$. The standard error is the honest
measure of the answer's precision. It falls as $n^{-1/2}$, so the error bar is always visible and
can always be tightened by spending more draws.

**Follow-ups.**
- *Why $n-1$ in the sample variance?* Dividing by $n-1$ removes the bias that comes from measuring
  deviations from the sample mean rather than the true mean.
- *What if the draws are correlated?* The $s/\sqrt{n}$ formula assumes independence. Batch means
  or the effective sample size fix the error bar.
- *Can you guarantee the error rather than estimate it?* Only with a bound that needs no CLT. For values
  in $[0,1]$, Hoeffding gives $\pm1.358/\sqrt{n}$ at 95% confidence, which is conservative next to the
  normal $\pm0.98/\sqrt{n}$ worst case.

## Q2. How many draws do you need for a $\pm 1\%$ answer?

**60-second answer.** Solve $z\sigma/\sqrt{n} = \varepsilon$, so $n = (z\sigma/\varepsilon)^2$. For
a relative error of 1% at 95% confidence, with a coefficient of variation $\sigma/\mu$ of 1, that is
$(1.96/0.01)^2 \approx 38{,}000$ draws. The answer scales with the square of the precision you ask
for, so 0.1% costs a hundred times more.

**Follow-ups.**
- *The variance is unknown.* Run a pilot of a few thousand draws, read off $s$, and size the
  main run from it.
- *How do you reduce the cost?* Variance reduction, such as antithetic or control variates, lowers
  $\sigma$. Importance sampling moves draws to where $f$ matters.

## Q3. Why is naive Monte Carlo a poor tool for rare events?

**60-second answer.** For an indicator of an event with probability $p$, the relative standard error
is $\sqrt{(1-p)/(np)}$. The number of draws needed for a fixed relative error grows like $1/p$. A
one-in-a-million event needs about $10^{10}$ draws for 1% relative error, which is not practical.
The fix is to change the sampling distribution so that the event is common, then correct the
weights (importance sampling).

**Follow-ups.**
- *What do you report when no event is observed?* A confidence bound from the number of draws, not
  a zero estimate with a zero error bar.
- *Does the normal interval still hold?* Not for rare events with few hits. Check the count of hits
  first.

## Common mistakes

- Dividing the variance by $n$ instead of $n-1$ and calling it the sample standard deviation.
- Reporting the standard error of the draws, $s$, instead of the standard error of the mean, $s/\sqrt{n}$.
- Quoting an estimate with no error bar, so the reader cannot tell whether two results differ.
- Using the one-pass formula $\mathbb{E}[Y^2] - \mathbb{E}[Y]^2$ on data with a large offset.
