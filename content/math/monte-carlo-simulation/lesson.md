---
title: Monte Carlo simulation
minutes: 12
min_viable: "Read The system and the Worked example, then solve p01."
---

## The system

To estimate an expectation $\mu = \mathbb{E}[f(X)]$, draw $n$ independent values $X_1, \dots, X_n$
from the distribution of $X$ and average the integrand values $Y_i = f(X_i)$.

| Quantity | Shape | Formula | Meaning |
|---|---|---|---|
| Estimate | scalar | $\hat\mu = \frac{1}{n}\sum_i Y_i$ | average of the integrand |
| Sample sd | scalar | $s = \sqrt{\frac{1}{n-1}\sum_i (Y_i - \hat\mu)^2}$ | spread of one draw, divisor $n-1$ |
| Standard error | scalar | $\mathrm{SE} = s/\sqrt{n}$ | error bar of $\hat\mu$ |
| Interval | pair | $\hat\mu \pm z\,\mathrm{SE}$ | normal-approximation confidence interval |

Here $z = \Phi^{-1}\!\big(\tfrac{1+c}{2}\big)$ is the standard normal quantile for confidence level
$c$: $z = 1.95996$ for $c = 0.95$, and $z = 1.64485$ for $c = 0.90$.

The standard error describes the error of $\hat\mu$, not the spread $s$ of the draws.

## Worked example

Estimate $\mathbb{E}[U^2]$ for $U \sim \mathrm{Uniform}(0,1)$. The truth is $1/3$. Take $n = 4$
draws, $U = (0.2, 0.5, 0.9, 0.4)$.

1. $Y = (0.04, 0.25, 0.81, 0.16)$.
2. $\hat\mu = 1.26 / 4 = 0.315$.
3. Deviations $Y - \hat\mu = (-0.275, -0.065, 0.495, -0.155)$. Their squares sum to $0.3489$.
4. $s^2 = 0.3489 / 3 = 0.1163$, so $s = 0.3410$ and $\mathrm{SE} = 0.3410 / 2 = 0.1705$.
5. $z\,\mathrm{SE} = 1.96 \times 0.1705 = 0.334$. The interval is $(-0.019, 0.649)$.

The interval covers $1/3$, but it is about 0.67 wide. Four draws say very little.

## Why it works

The law of large numbers gives $\hat\mu \to \mu$. The central limit theorem says that for large
$n$, $(\hat\mu - \mu)/(\sigma/\sqrt{n})$ is approximately standard normal, where
$\sigma^2 = \mathrm{Var}(f(X))$. Replacing $\sigma$ by $s$ and the normal quantile by $z$ gives
the interval. Metropolis and Ulam (1949) introduced the method.

The error $\sigma/\sqrt{n}$ falls as $n^{-1/2}$. Halving it takes four times the draws. Cutting it
tenfold takes a hundred times the draws. This is the price of simulation, and it does not depend on
the dimension of $X$.

The normal approximation is an approximation. For $f = \exp$ and $X \sim N(0,1)$ the true value is
$e^{1/2} = 1.6487$. With seeds $0,\dots,999$, each drawing `default_rng(seed).standard_normal(200)`, the
nominal 95% interval covered it 93.6% of the time. With seeds $0,\dots,299$ at $n = 2000$ it covered 95.0%.
Skew makes small-$n$ intervals too narrow.

**A bound without the CLT.** For $f$ with values in $[0,1]$, Hoeffding's inequality gives $P(|\hat\mu-\mu|\ge\varepsilon)\le 2e^{-2n\varepsilon^2}$ for every $n$. So with probability at least $1-\delta$, the error is at most $\sqrt{\ln(2/\delta)/(2n)}$. At $\delta=0.05$ that is $1.358/\sqrt n$, against the normal half-width $1.96\sigma/\sqrt n\le 0.98/\sqrt n$ when $\sigma\le 1/2$. The bound is conservative: use it for a guarantee, and the standard error for a typical error bar (the Library note's locator is Hoeffding 1963, Theorem 2, not checked against the article; [source: library-refresh/01-Mathematics/09-Learning-Theory/1963-2014 -- Hoeffding et al -- Concentration, Uniform Convergence and PAC Learning -- Study Note.md]).

## Signal vs. noise

- **Signal:** the estimator is a plain average of $f$ over the draws. Its error bar is
  $s/\sqrt{n}$, and the rate is $n^{-1/2}$.
- **Signal:** the interval is a normal approximation. Its accuracy depends on $n$ and on how skewed
  $Y$ is.
- **Variance reduction changes the constant, not the rate.** Antithetic variates pair each draw $U$ with $1-U$ and average $f(U)$ and $f(1-U)$. It reduces variance when $f$ is monotone, because $f(U)$ and $f(1-U)$ are then negatively correlated. Control variates subtract $c\,(g(X) - \mathbb{E}[g(X)])$ for a function $g$ with known mean. The coefficient $c$ is chosen to minimise variance. Both methods still give error $\propto n^{-1/2}$.
- **Noise for now:** quasi-random sequences, which need their own error theory.

## Where it breaks

- **Infinite variance.** If $\mathrm{Var}(f(X))=\infty$, as for a Pareto distribution with shape at
  most 2, the CLT does not apply. The sample sd is then not a reliable error estimate, and the
  normal interval can undercover. Check its coverage on a case you can compute.
- **Rare events.** For an indicator $f = \mathbf{1}_A$ with $P(A) = p$, the relative standard error
  is $\sqrt{(1-p)/(np)}$. A 1% relative error at $p = 10^{-6}$ needs about $10^{10}$ draws.
- **Cancellation.** The one-pass formula $\mathbb{E}[Y^2] - \mathbb{E}[Y]^2$ loses its precision
  when the mean is large compared with the spread. With draws $10^9 + Z_i$, $Z_i \sim N(0,1)$, from
  `default_rng(1)` at $n = 10$, `mean(y*y) - mean(y)**2` returns $128.0$, while the sample variance is
  $0.440$. Sum squared deviations from the mean in a second pass.
- **Seed dependence.** One seed gives one answer, so report the interval, not a single digit. Create one `default_rng(seed)` per estimate, and never use NumPy's global state.

## Interview angle

**60-second answer.** Write the quantity as an expectation. Draw $n$ independent samples. Report
the mean of $f$, the standard error $s/\sqrt{n}$, and a normal interval. The error falls as
$n^{-1/2}$, so the draw count sets the precision. For a 1% relative error at 95% with a coefficient
of variation of 1, you need $(1.96/0.01)^2 \approx 38{,}000$ draws.

**Follow-ups.**

- *How do you make it cheaper?* Variance reduction lowers the constant; importance sampling targets rare events.
- *When do you not trust the interval?* Heavy tails, rare events and small $n$ with skewed $Y$.

## Connections

- **Requires:** `cml/probability-basics`, `cml/descriptive-statistics` (the sample variance with
  divisor $n-1$).
- **Unlocks:** `math/inverse-transform-sampling` and `math/building-generators-by-rejection`, which
  make the draws. `rl/monte-carlo-prediction` is the same average taken over episodes.
- **Lineage:** Metropolis and Ulam (1949) named the method. The error bar is the central limit
  theorem applied to an average, which is older than the method.
