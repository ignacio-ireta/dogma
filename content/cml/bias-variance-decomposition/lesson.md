---
title: Bias-variance decomposition
minutes: 12
min_viable: "Read The system and the Worked example, then do p01."
---

## The system

Fix a test point $x$. Draw training sets $D$ from the same process, fit a model $\hat f_D$ to each, and look at the
prediction $\hat f_D(x)$ across those sets. Let $f(x)$ be the true function, $\varepsilon$ the label noise with
$\mathbb{E}\varepsilon = 0$ and $\operatorname{Var}\varepsilon = \sigma^2$, and $\bar f(x) = \mathbb{E}_D[\hat f_D(x)]$ the
average model.

| Term | Formula | Who controls it |
|---|---|---|
| Noise | $\sigma^2$ | nobody: it is in the data |
| Squared bias | $(\bar f(x) - f(x))^2$ | model choice, features, regularisation |
| Variance | $\mathbb{E}_D[(\hat f_D(x) - \bar f(x))^2]$ | model choice, data size, regularisation |

Expected squared test error at $x$ is the sum of the three.

## Worked example

Training sets of 60 points with $x \sim U(0, 1)$ from $y = \sin(2\pi x) + \varepsilon$, $\sigma = 0.3$, so $\sigma^2 = 0.09$.
Least-squares polynomial fits (`np.polyfit`) of degree 1 to 9, 300 training sets per degree, averaged over the 50 test
points `np.linspace(0, 1, 50)`. One run: `np.random.default_rng(2026)`, with the degrees fitted in order from that generator:

| Degree | Squared bias | Variance | Expected test error |
|---|---|---|---|
| 1 | 0.2122 | 0.0108 | 0.3131 |
| 3 | 0.0054 | 0.0074 | 0.1028 |
| 5 | 0.0001 | 0.0127 | 0.1027 |
| 9 | 0.0001 | 0.0866 | 0.1767 |

Check the identity on degree 3: $0.0054 + 0.0074 + 0.09 = 0.1028$. The noise term is 0.09 at every degree.

## Why it works

Let $y = f + \varepsilon$ and write $\hat f$ for $\hat f_D(x)$.

1. $\mathbb{E}[(y - \hat f)^2] = \mathbb{E}[(f + \varepsilon - \hat f)^2]$. Expand the square.
2. $= \mathbb{E}[\varepsilon^2] + \mathbb{E}[(f - \hat f)^2] + 2\,\mathbb{E}[\varepsilon(f - \hat f)]$.
3. The cross term is zero: $\varepsilon$ has mean zero and is independent of $\hat f$. So the first term is $\sigma^2$.
4. $\mathbb{E}[(f - \hat f)^2] = (f - \bar f)^2 + \mathbb{E}[(\hat f - \bar f)^2]$, since $\mathbb{E}[\hat f - \bar f] = 0$ kills the cross term.
5. Substitute: $\mathbb{E}[(y - \hat f)^2] = \sigma^2 + (\bar f - f)^2 + \mathbb{E}[(\hat f - \bar f)^2]$.

The only step people skip is 3. It is the reason the noise term cannot be reduced by any model.

## Signal vs. noise

- **Signal:** the squared bias and the variance. They move with the model, in opposite directions for most families.
- **Signal:** the total. A model with lower bias and higher variance can be worse overall, as degree 9 is here.
- **Noise:** the exact third decimal of any one term. Compare totals and ranges across seeds.

## Where it breaks

- **Not monotone at every step.** From degree 1 to degree 2 the squared bias rises slightly (0.2122 to 0.2128). Both
  fits miss $\sin(2\pi x)$, which is odd about $x = 0.5$, and degree 2 adds no odd term. The bias falls sharply at
  degree 3, where a cubic can capture that odd part.
- **Heavy-tailed variance.** Degree 9 alone, with 300 training sets per seed, has variance from 0.075 to 0.387 across
  seeds 0 to 6. A few training sets with unusual designs produce extreme predictions at the edges. Use many sets,
  check several seeds, and report the range.
- **Double descent.** The classical U shape can fail for over-parameterised models. In a seeded check (minimum-norm
  least squares, $n = 40$ training rows, 200 standard Gaussian feature columns of which the first $p$ are used, true
  coefficient vector $e_1$, noise sd 0.5, 30 seeds, test error against the noise-free target), the median test MSE was
  0.025 at 5 features, 24.8 at 40, 0.74 at 100 and 0.86 at 200. The closed form $(1 - n/p) + \sigma^2 n/(p - n - 1)$
  gives the mean at 100 and 200 features, which is 0.77 and 0.86. The peak at 40 is an ill-conditioning spike, and its
  mean is unstable across seeds. Belkin et al. (2019) study this phenomenon. The second descent is real, but it did not
  beat the small models here, so test the curve rather than assume it.

## Interview angle

**60-second answer.** Fix a test point and average over training sets. The expected squared error splits into noise,
squared bias and variance. The noise is irreducible. Bias and variance move against each other with model complexity and
regularisation, so the target is the total. The derivation needs only zero-mean, independent noise. I would still check
the curve on my own data, since over-parameterised models can descend a second time.

**Follow-ups.**

- *Which term does regularisation move?* It usually raises bias and lowers variance, by shrinking the fit toward a simpler
  model.
- *Why not report one seed's variance?* The estimate is heavy-tailed for flexible models, so the range across seeds is the
  honest number.

Full answers are in `interview.md`.

## Connections

- **Requires:** `cml/expectation-and-variance` (the mean, the variance and the cross terms) and
  `cml/over-and-underfitting-early-stopping` (the train and validation curves this decomposition explains).
- **Unlocks:** the regularisation topics, where the bias-variance trade-off is the reason penalties help.
- **Lineage:** Bishop (2006, section 3.2) derives the decomposition in regression. Hastie, Tibshirani and Friedman
  (ESL, chapter 7) cover the complexity trade-off. Belkin et al. (2019) cover the second descent.
