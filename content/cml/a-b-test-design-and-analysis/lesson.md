---
title: A/B test design and analysis
minutes: 12
min_viable: "Read The system and the Worked example, then solve p01."
---

## The system

An A/B test is a randomised comparison with its plan fixed before any outcome is read. The plan has six parts.

| Step | Decide | Example |
|---|---|---|
| 1. Metric | One primary metric (the decision criterion), plus guardrails | checkout conversion; refund rate as guardrail |
| 2. Unit | What is randomised, and what it is averaged over | user, not session, so repeat visits stay in one arm |
| 3. Effect | Minimum detectable effect (MDE): the smallest change worth acting on | 10% → 12%, an absolute 2 points |
| 4. Size | Per-arm $n$ from $\alpha$ (two-sided) and power $1-\beta$ | $n = 3841$ per arm at 80% power |
| 5. Duration | Whole weeks that cover $2n$ users at your traffic | $2 \times 3841 / 1000$ users per day $\approx 7.7$ days, so run 2 weeks |
| 6. Checks | Sample ratio mismatch (SRM) test and guardrails before the outcome | assignment counts consistent with the planned split |

Sample size for two proportions (per arm, two-sided, normal approximation): with $p_1$ the control rate,
$p_2 = p_1 + \text{MDE}$, $\bar p = (p_1+p_2)/2$,

$$n = \frac{\big(z_{1-\alpha/2}\sqrt{2\bar p(1-\bar p)} + z_{1-\beta}\sqrt{p_1(1-p_1)+p_2(1-p_2)}\big)^2}{(p_2-p_1)^2}.$$

The analysis is a pooled two-proportion z-test, $z = (\hat p_B - \hat p_A)/\sqrt{\hat p(1-\hat p)(1/n_A+1/n_B)}$.

## Worked example

Baseline $p_1 = 0.10$, MDE $0.02$, $\alpha = 0.05$, power $0.8$: $n = 3840.85$, so 3841 users per arm.
Simulated at $n = 3841$ with true rates 0.10 and 0.12 (20,000 experiments, `default_rng(20261024)`), the
pooled z-test rejects 79.8% of the time, close to the 80% target.

If 1,000 eligible users arrive per day, the test needs $7682/1000 \approx 7.7$ days. Run whole weeks, so
14 days. Weekday and weekend patterns then cancel out.

## Why it works

The test compares two sample proportions, each with variance $p(1-p)/n$. The significance level $\alpha$ is
the one error rate fixed in advance: the false-positive rate accepted under the null. Power is the probability of
clearing the critical value when the true difference is the MDE. It depends on the effect, $n$ and the
variance, so an underpowered plan misses the MDE even when executed correctly (Neyman and Pearson, per the
Library note). The formula sets $n$ so the MDE sits $z_{1-\alpha/2} + z_{1-\beta}$ standard errors from zero.
Randomisation makes the arms comparable in expectation, so the difference estimates the average effect.

## Signal vs. noise

- **Signal:** the plan is fixed before launch: metric, MDE, $n$, duration, and what counts as a win.
- **Signal:** the MDE is a business decision, not a statistical one. Smaller MDEs cost quadratically more users.
- **Noise for now:** ratio metrics, CUPED-style variance reduction and sequential designs; they refine the same plan.

## Where it breaks

- **Peeking.** Checking the p-value repeatedly and stopping at the first $p < 0.05$ inflates false positives.
  In an A/A simulation (Bernoulli(0.10) in both arms, up to 5,000 users per arm, two-sided $z$ at 1.96,
  100,000 experiments per row, seed 7, one seeded run), one look at the horizon rejected 4.96% of the time,
  five equally spaced looks 14.06%, and ten looks 19.19%.
  Fix the horizon in advance, or use sequential boundaries designed for repeated looks.
- **Sample ratio mismatch.** If a 50/50 design produces 50,600 and 49,400 assignments out of 100,000, the chi-square
  statistic is 14.4 ($p = 0.00015$). The assignment or logging is broken, so the outcome comparison is not
  trustworthy. Check the split before reading any metric.
- **Novelty and primacy.** A change can win in week one because users notice it, then fade. Look at the effect by
  week before calling the long-run result.
- **Practical versus statistical significance.** With a million users per arm and a 10% baseline, a
  0.1-point lift gives $p = 0.019$, yet its 95% interval runs from about 0.02 to 0.18 points. With a 1-point
  MDE the whole interval lies below the effect worth acting on. Report the interval and compare it with the MDE.
- **Observational comparisons.** Without randomisation, a difference is causal only under identification
  assumptions: exchangeability given measured confounders, positivity and consistency. The Library note's
  example reverses a crude +0.06 risk difference to a standardised −0.15 (Hernán and Robins, per the Library note).
- **Interference and many metrics.** Users in one arm can affect the other (marketplaces, social feeds). Testing
  twenty metrics makes one of them significant by chance.

## Interview angle

**60-second answer.** Metric first: one primary metric the decision depends on, plus guardrails. Randomise at the
unit that matches how users experience the change. Fix the MDE with the product owner, compute the per-arm sample
size for two proportions at the chosen $\alpha$ and power, and convert it to a duration in whole weeks. Before
reading any outcome, check the assignment for sample ratio mismatch. At the end, test the difference with a
pooled two-proportion z-test and report the interval against the MDE.

**Follow-ups.**
- *The result is significant on day three. Ship it?* Not on that basis. A look before the planned horizon
  inflates false positives, so the analysis must follow the pre-registered plan, or use sequential methods.
- *The split is 50.6 / 49.4. What now?* Treat it as a broken experiment until the cause is found. A small
  imbalance at large $n$ is still evidence of a bug.

## Connections

- **Requires:** `cml/t-tests-and-anova` (comparing two means and their standard errors), `cml/bernoulli-binomial-and-poisson`
  (the binomial model behind conversion counts).
- **Unlocks:** `cml/cuped-delta-method-and-design-effect` (variance reduction and clustered units), `cml/chi-square-tests`
  (the SRM check), `cml/multiple-testing-corrections` (many metrics).
- **Lineage:** Fisher's randomised experiments (1935) are the root. The online-experiment practice is laid out in
  Kohavi, Tang & Xu.
