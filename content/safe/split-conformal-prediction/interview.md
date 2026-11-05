# Interview notes — Split conformal prediction

## Q1. Build a 90% prediction interval for a regression model without distributional assumptions.

**60-second answer.** Hold out $n$ calibration points that the model never saw. For each, compute the absolute residual. Take the $\lceil (n+1)\cdot 0.9\rceil$-th smallest residual as the threshold $\hat q$, and output the prediction plus or minus $\hat q$. If calibration and test points are exchangeable, the new label falls inside with probability at least 0.9, averaged over calibration sets and test points. No Gaussian or homoscedastic assumption is needed, and the model itself is untouched.

**Follow-ups.** Why $n+1$ rather than $n$? The test point is one of the $n+1$ exchangeable values, so it has to count toward the rank. Using $n$ undercovers slightly in small samples. What if the threshold rank exceeds $n$? The interval is infinite, which is the honest answer when the calibration set is too small for the target level.

## Q2. Does conformal prediction give conditional coverage?

**60-second answer.** No. The guarantee is marginal: it averages over the calibration set and the test point. A 90% interval can cover 60% of one subgroup while covering 95% overall. Conditional coverage for every input is not achievable without assumptions on the data, so the honest move is to report coverage by group and by score bucket.

**Follow-ups.** What does a per-group report look like? Coverage and mean width for each group, with the group sizes. Can you fix it? Group-wise calibration gives per-group guarantees, at the cost of fewer points per group.

## Q3. What breaks a conformal interval in production?

**60-second answer.** Exchangeability. Drift changes the residual distribution, so a threshold calibrated last quarter can undercover today. Calibration leakage is the other common failure: if the calibration points were used in training, the residuals are optimistic and the intervals are too narrow. A recent held-out window reduces the gap, but it does not restore exchangeability, so I also monitor coverage on labels as they arrive.

**Follow-ups.** How would you detect drift in coverage? Track the empirical coverage on a rolling window of labelled outcomes against the target, with a binomial interval for the noise. Does an input shift with the same $P(Y\mid X)$ break it? Yes. The calibration scores then describe a different region of inputs. In a simulation with a fixed predictor, coverage fell from about 0.90 inside the calibration region to about 0.55 outside it. The guarantee does not cover this case, and any remedy needs a stated model of the shift.

## Common mistakes

- Using the $(1-\alpha)$ empirical quantile of $n$ scores without the $+1$ in $n+1$.
- Calculating the rank as a float product and rounding up at an integer boundary.
- Calibrating on training residuals, which gives intervals that are too narrow.
- Reporting marginal coverage as if it held for every input or subgroup.
