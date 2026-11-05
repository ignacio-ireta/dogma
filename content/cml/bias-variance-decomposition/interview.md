# Interview notes: Bias-variance decomposition

## Q1. Explain the bias-variance trade-off.

**60-second answer.** Fix a test point and draw training sets from the same process. The expected squared error of a
prediction splits into three terms: the noise variance, which no model can remove; the squared bias, which is the gap
between the average model and the truth; and the variance, which is how much the models disagree with each other. The
derivation uses only that the noise has mean zero and is independent of the model, and that the deviations from the
average model have mean zero. Model choice and regularisation move bias and variance in opposite directions, so the
goal is the total, not either term alone.

**Follow-ups.**
- *Which term can I change?* Bias and variance, through capacity, features and regularisation. The noise term is fixed
  by the data-generating process.
- *What does a more flexible model do?* It usually lowers bias and raises variance. The table in the lesson shows bias
  falling from 0.212 at degree 1 to 0.0001 at degree 5, while variance rises from 0.011 at degree 1 to 0.087 at
  degree 9 (it dips at degree 3).

## Q2. When does the classical trade-off break down?

**60-second answer.** The trade-off describes the typical case, and the decomposition is always true, but the picture
of error rising after a minimum can fail. Over-parameterised models, with more parameters than training rows, can show
a second descent: test error falls again after an interpolation peak. In one minimum-norm least-squares check, test
error peaked when the number of features equalled the number of rows, then fell again. I would test the curve rather
than assume a U shape.

**Follow-ups.**
- *Did the second descent beat the small models?* Not in that check. The median test MSE was 0.74 at 100 features and
  0.86 at 200, against 0.025 at 5 features, so the second descent was real but not the best result.
- *What drives the peak?* Near the interpolation threshold, the fit is nearly unstable, so small noise produces large
  swings in the coefficients.

## Q3. How would you estimate the variance of a model in practice?

**60-second answer.** I would train the model on many training sets drawn from the same process, store the predictions
on a fixed test grid, and compute the spread across the models at each point. For a high-variance model, the estimate
itself is unstable. I would check it across seeds and report the range, not one number. I would also check that the
decomposition identity holds on my own predictions.

**Follow-ups.**
- *How many training sets?* Enough that the estimate stops moving. Degree 9 with 60 rows varied about five-fold
  across seeds 0 to 6, so one seed is not enough.
- *Can I get the variance from one training set?* Not directly. Resampling methods approximate it, with their own bias.

## Common mistakes

- Dividing by $B - 1$ when the variance is over the models, not a sample.
- Squaring the average error across points instead of the error at each point.
- Forgetting the noise term when comparing a model's expected error with its validation error.
- Reading a single seed's variance as the model's variance.
- Assuming the error curve always rises after its minimum.
