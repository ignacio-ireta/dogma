# Interview notes: Regression metrics and robust losses

## Q1. Which regression metric would you report, and why?

**60-second answer.** I start from the decision the number supports. RMSE penalises large misses more
heavily, and it is in the units of the target. MAE is the typical absolute miss, and its minimiser is the
median, so it is the right metric when a big miss should not dominate. R² is relative to the mean predictor.
It is useful for comparing models on one dataset, but it is not a portable accuracy figure, because it
depends on the spread of the target. I would report the metric that matches the cost of an error, and I
would show the residual distribution alongside it.

**Follow-ups.**
- *Why does the mean minimise squared error?* Setting the derivative of the sum of squared deviations to zero
  gives the sample mean.
- *Can R² be negative?* Yes. It is negative when the model's squared error exceeds that of predicting the mean.

## Q2. How would you make a regression robust to outliers?

**60-second answer.** Replace the squared loss with one whose influence is bounded. The Huber loss is
quadratic within a threshold δ and linear outside it, so its gradient is capped at δ. Large residuals still count,
but their residual derivative is capped. High-leverage feature vectors can still have a large effect on coefficients. Alternatively, use the absolute loss, whose minimiser is the
median. The cost is a tuning choice (δ) and a non-smooth point at zero for absolute loss. Before changing the
loss, I would check whether the outliers are errors in the data, because cleaning them may be the better fix.

**Follow-ups.**
- *How do you choose δ?* Pick it from the scale of the residuals, for example a fixed multiple of their median
  absolute deviation, and tune it on held-out data.
- *Does a robust loss change the metric?* It changes the fit. Report the metric you care about, as well as the loss used.

## Q3. A model has R² = 0.85 on the training set and 0.20 on the holdout. What is happening?

**60-second answer.** This suggests overfitting, but a distribution shift or a different target spread can also explain the gap. For nested ordinary least squares on the same rows, training
R² never falls when a feature is added, so it cannot detect overfitting. I would check whether the holdout is drawn
from the same distribution, then reduce the model's flexibility or add regularisation, and compare models on
held-out data using CV. I would also check for leakage, since a leaked feature would inflate the training score.

**Follow-ups.**
- *Would a higher R² on holdout settle it?* Only if measured on the same target spread. Compare RMSE as well.
- *What if the holdout target is nearly constant?* Its R² denominator is small, so R² swings widely. Report RMSE.

## Common mistakes

- Reporting R² without the target's variance, then comparing it across datasets.
- Clipping R² at zero. A negative value is informative.
- Using MSE on heavy-tailed targets without checking the residuals for outliers.
- Calling MAPE a percentage error without guarding against zero targets.
- Computing R² on training data and treating it as generalisation performance.
