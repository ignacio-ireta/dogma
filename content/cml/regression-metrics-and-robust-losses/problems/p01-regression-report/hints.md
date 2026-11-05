## Hint 1 — concept

Every score is a summary of two things: the residuals `y - y_pred`, and the spread of `y_true` around
its own mean. R² compares the squared residuals with the squared deviations of `y_true`. The
denominator is zero only when every `y_true` value is equal, so that case must be rejected first.

## Hint 2 — formula / approach

MAE is the mean of the absolute residuals, MSE is the mean of the squared residuals, and RMSE is the
square root of MSE. R² is `1 - SS_res / SS_tot`, where `SS_tot` centres `y_true` at its own mean. Do
not clip R². Validate the inputs first: both 1-D, equal length, non-empty, finite. Then check the
zero-variance case before you divide.

## Hint 3 — code shape

Copy both inputs into float64 arrays, so you never write to the caller's data. Check the rules in
order and raise `ValueError` with a message for each. Compute `residual = truth - pred` once and reuse
it. Return a dict whose values are plain Python floats.
