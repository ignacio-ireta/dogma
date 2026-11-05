# Interview notes — Missing values and outliers

## Q1. How do you handle missing values in a numeric feature?

**60-second answer.** First I ask why the values are missing. If missingness is independent of both observed and unobserved values,
deletion is unbiased and simple. Under MAR, a correctly specified conditional model can support imputation, but a median fill is
still only a predictive baseline. Under MNAR, identification generally needs extra assumptions or external
information; I preserve missingness and discuss sensitivity. In practice I impute with the training median, add an indicator column, and fit both inside each
training fold so the held-out rows never shape the fill.
**Follow-ups.**
- *Why not drop the rows?* MCAR is sufficient for an unbiased complete-case sample; other special models can also justify deletion, and it throws away the rows that other
  features are still informative about.
- *How do you test the mechanism?* Compare the rows with and without the gap on the other variables. A systematic
  difference is evidence against MCAR, but finite-sample differences alone do not prove the mechanism, and
  observed data generally cannot distinguish MAR from MNAR.

## Q2. How do you detect and treat outliers?

**60-second answer.** I flag with a rule that is robust to the outliers it looks for. The median and the
median absolute deviation give a modified z-score, with the usual cutoff of 3.5. The classic z-score can
hide a single extreme value because the value inflates the mean and the standard deviation. Flagging is
not removal. I check whether the value is an error. If it is real, I keep it, transform the feature, or
cap it, and I say which, because each changes the target being modelled.
**Follow-ups.**
- *What if most values are identical?* The MAD is zero, so the score is undefined. The rule must state
  the fallback, which here is to flag values that differ from the median.
- *IQR fences or the modified z-score?* Both use robust order statistics. The IQR fences are simpler to explain, and
  the modified score gives one number per value.

## Common mistakes

- Filling with the mean of the whole table, which includes the test rows.
- Dropping every row with a gap without asking whether the gap is random.
- Removing an outlier without checking whether it is an error or a real observation.
- Using the classic z-score to find an outlier that is large enough to inflate its own statistics.
- Letting a zero MAD silently mean "no outliers".
