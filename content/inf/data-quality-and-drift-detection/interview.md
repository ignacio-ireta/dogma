# Interview notes: Data quality and drift detection

## Q1. Accuracy dropped in production. What do you check first?

**60-second answer.** I check the pipeline before the model: schema, null rates, value ranges and freshness, compared
with the last good week. These checks expose pipeline failures without a statistical test. If the data are clean, I compare
each input feature with its training reference, using the effect size (a KS distance or a PSI) and not the p-value
alone, because large samples make trivial differences significant. Labels often arrive late, so I check label shares
and error by segment only when they are available. The response depends on the stage: fix the pipeline, recalibrate,
or retrain on recent labelled data. I would not retrain first, because retraining on corrupted data repeats the bug.

**Follow-ups.**
- *How do you monitor without labels?* Watch the input distributions and the prediction distribution against the reference.
- *Why not retrain on a schedule?* It can be right, but it hides a data bug, and the labels it needs may not exist yet.

## Q2. How do you tell covariate drift from concept drift?

**60-second answer.** Covariate drift changes the inputs, P(x), and concept drift changes the rule from inputs to labels,
P(y given x). Covariate drift shows up in the feature distributions, so PSI or KS against the reference reveals it
without labels. Concept drift shows up only when the errors change for the same kind of input, so I need labelled
outcomes and error by segment over time. A feature shift with stable accuracy is a warning to watch, not an emergency.
A stable feature distribution with rising error points to concept drift or a pipeline change that alters the labels.

**Follow-ups.**
- *Can a KS test detect a concept drift?* Not directly, because it compares inputs. It can only show that inputs moved.
- *What is label drift?* A change in P(y). It is visible in label shares once labels arrive, and it can be corrected by adjusting priors or thresholds.

## Common mistakes

- Treating a pipeline bug as model drift and retraining on bad data.
- Choosing PSI bins from the current sample, which hides the drift.
- Reading a KS p-value without the effect size on a large sample.
- Using the 0.1 and 0.25 PSI thresholds as statistical tests.
- Running a drift test on hundreds of features at alpha 0.05 and treating every alert as real.
