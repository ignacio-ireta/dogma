# Interview notes: Random forests and feature importance

## Q1. Why does a random forest beat a single deep tree?

**60-second answer.** A single deep tree has low bias and high variance: small changes in the data move its splits. A random forest averages many trees, each grown on a bootstrap sample with a random subset of features at each split. Averaging reduces variance, and the feature subsampling makes the trees less correlated, which matters most, because the variance of an average is $\rho\sigma^2+\frac{1-\rho}{B}\sigma^2$. The first term is the correlation term, and feature subsampling shrinks it. The out-of-bag rows give an honest error estimate without a separate holdout.

**Follow-ups.**
- *Does more trees always help?* It helps the $1/B$ term only. Past a few hundred trees the gain is usually small.
- *What does OOB error cost?* Nothing extra: each row is predicted by the trees that did not see it.

## Q2. Which features matter in your forest, and how do you know?

**60-second answer.** Impurity importance is cheap, since it comes from the splits during training, but it is biased toward continuous and high-cardinality features. A pure-noise continuous column can get almost all of the impurity importance. I would use permutation importance on held-out data instead: shuffle one column, measure the score drop, and repeat to get a spread. A zero drop alone does not prove the model ignores the feature; redundancy and the metric can hide its effect. When features are correlated, the credit is shared, so I would test correlated groups together, by permuting the whole group.

**Follow-ups.**
- *Why not permute on training data?* The model has memorised part of it, so the drop mixes importance with overfitting.
- *What is the risk of permuting correlated features?* The shuffled rows can be unlikely combinations, and the drop then mixes importance with extrapolation.

## Common mistakes

- Reporting impurity importance as if it were a causal or stable ranking.
- Measuring permutation importance on the training set.
- Trusting OOB error on time-ordered or grouped data, where bootstrap rows are not independent.
- Permuting one of two correlated features and concluding that it is unimportant.
