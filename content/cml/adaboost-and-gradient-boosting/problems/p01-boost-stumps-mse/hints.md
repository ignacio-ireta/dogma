## Hint 1 — concept

Each round asks: where is the current model wrong, and which single cut of one feature best corrects it? The residuals y − F are the errors left to explain, so the next stump is fitted to them. Two things must match the contract exactly. Use the mean of the training targets as the starting point, and the stump's leaves are the means of the residuals on each side of the cut, not the parent mean.

## Hint 2 — formula / approach

Start with F₀ = mean(y). For each round, r = y − F. For each feature and each midpoint between consecutive distinct values, the gain is the drop in sum of squared errors when each child is predicted by its own mean. Equivalently, with child sums S_L, S_R and sizes n_L, n_R, the gain is S_L²/n_L + S_R²/n_R − S²/n. Keep the best candidate by scanning features in order and thresholds in order, and replace it only when a later gain is larger by more than 1e-9. Then update F ← F + learning_rate · stump.

## Hint 3 — code shape

Validate every argument first. Keep the fitted values in an array, and the stumps in a list of (feature, threshold, left leaf, right leaf). In the per-feature search, sort the column with `np.argsort`, take prefix sums of the sorted residuals, and read the candidate cuts where consecutive sorted values differ. Store each stump as you go. `predict` starts from F₀ and adds `learning_rate` times each stored stump, in the same order used in training, so the training predictions and `predict` on the training rows agree exactly.
