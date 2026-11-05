## Hint 1 — concept

A round reads three things: how much weight the classifier's mistakes carry, what vote that share earns, and how the weights move afterwards. Normalise the weights before anything else, so the error is a probability. A classifier at or above chance earns no vote, and a perfect one would earn an infinite one, so both need explicit handling.

## Hint 2 — formula / approach

With normalised weights w, the weighted error is ε = Σ wᵢ over the mistaken samples. If ε ≥ ½, return alpha 0 and the weights unchanged. If ε = 0, raise ValueError. Otherwise alpha = ½ · ln((1 − ε) / ε), and each weight is multiplied by e^(−alpha · yᵢ · predᵢ), which is e^alpha for a mistake and e^(−alpha) for a correct sample. Divide by the sum at the end.

## Hint 3 — code shape

Convert the three inputs to float arrays and check them in the order the contract lists: non-empty and one-dimensional, equal lengths, labels in {−1, +1}, predictions in {−1, +1}, non-negative finite weights, positive sum. Compute `w = weights / total` into a new array. Build a boolean mask of the mistakes and sum `w` over it. Branch on the error, then compute alpha with `math.log`, the multiplier per sample with `np.exp` on `-alpha * y * pred`, and return the normalised product.
