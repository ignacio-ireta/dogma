## Hint 1 — concept

Raw agreement can look high even when two raters agree by chance. Kappa measures how much of the
possible agreement beyond chance was actually achieved. It is zero when the raters agree no more
than their label frequencies predict, and one when they agree perfectly.

## Hint 2 — formula / approach

Let `p_o` be the fraction of items with equal labels. For each label, take the fraction of items
rater A gave it and the fraction rater B gave it, multiply them, and sum over the label set; that
sum is `p_e`. Then `kappa = (p_o - p_e) / (1 - p_e)`. The denominator is zero only when both raters
used one label throughout, and that case needs its own error.

## Hint 3 — code shape

Validate the label set, the lengths and the membership of every label before any arithmetic. Count
the matching positions for `p_o`. Loop over the label set and accumulate the product of the two
marginal fractions for `p_e`. Check the denominator, then return the quotient.
