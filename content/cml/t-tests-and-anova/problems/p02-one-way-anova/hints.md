## Hint 1 — concept

ANOVA splits the total variation of the data into a part explained by group membership and a part left
inside the groups. The ratio of the two, scaled by their degrees of freedom, is F. Large F means the group
means differ more than the within-group noise would explain.

## Hint 2 — formula / approach

With grand mean `m`, `SSB = Σ n_i (m_i − m)²` and `SSW = Σ (x − m_i)²`. Use `df1 = k − 1` and
`df2 = N − k`. Then `F = (SSB / df1) / (SSW / df2)`, and `p` is the upper tail of the F distribution with
those two degrees of freedom. `scipy.stats.f.sf` gives that tail.

## Hint 3 — code shape

Validate every group first: 1-D, non-empty, finite. Check `N > k` and `SSW > 0`. Compute `SSB` and `SSW`
from the group arrays, form F, and return `(F, f.sf(F, df1, df2))` as floats. Do not write `1 - f.cdf(...)`:
it returns exactly zero for small p-values.
