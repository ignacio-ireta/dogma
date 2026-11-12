# Cohen's kappa between two raters

Two judges, a human and an automated judge, label the same items from a fixed label set. Raw agreement
looks high when one label dominates, because two raters that both say "good" most of the time agree
by chance. Cohen's kappa corrects for that chance. Implement
`cohens_kappa(rater_a, rater_b, labels)`.

## Contract

- `rater_a`, `rater_b`: sequences of `str` of the same length `n >= 1`. Item `i` was labelled
  `rater_a[i]` by one rater and `rater_b[i]` by the other.
- `labels`: a non-empty sequence of distinct `str`. It is the full label set. Every value in `rater_a`
  and `rater_b` must belong to it, and labels that never occur are allowed.
- Let `p_o` be the observed agreement: the fraction of items where the two labels are equal.
- Let `p_a(l)` and `p_b(l)` be the fraction of items each rater gave label `l`. Expected agreement is
  `p_e = sum over labels l of p_a(l) * p_b(l)`.
- Returns `kappa = (p_o - p_e) / (1 - p_e)` as a `float`. It is `1.0` for perfect agreement and can be
  negative.
- **Degenerate case.** When `p_e == 1`, both raters gave the same single label to every item, so the
  denominator is zero. Raise `ValueError` with a message that says kappa is undefined. Do not return
  `nan`.
- Raises `TypeError` when a label is not a `str`, and `ValueError` when the sequences differ in length,
  are empty, a label is outside `labels`, or `labels` is empty or has duplicates.
- Must not mutate inputs.

## Examples

```python
cohens_kappa(["yes", "yes", "no", "no"], ["yes", "no", "no", "no"], ["yes", "no"])
# p_o = 3/4, p_a(yes) = 1/2, p_b(yes) = 1/4, p_e = 1/2*1/4 + 1/2*3/4 = 1/2
# kappa = (3/4 - 1/2) / (1 - 1/2) = 0.5
```

```python
cohens_kappa(["a", "b", "c"], ["a", "b", "c"], ["a", "b", "c", "d"])  # -> 1.0
cohens_kappa(["a", "a", "a"], ["a", "a", "a"], ["a", "b"])  # ValueError: p_e == 1
```

## Constraints

Allowed: Python standard library and NumPy. Not allowed: `sklearn` (it is the reference for the hidden
tests) and `statsmodels`. Count labels yourself, and compute the two marginals from those counts.

## Done looks like

The public examples pass, plus the perfect-agreement and degenerate cases. Hidden tests compare against
scikit-learn's `cohen_kappa_score` on seeded random label sequences, including unused labels in the
label set, check that raw agreement alone is rejected in a skewed-label example, check symmetry in the two
raters, check that negative kappa is possible, and check the degenerate cases and the input-validation
families.

The hidden tests draw their bulk inputs from the real datasets `breast_cancer` and `iris` (see
`docs/DATASETS.md`).
