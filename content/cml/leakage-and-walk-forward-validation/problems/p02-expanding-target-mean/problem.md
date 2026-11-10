# Leakage-safe target mean

Implement `expanding_target_mean(categories, target)`. Target encoding replaces each category by the mean
target of that category. Computed over all rows, the encoding leaks: a row's own target is part of its
feature. This function encodes each row from earlier rows only. Row `i` gets the mean target of the rows
before it that share its category, which is what a model could have known at prediction time.

## Contract

- `categories`: 1-D sequence of hashable labels (ints or strings), one per row, in time order.
- `target`: 1-D array-like of finite real numbers, one per row.
- Rows are ordered by position alone. The position is the clock.
- Returns: a float64 array of length `n`. Entry `i` is the mean of `target[j]` over all `j < i` with
  `categories[j] == categories[i]`. It is NaN when no earlier row shares that category.
- The encoding of row `i` must not depend on `target[i]` or on any later row.
- The encoding is the plain expanding mean over the whole earlier history of the category, however long
  that history is: no recent window, no decay, no smoothing, and no reset of the state partway through.
- Raises `ValueError` when the two inputs have different lengths, when either is not 1-D, or when
  `target` contains NaN or infinity.
- Must not mutate inputs.

## Examples

```python
>>> expanding_target_mean(["a", "b", "a", "a"], [1.0, 5.0, 3.0, 2.0])
array([nan, nan, 1., 2.])      # row 3 sees rows 0 and 2 of category "a": (1 + 3) / 2 = 2
>>> expanding_target_mean([1, 1, 1], [4.0, 6.0, 2.0])
array([nan, 4., 5.])           # row 2 sees 4 and 6: mean 5
>>> expanding_target_mean(["a"], [1.0, 2.0])
ValueError                     # the lengths differ
```

On real `adult` (`load_frame`, 6000 rows in file order as the clock), `expanding_target_mean(df["native.country"],
(df["income"] == ">50K").astype(float))` gives the 5350 United-States rows a mean over all earlier United-States rows.
Here `?` is an ordinary category, not a missing value, so the 41 levels of `native.country` include `?`,
and the first row of each of them is NaN. The encoding of the last United-States row is the mean of its
5349 predecessors, not of the last 500.

## Constraints

- Allowed: NumPy and the standard library.
- Not allowed: `pandas`. Its `groupby`, `shift` and `expanding` idiom is the reference the tests use, so
  you write the running state yourself.

## Done looks like

All public tests pass, then **Submit** passes the hidden tests. They compare the encoding with a brute-force
reference and with pandas' `groupby`, `shift` and `expanding` on random integer and string categories,
including alternating categories where a category's rows are never adjacent, and mixed integer and string
labels. On real `adult` keys (`occupation` and `native.country`, with a dominant level and many rare ones; `?` is an ordinary level) and
the real income target, in file order, they check the same idiom over a long history. They check that a row's
encoding does not change when its own target or any later target changes, that a category's first row is NaN,
that the second row of a category equals the first target, and that large and small magnitudes stay accurate,
including a repeated huge target whose raw sum would overflow. They check empty input, non-finite and non-1-D
inputs, two-dimensional labels (which must raise `ValueError`), and that the inputs are left unchanged.
