# Train-fitted preprocessing for mixed columns

Implement `fit_transform_pipeline(train, test, numeric, categorical)`. It preprocesses a training
table and a test table with one set of fitted rules. Every rule is learned from the training table
alone, and the test table is only transformed with those rules. The output columns, their order and
the handling of missing or unseen values are part of the contract, so two correct implementations
return the same arrays.

## Contract

- `train`, `test`: pandas DataFrames. Every column named in `numeric` or `categorical` must be present
  in both frames.
- `numeric`: list of column names with numeric values. Missing values are NaN.
- `categorical`: list of column names with string values. Missing values are NaN.
- A column may appear at most once across both lists. At least one column must be named.
- Returns `(train_out, test_out)`: two float64 NumPy arrays with shapes `(len(train), d)` and
  `(len(test), d)`. Both have the same `d` columns, in this order: first the numeric columns, in the
  order given, then the one-hot block of each categorical column, in the order given.
- Raises `ValueError` when no column is named, when a column is named twice, when a named column is
  missing from a frame, when a numeric column has no observed value in `train`, or when a categorical
  column has no observed value in `train`.
- Must not mutate `train` or `test`.

**Protocol (exact).**

*Numeric column.*

1. Median of the observed training values. The median is the usual one, the mean of the two middle
   values for an even count.
2. Fill every missing value in `train` and `test` with that training median.
3. Let `mean` and `scale` be the mean and the population standard deviation (denominator `n`) of the
   filled training column. If the filled training column is constant (its maximum equals its
   minimum), `scale` is `1.0`. Compute both without overflow or underflow: columns of values near
   `1e200` or `1e-200` still have a finite, nonzero `scale`.
4. Transform both columns with `(x - mean) / scale`.

*Categorical column.*

1. Mode of the observed training values. Ties go to the smallest value.
2. Fill every missing value in `train` with that mode.
3. `categories` are the sorted distinct values of the filled training column.
4. A row gets a `1.0` in the column of its category and `0.0` in the others. In `test`, a missing
   value is treated as the training mode. A value that is not in `categories` gives an all-zero block.

## Examples

```python
>>> train = pd.DataFrame({"age": [20.0, np.nan, 40.0, 30.0],
...                       "city": ["north", np.nan, "south", "north"]})
>>> test = pd.DataFrame({"age": [np.nan, 50.0], "city": ["west", np.nan]})
>>> train_out, test_out = fit_transform_pipeline(train, test, ["age"], ["city"])
>>> train_out     # age: median 30 fills the gap; city: mode "north" fills it; categories north, south
[[-1.41421356,  1.,  0.],
 [ 0.        ,  1.,  0.],
 [ 1.41421356,  0.,  1.],
 [ 0.        ,  1.,  0.]]
>>> test_out      # "west" is unseen, so its block is zeros; the missing city becomes "north"
[[ 0.        ,  0.,  0.],
 [ 2.82842712,  1.,  0.]]
```

```python
>>> train = pd.DataFrame({"plan": ["pro", "basic", "pro", "basic", np.nan]})
>>> test = pd.DataFrame({"plan": [np.nan, "pro"]})
>>> fit_transform_pipeline(train, test, [], ["plan"])
# "basic" and "pro" tie for the mode, so the smallest, "basic", fills the gap
# train: [[0,1],[1,0],[0,1],[1,0],[1,0]]    test: [[1,0],[0,1]]
```

## Constraints

- Allowed: pandas, NumPy, and the standard library (for example `collections.Counter`).
- Not allowed: `sklearn`. The point is to write the fit and transform rules yourself, so that you can
  see where a leak would enter. Do not build the test table's dummy columns with `get_dummies` on a
  concatenated frame.

## Done looks like

All public tests pass, then **Submit** passes the hidden tests. Those compare your output with a
scikit-learn `ColumnTransformer` (median imputation and standardization for the numeric columns,
most-frequent imputation and a one-hot encoder that ignores unknown categories for the categorical
ones) on random mixed tables with missing values, unseen categories and a constant column. They also
check: that the test table never changes the fitted training output; that missing test values take the
training median and mode; that unseen test categories give all-zero blocks; the mode tie rule; column
order; output shape; a train column with no observed values; invalid column names and duplicates; and
inputs left unchanged.
