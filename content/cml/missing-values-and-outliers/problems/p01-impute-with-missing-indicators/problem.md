# Imputation with missing-value indicators

Implement `impute_with_indicators(df, numeric, strategy="median")`. It fills the missing values of the
named numeric columns with a statistic of their observed values, and it records where the values were
missing in extra indicator columns. Missingness is often informative, so the indicator keeps that
signal after the fill. The output layout is part of the contract: two correct implementations return
the same columns, in the same order.

## Contract

- `df`: a pandas DataFrame.
- `numeric`: list of column names. Each name must be a column of `df` with a numeric dtype (booleans
  are not numeric here). Each name may appear once. The list may be empty.
- `strategy`: `"median"` or `"mean"`. Any other value raises `ValueError`.
- Missing values are NaN.
- Returns a new DataFrame, with:
  - the same index and the same original columns, in the same order;
  - each named column that has at least one missing value converted to float64, with its missing
    values filled by the chosen statistic of that column's observed values in `df`;
  - a column `"<name>_was_missing"` of dtype `int8` (1 where the value was missing, 0 otherwise),
    **only** for named columns that have at least one missing value in `df`. These indicator columns
    come after all original columns, in the order the names appear in `numeric`;
  - every other column copied unchanged, including its dtype.
- Raises `ValueError` when `strategy` is unknown; when a name is missing from `df` or is not numeric;
  when a name is listed twice; when a column needs imputation but has no observed value; and when the
  indicator `<name>_was_missing` of a column with a gap already exists in `df`.
- Must not mutate `df`.

## Examples

```python
>>> df = pd.DataFrame({"age": [20.0, np.nan, 40.0, 30.0], "income": [1, 2, 3, 4]})
>>> impute_with_indicators(df, ["age", "income"])
    age  income  age_was_missing
0  20.0       1                0
1  30.0       2                1
2  40.0       3                0
3  30.0       4                0
    # median of the observed ages 20, 40, 30 is 30; income has no gap, so it gets no indicator
```

```python
>>> impute_with_indicators(df, ["age"], strategy="mean")["age"].tolist()
[20.0, 30.0, 40.0, 30.0]
>>> impute_with_indicators(df, ["age"], strategy="max")
ValueError              # strategy must be "median" or "mean"
```

## Constraints

- Allowed: pandas and NumPy.
- Not allowed: `sklearn` and `scipy`. Write the fill and the indicator yourself; the point is to decide
  which columns get an indicator and where the new columns go.

## Done looks like

All public tests pass, then **Submit** passes the hidden tests. Those compare the imputed values and the
indicator values with scikit-learn's `SimpleImputer(add_indicator=True)` on random tables with missing
values in some columns and none in others, for both strategies. They also check: the exact column
layout, with indicators only for columns that had gaps and appended in the order named; the `int8`
indicator dtype; that columns without gaps keep their dtype; that the input frame is unchanged; the
median and mean on a column with an even count and with a single value; large-magnitude values; an
empty frame; an empty list of names; and the invalid cases (unknown strategy, unknown or non-numeric
column, duplicate name, a column with no observed value, and an indicator name that already exists).
