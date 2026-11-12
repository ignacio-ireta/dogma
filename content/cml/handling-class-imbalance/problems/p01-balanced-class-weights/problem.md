# Balanced class weights

Implement `balanced_class_weights(y)`. It returns one weight per class such that every class has the same total sample weight. Multiplying each sample's loss by that weight balances class counts; realised losses can still differ because the prediction errors differ. This is the rule behind scikit-learn's `class_weight="balanced"`. The weights belong to the training loss. They say nothing about how the model should be evaluated.

## Contract

- `y`: a 1-D sequence or NumPy array of labels (integers or strings), length n >= 1, with k distinct labels.
- Returns a dict with one entry per label that occurs in `y`. The value is the Python float n / (k * count), where count is the number of samples with that label.
- The order of the dict is not specified. The keys are the labels exactly as they appear in `y`.
- Raises `ValueError` for empty input and for input that is not 1-D.
- Must not mutate `y`.

## Examples

```python
>>> balanced_class_weights([0, 0, 0, 1])
{0: 0.6666666666666666, 1: 2.0}        # n = 4, k = 2: 4 / (2 * 3) and 4 / (2 * 1)
```

```python
>>> balanced_class_weights(["a", "b", "a", "a"])
{'a': 0.6666666666666666, 'b': 2.0}
```

```python
>>> balanced_class_weights([2, 0, 0, 1, 0, 0, 2, 0, 1])
{0: 0.6, 1: 1.5, 2: 1.5}                # n = 9, k = 3: 9 / (3 * 5), 9 / (3 * 2), 9 / (3 * 2)
```

## Constraints

- Allowed: the Python standard library and NumPy.
- Not allowed: `sklearn`. Do not call `compute_class_weight`. The formula is the point.

## Done looks like

All public tests pass, then **Submit** passes the hidden tests. They compare with scikit-learn's `compute_class_weight("balanced")` on random labels. They check the two identities of the rule: each class contributes n / k to the weighted count, and the total weighted count is n. They check that weights are floats even when the division is exact, that the minority class gets the larger weight, that the weights follow each label and not the order in which labels first appear, string labels, the single-class case, empty and two-dimensional input raising `ValueError`, and that the input is left unchanged.

The hidden tests draw their bulk inputs from the real dataset `adult` (see `docs/DATASETS.md`).
