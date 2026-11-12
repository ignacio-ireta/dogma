# Position-debiased pairwise win rate

A pairwise judge sees two answers and says which is better. Judges favour a position: many prefer the
answer shown first. To cancel that, every pair is judged twice, once with candidate `X` in position A and
`Y` in position B, and once with the order swapped. Implement `debiased_win_rate(judgments)`, which turns
the two verdicts for each pair into one score for `X` and returns the average.

## Contract

- `judgments`: a non-empty list or tuple. Each element is a pair `(forward, reverse)`, one entry per
  comparison item.
  - `forward` is the verdict when `X` is in position A and `Y` is in position B.
  - `reverse` is the verdict when `Y` is in position A and `X` is in position B.
  - A verdict is `"A"` (the answer in position A is better), `"B"` (position B is better) or `"tie"`.
- **Per-order reading for `X`.** In `forward`, `"A"` means `X` wins, `"B"` means `Y` wins. In `reverse`,
  `"B"` means `X` wins, `"A"` means `Y` wins. `"tie"` is a tie in either order.
- **Item score for `X`.** Score `1.0` when `X` wins in both orders. Score `0.0` when `Y` wins in both
  orders. Every other item scores `0.5`. This covers ties, one tie with one decisive verdict, and any
  pair where the two orders disagree. Inconsistent pairs count as ties.
- **Win rate.** The arithmetic mean of the item scores, a `float` in `[0, 1]`. Swapping `X` and `Y`
  turns the rate into `1 - rate`.
- Raises `ValueError` when `judgments` is empty, or when a verdict is a `str` outside `{"A", "B", "tie"}`.
- Raises `TypeError` when an element is not a 2-item tuple or list, or a verdict is not a `str`.
- Must not mutate the input.

## Examples

```python
debiased_win_rate([("A", "B"), ("B", "A"), ("A", "A"), ("tie", "tie")])
# -> 0.5    item scores: 1.0 (X both orders), 0.0 (Y both), 0.5 (A then A disagrees), 0.5 (tie)
```

```python
debiased_win_rate([("A", "A"), ("A", "A"), ("A", "A")])
# -> 0.5    pure position bias: every pair is inconsistent, so no judge preference survives
```

## Constraints

Allowed: Python standard library. Not allowed: `statistics.fmean`, `statistics.mean` (use your own
sum and count). Read each pair once; do not fit a model.

## Done looks like

The public examples pass, plus the empty input case. Hidden tests compare against an independent lookup
table of the nine verdict combinations on seeded random judgments, check the position-bias case
(always "A" or always "B") gives one half, check a consistent judge gives one or zero, check that a tie
and a decisive verdict do not count as a win, check that the order of items does not change the rate,
and cover every invalid-input family.

The hidden tests draw their bulk inputs from the real dataset `mt_bench_judgments` (see
`docs/DATASETS.md`).
