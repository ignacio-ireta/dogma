## Hint 1 — concept

Two sums of raw values lose precision when the values are large compared with their spread. Welford's
update avoids that by tracking the deviation from the running mean. You only need three numbers at any
time: how many values you have seen, their mean, and M2.

## Hint 2 — formula / approach

For each new value x, with count n after including it: delta = x − mean (using the old mean),
mean ← mean + delta / n, and M2 ← M2 + delta · (x − mean) (using the new mean). At the end,
the sample variance is M2 / (n − 1). Raise `ValueError` before dividing when the count is zero.

## Hint 3 — code shape

Initialise `count = 0`, `mean = 0.0`, `m2 = 0.0`. Loop with `for value in xs:` once, converting with
`float(value)`, and apply the three updates. After the loop, check `count == 0` and raise, then return
`(count, mean, m2 / (count - 1) if count > 1 else math.nan)`.
