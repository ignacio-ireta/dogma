# I-32 catalogue: typical mistakes and allowed freedoms, by kind of problem

*Read the one section your runbook points to. A line here is a prompt, not a requirement. A
mistake counts only if it breaks a sentence in that problem's `problem.md`. A freedom counts
only if no sentence there fixes the choice.*

Each section has three lists:

- **Mistakes:** what learners get wrong. Turn one into a wrong solution (`reject_*.py`).
- **Freedoms:** choices a correct solution may make differently from the reference. Turn one
  into a correct solution (`accept_*.py`). If the tests reject it, that is a finding.
- **Inputs that expose it:** what a test input needs so that the mistake changes the result.

## Numeric

Mistakes:

- **Divisor.** `n` where the Contract says `n - 1`, or the reverse. A sum where a mean is asked.
- **Axis.** Reduces over rows instead of columns. Drops `keepdims`, so `(n,)` broadcasts wrongly
  against `(n, d)`.
- **Shape of the result.** `(n, 1)` for `(n,)`. A 0-d array where a 1-d array is asked.
- **Integer input.** No cast to float: division truncates, or an int array stores rounded values.
- **In-place change of an input.** `x -= x.mean()`, `x /= scale`, sorting in place. Note that
  `np.asarray(x, dtype=np.float64)` returns the caller's own array when it is already float64.
- **Unstable formula** where the Contract asks for stability: `log(1 + exp(x))`, `exp(x)`
  without subtracting the maximum, `1 / (1 + exp(-x))` for very negative `x`, variance as
  `E[x²] − E[x]²`, a squared length that overflows.
- **Guard missing or too narrow.** Empty input, a zero denominator, a non-finite entry, `<` where
  the bullet needs `<=`.
- **Ties.** First maximum against last. An unstable sort where the order of equals is stated.
- **NaN.** Propagates where the Contract says ignore, or the reverse.
- **Off by one** in a window, a lag or a slice end.
- **Convention.** Ascending against descending eigenvalues. Forgetting to centre. Forgetting the
  intercept column. A documented default value changed.

Freedoms:

- A NumPy scalar where the reference returns a Python float, when the Contract says "a float".
- Another rounding path: summing in another order, going through a normalised intermediate, a
  closed form against a stable iteration. Only the last digits differ.
- The sign of an eigenvector or singular vector, when the Contract leaves the sign open.
- The order among exact ties, when the Contract does not state it.
- A loop where the reference is vectorised.

Inputs that expose it: sizes 1 and 2; `n ≠ d`, so that an axis mix-up changes the shape; columns
on different scales; an input whose mean is not 0, whose largest entry is not 1 and whose norm is
not 1; integer arrays; a float64 array passed as it is (it shows in-place writes); ties; negative
values.

## Torch

Mistakes:

- **Breaks the graph.** `.detach()`, `.item()`, `.numpy()`, or `torch.no_grad()` around a value
  that needs gradients. A result rebuilt from Python floats.
- **dtype or device.** `torch.tensor(...)` or `torch.zeros(...)` without the input's `dtype` and
  `device`, so a float32 input gives a float64 result.
- **In-place operation** on an input or on a tensor the backward pass needs (`x.add_`, `+=`,
  `x[mask] = 0`).
- **Wrong `dim`,** or a missing `keepdim`, in a mean, sum, norm or softmax.
- **Epsilon in the wrong place.** Inside against outside the square root. Added to the variance
  against the standard deviation.
- **Train against eval.** Batch statistics used at eval. Dropout active at eval. Running
  statistics not updated, or updated while tracking gradients.
- **Biased against unbiased variance.**
- **Masks.** Applied after the softmax. A causal mask off by one. A padding mask that broadcasts
  over the wrong axis. A boolean mask where an additive one is asked.
- **Parameters.** A plain tensor attribute instead of `nn.Parameter`. A missing bias.
- **Optimiser steps.** No bias correction. Weight decay added to the gradient where it should be
  decoupled. A momentum buffer that starts wrongly on the first step.
- **Scaling.** A missing `1 / sqrt(d_k)`. Dropout that does not divide by the keep probability.
- **Shape.** Fails for a batch of one or a sequence of one. `view` on a non-contiguous tensor.

Freedoms:

- `einsum`, `matmul` or `@`. `reshape` or `contiguous().view`.
- Equal formulas whose last digits differ (`x * rsqrt(v)` against `x / sqrt(v)`).
- A new tensor against a view, when the Contract does not say.

Inputs that expose it: float32 inputs; inputs with `requires_grad=True` and a backward call;
a batch of one; a tensor made non-contiguous by `transpose`; values far from zero, so that the
place of epsilon matters; both `train()` and `eval()` modes.

## Plain Python and algorithms

Mistakes:

- **Boundary.** `<` against `<=`. `range(n)` against `range(n - 1)`. The first or last element
  skipped. Empty and single-element inputs.
- **Wrong thing returned.** The upper bound for the lower bound. A value for an index. `-1` for
  `len`.
- **Duplicates and ties.** Counted once where they count each time. Ties in the wrong order.
  Insertion order lost where it is promised.
- **Mutates the input** (sorts in place, pops).
- **Complexity,** only when Constraints promise one: a linear scan for a binary search, a nested
  loop for a hash map or heap, a full re-sort at every step.
- **Recursion** where the input can be deep and the Contract says it must be handled.
- **Type strictness,** only when stated: `bool` accepted as `int`, a tuple where a list is asked.
- **Division and rounding.** `/` against `//`. Floor against ceiling for a count.
- **Units in sizing arithmetic.** Bits against bytes, MB against MiB, a missing factor (layers,
  heads, two tensors for keys and values).
- **Strings.** Case, surrounding spaces, the empty string, splitting on one space against any
  whitespace.
- **State across calls.** A mutable default argument. A module-level cache.

Freedoms:

- Iterative against recursive.
- Any order, when the Contract says "in any order".
- Another valid tie-break, when none is stated.

Inputs that expose it: empty; one element; all equal; already sorted and reverse sorted;
duplicates at both ends; negative numbers; very large ints; the same function called twice.

## pandas

Mistakes:

- **Missing keys.** `groupby` drops NaN keys by default. A `merge` that is inner where rows
  without a match must stay.
- **Duplicate keys** on both sides of a `merge` multiply rows.
- **`count`, `size`, `nunique`** mixed up (they treat NaN differently).
- **Shape of the frame.** Index not reset. Columns in the wrong order or with the wrong names.
  An int column turned float by NaN.
- **Order.** Not sorted as stated, or ties left in arbitrary order.
- **Mutates the input frame** (assigns a column on the argument).
- **Cleaning.** Case or spaces not normalised where the Contract says they are.
- **Mean of group means** where the overall mean is asked. A division by an empty group.

Freedoms:

- `groupby().agg`, `pivot_table` or an explicit `merge`.
- Row order, when the Contract says it does not matter.

Inputs that expose it: a NaN key; duplicate keys on both sides; a group of one row; an empty
frame; unsorted rows; keys that differ only in case or spaces.

## SQL

Mistakes:

- **`JOIN` against `LEFT JOIN`.** Rows with no match disappear, or appear as NULL groups.
- **`COUNT(*)`, `COUNT(col)`, `COUNT(DISTINCT col)`** mixed up.
- **NULL.** `= NULL`. `NOT IN` over a set that holds NULL. A missing or misplaced `COALESCE`.
- **`WHERE` against `HAVING`.** A filter applied on the wrong side of an aggregation or a window.
- **Windows.** `ROW_NUMBER`, `RANK` and `DENSE_RANK` on ties. A missing `PARTITION BY`. The
  default frame, which includes peers of the current row, where `ROWS` is needed. No tie-breaker
  in `ORDER BY`.
- **Dates.** An inclusive end where it is exclusive. A seven-day window that covers eight days.
  Truncating to the month against comparing text.
- **Fan-out.** Joining two one-to-many tables before aggregating doubles the sums.
- **`DISTINCT`** forgotten, or added where duplicates must count.
- **Integer division,** and `AVG` over integers.
- **`UNION` against `UNION ALL`.**
- **Top N.** `LIMIT` that cuts a tie which must be kept whole.

Freedoms:

- A CTE or a subquery. `ON` or `USING`. `CASE` or `FILTER`. A window or a self-join.
- Row order, when the Contract does not require `ORDER BY`.

Inputs that expose it: the wrong query shows which rows the fixtures lack. Look for NULL values,
ties, a user with one row, an empty group, a row exactly on a date boundary.

## Random draws

Read this with your main section when the entry takes `rng` or `seed`.

Mistakes:

- Ignores the generator it was given and uses the global one or a fresh `default_rng()`.
- Draws in another order, or another number of values, than the Contract fixes.
- With replacement where it must be without, or the reverse.
- `<` against `<=` on a uniform draw. Off by one in the sampled range.
- Wrong scale: no division by the keep probability, weights not normalised.
- One draw reused for two purposes.

Freedoms:

- Any way of using the generator, **when no clause fixes the order of draws**. If the tests
  compare exact random streams and no clause fixes the order, a correct solution that draws
  differently will fail. Write it, and record the refusal as `overreach`.

Inputs that expose it: two calls with the same seed (results must match); two different seeds
(results must differ); a global seed set before the call (it must not change the result).

## Classes

Read this with your main section when the entry is a class.

Mistakes:

- State shared between instances (a class attribute, a mutable default).
- The constructor keeps the caller's list and changes it later.
- A method fails when called first (`get` before any `put`). Capacity 0 or 1.
- Updating an existing key takes a different path from inserting, and skips a step.
- Returns its internal list or dict, so the caller can change the object's state.

Freedoms:

- The internal data structure and the names of private attributes. Tests may use only the
  interface the Contract names.

Inputs that expose it: two instances used in turn; the smallest capacity; the same key written
twice; a returned value changed by the caller, then read again.
