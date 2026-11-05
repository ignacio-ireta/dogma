## Hint 1 — concept

A subarray is the difference of two running totals. If you know the total of everything before each
position, then "the subarray between positions i and j sums to k" becomes a question about two totals,
not about every pair of positions. Ask which earlier totals you would need to remember, and why a
dictionary is the right container for them.

## Hint 2 — formula / approach

Let `P_j` be the sum of the first `j` entries, with `P_0 = 0`. The subarray `nums[i:j]` sums to `k` exactly
when `P_j - P_i = k`, so for each `j` you need the number of earlier `i` with `P_i = P_j - k`. Two details
decide the answer. Start the dictionary with the empty prefix `0` already counted once, so subarrays that
begin at position 0 are found. And look up `P_j - k` before you add `P_j` to the dictionary, so that a target
of zero does not match the current prefix with itself.

## Hint 3 — code shape

Validate the argument types first, with `type(x) is int` for `k` and for each entry. Then make one pass.
The skeleton has the three steps in order; fill in the blanks yourself:

```python
counts = {0: ...}  # the empty prefix, already seen once
prefix = 0
total = 0
for value in nums:
    prefix += ...  # update the running total
    total += counts.get(...)  # earlier prefixes that make a subarray sum to k
    counts[prefix] = ...  # record this prefix, after the lookup
return total
```

Test your loop on `[1, 2, 1, 2]` with `k = 3` by hand before you run the tests: it should give 3.
