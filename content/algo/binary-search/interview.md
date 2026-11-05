# Interview notes: Binary search

## Q1. Find the first position where a sorted array is at least x.

**60-second answer.** Use a half-open interval `[lo, hi)` that starts as `[0, n)`. The invariant is
that every index before `lo` is smaller than `x`, and every index from `hi` on is at least `x`. Test
the middle element. If it is smaller than `x`, set `lo = mid + 1`. Otherwise set `hi = mid`, because
it might be the answer. Loop while `lo < hi`, then return `lo`. This takes O(log n) comparisons. I
check the empty array, an `x` below everything, an `x` above everything, and a run of duplicates,
because those are where off-by-one errors live.

**Follow-ups.**
- *How does the upper bound differ?* Use `<=` in the test, so the loop moves past the run of equal
  values.
- *Why does `hi` start at `n`?* Because `n` is a valid answer when nothing is at least `x`.

## Q2. What is the smallest shipping capacity that delivers everything in D days?

**60-second answer.** This is a binary search over the answer. A capacity is feasible if a greedy
pass can ship everything in at most D days. Feasibility is monotone: a larger capacity never needs
more days. The answer lies in `[max(weights), sum(weights)]`, because one truck must carry the
heaviest parcel and one truck can carry everything. Search that range with the same update rules as
Q1, which costs O(n log(sum)) in total.

**Follow-ups.**
- *Why is the lower end the heaviest parcel and not zero?* A capacity below a single parcel cannot
  ship it, so the check must not accept it.
- *What is the feasibility check?* Fill each day with consecutive parcels until the next one would
  exceed the capacity, then start a new day. Count the days.

## Q3. My binary search loops forever, or is off by one. How do you debug it?

**60-second answer.** State the invariant in one sentence, and check that each update keeps the
answer inside the interval. Then trace the smallest inputs by hand: an empty list, one element, two
elements, and the target below, equal to, and above the element. Two bugs show up at once: `lo = mid`
never ends when the interval has two elements, and `hi = len - 1` misses the case where the answer is
`n`. Finally, compare with the library function on random inputs.

**Follow-ups.**
- *What is the cheapest test that catches both bugs?* A small list of duplicates, and a target above
  every element.

## Common mistakes
- `lo = mid` in place of `lo = mid + 1`, which loops forever on two elements.
- `hi = len - 1`, which misses the answer when every element is smaller than the target.
- Using `<=` in a lower-bound test, which returns the upper bound.
- Searching over answers without checking that the feasibility test is monotone.
- Forgetting that the answer for the smallest capacity must be at least the heaviest parcel.
