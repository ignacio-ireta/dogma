# Interview notes — Prefix sums

## Q1. Count subarrays with a target sum when values can be negative.
**60-second answer.** A subarray ending at j has sum k when an earlier prefix
equals the current prefix minus k. Keep frequencies of earlier prefixes, starting
with the empty prefix zero counted once. Look up matches before adding the current
prefix, which avoids counting empty subarrays for a zero target. Expected time
and space are linear in the number of entries under ordinary dictionary costs.
**Follow-ups.** Why frequencies rather than a set? Repeated prefix values represent
different starting positions. Why not two pointers? Negative values destroy the
monotone relationship used to shrink a window. What about huge integers? Python
integers preserve exact sums; the bit cost still grows with their size.

## Q2. How would you answer many rectangle-sum queries?
**60-second answer.** Build a table with a zero border that stores sums above and
left of each corner. A rectangle is the lower-right prefix minus two strips plus
their overlap. Construction takes O(RC) arithmetic operations and each static
query takes four table lookups. State whether coordinates are inclusive.
**Follow-ups.** What if data changes? Use a structure supporting updates, such as
a two-dimensional Fenwick tree, or rebuild the static table. Why do floats cause
problems? Large accumulated sums can erase small differences through cancellation.

## Common mistakes
- Forgetting the empty-prefix seed.
- Inserting the current prefix before looking up matches.
- Storing only the presence of a prefix when multiplicity matters.
- Applying an integer-exact argument to floating-point equality without a policy.
