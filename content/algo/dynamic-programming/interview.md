# Interview notes — Dynamic programming

## Q1. What is the minimum number of coins for an amount?
**60-second answer.** State: the fewest coins for each value up to the amount. Base case: zero
coins make zero. Transition: one more than the best of the smaller values reached by subtracting
each coin. Order: increasing value, so every read is final. Answer: the entry for the amount,
mapped to -1 when it is unreachable. Cost: the amount times the number of coin types. I would not
use greedy, because the largest coin first can use more coins.
**Follow-ups.** How do I return the coins? Keep a pick table and walk back from the amount. Why not
recursion? It repeats work without a cache, and a memoised version recurses as deep as the amount.

## Q2. How would you find a longest common subsequence and one example of it?
**60-second answer.** State: the LCS length of the first i characters of one string and the first
j of the other. Base: an empty prefix gives 0. Transition: if the last characters match, add one to
the diagonal entry; otherwise take the larger neighbour. Answer: the corner entry. Cost: the product
of the lengths in time, and the same in memory for the full table. To recover a subsequence, walk
back from the corner, taking matches and moving toward the larger neighbour.
**Follow-ups.** Can I use less memory? For the length, two rows are enough, because each row only
reads the one above. For the subsequence, the walk needs the table, or a second pass. Is it the
same as edit distance? The same table shape with different transitions.

## Common mistakes
- Stating the recurrence before the state meaning is fixed.
- Filling the table in an order that reads entries not yet computed.
- Using 0 as the sentinel for unreachable values.
- Confusing a subsequence with a contiguous substring.
- Recursing one frame per unit of amount or per character without considering the limit.
