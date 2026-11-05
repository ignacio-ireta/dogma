## Hint 1 — concept

Consider the last character of each string. If they are equal, can a longest common subsequence
ever be improved by ignoring them? If they differ, which string should you drop one character
from? Answer for every pair of prefixes, not only for the full strings.

## Hint 2 — formula / approach

Let `dp[i][j]` be the length of the longest common subsequence of the first `i` characters of `a`
and the first `j` characters of `b`. Each row and column of zeros is the base case. When
`a[i-1] == b[j-1]`, add one to `dp[i-1][j-1]`. Otherwise take the larger of `dp[i-1][j]` and
`dp[i][j-1]`. The answer is `dp[len(a)][len(b)]`. For the witness, walk back from that corner and
collect the matched characters; reverse them at the end.

## Hint 3 — code shape

Build `dp` with `(n + 1)` rows of `(m + 1)` zeros. Fill it with two nested loops over `i` and `j`
from 1 upward, using the one-based prefix lengths. For the walk back, a `while i > 0 and j > 0`
loop: take the match and move both indices, or move the index whose neighbour holds the larger
value. Check the witness length against the table before returning.
