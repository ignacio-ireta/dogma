---
title: Prefix sums
minutes: 12
min_viable: "Read The system and the Worked example, then solve p01."
---

## The system

Take a list $x_1,\dots,x_n$ and define the **prefix sums** $P_0 = 0$ and $P_j = x_1 + \dots + x_j$ for $1 \le j \le n$. Then

$$\sum_{t=i+1}^{j} x_t = P_j - P_i, \qquad 0 \le i < j \le n.$$

Two consequences. Building $P$ costs one pass. After that, any range sum is one subtraction. The prefix array is the discrete integral of $x$, and the range sum is the discrete version of "area under the curve between two points."

**Subarray count.** A subarray $x_{i+1..j}$ sums to $k$ exactly when $P_i = P_j - k$. So the count is the number of pairs $i < j$ with $P_i = P_j - k$. Scan $j$ from $1$ to $n$, keep a dictionary of the prefix values seen so far, and count the earlier matches:

| Step | Rule | Why |
|---|---|---|
| Seed | `counts = {0: 1}` | $P_0 = 0$ is a prefix. Without it, subarrays that start at position 1 are missed. |
| Lookup | `total += counts[P_j - k]` | every earlier prefix with that value gives one subarray |
| Insert | `counts[P_j] += 1`, **after** the lookup | with $k = 0$, inserting first would match $P_j$ with itself, which counts the empty subarray $x_{j+1..j}$ |

**Summed-area table (2-D).** Let $S$ be an $(R+1) \times (C+1)$ table with a zero first row and column, where $S[i][j]$ is the sum of the grid entries with row $< i$ and column $< j$. For inclusive corners $(r_1, c_1)$ and $(r_2, c_2)$:

$$\text{sum} = S[r_2{+}1][c_2{+}1] - S[r_1][c_2{+}1] - S[r_2{+}1][c_1] + S[r_1][c_1].$$

This is inclusion–exclusion. Two strips are subtracted, and their overlap corner is added back.

## Worked example

**Subarray count.** `nums = [1, 2, 1, 2]`, `k = 3`. The prefix sums are $P = [0, 1, 3, 4, 6]$.

| $j$ | $x_j$ | $P_j$ | look up $P_j - 3$ | hits | total | counts after the insert |
|---|---|---|---|---|---|---|
| 1 | 1 | 1 | $-2$ | 0 | 0 | $\{0{:}1, 1{:}1\}$ |
| 2 | 2 | 3 | $0$ | 1 | 1 | $+ \{3{:}1\}$ |
| 3 | 1 | 4 | $1$ | 1 | 2 | $+ \{4{:}1\}$ |
| 4 | 2 | 6 | $3$ | 1 | 3 | $+ \{6{:}1\}$ |

The answer is 3: $[1,2]$ (entries 1–2), $[2,1]$ (entries 2–3) and $[1,2]$ (entries 3–4), counting entries from 1 as in the definition of $P$.

**Summed-area table.** The grid $\begin{smallmatrix}1&2&3\\4&5&6\\7&8&9\end{smallmatrix}$ gives

$$S = \begin{pmatrix} 0&0&0&0\\ 0&1&3&6\\ 0&5&12&21\\ 0&12&27&45\end{pmatrix}.$$

The query $(r_1, c_1, r_2, c_2) = (1, 1, 2, 2)$, the bottom-right $2 \times 2$ block, gives $S[3][3] - S[1][3] - S[3][1] + S[1][1] = 45 - 6 - 12 + 1 = 28$, which is $5 + 6 + 8 + 9$.

## Why it works

Each range sum is a difference of two prefixes, so the $\binom{n+1}{2}$ pairs of positions collapse into lookups in a table of prefix values. A dictionary answers "how many earlier prefixes equal $P_j - k$" in expected O(1). The whole count is O(n) time and O(n) space.

The method does not need the values to be non-negative. Two pointers do, because a sliding window needs its sum to change monotonically as the window grows. `algo/two-pointers-and-sliding-windows` covers that case. For the 2-D table, inclusion–exclusion gives O(1) per query after an O(RC) build. Viola and Jones (2001) used this identity as integral images for rectangle features; their [original paper record](https://www.merl.com/publications/TR2004-043) identifies the CVPR publication.

## Signal vs. noise

- **Signal:** the seed $P_0 = 0$, and lookup-before-insert. Those two lines decide almost every wrong answer.
- **Signal:** the query formula has four terms and the table is one larger in each direction. Zero padding makes the formula uniform, with no special case at the edges.
- **Noise for now:** alternative data structures before the prefix-difference identity and update order are clear.

## Where it breaks

- **Missing leading zero.** Counting only the prefixes $P_1, \dots, P_n$ undercounts every subarray that starts at position 1.
- **Lookup after insert.** With $k = 0$ the subarray $x_{j+1..j}$ is counted. It has no entries, so it is not a subarray.
- **Off-by-one corners.** Using $S[r_2][c_2]$ instead of $S[r_2{+}1][c_2{+}1]$ drops the last row and column. Inclusive queries need the $+1$.
- **Float cancellation.** With floats, large prefix sums absorb small entries. For $x = [10^{16}, 1.0, 1.0]$, the float prefixes are $[0, 10^{16}, 10^{16}, 10^{16}]$. So $P_2 - P_1 = 0$, not 1. Use exact integers when you can.
- **Stale table.** The table is a snapshot. If the grid changes, the table is wrong until it is rebuilt. A Fenwick tree handles updates in O(log R log C) per update and per query.

## Interview angle

*60-second answer:* "Subarray sum equals $k$, with negatives allowed: prefix sums turn the range condition into $P_j - P_i = k$. One pass, a count map seeded with 0, and lookup before insert. O(n) time and O(n) space. A sliding window fails because negatives make the sum non-monotone."

*Likely follow-ups:* Why not two pointers? (Negatives break the monotone shrink step.) How does the 2-D version work? (Build the padded table once in O(RC), then use four lookups per query.) What if the grid changes? (A Fenwick tree instead of a static table.)

## Connections

- **Requires:** `algo/hash-maps-and-sets` (the count dictionary), `algo/two-pointers-and-sliding-windows` (the non-negative contrast).
- **Reuses the same cumulative-sum idea as:** exact greedy split finding in gradient-boosted trees, which scans cumulative gradient sums in sorted order (`cml/xgboost-second-order-objective`), integral images in vision, and rolling sums in time series.
- **Application:** [Viola and Jones (2001)](https://www.merl.com/publications/TR2004-043) use integral images for fast rectangle-feature evaluation.
