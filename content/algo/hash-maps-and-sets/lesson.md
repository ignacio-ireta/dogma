---
title: Hash maps and sets
minutes: 12
min_viable: "Read The system and the Worked example, then solve p01."
---

## The system

A hash map (Python `dict`) stores key-value pairs. To find a key it computes $\operatorname{hash}(k)$,
picks a bucket from that value, and compares only the keys in that bucket. A set is a hash map
without values: it stores keys only, and answers "is $x$ present?".

| Operation | Average cost | Worst case | Note |
|---|---|---|---|
| `d[k]`, `k in d` | $O(1)$ | $O(n)$ | worst case needs many keys in one bucket |
| `d[k] = v` | $O(1)$ amortised | $O(n)$ | occasional resize |
| `x in s` (set) | $O(1)$ | $O(n)$ | same mechanism as the dict |
| `x in L` (list) | $O(n)$ | $O(n)$ | linear scan, every time |

Three idioms cover most interview problems:

1. **Membership:** `seen = set()`, then `if x in seen` and `seen.add(x)`.
2. **Counting:** `counts[t] = counts.get(t, 0) + 1`. The `get` supplies 0 on a first occurrence.
3. **Grouping:** `groups.setdefault(key, []).append(item)`.

For top-k, you have $d$ distinct items and want the $k$ with the largest counts:

| Method | Cost | Use when |
|---|---|---|
| Sort all $d$ items | $O(d \log d)$ | $k$ is large, or you need the whole ranking |
| Heap of size $k$ | $O(d \log k)$ | $k$ is small compared with $d$ |

## Worked example

Tokens `[b, a, b, c, a, b]`, $k = 2$.

1. **Count:** `b` → 1, `a` → 1, `b` → 2, `c` → 1, `a` → 2, `b` → 3. Result: `{b: 3, a: 2, c: 1}`.
   Six lookups, each $O(1)$ on average, so the pass is $O(n)$.
2. **Rank** by the key $(-\text{count}, \text{token})$: `(b, 3)`, `(a, 2)`, `(c, 1)`.
   The negated count sorts the largest first, and the token breaks ties in one sort.
3. **Take** the first two: `[(b, 3), (a, 2)]`.

Tie variant: tokens `[b, a]`, $k = 1$. Both count 1. Ranking by $(-\text{count}, \text{token})$ gives
`a` first, although `b` appeared first. A sort on count alone would keep the first-appearance order
and return `b`, which is the wrong answer under this problem's rule.

## Why it works

The counting pass is $O(n)$ because a hash lookup is $O(1)$ on average. A hash function spreads keys
across $m$ buckets. With chaining, an unsuccessful search costs $1 + \alpha$ on average, where
$\alpha = n/m$ is the load factor. The table grows when $\alpha$ gets large, so $\alpha$ stays
bounded and the average cost stays constant (Cormen et al., ch. 11).

The ranking step is an ordinary sort. Python compares tuples element by element, so a key of
$(-\text{count}, \text{token})$ encodes both rules at once: count descending, then token ascending.

## Signal vs. noise

- **Signal:** average $O(1)$ against worst-case $O(n)$. The counting idiom. Set membership against
  list membership. The negated-count key for descending order with an ascending tie-break.
- **Signal:** sort against heap. $O(d \log d)$ and $O(d \log k)$ differ only when $k \ll d$.
- **Noise for now:** resizing strategy, the hash function's internals, and open addressing against
  chaining. Know that they exist, and that the average-case claim depends on them.

## Where it breaks

- **Unhashable keys.** A list or dict cannot be a key, because it is mutable and its hash could
  change. Python raises `TypeError`. Use a tuple of hashable values.
- **Hash collisions.** Two keys in one bucket are resolved by the table, so the answer stays correct,
  but a bad hash or adversarial keys can push every lookup toward $O(n)$.
- **Dict order is not ranking.** A dict keeps insertion order, and a set has no order you may rely
  on. A top-k answer must be sorted explicitly.
- **Stable sort keeps first appearance.** Python's `sorted` is stable, so sorting by count alone keeps
  the insertion order among ties. That is the trap this problem tests.
- **Memory.** A hash map of $n$ distinct keys costs $O(n)$ memory. For a stream too large for memory,
  use approximate counting (Count-Min sketch, Misra-Gries). That is beyond this topic.

## Interview angle

"Find the $k$ most frequent words in a document." Answer shape: one pass with a dict for counts, $O(n)$.
Then top-k: sort the $d$ distinct words in $O(d \log d)$, or keep a heap of size $k$ in
$O(d \log k)$ when $k$ is small. State the tie-break before you code it. Follow-ups:

- *The data does not fit in memory?* Stream it, and use approximate counts or an external sort.
- *Worst-case lookup?* $O(n)$ under heavy collisions. A balanced tree gives $O(\log n)$ and keeps
  keys ordered, at the cost of a slower average.

## Connections

- **Unlocks:** `algo/heaps-and-priority-queues` (the heap used for top-k), `algo/sort-then-sweep`,
  and `algo/prefix-sums` (a dict of prefix counts).
- **Used by:** `sql/sql-group-by-and-having` (grouping is the same counting pattern) and
  `cml/tf-idf-and-bm25` (term counts per document).
- **Lineage:** hashing for fast lookup dates to the 1950s, and Knuth's *The Art of Computer
  Programming*, vol. 3, treats it in depth.
