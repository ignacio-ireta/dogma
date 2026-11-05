---
title: Binary search
minutes: 12
min_viable: "Read The system and the Worked example, then solve p01."
---

## The system

Binary search finds a boundary in a range where a yes/no test changes only once. Sorted data gives
that: for a target $x$, the test "is $a_i \ge x$?" is FALSE for every index before some point and
TRUE from that point on.

The **lower bound** of $x$ is the first index $i$ with $a_i \ge x$. If no element qualifies, it is
$n = \text{len}(a)$. Python's `bisect.bisect_left` returns this value.

| Name | Test | Python |
|---|---|---|
| lower bound | first $i$ with $a_i \ge x$ | `bisect_left` |
| upper bound | first $i$ with $a_i > x$ | `bisect_right` |

Use a **half-open** interval $[lo, hi)$ with the invariant: every index before $lo$ fails the test,
and every index from $hi$ on passes it, or $hi = n$. Start with $lo = 0$ and $hi = n$. Then:

1. While $lo < hi$, take $mid = lo + \lfloor (hi - lo)/2 \rfloor$.
2. If $a_{mid} < x$, the test fails at $mid$, so the answer is after it: $lo = mid + 1$.
3. Otherwise $mid$ passes and may be the answer: $hi = mid$.
4. When $lo = hi$, the interval is empty and $lo$ is the answer.

## Worked example

Find the lower bound of $x = 3$ in $a = [1, 3, 3, 3, 7, 9]$ ($n = 6$).

| step | lo | hi | mid | $a_{mid}$ | test | update |
|---|---|---|---|---|---|---|
| 1 | 0 | 6 | 3 | 3 | $3 \ge 3$ passes | $hi = 3$ |
| 2 | 0 | 3 | 1 | 3 | passes | $hi = 1$ |
| 3 | 0 | 1 | 0 | 1 | $1 < 3$ fails | $lo = 1$ |

Now $lo = hi = 1$, so the answer is 1. The upper bound of 3 is 4: the first index whose value is
strictly greater than 3.

## Why it works

Each step tests one middle index and removes the half that cannot hold the answer. An interval of
width $w$ becomes at most $\lfloor w/2 \rfloor$ wide, so a search over $n$ items makes at most
$\lfloor \log_2 n \rfloor + 1$ probes. Knuth covers the ordered-table search and its variants
(*TAOCP* vol. 3, sec. 6.2.1). The invariant is what makes the answer correct. Every update keeps the
answer inside $[lo, hi)$, and the loop ends only when that interval is empty.

**Binary search on the answer.** The same loop works on a range of candidate answers, when a
feasibility test $f(c)$ is monotone: once it is TRUE, it stays TRUE for every larger $c$. Then the
first TRUE $c$ is the answer. Start with a range that surely contains it, such as $[\max w, \sum w]$
for shipping problems, and apply the same update rules.

## Signal vs. noise

- **Signal:** the invariant, and the two update rules. `lo = mid + 1` and `hi = mid` are the only
  pair that works for a lower bound.
- **Signal:** the monotone predicate. It is the precondition for searching over answers.
- **Noise for now:** the exact constant in the probe count, and micro-optimisations of `mid`.

## Where it breaks

- **`lo = mid` instead of `lo = mid + 1`.** On $[1, 3]$ with $x = 2$, `lo` stays at 0 and the loop
  never ends.
- **`hi = mid - 1` in the true branch.** On $[1, 3]$ with $x = 2$, it returns 0, but the answer is 1.
- **`hi = len - 1`.** A target above every element returns $n - 1$ instead of $n$.
- **`<=` in the test.** It returns the upper bound. On $[1, 2, 2, 2, 5]$ with $x = 2$ it returns 4,
  not 1.
- **Overflow.** In fixed-width integer languages, `lo + hi` can overflow. Write
  `lo + (hi - lo) // 2`. Python integers do not overflow.

## Interview angle

"Find the first position where a sorted array is at least $x$." Answer shape: state the invariant
over $[lo, hi)$, the two update rules, the loop condition, and the return value. The cost is $O(\log
n)$ comparisons. Test duplicates, a target below everything, one above everything, and the empty
list. Follow-ups:

- *How does the upper bound differ?* One comparison: `<=` in place of `<`.
- *Why is `hi` starting at `n` and not `n - 1`?* Because $n$ is a valid answer when nothing qualifies.

## Connections

- **Prerequisites:** none in the syllabus. You need sorted sequences and loop invariants.
- **Unlocks:** `algo/sort-then-sweep`, where sorting first makes a search possible, and
  `algo/dynamic-programming` for other monotone decisions.
- **Lineage:** the idea is classic, and Knuth's treatment is the standard reference. Many languages
  ship a bisect routine, but interviewers expect you to write the loop.
