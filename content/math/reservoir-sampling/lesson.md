---
title: Reservoir sampling
minutes: 12
min_viable: "Read The system and the Worked example, then solve p01."
---

## The system

You want $k$ items chosen uniformly from a stream of $n$ items, but you do not know $n$ in advance
and you can read each item only once. Keep a **reservoir** of $k$ slots, numbered $0, \dots, k-1$.
Items are numbered $i = 0, 1, \dots$ as they arrive.

| Step | Item index $i$ | Action |
|---|---|---|
| Fill | $i < k$ | store item $i$ in slot $i$ |
| Replace | $i \ge k$ | draw $j \sim \mathrm{Uniform}\{0, 1, \dots, i\}$ (inclusive); if $j < k$, store item $i$ in slot $j$, otherwise discard it |

That is the whole algorithm. The draw covers $i + 1$ values, so item $i$ enters the reservoir with
probability $k/(i+1)$. Each item's draw is independent of the others.

The reservoir holds $k$ items, whatever $n$ is. Memory is $O(k)$, and the work per item is one
random integer.

## Worked example

Stream $a, b, c, d, e$ with $k = 2$.

1. $i=0$: $a$ goes to slot 0. $i=1$: $b$ goes to slot 1. Reservoir $= [a, b]$.
2. $i=2$: draw $j \in \{0,1,2\}$. Say $j = 1$. Slot 1 takes $c$. Reservoir $= [a, c]$.
3. $i=3$: draw $j \in \{0,1,2,3\}$. Say $j = 3$. Since $3 \ge k$, $d$ is discarded. Reservoir $= [a, c]$.
4. $i=4$: draw $j \in \{0,1,2,3,4\}$. Say $j = 0$. Slot 0 takes $e$. Reservoir $= [e, c]$.

Every item ends with probability $k/n = 2/5$. The draws above are one possible outcome, not the
answer.

## Why it works

Claim: after $n \ge k$ items, each of them is in the reservoir with probability $k/n$.

- **Base, $n = k$.** All $k$ items are stored, so each has probability $1 = k/k$.
- **Step, $n \to n+1$.** The new item enters with probability $P(j < k) = k/(n+1)$, because $j$ is
  uniform on $\{0, \dots, n\}$. An item already stored survives unless the draw hits its own slot,
  which happens with probability $1/(n+1)$. So it survives with probability $n/(n+1)$. The
  probability it is still stored is $\frac{k}{n}\cdot\frac{n}{n+1} = \frac{k}{n+1}$.

Both cases give $k/(n+1)$, which closes the induction. Vitter (1985) presents this algorithm.

The expected number of replacements after $n$ items (items $i \ge k$ that land in a slot) is
$\sum_{i=k}^{n-1} k/(i+1)$. For large $n$ this is roughly $k \ln(n/k)$, which gives $3.61$ at $n = 10$, $k = 3$. The
exact value there is $2761/840 \approx 3.287$. A simulation with `default_rng(seed)` over seeds $0,\dots,19999$,
using the protocol draws, gave $3.288$.

## Signal vs. noise

- **Signal:** the replacement rule, one uniform integer per item, and the rule $j < k$. The rest
  is bookkeeping.
- **Signal:** the sample is a uniform $k$-subset. Every $k$-subset is equally likely.
- **Noise:** the slot order. Slot numbers are a storage detail. The reservoir is uniform as a set,
  and the order only matters for reproducing a run.
- **Weighted variant, one paragraph.** When item $i$ has weight $w_i$, draw a key $u_i^{1/w_i}$
  with $u_i$ uniform, and keep the $k$ items with the largest keys. Efraimidis and Spirakis (2006)
  call this A-Res. It is a heap of size $k$ with one comparison per item, and the uniform algorithm is
  its special case $w_i = 1$.

## Where it breaks

- **Unweighted rule with weights.** Algorithm R ignores weights. Use A-Res, or a different design,
  when items differ in importance.
- **Throughput.** One random integer per item is cheap, but at very high rates the generator cost
  matters. Skip-based variants from the same 1985 paper draw far fewer random numbers.
- **Sample with repetition.** Reservoir sampling gives a sample without replacement. For
  independent draws, run $k$ single-item reservoirs.
- **Time-dependent interest.** A uniform reservoir treats the first and the last item alike. If
  recent items matter more, use a decayed or windowed scheme.

## Interview angle

**60-second answer.** Keep a reservoir of $k$ items. The first $k$ items fill it. For item $i \ge k$,
draw $j$ uniform on $0..i$, and replace slot $j$ if $j < k$. Memory is $O(k)$, the stream is read
once, and the induction shows each item is kept with probability $k/n$. It is the standard tool
for sampling logs, events or requests when the total is unknown.

**Follow-ups.**

- *Why is the inclusion probability $k/n$?* Induction on $n$: a new item enters with probability
  $k/(n+1)$, and an existing item survives with probability $n/(n+1)$.
- *How would you weight items?* A-Res: keys $u^{1/w}$, keep the top $k$ with a heap.

## Connections

- **Requires:** `cml/probability-basics` (uniform discrete draws, inclusion probabilities).
- **Related:** `math/inverse-transform-sampling` (turning uniforms into other distributions),
  `math/monte-carlo-simulation` (estimating from a sample).
- **Lineage:** Vitter (1985) gave the algorithm and the skip-based variants. Efraimidis and Spirakis
  (2006) gave the weighted version.
