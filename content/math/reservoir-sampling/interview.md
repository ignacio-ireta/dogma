# Interview notes — Reservoir sampling

## Q1. You have a log stream of unknown length. How do you keep a uniform sample of k lines in one pass?

**60-second answer.** Keep a reservoir of $k$ slots. The first $k$ items fill it. For item $i \ge k$,
draw $j$ uniform over $0, \dots, i$. If $j < k$, replace slot $j$ with the item. Otherwise discard
it. Memory is $O(k)$, the stream is read once, and the length does not need to be known. Each item
ends in the sample with probability $k/n$.

**Follow-ups.**
- *How do you know the sample is uniform?* Induction on $n$. A new item enters with probability
  $k/(n+1)$. An existing item survives with probability $n/(n+1)$. The product is $k/(n+1)$.
- *What about fewer than $k$ lines?* The reservoir holds all of them, in arrival order.

## Q2. Prove that each item is kept with probability $k/n$.

**60-second answer.** Base case: after $k$ items, every item is kept. Step: when item $n+1$ arrives,
it is kept with probability $P(j < k) = k/(n+1)$, because $j$ is uniform on $0..n$. An item already
kept is evicted only when $j$ equals its own slot, which has probability $1/(n+1)$, so it survives
with probability $n/(n+1)$. Multiply: $\frac{k}{n}\cdot\frac{n}{n+1} = \frac{k}{n+1}$, and the claim
holds for $n+1$.

**Follow-ups.**
- *Is the set uniform over all $k$-subsets?* Yes. The same induction, applied to subsets, gives
  equal probability to every subset.
- *Does the slot order matter?* For uniformity of the set, no. It only fixes how a run is reproduced.

## Q3. How would you weight the items?

**60-second answer.** Use the weighted reservoir A-Res. Give item $i$ the key $u_i^{1/w_i}$, with $u_i$
uniform, and keep the $k$ items with the largest keys in a min-heap. Each item costs one key and at
most one heap operation. With all weights equal, the rule reduces to the uniform case.

**Follow-ups.**
- *Why does $u^{1/w}$ work?* It is the standard weighted sampling without replacement scheme, and it
  gives a larger key to heavier items. Efraimidis and Spirakis (2006) prove the inclusion
  probabilities.
- *What if you need sampling with replacement?* Run $k$ independent single-item reservoirs.

## Common mistakes

- Replacing slot $j$ when $j \le k$. Slot $k$ does not exist, so this is an index error or a wrong
  distribution.
- Drawing $j$ from $0..i-1$ instead of $0..i$. A new item then enters with probability $k/i$ instead of
  $k/(i+1)$, so the sample leans toward later items.
- Calling `list(stream)` first. That breaks the $O(k)$ memory bound.
- Stopping the stream after $k$ items, which makes the sample just the first $k$.
