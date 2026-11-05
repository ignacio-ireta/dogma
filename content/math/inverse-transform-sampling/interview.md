# Interview notes — Inverse-transform sampling

## Q1. Given only a uniform random number generator, how do you sample from another distribution?

**60-second answer.** If the distribution's CDF $F$ has an inverse you can evaluate, draw
$U \sim \mathrm{Uniform}(0,1)$ and return $F^{-1}(U)$. Then $P(F^{-1}(U) \le x) = F(x)$, so the law is
exact. Each draw costs one uniform and one evaluation of the inverse. If the inverse has no closed
form, I would use rejection sampling: draw from a proposal that is easy to sample, and accept each
draw with a probability proportional to the target density over the proposal's.

**Follow-ups.**
- *Why is $F^{-1}(U)$ distributed as $F$?* For continuous $F$, the events $\{F^{-1}(U) \le x\}$ and
  $\{U \le F(x)\}$ are the same event, and the second has probability $F(x)$.
- *What about a discrete distribution?* The same rule, with $F$ a step function. Search the cumulative
  sums for $U$.

## Q2. How would you sample a discrete distribution with many categories, many times?

**60-second answer.** Build the cumulative sums once, in $O(k)$. For each uniform draw, search the
sums for the first entry greater than the draw, using a binary search, so each draw costs $O(\log k)$.
Validate the probabilities first: non-negative, finite, and summing to one within rounding. Clip the
index to the last category, since rounding can leave the final cumulative sum just below one.

**Follow-ups.**
- *Can it be faster?* An alias table gives $O(1)$ per draw after an $O(k)$ build.
- *What is the most common bug?* Using `side='left'` puts boundary draws in the wrong category, and
  a zero-probability category can then be returned. The half-open rule $c_{i-1} \le u < c_i$ avoids both.

## Common mistakes

- Using $\ln(u)$ where $\ln(1-u)$ was meant. The two are both exponential, but they are different maps.
- Taking $\log(1-u)$ when $u$ can be tiny, so the draw collapses to zero.
- Checking that probabilities sum to exactly one with `==`, which rejects valid vectors that round.
- Forgetting the top of the CDF: the last cumulative sum can be $1 - 2^{-53}$, so an index can run past
  the last category.
