# Interview notes — Building tensors by broadcasting

## Q1. How do you compute all pairwise distances without a Python loop, and what does it cost?

**60-second answer.** Broadcasting. The difference tensor `X[:, None, :] - Y[None, :, :]` has shape
$(n, m, d)$. Squaring it and summing over the feature axis gives the $(n, m)$ matrix exactly, but it
costs $O(nmd)$ memory. The expansion $\lVert x\rVert^2 - 2x\cdot y + \lVert y\rVert^2$ needs only
$O(nm)$ and one matrix product, so it is what large inputs use. Its weakness is precision far from
the origin, which centring the data fixes. For a large input I use the expansion on centred data,
and I check it against the difference form on a small one.

**Follow-ups.**
- *How big does the difference tensor get?* For $n = m = 10^4$ and $d = 512$, about $5 \times 10^{10}$
  floats, roughly 410 GB in float64. That rules it out at this scale.
- *What if the data are float32?* The cancellation is worse, so centring matters more.

## Q2. Why does the expansion trick need clipping, and why can it be inaccurate?

**60-second answer.** Each of the three terms is about the size of the norms. When two nearby points
are far from the origin, the terms are large and nearly equal, and their difference is the small
true distance. Rounding error scales with the large terms, so it can swamp the answer, and it can
make a true zero come out negative. Clipping at zero stops `sqrt` from returning NaN. Centring both
sets on a shared point keeps the terms small, which fixes the accuracy problem without changing
the distances.

**Follow-ups.**
- *Does centring change the distances?* No. Subtracting the same vector from every point leaves
  every difference $x_i - y_j$ unchanged.
- *Is clipping a fix for accuracy?* No. It fixes sign only. Large errors still need centring.

## Q3. How would you find a silent broadcasting bug?

**60-second answer.** A broadcast that is valid but wrong raises nothing, so I look at shapes, not
exceptions. I print the shape at each step, and I check the result on a tiny input whose answer I
know by hand. The shapes that hide bugs are square ones, where $(n,)$ and $(m,)$ with $n = m$ combine
silently, and mismatched feature counts, where a size-1 axis stretches without complaint. I make
both explicit: reshape to $(n, 1)$ and $(1, m)$, and check the feature counts before the operation.

**Follow-ups.**
- *Why not just use `scipy.spatial.distance.cdist`?* In production, I would. For an interview, the
  point is to show I can build it and to explain its costs.
- *Where else does broadcasting matter?* Adding a bias to every row of a batch, and the attention
  score matrix, which broadcasts the same way.

## Common mistakes

- Broadcasting $(n,)$ against $(m,)$ and expecting an outer result. It is an error unless $n = m$,
  and when they are equal it is wrong.
- Summing over the wrong axis, so the $(n, m)$ result collapses to a vector.
- Trusting the expansion trick for points far from the origin without centring.
- Allocating the $n \times m \times d$ difference tensor for large inputs.
