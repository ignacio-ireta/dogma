---
title: Building tensors by broadcasting
minutes: 12
min_viable: "Read The system and the Worked example, then solve p01."
---

## The system

A tensor is an array with any number of axes, and its shape gives one size per axis. Two arrays
combine under **broadcasting** by these rules:

1. **Align from the right.** Line up the trailing axes. A missing leading axis counts as size 1.
2. **Compare each aligned pair of sizes.** They must be equal, or one of them must be 1.
3. **Stretch the 1s.** An axis of size 1 is repeated to match the other size, with no copying.
4. **Otherwise, error.** Two unequal sizes, neither of them 1, cannot be combined.

| Shapes | Aligned as | Result | Note |
|---|---|---|---|
| $(3,)$ and $(2, 3)$ | $(1, 3)$ and $(2, 3)$ | $(2, 3)$ | the row is added to every row |
| $(2, 1)$ and $(1, 4)$ | $(2, 1)$ and $(1, 4)$ | $(2, 4)$ | an outer grid |
| $(3,)$ and $(4,)$ | $3$ vs $4$ | error | neither size is 1 |
| $(n,)$ and $(n, 1)$ | $(1, n)$ and $(n, 1)$ | $(n, n)$ | valid, and usually a bug |

Inserting an axis of size 1 is how you ask for a particular broadcast. `x[:, None, :]` turns an
$(n, d)$ array into $(n, 1, d)$.

The task is the all-pairs distance matrix. Let $X \in \mathbb{R}^{n \times d}$ and
$Y \in \mathbb{R}^{m \times d}$, with rows $x_i$ and $y_j$. We want $D \in \mathbb{R}^{n \times m}$
with $D_{ij} = \lVert x_i - y_j \rVert^2$. Two broadcast constructions do it:

- **Difference tensor.** `X[:, None, :] - Y[None, :, :]` has shape $(n, m, d)$. Square it and sum over the last axis. Memory is $O(nmd)$.
- **Expansion.** $D_{ij} = \lVert x_i \rVert^2 - 2\,x_i\cdot y_j + \lVert y_j \rVert^2$. The three terms have shapes $(n, 1)$, $(n, m)$ and $(1, m)$. The middle term is one matrix product $XY^\top$. Memory is $O(nm)$, but it subtracts large, nearly equal numbers.

## Worked example

Let $X = \begin{pmatrix} 0 & 0 \\ 1 & 0 \end{pmatrix}$ (shape $(2, 2)$) and $Y = \begin{pmatrix} 0 & 1 \end{pmatrix}$ (shape $(1, 2)$). The target is shape $(2, 1)$.

Difference tensor:

1. $X[:, \text{None}, :]$ has shape $(2, 1, 2)$. $Y[\text{None}, :, :]$ has shape $(1, 1, 2)$.
2. Broadcast the subtraction: the result has shape $(2, 1, 2)$. Pair $(0, 0)$ gives $(0 - 0,\; 0 - 1) = (0, -1)$. Pair $(1, 0)$ gives $(1 - 0,\; 0 - 1) = (1, -1)$.
3. Square and sum over the last axis: $0 + 1 = 1$ and $1 + 1 = 2$.
4. Result: $D = \begin{pmatrix} 1 \\ 2 \end{pmatrix}$, shape $(2, 1)$.

The expansion gives the same answer: $(0, 1)^\top + 1 - 2\,(0, 0)^\top = (1, 2)^\top$.

## Why it works

Broadcasting is indexing with size-1 axes. Subtracting a $(1, m, d)$ array from an $(n, 1, d)$ array repeats each along its length-1 axis, so one line covers all $nm$ pairs.

The expansion starts from $\lVert x - y \rVert^2 = (x - y)\cdot(x - y) = \lVert x\rVert^2 - 2x\cdot y + \lVert y\rVert^2$. Each term has a natural shape: one norm per point, and all pairwise dot products as one matrix product.

Precision is the cost of the expansion. Near a centre $c$ far from the origin, each norm term is about $\lVert c\rVert^2$, and rounding error scales with it. Shifting both sets by the same vector leaves distances unchanged, so subtract a shared centre first.

## Signal vs. noise

- **Signal:** the four broadcast rules. Align from the right, match or 1, stretch. Write the result shape before you write the line.
- **Signal:** silent broadcasting. A valid broadcast is not necessarily the one you meant. $(n,)$ with $(n, 1)$ gives $(n, n)$ with no error.
- **Signal:** precision. Subtracting two large, nearly equal numbers loses digits. Centre the data before an expansion trick.
- **Noise for now:** strides, `einsum` syntax and GPU memory layouts.

## Where it breaks

- **Silent broadcasting.** `sq_x + sq_y` with shapes $(n,)$ and $(m,)$ is valid only when $n = m$, and then it adds the wrong pairs. Reshape to $(n, 1)$ and $(1, m)$ first.
- **Feature mismatch hidden by broadcasting.** An $(n, 1)$ array against an $(m, d)$ array broadcasts over $d$ with no error. Check the feature counts explicitly.
- **Memory.** For $n = m = 10^4$ and $d = 512$, the difference tensor holds $5.1 \times 10^{10}$ floats, about 410 GB in float64. Use the expansion, or process rows in chunks.
- **Cancellation.** The expansion can return a tiny negative number for a true zero. Clip at zero before any square root.

## Interview angle

*60-second shape:* all pairwise distances by broadcasting. The difference tensor is exact but costs $O(nmd)$ memory. The expansion costs $O(nm)$ and one matrix product, but it needs centring far from the origin.

*Likely follow-ups:* why clip at zero (rounding can make a true zero negative, and the square root of that is NaN), and how this feeds k-NN (keep the $k$ smallest entries per row, for example with `np.argpartition`). Full answers are in `interview.md`.

## Connections

- **Prerequisites:** `cml/vector-operations` (norms and dot products), `cml/matrix-shapes-and-transpose` (axes and reshaping). `cml/matrix-multiplication` supplies $XY^\top$ in the expansion.
- **Unlocks:** `cml/nearest-neighbour-search-and-knn`, `dl/scaled-dot-product-attention` (its score matrix is a broadcast of the same kind), and `cml/feature-scaling`, where centring is the same operation applied to features.
- **Lineage:** nearest-neighbour discrimination goes back to Fix and Hodges (1951).
