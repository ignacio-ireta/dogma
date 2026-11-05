---
title: Matrix shapes and transpose
minutes: 12
min_viable: "Read The system and the Worked example, then solve p01."
---

## The system

A matrix $A$ of shape $(r, c)$ has $r$ rows and $c$ columns. Its entry in row $i$, column $j$ is $A_{ij}$. Rows always come first. Before you compute anything, write the shape next to every object in the line.

| Object | Shape | NumPy | Orientation |
|---|---|---|---|
| Matrix $A$ | $(r, c)$ | `np.zeros((r, c))` | $r$ rows, $c$ columns |
| Column vector $\mathbf{x}$ | $(n, 1)$ | `x[:, None]` | The linear-algebra default |
| Row vector $\mathbf{x}^\top$ | $(1, n)$ | `x[None, :]` | Transpose of a column |
| 1-D array $\mathbf{x}$ | $(n,)$ | `np.array([1, 2, 3])` | No orientation: a NumPy shortcut |

The transpose swaps the two indices:

$$(A^\top)_{ij} = A_{ji}, \qquad A \in \mathbb{R}^{r \times c} \Rightarrow A^\top \in \mathbb{R}^{c \times r}.$$

Three rules cover almost every transpose you will write:

1. $(A^\top)^\top = A$. Transposing twice gives the input back.
2. $(A + B)^\top = A^\top + B^\top$ and $(\alpha A)^\top = \alpha A^\top$. Transpose is linear.
3. For $A \in \mathbb{R}^{r \times k}$ and $B \in \mathbb{R}^{k \times c}$: $(AB)^\top = B^\top A^\top$. The order of the factors reverses.

Rule 3 is the one people get wrong. The shapes show why. $AB$ is $(r, c)$, so $(AB)^\top$ is $(c, r)$. $B^\top A^\top$ is $(c, k)(k, r)$, which is also $(c, r)$. The product $A^\top B^\top$ is not defined in general: it is $(k, r)(c, k)$, and its inner dimensions are $r$ and $c$.

## Worked example

Let $A = \begin{pmatrix} 1 & 2 & 3 \\ 4 & 5 & 6 \end{pmatrix}$, shape $(2, 3)$.

1. Shape: $r = 2$, $c = 3$. So $A^\top$ has shape $(3, 2)$: three rows, each of length 2.
2. Row $j$ of $A^\top$ is column $j$ of $A$. Column 1 of $A$ is $(1, 4)$, so row 1 of $A^\top$ is $(1, 4)$. Column 2 is $(2, 5)$. Column 3 is $(3, 6)$.
3. Assemble: $A^\top = \begin{pmatrix} 1 & 4 \\ 2 & 5 \\ 3 & 6 \end{pmatrix}$.
4. Spot-check the entry rule: $(A^\top)_{32} = A_{23} = 6$. Row 3, column 2 of the transpose is row 2, column 3 of $A$.

The trap is the 1-D array. In NumPy, `x = np.array([1, 2, 3])` has shape $(3,)$, and `x.T` is still $(3,)$. To get a column, write `x[:, None]`, which has shape $(3, 1)$. Its transpose has shape $(1, 3)$.

## Why it works

Expand one entry of $(AB)^\top$. Let $A \in \mathbb{R}^{r \times k}$ and $B \in \mathbb{R}^{k \times c}$:

$$(AB)^\top_{ij} = (AB)_{ji} = \sum_{t=1}^{k} A_{jt}\, B_{ti}, \qquad (B^\top A^\top)_{ij} = \sum_{t=1}^{k} B_{ti}\, A_{jt}.$$

The two sums are identical. The summed index $t$ sits between the factors. Transposing swaps the roles of $i$ and $j$, not of $t$, so $t$ must stay the inner index, and the factors have to swap to keep it there.

Most bugs in ML code are shape bugs. A model is a chain of products such as $XW$, and every factor's shape is a convention you must match. Many libraries store a linear layer's weight as (out, in). PyTorch's `nn.Linear` does this, so its forward pass is $XW^\top + b$. Forgetting that one transpose is a common shape bug. Strang (ch. 2) and Deisenroth (§2.2) both state the rule $(AB)^\top = B^\top A^\top$.

## Signal vs. noise

- **Signal:** write the shape of every line before running it. Rows first. A transpose swaps the two numbers and changes nothing else.
- **Signal:** a 1-D array has no row or column. `.T` does nothing to it, so add an axis when you mean a column.
- **Noise for now:** the conjugate transpose (matters only for complex numbers), `np.transpose` with an `axes` argument for tensors of rank 3 and higher (you meet it in attention, not here), and block matrices.

## Where it breaks

- **Ragged rows.** `zip(*rows)` silently truncates to the shortest row. `zip(*[[1, 2], [3]])` returns `[(1, 3)]` and raises nothing. Check every row length before you transpose.
- **Reshape is not transpose.** `A.reshape(3, 2)` reads the flat data in row-major order and returns `[[1, 2], [3, 4], [5, 6]]`. The transpose is `[[1, 4], [2, 5], [3, 6]]`. Same shape, different matrix.
- **Broadcasting hides shapes.** $(n,)$ minus $(n, 1)$ gives $(n, n)$ with no error. Check output shapes, not just the absence of exceptions.
- **Views share memory.** `A.T` in NumPy is a view. Writing into it changes `A`. Call `.copy()` when you need an independent matrix.
- **Zero width.** An $(r, 0)$ matrix transposes to $(0, r)$, which a nested list cannot hold, so the answer is `[]`. Transposing back does not recover it. This is a representation limit, not a maths error.

## Interview angle

*60-second shape:* name every shape in $Y = XW + b$, check the inner dimension, then state the output shape. Weights stored as (out, in), as PyTorch's `nn.Linear` stores them, give the product $XW^\top + b$.

*Likely follow-ups:* is transposing expensive (no: `.T` is a view), and how do you debug a shape error you did not write (run one example with batch size 1 and print each shape). Full answers are in `interview.md`.

## Connections

- **Prerequisites:** `cml/vector-operations`. Column vectors, and dot products written as $\mathbf{x}^\top\mathbf{y}$.
- **Unlocks:** `cml/matrix-multiplication` (the shape rule, and $(AB)^\top$), `dl/building-tensors-by-broadcasting` (shapes of stretched axes), `cml/normal-equation` ($X^\top X$ needs the shape of $X^\top$), `dl/scaled-dot-product-attention` (scores are $QK^\top$).
- **Lineage:** matrix algebra as a notation for linear maps goes back to Cayley's 1858 memoir. The transpose is the same index bookkeeping, with rows and columns swapped.
