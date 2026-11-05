---
title: Matrix multiplication
minutes: 12
min_viable: "Read The system and the Worked example, then solve p01."
---

## The system

For $A \in \mathbb{R}^{m \times n}$ and $B \in \mathbb{R}^{n \times p}$, the product $C = AB \in \mathbb{R}^{m \times p}$ is defined only when the **inner dimensions match**: $A$ has $n$ columns and $B$ has $n$ rows. The outer dimensions give the shape of the result.

$$(m \times \underline{n})\,(\underline{n} \times p) = (m \times p)$$

The entry rule is row-times-column:

$$C_{ij} = \sum_{t=1}^{n} A_{it}\,B_{tj} = (\text{row } i \text{ of } A)\cdot(\text{column } j \text{ of } B).$$

Every entry of $C$ is a dot product, as in `cml/vector-operations`. The same product has two other readings:

| View | Statement | Use it to |
|---|---|---|
| Entry | $C_{ij} = (\text{row } i \text{ of } A)\cdot(\text{column } j \text{ of } B)$ | write the loop, check one number |
| Column | column $j$ of $C$ $= \sum_t B_{tj}\,(\text{column } t \text{ of } A)$ | see $AB$ as a mixture of $A$'s columns |
| Outer product | $AB = \sum_{t=1}^{n} (\text{column } t \text{ of } A)(\text{row } t \text{ of } B)$ | see $AB$ as a sum of $n$ rank-one matrices |

The product costs $mnp$ multiply-adds: each of the $mp$ entries takes $n$ of them.

## Worked example

Let $A = \begin{pmatrix} 1 & 2 & 3 \\ 4 & 5 & 6 \end{pmatrix}$ (shape $(2, 3)$) and $B = \begin{pmatrix} 7 & 8 \\ 9 & 10 \\ 11 & 12 \end{pmatrix}$ (shape $(3, 2)$). The inner dimensions are 3 and 3, so $AB$ is $(2, 2)$.

1. $C_{11} = 1\cdot 7 + 2\cdot 9 + 3\cdot 11 = 58$
2. $C_{12} = 1\cdot 8 + 2\cdot 10 + 3\cdot 12 = 64$
3. $C_{21} = 4\cdot 7 + 5\cdot 9 + 6\cdot 11 = 139$
4. $C_{22} = 4\cdot 8 + 5\cdot 10 + 6\cdot 12 = 154$

So $AB = \begin{pmatrix} 58 & 64 \\ 139 & 154 \end{pmatrix}$.

Two checks. $BA$ is also defined, but it is $(3, 3)$, a different shape. $A\cdot A$ is undefined: $(2, 3)(2, 3)$ has inner dimensions 3 and 2.

For cost, take $A$ of shape $(10, 100)$, $B$ of $(100, 5)$, and $C$ of $(5, 50)$. Then $(AB)C$ costs $10\cdot100\cdot5 + 10\cdot5\cdot50 = 7{,}500$ operations. $A(BC)$ costs $100\cdot5\cdot50 + 10\cdot100\cdot50 = 75{,}000$. Same matrix, ten times the work.

## Why it works

Matrix multiplication is composition. $B$ maps $\mathbf{x} \in \mathbb{R}^p$ to $B\mathbf{x} \in \mathbb{R}^n$, and $A$ maps that to $\mathbb{R}^m$. The product $AB$ is the single map that does both steps. Expanding the composition fixes every entry:

$$\big(A(B\mathbf{x})\big)_i = \sum_{t} A_{it}\sum_{j} B_{tj}\,x_j = \sum_{j}\Big(\sum_{t} A_{it}B_{tj}\Big)x_j = \sum_{j} (AB)_{ij}\,x_j.$$

The inner-dimension rule says the output of $B$ must have the size that $A$ takes as input. Composition is associative, so $(AB)C = A(BC)$ always holds. Only the cost depends on the bracketing. Strang (ch. 2) and Deisenroth (§2.2) both state associativity as a rule of matrix products.

## Signal vs. noise

- **Signal:** inner dimensions must match, and the result takes the outer ones. Write the shape beside each product before computing it.
- **Signal:** $AB \neq BA$ in general. The shapes may not even match, and the values almost always differ. Keep the order.
- **Signal:** a chain's cost depends on the bracketing. Count the operations for each option before you choose.
- **Noise for now:** Strassen and other sub-cubic algorithms, cache blocking, GPU tiling. Libraries handle these. You need the operation count, not the kernel.

## Where it breaks

- **`*` is not `@`.** In NumPy, `A * B` multiplies entry by entry, and `A @ B` is the matrix product. `A * B` raises no error when the shapes broadcast, so it returns a plausible wrong matrix.
- **Attention reordering.** Without a softmax between them, $(QK^\top)V = Q(K^\top V)$. The right-hand side never builds the $T \times T$ score matrix. The two orders cost $2T^2 d$ and $2Td^2$ multiply-adds, so the right-hand side is cheaper when $T > d$. Standard attention has a softmax between $QK^\top$ and $V$, so this reordering is not allowed there.
- **Floating point.** Summation order changes the last bits, so $(AB)C$ and $A(BC)$ agree only up to rounding. Compare with a tolerance, not `==`.
- **Integer overflow.** A narrow integer dtype can wrap silently in a long sum. Use `int64` or `float64` for large sums.
- **Batched shapes.** For `(b, m, n) @ (b, n, p)`, the product runs per batch and the batch axis is kept. For `(b, m, n) @ (n, p)`, one weight matrix is shared across the batch. Confirm which one you wrote.

## Interview angle

*60-second shape:* an $m \times n$ by $n \times p$ product costs $mnp$ multiply-adds. In a chain, the bracketing does not change the result, only the work, so pick the order that keeps the intermediate sizes small.

*Likely follow-ups:* is there a faster algorithm (Strassen is sub-cubic but less stable, and rarely decisive at practical sizes), and is $AB = BA$ (no: check the shapes first, then the values). Full answers are in `interview.md`.

## Connections

- **Prerequisites:** `cml/matrix-shapes-and-transpose` (the shape rule, and $(AB)^\top = B^\top A^\top$) and `cml/vector-operations` (dot products).
- **Unlocks:** `cml/matrix-inverse-and-rank`, `cml/normal-equation` ($X^\top X$ and $X^\top y$), `cml/eigenvalues-and-eigenvectors`, `dl/scaled-dot-product-attention` (batched $QK^\top$ and weights $\cdot\, V$).
- **Lineage:** Cayley's 1858 memoir defined the matrix product as the composition of linear substitutions. The entry rule above is that definition written out in indices.
