---
title: "Scaled dot-product attention"
minutes: 12
min_viable: "Read The system and Why it works, then implement p01 with Hint 2 open."
---

## The system

Attention takes queries, keys and values and returns, for each query, a weighted average of the values. Name every shape first. Let $L$ be the number of queries, $S$ the number of keys and values, $d_k$ the key width, and $d_v$ the value width:

| Tensor | Shape | Meaning |
|---|---|---|
| $Q$ | $L\times d_k$ | one query row per output position |
| $K$ | $S\times d_k$ | one key row per source position |
| $V$ | $S\times d_v$ | one value row per source position |
| $Z = QK^T/\sqrt{d_k}$ | $L\times S$ | scaled scores |
| $A = \mathrm{softmax}_{\text{row}}(Z)$ | $L\times S$ | attention weights |
| $O = AV$ | $L\times d_v$ | output |

Each row of $A$ is a probability vector over the $S$ keys, so $\sum_j A_{ij}=1$ for every query $i$. Columns carry no such constraint. Leading batch axes (for example batch and head) are shared and do not change any rule; the last two axes are always the ones above.

**Masks.** A boolean mask $M\in\{0,1\}^{L\times S}$ marks the allowed pairs, with $M_{ij}=1$ meaning query $i$ may attend key $j$. Disallowed scores are removed before the softmax, so $A_{ij}=0$ where $M_{ij}=0$. An additive mask adds $0$ to allowed scores and $-\infty$ to disallowed ones. The two are equivalent for every row with at least one allowed key.

## Worked example

Take $L=1$, $S=2$, $d_k=2$, $d_v=2$. Let $Q=[1,0]$, $K=\begin{pmatrix}1&0\\0&1\end{pmatrix}$ and $V=\begin{pmatrix}10&0\\0&10\end{pmatrix}$.

1. Raw scores $QK^T = [1, 0]$.
2. Scale by $1/\sqrt{2}\approx0.7071$: $Z=[0.7071, 0]$.
3. Softmax: $e^{0.7071}\approx2.028$ and $e^0=1$, so $A=[2.028,1]/3.028\approx[0.670, 0.330]$. The row sums to 1.
4. Output: $AV = [0.670\cdot10,\ 0.330\cdot10]\approx[6.70,\ 3.30]$.

With the mask $M=[1,0]$, the second score is removed, so $A=[1,0]$ and $O=[10,0]$. With $M=[0,0]$ no key is allowed; the row has no defined softmax, and the contract in p01 sets it to zeros.

## Why it works

**Why divide by $\sqrt{d_k}$.** Suppose the components of $q$ and $k$ are independent with mean $0$ and variance $1$. Then $q\cdot k=\sum_{i=1}^{d_k}q_ik_i$ has mean $0$ and variance $d_k$. Without scaling, the typical score grows like $\sqrt{d_k}$, and the softmax saturates toward a one-hot vector where its gradient is nearly zero. Dividing by $\sqrt{d_k}$ restores unit variance per score. Vaswani et al. (2017, §3.2.1 and its footnote 4) give this argument.

**Why subtract the maximum.** The softmax is unchanged by subtracting the same constant $c$ from every entry of a row, because $e^{z_j-c}/\sum_k e^{z_k-c}=e^{z_j}/\sum_k e^{z_k}$. Choose $c=\max_k z_k$. Then every exponent is at most $0$, so $e^{z_j-c}\le1$ and nothing overflows. The same identity is the reason masked rows need care, covered below.

## Signal vs. noise

Signal: the three shapes, the scale, the row-wise softmax and the output $AV$. Noise: which backend kernel runs the product, and batch layout. Ignore dropout for this session; it multiplies $A$ by a random mask and changes nothing in the rules above.

## Where it breaks

- **Forgetting the scale** still runs, but the softmax saturates for large $d_k$. The oracle test with $d_k=64$ exposes it; a shape test does not.
- **Softmax over the wrong axis.** Using `dim=-2` normalises columns. Check that every row of the returned weights sums to 1.
- **Overflow without the maximum.** Scaling the queries by $10^3$ makes naive $e^{z}$ overflow to `inf` and the ratio to `NaN`. Use the maximum subtraction.
- **Fully masked rows.** If row $i$ has no allowed key, $\max_j z_{ij}$ over the masked row is $-\infty$, and $z-\max$ is `NaN`. A naive implementation then returns `NaN` in the output row. The contract in p01 chooses zeros for weights and output and says so. Frameworks differ in this case, and some return NaN: in a check with torch 2.14.1, `torch.nn.MultiheadAttention` gave NaN for a fully blocked row, while `F.scaled_dot_product_attention` gave zeros on CPU. Do not rely on either behaviour without stating it. The causal-masking topic returns an all-`False` mask row instead of a score row, and multi-head attention raises; each topic states its own rule.
- **Mask polarity.** A mask where `True` blocks attention is the inverse of this contract. Shape checks do not catch it; the oracle does.

## Interview angle

Sixty-second answer: attention computes $\mathrm{softmax}(QK^T/\sqrt{d_k})V$. Shapes are $L\times d_k$, $S\times d_k$ and $S\times d_v$; the weights are $L\times S$ with each row summing to 1. The $\sqrt{d_k}$ keeps score variance near 1 so the softmax does not saturate. Masks remove pairs before normalisation, and every row needs at least one allowed key or an explicit policy. Likely follow-up: why not scale by $d_k$? Variance grows like $d_k$, so the standard deviation grows like $\sqrt{d_k}$, and dividing by $\sqrt{d_k}$ is the correct normalisation. Second follow-up: why not additive masks with $-10^9$? Large finite negatives can leak when scores are large, and they do not express a fully masked row.

## Connections

This combines the softmax (cml softmax topic) with the matrix products (cml matrix multiplication topic). Masking choices for autoregressive models come next in the causal masking topic. The attention output feeds the residual block in the residual connections topic. Historical lineage: Vaswani et al. (2017), arXiv:1706.03762, introduced scaled dot-product attention as the core of the transformer, with the decoder mask in §3.2.3.
