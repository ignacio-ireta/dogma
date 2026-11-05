---
title: "Causal masking"
minutes: 12
min_viable: "Read The system and Worked example, then implement p01 with Hint 1 open."
---

## The system

An autoregressive model predicts token $t$ from tokens $<t$ only. Attention enforces this with a boolean mask $M\in\{0,1\}^{L\times S}$, where $L$ is the number of queries, $S$ the number of keys, and $M_{ij}=1$ means query $i$ may attend key $j$. Two rules combine:

1. **Causal rule (bottom-right aligned).** Query $i$ is the token at absolute position $S-L+i$, so the last query lines up with the last key. It may attend key $j$ when $j\le S-L+i$. For $L=S$ this is the ordinary lower triangle, $M_{ij}=[j\le i]$.
2. **Padding rule.** With a padding indicator $V\in\{0,1\}^{N\times S}$ ($V_{nj}=1$ for a real token, $0$ for padding), a key is usable only if $V_{nj}=1$. The combined mask is $M_{nij} = [j\le S-L+i]\cdot V_{nj}$, shape $N\times L\times S$.

The causal part depends only on $(L,S)$. The padding part depends only on the batch row $n$. Keep them separate in code, then combine.

## Worked example

Take $L=2$, $S=4$: the last two tokens of a four-token context, as in a decoding step with a key-value cache. The offset is $S-L=2$.

| query $i$ | absolute position | allowed keys $j$ | row |
|---|---|---|---|
| 0 | $2$ | $j\le2$ | $[1,1,1,0]$ |
| 1 | $3$ | $j\le3$ | $[1,1,1,1]$ |

Now add padding $V=[1,1,0,1]$ for one sequence. The combined row for query 0 is $[1,1,1,0]\cdot[1,1,0,1]=[1,1,0,0]$, and for query 1 it is $[1,1,0,1]$. Key 2 is a real position that is padding in this sequence, so no query may attend it.

With $L=3$, $S=2$ (more queries than keys), the offset is $-1$. Query 0 has absolute position $-1$: no key qualifies, so its row is all zeros. That is a fully masked row.

## Why it works

Vaswani et al. (2017, §3.2.3) prevent "leftward information flow in the decoder to preserve the auto-regressive property", by masking the scores of illegal connections to $-\infty$ before the softmax. Here that masking is written as the boolean rule above. The offset matters because during decoding the queries are the newest tokens, and they sit at the end of the key sequence, not at its start.

Torch's `scaled_dot_product_attention(..., is_causal=True)` documents an upper-left (top-left) alignment for non-square masks. Checked in torch 2.14.1: for $L=2$, $S=4$ the flag gives query 0 weight only on key 0, and its output matches an explicit `tril(diagonal=0)` mask exactly. It does not match the bottom-right rule above, so a decoding step needs an explicit bottom-right mask, not the flag.

## Signal vs. noise

Signal: the offset $S-L$ and the combination of the two masks. Noise: how the mask is stored (boolean or additive) and whether the batch axis is broadcast. The rule is the same either way. The alignment is the one place people go wrong, so check it against the worked example.

## Where it breaks

- **Top-left alignment.** Using `is_causal=True` with $L\neq S$ silently shifts every query to the start of the keys. Outputs look plausible but cached decoding is wrong. Test a case with $L<S$.
- **Fully masked rows.** A row with no allowed key has maximum $-\infty$. A naive softmax returns `NaN`, which then spreads through the residual stream. Library implementations are not a safe reference here: in torch 2.14.1, `torch.nn.MultiheadAttention` returned NaN for a fully blocked row, while `F.scaled_dot_product_attention` returned zeros on CPU. This happens in two cases: $L>S$ (the early query rows) and a sequence whose keys are all padding. The contract in p01 returns an all-`False` row and never raises. Attention then has to treat such rows explicitly (see the attention topic for zero weights).
- **Padding on the wrong axis.** Padding is a property of keys (columns), so it broadcasts along the query axis. Applying it to rows removes queries rather than keys.
- **Mask polarity.** `True` means "may attend" here. A library convention that uses `True` for "blocked" inverts the rule. Check the polarity against the worked example.

## Interview angle

Sixty-second answer: a causal mask is a lower triangle; with a cache of $S-L$ earlier tokens, the rule becomes "query $i$ sees keys up to $S-L+i$", a bottom-right alignment. Padding is a per-batch key mask that is ANDed in. Any row with no allowed key must have a defined result, typically zeros, or softmax returns NaN. Likely follow-up: why not `is_causal=True`? It is top-left aligned for non-square masks, which is wrong during cached decoding. Second follow-up: when does a row become fully masked? When the query is earlier than every key, which happens if $L>S$, or when a whole sequence is padding.

## Connections

This builds directly on [scaled dot-product attention](../scaled-dot-product-attention/), which applies the mask. The same alignment rule governs key-value caching during inference. The fully masked row policy also matters for cross-attention and for flash-attention kernels, which handle masks in blocks. Historical lineage: Vaswani et al. (2017), arXiv:1706.03762, §3.2.3, introduced decoder self-attention masking.
