---
title: "Document packing and cross-document masks"
minutes: 12
min_viable: "Read The system and Worked example, then implement p01 with Hint 1 open."
---

## The system

Pretraining data is many short documents. Padding each one to the row length $T$ spends compute on
pad positions. Packing puts several whole documents into one row of $T$ tokens instead. The row
carries a document-id vector $\mathbf{d}\in\mathbb{Z}^{T}$, where $d_i$ is the document of token $i$.
Four rules define a packed row:

1. **Documents are runs.** A document is a maximal run of equal ids, and each id forms one run.
   The row $[0,0,1,0]$ is invalid: its last $0$ would attend the first document.
2. **Block-diagonal causal mask.** Query $i$ may attend key $j$ only when both tokens share a
   document and $j$ is not in the future. With $M\in\{0,1\}^{T\times T}$, where $M_{ij}=1$ means
   "may attend":
   $$M_{ij}=[\,d_i=d_j\,]\cdot[\,j\le i\,].$$
3. **Positions restart.** With $s(i)$ the index of the first token of $i$'s document, the position
   id is $p_i=i-s(i)$.
4. **Loss at boundaries.** The target at position $t$ is the token at $t+1$. When $d_t\neq d_{t+1}$,
   that target starts another document, so its label is $-100$, the ignore index from the padding
   topic.

## Worked example

Three documents of lengths 3, 2 and 3, with ids 0, 1 and 2. The tokens are $A_1A_2A_3\,B_1B_2\,C_1C_2C_3$,
so $\mathbf{d}=(0,0,0,1,1,2,2,2)$.

| $i$ | token | allowed keys $j$ | row count | $p_i$ |
|---|---|---|---|---|
| 0 | $A_1$ | {0} | 1 | 0 |
| 1 | $A_2$ | {0, 1} | 2 | 1 |
| 2 | $A_3$ | {0, 1, 2} | 3 | 2 |
| 3 | $B_1$ | {3} | 1 | 0 |
| 4 | $B_2$ | {3, 4} | 2 | 1 |
| 5 | $C_1$ | {5} | 1 | 0 |
| 6 | $C_2$ | {5, 6} | 2 | 1 |
| 7 | $C_3$ | {5, 6, 7} | 3 | 2 |

The mask has 15 True entries. A plain causal mask over the same eight tokens has 36. The other 21
are attention from a token to an earlier, unrelated document. That is the leak.

For the loss, take a 9-token window: the same ids plus one more $C$ token. Inputs are positions 0 to 7 and targets are positions 1 to 8. The input ids are $(0,0,0,1,1,2,2,2)$, so the
target ids $d_1,\dots,d_8$ are $(0,0,1,1,2,2,2,2)$. Targets at $t=2$ (A to B) and
$t=4$ (B to C) cross a boundary, so their labels become $-100$. Six of the eight targets remain.

## Why it works

A causal mask keeps only earlier keys. Vaswani et al. (2017, §3.2.3) block leftward information
flow in the decoder by setting illegal scores to $-\infty$ before the softmax. Packing adds one
condition to the legal set: the key must share the query's document. The allowed keys for query
$i$ are then the contiguous range $\{s(i),\dots,i\}$, which has $p_i+1$ elements. Each row of $M$
therefore has $p_i+1$ True entries, and the example sums to $1+2+3+1+2+1+2+3=15$. With one document,
$s(i)=0$ everywhere, so the mask is the plain causal mask.

Restarting positions matters only through the position encoding. With RoPE, the attention score
depends on the position difference, which is the same inside a document either way. With absolute
position embeddings, a document at offset 5 sees different inputs from the same document alone.
The restart makes every packed document see the positions it sees alone. The loss rule follows the
same logic: a target that starts a new document cannot be predicted from the previous document.

## Signal vs. noise

**Signal:** the two conditions and their AND, the row count $p_i+1$, the restart rule, and the
boundary label rule.

**Noise:** storage. Variable-length kernels take cumulative document lengths instead of a dense mask
and encode the same blocks.

## Where it breaks

- **Plain causal mask on packed rows.** A token in document $B$ attends to $A$, so training uses
  context the document never has when it is used alone.
- **Reused ids.** If id 0 returns after id 1, an equality rule lets the later run attend the earlier
  one. The contract rejects such rows.
- **Forgotten boundary labels.** The model learns to predict the first token of an unrelated
  document from the end of the previous one.
- **Off-by-one restart.** Position ids that start at 1 shift every document by one position, so
  single-document use disagrees with training.
- **Tail padding.** Pads given the id of the last real document share its run and count as its tokens.

## Interview angle

Sixty-second answer: packing fills each row with whole documents, so padding waste goes away.
Attention then needs a block-diagonal causal mask, the causal rule restricted to one document. Positions
restart per document, and the loss is ignored where the next token starts a new document. Likely
follow-up: why not a plain causal mask plus an EOS token? The EOS marks the end of a document but does
not stop attention from reading across it. Second follow-up: cost? Dense masks stay $T\times T$;
variable-length kernels pay for the sum of squared document lengths.

## Connections

- Prerequisites: [Next-token targets and batches](../next-token-targets-and-batches/) gives the window
  and the EOS separator. [Causal masking](../../dl/causal-masking/) gives the mask this topic extends.
  [Padding, truncation and pad masks](../padding-truncation-and-pad-masks/) gives the $-100$ rule.
- Lineage: the causal mask is from Vaswani et al. (2017). Packing practice has no primary source in
  this repository, so its rules are checked by the fixtures, not cited.
