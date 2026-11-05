# Interview notes — Multi-head attention

## Q1. Walk me through multi-head self-attention.

**60-second answer.** Start from the input $X$ with shape $(B, L, d)$. Project it three times, with $W_q$, $W_k$ and $W_v$, to get queries, keys and values, each $(B, L, d)$. Split the feature axis into $h$ heads of width $d_k = d/h$ by reshaping to $(B, L, h, d_k)$ and transposing to $(B, h, L, d_k)$. In each head, compute $\mathrm{softmax}(QK^\top/\sqrt{d_k})V$. Merge the heads back to $(B, L, d)$ and apply the output projection $W_o$. The cost is roughly the same as one attention head at full width, because the total feature width is still $d$.

**Follow-ups.**

- What do the heads buy? Each head has its own projections, so it can attend by a different rule, for example to nearby tokens or to a particular feature. One wide head can only produce one weight pattern per query.
- How many parameters? $4(d^2+d)$ with the biases. The head count does not change it, because $h$ divides $d$ and the projections are full $d\times d$ matrices.

## Q2. What is the most common bug in a head implementation?

**60-second answer.** Forgetting the transpose after the reshape. Reshaping $(B, L, d)$ straight to $(B, h, L, d_k)$ reads the feature axis in blocks that cross token boundaries, so heads mix tokens and features. The code still runs, and the output is a valid tensor of the right shape. The check is a test against a reference layer loaded with the same weights.

**Follow-ups.**

- Where does the $1/\sqrt{d_k}$ come from? Unit-variance inputs make each dot product have variance $d_k$. Without the scale, the softmax saturates as width grows.
- What does a mask do? It sets blocked scores to $-\infty$ before the softmax, so they get zero weight. The convention for "allowed" differs across libraries; PyTorch's `MultiheadAttention` uses `True` for blocked.

## Common mistakes

- Reshape without transpose, so heads mix tokens.
- Dividing by $\sqrt{d_{model}}$ instead of $\sqrt{d_k}$.
- Using $W\,x$ instead of $x\,W^\top$ for a layer stored in `nn.Linear` layout.
- Treating `True` in a mask as "allowed" when the library means "blocked".
