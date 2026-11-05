## Hint 1 — concept

Attention runs once per head, on a slice of the features. Each head sees the same tokens but a
different block of the projected queries, keys and values. The heads are then joined back together
and projected once more. Think of every head as an independent small attention layer.

## Hint 2 — formula / approach

Project with `y = x @ W.T + b`, because the weights are stored as `(out, in)`. Divide the feature
axis into `num_heads` blocks of width `d_k = d_model // num_heads`. Each head computes
`softmax(Q_j K_j^T / sqrt(d_k)) V_j`. After the heads are joined, apply the output projection. Check
every rule before computing: the rank, the weight and bias shapes, the head count, and the mask.

## Hint 3 — code shape

Reshape a `(B, L, d)` tensor to `(B, L, h, d_k)` and then transpose dimensions 1 and 2 to get
`(B, h, L, d_k)`. Scores are `Qh @ Kh.transpose(-2, -1) / math.sqrt(d_k)`. Fill masked entries with
`masked_fill(~mask, float("-inf"))` before `torch.softmax(..., dim=-1)`. Undo the transpose and
reshape before the final `@ W_o.T + b_o`.
