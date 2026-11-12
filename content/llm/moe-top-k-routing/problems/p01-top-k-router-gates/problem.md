# Top-k router gates for a mixture of experts

A mixture-of-experts layer has `E` expert networks. A small router scores every token against
every expert, producing one logit per (token, expert) pair. This function turns those logits into
routing decisions. For each token it picks the `k` experts with the largest logits, gives each
chosen expert a gate weight, and counts how many (token, slot) assignments each expert receives.
The gate weights are the weights the layer later uses to mix the chosen experts' outputs.

## Contract

- `top_k_route(logits, k)` with:
  - `logits`: `torch.float64` tensor of shape `(T, E)`, `T >= 0` tokens and `E >= 1` experts. Every
    value is finite. Otherwise `ValueError` (wrong rank, wrong dtype, or a NaN or infinite value).
  - `k`: a Python `int` with `1 <= k <= E`. `bool` is rejected with `TypeError`, as is any
    non-int. A value outside `1 .. E` raises `ValueError`.
- **Selection.** Rank the experts of each token by logit, largest first. Take the first `k`. Ties in
  logit go to the **lower expert index**. Selection uses the logits, not the probabilities, so the
  order stays exact even where a probability underflows to zero.
- **Slots.** Slot `j` of a token is its `j`-th ranked chosen expert (slot 0 has the largest logit).
- **Gates.** The gate of a chosen expert is its softmax probability renormalised over the `k` chosen
  experts (the renormalisation of the top-2 router in ST-MoE). For token `t` with chosen set `S_t`, the gate of expert
  `e` is `p[t, e] / sum over s in S_t of p[t, s]`, where `p[t, :]` is the softmax of the full row of
  logits. The gates of a token therefore sum to 1. This is the same as a softmax over the `k`
  selected logits, but you should compute it with the max subtracted for stability.
- Returns a tuple `(expert_indices, gate_weights, expert_counts)`:
  - `expert_indices`: `torch.int64`, shape `(T, k)`. `expert_indices[t, j]` is the expert of slot `j`.
  - `gate_weights`: `torch.float64`, shape `(T, k)`. Each row sums to 1 and every entry is `>= 0`.
  - `expert_counts`: `torch.int64`, shape `(E,)`. `expert_counts[e]` is the number of entries of
    `expert_indices` equal to `e`, counted over **all k slots**. Their sum is `T * k`.
- Must not mutate `logits`.
- `T = 0` is valid and returns empty `(0, k)` tensors with all-zero counts.

## Examples

```python
>>> logits = torch.tensor([[2.0, 1.0, 0.0]], dtype=torch.float64)
>>> idx, gates, counts = top_k_route(logits, k=2)
>>> idx
tensor([[0, 1]])
>>> gates          # softmax([2, 1]) = [0.7311, 0.2689]; the third expert is not chosen
tensor([[0.7311, 0.2689]], dtype=torch.float64)
>>> counts
tensor([1, 1, 0])
```

```python
>>> logits = torch.tensor([[1.0, 1.0, 1.0]], dtype=torch.float64)   # all tied
>>> idx, gates, counts = top_k_route(logits, k=2)
>>> idx.tolist(), gates.tolist()
([[0, 1]], [[0.5, 0.5]])            # ties go to the lower expert indices 0 and 1
```

```python
>>> logits = torch.tensor([[0.0, 3.0], [5.0, -1.0]], dtype=torch.float64)
>>> idx, gates, counts = top_k_route(logits, k=1)
>>> idx.tolist(), gates.tolist(), counts.tolist()
([[1], [0]], [[1.0], [1.0]], [1, 1])
```

## Constraints

- Allowed: the `torch` package in general, for example `torch.softmax`, `torch.gather`,
  `torch.sort` (with `stable=True` if you need a defined tie order), `torch.bincount`, and `math`.
- Not allowed: `torch.nn` in any form, including `torch.nn.functional`. The layer and its routing
  belong to you. Also not allowed: `torch.topk` and the `.topk` method. Selecting the `k` largest
  entries is the step this problem is about, so the library call that does it for you is banned.

## Done looks like

All public tests pass. Submit then runs the hidden tests. They cover agreement with a pure-Python
reference that computes the gates from the probabilities (on seeded random logits of several
shapes), tie-breaking to the lower expert index (including a tie at the k-th boundary), the
renormalisation of the gates, counts over every slot, a large-magnitude case where a naive exp
overflows, a case where probabilities underflow and only the logits can order the experts, `T = 0`,
`k = E`, the range and type rules for `k`, the dtype and rank rules for `logits`, non-finite
values, the output dtypes and shapes, and inputs left unchanged.

The hidden tests draw their bulk inputs from the real dataset `breast_cancer` (see
`docs/DATASETS.md`).
