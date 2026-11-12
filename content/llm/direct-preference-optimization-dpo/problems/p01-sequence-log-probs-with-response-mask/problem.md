# Sequence log-probabilities with a response mask

DPO compares whole responses, so it needs the log-probability of each response under a model.
A response's log-probability is the sum of its token log-probabilities, counting only the response
tokens. Prompt tokens are masked out.

Implement `sequence_log_probs(logits, input_ids, response_mask)`. It returns, for each row of a
batch, the sum of the log-probabilities the model assigns to the response tokens of that row.

Remember the shift. The logits at position `t` are the model's scores for the token at position
`t + 1`. So the token at position `t` is scored by the logits at position `t - 1`.

## Contract

- `logits`: floating-point tensor, shape `(B, T, V)`, with `B >= 1`, `T >= 1`, `V >= 1`. These are
  unnormalised scores. Do not apply a softmax before the call.
- `input_ids`: integer tensor (not `bool`), shape `(B, T)`, values in `[0, V)`.
- `response_mask`: boolean tensor, shape `(B, T)`. `response_mask[b, t]` is `True` when the token at
  position `t` of row `b` belongs to the response. The mask describes the token being scored.
- Returns a tensor of shape `(B,)` with the dtype of `logits`. Element `b` is
  $\sum_{t \in [1, T):\ \text{response\_mask}[b,t]} \log p(\text{input\_ids}[b,t] \mid \text{logits}[b, t-1])$,
  where $p$ is the softmax over the vocabulary.
- A row with no `True` in its mask has sum `0`.
- Raises `ValueError` when: `logits` is not 3-D or not floating-point; `input_ids` is not an integer
  tensor of shape `(B, T)`; `response_mask` is not a boolean tensor of shape `(B, T)`; `B == 0`;
  `response_mask[:, 0]` contains a `True` (the first token has no prediction); any `input_ids` value is
  outside `[0, V)`.
- Must not mutate inputs.

## Examples

Tensors are shown as nested lists for brevity. Pass `torch.tensor(...)` values with the dtypes above.

```python
logits = [
    [[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]]
]  # B=1, T=3, V=4: every token has probability 1/4
input_ids = [[0, 2, 1]]
response_mask = [[False, True, True]]  # score tokens 1 and 2
sequence_log_probs(logits, input_ids, response_mask)
# -> [-2.7725887]   (two tokens, each log(1/4) = -1.3862944)
```

```python
logits = [
    [
        [0.0, 1.0986123],  # position 0 predicts token 1: softmax = [0.25, 0.75]
        [0.0, 0.0],
    ]
]  # position 1, unused here
input_ids = [[0, 1]]
response_mask = [[False, True]]  # token 1 is scored by logits at position 0
sequence_log_probs(logits, input_ids, response_mask)
# -> [-0.2876821]   (log 0.75)
```

```python
logits = [[[0.0, 0.0], [0.0, 0.0]], [[0.0, 0.0], [0.0, 0.0]]]
input_ids = [[0, 0], [1, 1]]
response_mask = [[False, False], [False, True]]
sequence_log_probs(logits, input_ids, response_mask)
# -> [0.0, -0.6931472]   (row 0 has no response tokens; row 1 scores one token at probability 1/2)
```

## Constraints

- Allowed: Python standard library and PyTorch tensor operations, such as `log_softmax`, `gather`,
  `sum`, indexing and masking.
- Not allowed: `torch.nn` (this covers `torch.nn.functional`, so `F.cross_entropy` and
  `F.log_softmax` are banned too). Write the log-probability lookup yourself from the definition.

## Done looks like

The three examples above pass. The hidden tests compare with an independent cross-entropy reference
on seeded random batches with random masks. They check the one-token shift, a row with no response
tokens, a batch with no scored tokens, logits with very large magnitudes and very small
probabilities, float32 dtype, a mask on the first token, out-of-range token ids, wrong dtypes and
shapes, a float64 finite-difference gradient check through the logits, and no in-place mutation.

The hidden tests draw their bulk inputs from the real dataset `breast_cancer` (see
`docs/DATASETS.md`).
