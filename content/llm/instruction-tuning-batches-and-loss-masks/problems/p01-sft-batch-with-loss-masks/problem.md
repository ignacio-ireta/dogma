# Supervised fine-tuning batch with response-only loss masks

Implement `build_sft_batch(pairs, pad_id)`. Each pair is one training example: a tokenized prompt and
the tokenized response the model should learn to produce. Build one rectangular batch with three
tensors:

- `input_ids`: the prompt followed by the response, right padded with `pad_id`.
- `labels`: the same positions as `input_ids`. A response token keeps its id. Every prompt position and
  every padding position is `-100`, the ignore index of the cross-entropy loss.
- `attention_mask`: `1` on real tokens (prompt or response), `0` on padding.

**Label alignment.** `labels` is not shifted. Position `t` of `labels` holds the id that position `t`
of the input is. The one-token shift belongs to the loss: the logits at position `t - 1` are scored
against `labels[t]`. The batch builder therefore only decides *which* positions count, and the loss
does the shift.

## Contract

- `pairs`: a `list` or `tuple` of `B >= 0` items. Each item is a `list` or `tuple` of exactly two
  elements, `(prompt_ids, response_ids)`. Both are `list` or `tuple` of Python `int` token ids, each
  id with `0 <= id <= 2**63 - 1`.
- `prompt_ids` must have at least one token. The first response token needs a preceding position to
  be predicted from, so an empty prompt is invalid.
- `response_ids` must have at least one token.
- `pad_id`: Python `int` in `0..2**63 - 1`. A `bool` is invalid.
- Let `n_i = len(prompt_i) + len(response_i)` and `L = max(n_i)`. Each row has length `L`.
  - `input_ids[i]` holds the prompt, then the response, then `pad_id` in the columns `n_i..L-1`.
  - `labels[i]` holds `-100` for columns `0..len(prompt_i)-1`, then the response ids, then `-100` in
    columns `n_i..L-1`.
  - `attention_mask[i]` holds `1` in columns `0..n_i-1` and `0` after.
  - The masks come from lengths, not from token values. A response token whose id equals `pad_id`
    is a real token: its label keeps its id and its attention mask is `1`.
- Returns: `(input_ids, labels, attention_mask)`, each a `torch.int64` tensor of shape `(B, L)`. When
  `pairs` is empty, all three have shape `(0, 0)`.
- Raises `ValueError` when: `pairs` is not a list or tuple; an item is not a two-element list or
  tuple; a prompt or response is not a list or tuple; the prompt or the response is empty; an id is
  not a Python `int`, is a `bool`, is negative, or exceeds `2**63 - 1`; `pad_id` is not a Python
  `int`, is a `bool`, is negative, or exceeds `2**63 - 1`.
- Must not mutate `pairs` or any list inside it.

## Examples

```python
>>> input_ids, labels, mask = build_sft_batch([([5, 6], [7, 8, 9]), ([4], [3])], pad_id=0)
>>> input_ids
tensor([[5, 6, 7, 8, 9],
        [4, 3, 0, 0, 0]])
>>> labels          # -100 on the prompt and on padding; response ids keep their positions
tensor([[-100, -100,    7,    8,    9],
        [-100,    3, -100, -100, -100]])
>>> mask
tensor([[1, 1, 1, 1, 1],
        [1, 1, 0, 0, 0]])
```

```python
>>> input_ids, labels, mask = build_sft_batch([([1], [0, 2])], pad_id=0)
>>> labels          # the response token 0 equals pad_id, but it is a real target
tensor([[-100,    0,    2]])
>>> mask
tensor([[1, 1, 1]])
```

```python
>>> build_sft_batch([([], [3])], pad_id=0)
ValueError: prompt must contain at least one token
```

## Constraints

- Allowed: Python list operations, integer arithmetic, and torch tensor creation and indexing, such as
  `torch.full`, `torch.zeros`, `torch.tensor` and slice assignment.
- Not allowed: `torch.nn` in any form (this includes `torch.nn.functional`, `torch.nn.utils.rnn` and
  its `pad_sequence`, and `torch.nn.CrossEntropyLoss`), and `torch.utils.data` (its collate helpers
  pad and mask for you). Write the padding, the labels and the mask yourself.

## Done looks like

All public tests pass. **Submit** then runs the hidden tests. They check the batch against a plain-Python
reference on seeded random pairs, and they check the alignment end to end: with random logits, the
shifted cross-entropy over `labels` must equal the negative log-likelihood of the response tokens
computed by hand. They also cover response tokens equal to `pad_id`, prompt-only positions that must
never be targets, the first response token right after the prompt, a single pair with no padding, a
batch of many lengths, very large token ids that must stay exact in `int64`, the output dtype and
shape, the empty batch, every `ValueError` rule, and inputs left unchanged.

The hidden tests draw their bulk inputs from the real datasets `breast_cancer` and `gsm8k` (see
`docs/DATASETS.md`).
