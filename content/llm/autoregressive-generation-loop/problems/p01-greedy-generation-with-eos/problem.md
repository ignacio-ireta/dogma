# Greedy generation with per-sequence EOS

Implement `greedy_generate(next_logits, prompts, max_new_tokens, eos_id, pad_id)`. It runs greedy
decoding for a batch of prompts that all have the same length. At each step, every unfinished
row takes the argmax of its next-token logits and appends it. A row is finished after it emits
`eos_id`, and from then on it appends `pad_id`. The loop stops when every row is finished or when
`max_new_tokens` new tokens have been generated. The model is a black box: you only call
`next_logits`.

## Contract

- `next_logits`: a callable. You call it with an `int64` tensor `tokens` of shape `(B, T)`, where
  `T >= 1`. It holds the prompt followed by every token appended so far. Finished rows carry
  `pad_id` at the appended positions. It returns a tensor of shape `(B, V)`, `V >= 1`, with the
  logits for the token that follows each row's last position. Logits of finished rows are ignored.
- `prompts`: `torch.int64` tensor of shape `(B, P)`, `B >= 1` and `P >= 1`. Each row is a
  complete prompt with no padding.
- `max_new_tokens`: `int`, `>= 0`. A `bool` is rejected.
- `eos_id`, `pad_id`: `int`. A `bool` is rejected. They are not checked against `V`, and
  `pad_id` may equal `eos_id`.
- Loop: before step `s` (`s = 1, 2, ...`), if every row is finished, stop. Otherwise call
  `next_logits(tokens)`, where `tokens` has width `P + s - 1`. For each unfinished row, append
  the argmax of its logits. Ties go to the lowest token id. For each finished row, append
  `pad_id`. A row that appends `eos_id` becomes finished. The EOS token is part of the output.
- Returns: `torch.int64` tensor of shape `(B, P + G)`, where `G` is the number of steps run,
  `0 <= G <= max_new_tokens`. Row `b` holds its prompt, then its generated tokens up to and
  including its EOS (or up to the cap), then `pad_id` up to width `P + G`. `G` equals the length of
  the longest generated sequence.
- Raises `ValueError` when: `prompts` is not a 2-D `int64` tensor; `B = 0` or `P = 0`;
  `max_new_tokens` is negative or not an `int`; `eos_id` or `pad_id` is not an `int`; or
  `next_logits` returns something that is not a 2-D tensor of shape `(B, V)` with `V >= 1`.
- Must not mutate `prompts`. The returned tensor must not share storage with `prompts`.

## Examples

The toy model `next_logits` below returns logits whose argmax is `(last + 1) % 4`. Here `eos_id = 3`
and `pad_id = 0`. Row 0 is the prompt `[0]` and row 1 is the prompt `[1]`.

```python
>>> greedy_generate(model, torch.tensor([[0], [1]]), max_new_tokens=5, eos_id=3, pad_id=0)
tensor([[0, 1, 2, 3],
        [1, 2, 3, 0]])
```
Row 1 hits EOS at step 2 and is padded at step 3. The loop stops after step 3, when every row is
finished, so the output has width `1 + 3`.

```python
>>> greedy_generate(model, torch.tensor([[0], [1]]), max_new_tokens=2, eos_id=3, pad_id=0)
tensor([[0, 1, 2],
        [1, 2, 3]])
```
The step cap stops the loop before row 0 reaches EOS. No row is padded.

```python
>>> greedy_generate(model, torch.tensor([[0], [1]]), max_new_tokens=0, eos_id=3, pad_id=0)
tensor([[0],
        [1]])
```

## Constraints

- Allowed: `torch` tensor operations, such as `torch.argmax`, `torch.where`, `torch.full_like`,
  `torch.cat` and `clone`; plain Python `int` and `bool` checks.
- Not allowed: `torch.nn` in any form, `torch.distributions` and `torch.multinomial`. Sampling is
  the next topic. Here the choice is always the argmax.

## Done looks like

All public tests pass. Then **Submit** runs the hidden tests. They cover: agreement with an
independent one-row-at-a-time reference on seeded random bigram models; a row that emits EOS not
stopping the others; finished rows padded with a non-zero `pad_id`; the EOS token kept in the
output; ties broken toward the lowest id; the step cap counting only new tokens; the model seeing
a prefix that grows by one column per call, with `pad_id` in finished rows; a zero cap returning
a copy and not an alias; the prompt left unchanged; a batch of one; a large vocabulary; the output
dtype; and the `ValueError` rules for invalid arguments and for a model that returns the wrong
shape.

The hidden tests draw their bulk inputs from the real datasets `gutenberg_frankenstein`,
`gutenberg_pride_prejudice` and `gutenberg_walden` (see `docs/DATASETS.md`).
