---
title: Padding, truncation and pad masks
minutes: 12
min_viable: "Read The system and Worked example, then implement p01 with Hint 1 open."
---

## The system

A batch is a rectangle: $B$ rows, each of length $L$. Documents are not rectangles, so collation turns a ragged list of token-id lists into one, in a fixed order:

1. **Truncate** each sequence to at most `max_length` tokens. Keep the first tokens (`truncation_side="right"`) or the last tokens (`"left"`).
2. **Pad** the kept tokens with `pad_id` up to `max_length`. Pads go after the tokens (`padding_side="right"`) or before them (`"left"`).
3. **Mask.** Build `attention_mask` with the same shape: `1` for a real token, `0` for a pad.

Rules that decide most bugs:

- **The mask follows length, not value.** A pad is a column past the kept length. A token id can equal `pad_id` (for example `0` or the end-of-text id) and still be real. Writing `mask = ids != pad_id` drops those tokens.
- **Output contract:** `input_ids` and `attention_mask` are `int64` tensors of shape $(B, L_{\max})$, where $L_{\max}$ is `max_length`.

| choice | `"right"` | `"left"` |
|---|---|---|
| padding puts pads | after the tokens | before the tokens |
| truncation keeps | the first `max_length` tokens | the last `max_length` tokens |

Two masks, two jobs:

| tensor | values | read by | meaning |
|---|---|---|---|
| attention mask | `1` real, `0` pad | attention | which keys a query may read |
| target labels | the token id, or `-100` at pad | cross-entropy | which positions contribute to the loss |

## Worked example

Take `sequences = [[5, 6, 7], [8]]`, `pad_id = 0`, `max_length = 4`. Row 0 has 3 tokens and gets 1 pad. Row 1 has 1 token and gets 3 pads.

Right padding:

- `input_ids = [[5, 6, 7, 0], [8, 0, 0, 0]]`
- `attention_mask = [[1, 1, 1, 0], [1, 0, 0, 0]]`

Left padding:

- `input_ids = [[0, 5, 6, 7], [0, 0, 0, 8]]`
- `attention_mask = [[0, 1, 1, 1], [0, 0, 0, 1]]`

Truncation of `[1, 2, 3, 4, 5]` to `max_length = 3`: `"right"` keeps `[1, 2, 3]`, `"left"` keeps `[3, 4, 5]`. The mask is `[1, 1, 1]` in both cases, because no pad is needed.

Mask by length. With `pad_id = 0`, `[0, 4]` padded to 3 on the right gives `ids = [0, 4, 0]` and `mask = [1, 1, 0]`: the first `0` is real.

**Why generation pads on the left.** Decoding reads next-token logits at the last column, `logits[:, -1, :]`. Take rows `[5, 6, 7]` and `[8]` with `max_length = 4`. Right padding gives `[5, 6, 7, 0]` and `[8, 0, 0, 0]`: the last column is a pad for row 1, so its logits come from a pad position. Left padding gives `[0, 5, 6, 7]` and `[0, 0, 0, 8]`: column 3 is the last real token in every row, so one slice serves the whole batch. The attention mask still hides the left pads, and position ids must count only real tokens.

## Why it works

Padded keys can be removed the same way as future keys in the causal mask: the score is set to $-\infty$ before the softmax, so the weight is $0$. The paper gives this for illegal connections in the decoder (Vaswani et al., 2017, §3.2.3, p.5). Applying it to pad keys is standard practice, not something the paper states.

The loss mask is a different mechanism. Cross-entropy takes a target per position. The PyTorch default `ignore_index = -100` tells the loss to skip those positions. With the default `reduction="mean"`, the loss divides by the number of non-ignored targets, not by $B \cdot T$. A padded position therefore neither adds loss nor dilutes it.

## Signal vs. noise

Signal: the mask comes from lengths, the two sides are independent, and the attention mask and the loss labels are different tensors. Noise: which pad token string the tokenizer uses. Both tensors here are `int64`.

## Where it breaks

- **Mask by value.** Dropping every token equal to `pad_id` loses real tokens, which is common when the pad id is `0`.
- **Right padding in generation.** The next-token logits are read from a pad column. The output looks plausible and is wrong for every short row in the batch.
- **Truncating the wrong side.** Keeping the start of a long chat prompt drops the latest user turn. Decoder-only prompts usually keep the last tokens.
- **No loss mask.** Pads become targets, and the model learns to predict the pad token.
- **Ids in a float tensor.** float32 holds integers exactly only up to $2^{24}$; larger ids change silently.

## Interview angle

Sixty-second answer: collation truncates, pads and builds a mask, in that order. The attention mask marks real tokens by length and hides pads from attention. The loss mask is a separate tensor, usually labels set to $-100$, so pads never count as targets. Decoder-only generation pads on the left, because the next token is appended at the last column. Likely follow-up: why is the pad id not enough to identify padding? Because a real token can equal the pad id. Second follow-up: which side should truncation keep? For a decoder prompt, the last tokens, which hold the most recent context.

## Connections

- Prerequisite: [Causal masking](../../dl/causal-masking/), where the same boolean mask idea is built for attention.
- Next: [Next-token targets and batches](../next-token-targets-and-batches/), which shifts the inputs by one, packs documents and sets the loss labels.
