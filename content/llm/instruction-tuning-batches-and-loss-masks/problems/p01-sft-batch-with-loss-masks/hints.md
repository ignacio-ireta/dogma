## Hint 1 — concept

A training row is a prompt and a response glued together. The model is only supposed to learn the
response, so the loss must ignore every position that belongs to the prompt or to padding. The
three tensors are the same grid with three different jobs: the ids, the targets, and the attention.

## Hint 2 — formula / approach

Let `n = len(prompt) + len(response)` for each row, and let `L` be the longest `n`. Start each tensor
from its padding value: `pad_id` for the ids, `-100` for the labels, `0` for the mask. Then write the
real values into columns `0..n-1`. The labels are **not** shifted. Column `t` of the labels holds the
id of input token `t` only when `t` is a response position. The loss shifts them later.

## Hint 3 — code shape

Validate every pair first: each prompt and response must be a non-empty list or tuple of plain ints.
Compute `B` and `L` from the validated lengths. Create three `(B, L)` `int64` tensors with
`torch.full` and `torch.zeros`. For each row, slice-assign the prompt-then-response ids into the first
`n` columns of the ids and the attention mask. Slice-assign only the response ids into the labels, at
columns from `len(prompt)` to `n`. Do not look at token values to decide what is masked.
