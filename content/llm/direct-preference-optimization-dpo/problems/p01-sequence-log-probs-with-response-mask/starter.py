from __future__ import annotations

import torch


def sequence_log_probs(
    logits: torch.Tensor,
    input_ids: torch.Tensor,
    response_mask: torch.Tensor,
) -> torch.Tensor:
    """Return the (B,) sum of response-token log-probabilities, with the one-token shift.

    logits: (B, T, V) unnormalised scores; logits[:, t] scores the token at position t + 1.
    input_ids: (B, T) integer token ids. response_mask: (B, T) bool; True marks a response token
    at that position. Element b of the result is the sum over t >= 1 with response_mask[b, t] of
    log softmax(logits[b, t - 1])[input_ids[b, t]], in the dtype of logits. Rows with no response
    tokens sum to 0. Raise ValueError for shape, dtype, mask-on-first-token and out-of-range id
    violations, and never mutate the inputs.
    """
    raise NotImplementedError
