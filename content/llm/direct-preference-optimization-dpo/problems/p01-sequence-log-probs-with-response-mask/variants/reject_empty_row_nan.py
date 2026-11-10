from __future__ import annotations

import torch


def _validate(logits: object, input_ids: object, response_mask: object) -> None:
    """Reject inputs outside the contract before any arithmetic runs."""
    if not isinstance(logits, torch.Tensor) or logits.ndim != 3:
        raise ValueError("logits must be a 3-D tensor of shape (B, T, V)")
    if not logits.is_floating_point():
        raise ValueError("logits must be floating-point unnormalised scores")
    batch, seq_len, vocab = logits.shape
    if batch == 0 or seq_len == 0 or vocab == 0:
        raise ValueError("logits must have at least one row, position and vocabulary entry")
    if not isinstance(input_ids, torch.Tensor) or input_ids.shape != (batch, seq_len):
        raise ValueError("input_ids must be an integer tensor with the shape (B, T) of logits")
    if input_ids.dtype == torch.bool or input_ids.is_floating_point() or input_ids.is_complex():
        raise ValueError("input_ids must be integer token ids, not floats or booleans")
    if not isinstance(response_mask, torch.Tensor) or response_mask.shape != (batch, seq_len):
        raise ValueError("response_mask must have the shape (B, T) of logits")
    if response_mask.dtype != torch.bool:
        raise ValueError("response_mask must be a boolean tensor: True marks a response token")
    if bool(response_mask[:, 0].any()):
        # The first token has no logits before it, so nothing can score it.
        raise ValueError("response_mask marks the first token, which has no prediction")
    if bool(((input_ids < 0) | (input_ids >= vocab)).any()):
        raise ValueError("input_ids must lie in [0, V): an id is outside the vocabulary")


def sequence_log_probs(
    logits: torch.Tensor,
    input_ids: torch.Tensor,
    response_mask: torch.Tensor,
) -> torch.Tensor:
    """Wrong on purpose: return the (B,) sum of response-token log-probabilities, with the one-token shift.

    The logits at position t are the scores for the token at position t + 1, so the token at
    position t is scored by logits[:, t - 1]. The mask belongs to the scored token.
    """  # noqa: E501
    _validate(logits, input_ids, response_mask)
    # Step 1: log-softmax over the vocabulary at every position. It subtracts the row maximum
    # internally, so large logits do not overflow and very small probabilities do not underflow
    # to log(0).
    log_probs = torch.log_softmax(logits, dim=-1)  # (B, T, V)
    # Step 2: the one-token shift. Predictions for tokens 1..T-1 come from positions 0..T-2.
    predictions = log_probs[:, :-1, :]  # (B, T-1, V)
    targets = input_ids[:, 1:].long()  # (B, T-1); gather needs int64 indices
    # Step 3: pick the log-probability of each actual next token.
    token_log_probs = torch.gather(predictions, -1, targets.unsqueeze(-1)).squeeze(-1)
    # Step 4: keep only response tokens. The mask is aligned with the scored token, so use
    # mask[:, 1:], not mask[:, :-1].
    kept = torch.where(response_mask[:, 1:], token_log_probs, torch.zeros_like(token_log_probs))
    # Step 5: sum over positions, one total per row. Empty rows sum to zero.
    return kept.sum(dim=1).masked_fill(~response_mask[:, 1:].any(dim=1), float("nan"))
