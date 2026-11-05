## Hint 1 — concept

Each step is one call to the model, one argmax per row, and one append. Rows do not interact, but
they share one loop. A finished row is not removed from the batch. It keeps a place in the tensor,
and that place is filled with `pad_id`.

## Hint 2 — formula / approach

Keep a boolean `finished` flag per row, of length `B`. At each step, compute the argmax of the
logits for all rows. Replace the argmax with `pad_id` where the flag is set, using
`torch.where`. Then set the flag for rows whose new token equals `eos_id`. Stop when every flag is
set or after `max_new_tokens` steps. Check every rule before the loop starts: the rank, the dtype,
the batch and prompt sizes, the integer types, and the cap.

## Hint 3 — code shape

Start from `tokens = prompts.clone()`. Inside the loop, call `logits = next_logits(tokens)`, take
`choice = torch.argmax(logits, dim=-1)`, and build `next_tok` with `torch.where(finished, ...,
choice)`. Append it with `torch.cat([tokens, next_tok.unsqueeze(1)], dim=1)`. Update `finished`
after the append, and return `tokens` after the loop.
