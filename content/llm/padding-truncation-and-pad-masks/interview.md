# Interview notes — Padding, truncation and pad masks

## Q1. Why does a decoder-only model pad on the left for generation?

**60-second answer.** Generation appends one token per step and reads the logits at the last column. With right padding, a shorter row ends in pad columns, so the new token is written after the pads and the logits come from a pad position. With left padding, every row ends in its last real token, so one column serves all rows. The attention mask still hides the left pads, and the position ids count only real tokens, so each row keeps its own offset.

**Follow-ups.** Is left padding needed for training? No. Training can pad on either side as long as the mask and the loss labels are right. Why not just set the pads to a special id? Because the mask must come from length, not value, since a real token can share the id.

## Q2. What is the difference between an attention mask and a loss mask?

**60-second answer.** The attention mask is a boolean or 0/1 tensor the same shape as the inputs. It removes padded keys from attention before the softmax. The loss mask is a set of targets that cross-entropy ignores, usually labels set to `-100`. They use different mechanisms and are read by different parts of the model. Setting pads to `-100` in the labels does nothing for attention, and masking attention does nothing for the loss.

**Follow-ups.** Does the mean include the ignored positions? In PyTorch with `reduction="mean"`, no: the division uses the count of non-ignored targets. What if a whole row is padding? Its attention row is all masked, so the implementation must define that case explicitly.

## Q3. Which side should truncation keep?

**60-second answer.** For a decoder prompt, keep the last tokens, because the most recent turn is what the model must continue. For an encoder that reads a whole document, keep the first tokens if the start carries the signal, and say so in the contract. Whichever side is chosen, the truncation side should be a named argument, not a default that nobody reads.

## Common mistakes

- Building the mask as `ids != pad_id`, which drops real tokens equal to the pad id.
- Reading next-token logits at the last column of a right-padded batch.
- Truncating from the wrong side and dropping the most recent context.
- Forgetting the loss mask, so pads become training targets.
