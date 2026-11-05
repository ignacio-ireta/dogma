# Interview notes — Instruction-tuning batches and loss masks

## Q1. How do you build a batch for supervised fine-tuning, and which positions contribute to the loss?
**60-second answer.** Each example is a prompt and a response. I concatenate them, right pad to the longest
row, and build three aligned tensors. The inputs hold prompt and response. The attention mask is 1 on
real tokens, from lengths. The labels equal the inputs on response positions and -100 on the prompt and
on padding, so only response tokens are scored. The labels are not shifted. The loss shifts the logits by
one position, so the logits at t-1 are scored against the label at t. I check the batch by computing the
shifted loss on random logits and comparing it with the hand-computed response NLL.
**Follow-ups.** Why mask the prompt? The objective is the likelihood of the response given the prompt, and
loss on user text spends capacity on text the model should not generate. Why not mask pad ids by value?
A real token can equal the pad id, and the mask must follow lengths.

## Q2. What changes with several assistant turns or packed conversations?
**60-second answer.** Mask by role span, not by position. Every assistant span is a target and every user
or system span is masked. Packing puts several conversations in one row to avoid padding. Each
conversation keeps its own prompt mask, and attention must not cross the boundary. Otherwise the second
conversation can read the first one's text.
**Follow-ups.** Does packing need a block-diagonal attention mask? Yes, if attention is to stay within a
conversation. Which token ends a span? Its label must be the end-of-turn token when the model should learn
to stop.

## Common mistakes
- Shifting the labels in the batch and again in the loss.
- Masking by token value, which drops a real token equal to the pad id.
- Off-by-one on the prompt length, which leaves the last prompt token as a target.
- Forgetting -100 on padding, so the model learns to emit the pad id.
