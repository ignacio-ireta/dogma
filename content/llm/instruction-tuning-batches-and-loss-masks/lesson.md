---
title: "Instruction-tuning batches and loss masks"
minutes: 12
min_viable: "Read The system and the Worked example, then implement p01 with Hint 1 open."
---

## The system

Supervised fine-tuning (SFT) trains on pairs $(p, r)$: a prompt of $P$ token ids and a response of $R$ token ids. The
model should produce $r$ after $p$, not reproduce $p$. Symbols: $B$ rows, $L$ the longest row, and $n_i = P_i + R_i$ the
real length of row $i$.

The batch has three $(B, L)$ tensors, each with one job:

| tensor | row $i$, column $t$ | read by |
|---|---|---|
| `input_ids` | the token id: prompt, then response, then `pad_id` | the embedding lookup |
| `attention_mask` | $1$ if $t < n_i$, else $0$ | attention, so real tokens never attend to pads |
| `labels` | the response id if $P_i \le t < n_i$; $-100$ otherwise | the loss, which skips $-100$ |

Three rules decide most bugs:

- **Labels are aligned to inputs, not shifted.** Column $t$ of `labels` describes input token $t$. The loss does the
  one-token shift: the logits at column $t-1$ are scored against `labels[t]`. Shifting in the batch builder and again
  in the loss scores every target one position late.
- **The first response token is predicted from the last prompt token.** The response starts at column $P_i$, so
  `labels[P_i]` holds it. The logits that predict it sit at column $P_i - 1$, the last prompt position. That is why a
  prompt needs at least one token.
- **Masks come from lengths, not token values.** A response token can equal `pad_id`. It is still a target and a real token.

## Worked example

Take $(p, r) = ([5, 6], [7, 8, 9])$ and $([4], [3])$, with `pad_id = 0`. Row 0 has $P = 2$, $R = 3$, $n = 5$. Row 1 has
$P = 1$, $R = 1$, $n = 2$. So $L = 5$.

| row | `input_ids` | `labels` | `attention_mask` |
|---|---|---|---|
| 0 | 5 6 7 8 9 | −100 −100 7 8 9 | 1 1 1 1 1 |
| 1 | 4 3 0 0 0 | −100 3 −100 −100 −100 | 1 1 0 0 0 |

Now the shifted loss on row 0. The logits at column 1 (token 6, the last prompt token) are scored against
`labels[2] = 7`. The logits at column 2 are scored against `labels[3] = 8`, and so on. The loss sees exactly the response
tokens 7, 8, 9, each predicted from the token before it. Row 1 works the same way: the logits at column 0 (token 4)
predict 3.

The trap. Take $([1], [0, 2])$ with `pad_id = 0`. The labels are `[-100, 0, 2]`. The response token 0 is kept because it
sits in a response column, not because of its value. A value-based mask (`labels[input_ids == pad_id] = -100`) would drop it.

## Why it works

Maximum likelihood on the response is the SFT objective: the mean negative log-likelihood of the response tokens given
the prompt. Loss on prompt tokens adds terms for text the user wrote, which the model should not generate. The
InstructGPT pipeline starts with this stage, where demonstrators' responses teach the format (Ouyang et al., 2022,
arXiv:2203.02155) [library:AI-ML/12-Frontiers/Ouyang InstructGPT RLHF 2022.md]. The masking rule is an
implementation convention, not a claim from the paper. The shift is the next-token one; see
[next-token targets](../../llm/next-token-targets-and-batches/).

PyTorch's cross-entropy skips targets equal to `ignore_index` ($-100$ by default). Its mean divides by the count of
non-ignored targets, so masked positions neither add loss nor dilute it.

## Signal vs. noise

Signal: which positions are targets, the one-token offset, and that the mask comes from lengths. Noise: the tokenizer, the chat template's special tokens, the optimizer. They change the content of the rows,
not the masking rule.

## Where it breaks

- **No prompt mask.** The model trains on the user's text and learns to write user turns.
- **Off-by-one on the prompt length.** Masking $P - 1$ positions makes the last prompt token a target. Masking $P + 1$
  drops the first response token. A reference comparison catches both.
- **Shifting in two places.** The labels are shifted in the batch and again in the loss. The model learns to predict
  two steps ahead.
- **Multi-turn chats (beyond this problem).** Every assistant span is a target and every user or system span is masked.
  Masking "everything after the first response" leaves later user turns in the loss.
- **Packing (beyond this problem).** Conversations packed into one row attend to each other unless the attention mask is
  block-diagonal. Each keeps its own prompt mask.

## Interview angle

Sixty-second answer: three aligned grids. Ids hold prompt and response; labels equal the ids on response positions and
$-100$ elsewhere; the attention mask comes from lengths. Labels are not shifted: the loss scores the logits at $t-1$
against the label at $t$. Follow-up: why mask the prompt? The objective is the response's likelihood, and loss on user
text spends capacity on text you do not want generated. Second follow-up: several assistant turns or packed
conversations? Each assistant span is a target, every other span is masked, and packed rows need boundaries that
attention and labels both respect.

## Connections

- Prerequisites: [Padding, truncation and pad masks](../padding-truncation-and-pad-masks/) for the two masks, and
  [Negative log-likelihood and perplexity](../negative-log-likelihood-and-perplexity/) for the masked mean loss.
- Next: [LoRA adapters](../lora-adapters/), which changes which weights get the gradient from this loss.
