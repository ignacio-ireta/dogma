---
title: "Autoregressive generation loop"
minutes: 12
min_viable: "Read The system and Worked example, then implement p01 with Hint 1 open."
---

## The system

Let $B$ be the batch size, $P$ the prompt length and $V$ the vocabulary size. A language model maps a token prefix of length $T$ to logits $z\in\mathbb{R}^{V}$ for the next token. Generation is a loop over three phases:

1. **Prefill.** Run the whole prompt, shape $(B,P)$, through the model once. This gives the logits for the first new token.
2. **Decode.** For each step $s=1,2,\dots$: choose the next token from the logits of the last position, append it to each row, and run the model again. Each step adds exactly one token per row.
3. **Stop.** A row stops when it emits the end-of-sequence token EOS. The batch stops when every row has stopped, or when $s$ reaches `max_new_tokens`.

Greedy decoding picks $y_s=\arg\max_v z_v$. The contract in p01 fixes three rules that people often leave implicit:

| Rule | What it says |
|---|---|
| Finished flag | Once a row emits EOS it stays finished. Every later step appends `pad_id` to that row. |
| Batch stop | The loop ends when all rows are finished or `max_new_tokens` new tokens exist. |
| Output width | Output is $(B,\,P+G)$, where $G$ is the number of steps actually run. Shorter rows are padded with `pad_id` on the right. |

The p01 callable receives the whole prefix on each call. A KV cache changes the cost of a step, not the tokens.

## Worked example

Use a toy model whose next token is $(\text{last}+1) \bmod 4$, with EOS $=3$, `pad_id` $=0$, and two prompts of length one: row 0 is $[0]$ and row 1 is $[1]$. Set `max_new_tokens` $=5$.

| step $s$ | row 0 | row 1 | what happens |
|---|---|---|---|
| 1 | $0\to1$ | $1\to2$ | both append |
| 2 | $1\to2$ | $2\to3$ (EOS) | row 1 is finished |
| 3 | $2\to3$ (EOS) | finished, appends $0$ | row 0 is finished; every row is now finished |

The loop stops after step 3, so $G=3$, not 5. The output is

$$
\begin{pmatrix}0&1&2&3\\1&2&3&0\end{pmatrix}.
$$

Row 1 has one trailing pad because it finished one step earlier. With `max_new_tokens`$=2$ the output is $\begin{pmatrix}0&1&2\\1&2&3\end{pmatrix}$, and no row is padded.

## Why it works

A language model factorises the sequence probability as $p(y_{1:S}\mid x)=\prod_{s} p(y_s\mid x,y_{<s})$. The loop evaluates one factor per step. Greedy decoding takes the mode of each factor. That is a local choice, so it does not return the most likely whole sequence. Beam search is the usual fix, and it is outside this topic.

Decoding is memory-bound. A decode step reads every weight once and produces one token per row. A weight of $2$ bytes (fp16) takes $2$ FLOPs per row for the multiply-add, so the arithmetic intensity is about $B$ FLOPs per byte read. Accelerators need on the order of a hundred FLOPs per byte before compute, rather than memory traffic, limits the speed. So throughput grows almost linearly with $B$ until the batch is large. A finished row keeps its slot until the batch ends, the cost of a rectangular tensor.

The loop is valid because the model is causal: the logits at position $t$ depend only on tokens $\le t$ (Vaswani et al. 2017, section 3.2.3), so appending a token never changes earlier logits.

## Signal vs. noise

Signal: the three phases, the finished flag, the padding rule, and the fact that one decode step adds one token per row. Noise: sampling (next topic), chat templates, streaming APIs and beam search.

## Where it breaks

- **Stopping at the first EOS in the batch.** This truncates every other row. Check that a single row's EOS leaves the rest running.
- **Writing argmax tokens into finished rows.** Finished rows must emit `pad_id`. If they keep generating, their output is wrong even though the other rows are right.
- **Counting the prompt toward `max_new_tokens`.** The cap counts new tokens only. The worked example with cap 2 has width $P+2$.
- **Dropping the EOS token.** The caller cannot tell where the answer ended. EOS is part of the output.
- **Feeding only the prompt on every step.** For a bigram toy model the output looks plausible but is wrong. Each call must see the growing prefix.
- **Undefined ties.** Argmax on equal logits must follow a stated rule. This contract uses the lowest token id.

## Interview angle

Sixty-second answer: prefill runs the prompt in one parallel pass and gives the first logits. Each decode step appends one token per row, and each row stops at EOS. The batch stops when all rows are finished or the cap is reached. A decode step is memory-bound because it rereads the weights to produce one token per row, so batching is the lever. Likely follow-ups: why greedy is not the most likely full sequence (it is a local choice per step), and why finished rows are not removed from the batch (the tensor stays rectangular; continuous batching removes them).

## Connections

Prerequisite: [causal masking](../../dl/causal-masking/lesson.md). Next: [temperature, top-k and top-p sampling](../temperature-top-k-and-top-p-sampling/lesson.md) replaces the argmax with a random draw, and [incremental decoding with a KV cache](../incremental-decoding-with-a-kv-cache/lesson.md) removes the prefix recomputation. The lineage is the decoder side of Vaswani et al. (2017).
