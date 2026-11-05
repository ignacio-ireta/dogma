---
title: Next-token targets and batches
minutes: 12
min_viable: "Read The system and Worked example, then implement p01 with Hint 1 open."
---

## The system

A causal language model is trained to predict each token from the tokens before it. For a token stream $S = (s_1, \dots, s_N)$ of $N$ token ids, the training pairs are

$$x_t = s_t \qquad \text{and} \qquad y_t = s_{t+1}, \qquad t = 1, \dots, N-1.$$

So the **targets are the inputs shifted left by one**. The shift is the whole objective. A model that receives its own input as the target learns nothing about what comes next.

Build fixed-length blocks from the stream with three rules:

1. **Separate documents with EOS.** Each document $d$ (a list of token ids) contributes its tokens followed by one end-of-sequence id. The stream is $d_1 \,\|\, [\text{EOS}] \,\|\, d_2 \,\|\, [\text{EOS}] \,\|\, \cdots$. The EOS teaches the model where a document ends.
2. **Window of $T+1$, stride $T$.** Block $b$ (from 0) covers $S_{bT+1}, \dots, S_{bT+T+1}$ in one-based indexing. Its inputs are the first $T$ tokens of that window, and its targets are the last $T$. Consecutive windows share one token, so every token is a target exactly once except the first.
3. **Drop the remainder.** With $N$ tokens, the number of full blocks is $B = \lfloor (N-1)/T \rfloor$. The last $N-1-BT$ tokens do not fill a window. Drop them. Do not pad them here: a padded block adds pad targets, and the loss then needs a mask.

Shapes: `inputs` and `targets` are `int64` tensors of shape $(B, T)$.

## Worked example

Two documents, `[[1, 2], [3]]`, EOS id $0$. The stream is $S = (1, 2, 0, 3, 0)$, so $N = 5$. Block size $T = 2$ gives $B = \lfloor 4/2 \rfloor = 2$ blocks.

| block | window | inputs $x$ | targets $y$ |
|---|---|---|---|
| 0 | $S_1..S_3 = (1,2,0)$ | $(1,2)$ | $(2,0)$ |
| 1 | $S_3..S_5 = (0,3,0)$ | $(0,3)$ | $(3,0)$ |

So `inputs = [[1, 2], [0, 3]]` and `targets = [[2, 0], [3, 0]]`. The token $0$ at position $S_3$ is the target of block 0 and the input of block 1. That shared token is the overlap.

Remainder check: with $T = 3$, $B = \lfloor 4/3 \rfloor = 1$. The window is $(1, 2, 0, 3)$ and the last token $0$ is dropped.

## Why it works

Language modelling factorizes a sequence probability into conditionals. Bengio et al. (2003, p. 2) write the probability of $w_1^T$ as the product $\prod_{t=1}^{T} \hat P(w_t \mid w_1^{t-1})$. Training maximizes the log of that product, so each term needs exactly one (context, next token) pair. The shift produces those pairs. The EOS makes the stream a single sequence even when it holds many documents. Windows of $T+1$ with stride $T$ make the targets of consecutive blocks tile the stream, so every token after the first is a target exactly once.

## Signal vs. noise

Signal: the shift by one, the EOS between documents, and the rule for the remainder. Noise: whether the loader keeps blocks in order, shuffles them or uses multiple workers. Those choices change the batch order, not the pairs. State the remainder rule, because a silent change in it changes the number of training tokens.

## Where it breaks

- **Shift applied twice, or not at all.** Targets equal to inputs train the model on an identity map. The loss can look low while the model learns nothing about the next token.
- **No EOS between documents.** The documents run together. The model never learns that a document ends, and it continues a finished text.
- **Packing without masks.** A packed block holds several documents, and plain causal attention lets a token in the second document attend to the first. The EOS is the only boundary signal. Document masks (the next topic in this track) stop that.
- **Padding a short tail with zeros and no loss mask.** Pad ids become targets, and the loss averages over them. Targets for pad positions must be $-100$, so the loss ignores them.

## Interview angle

Sixty-second answer: the targets are the inputs shifted left by one, so a window of $T+1$ tokens with stride $T$ gives an input block and a target block that overlap by one token. Documents are joined with an EOS so the model sees where each one ends. The leftover tokens are dropped, or padded with a loss mask, and the choice is stated. Packing is cheap, but without a document mask, attention crosses document boundaries. Likely follow-up: why not pad each document to $T$? That wastes compute on pads, and packing uses every token. Second follow-up: how many training tokens does a block set contain? $B \cdot T$ inputs and $B \cdot T$ targets, with $N - 1 - BT$ tokens dropped.

## Connections

- Prerequisite: [Padding, truncation and pad masks](../padding-truncation-and-pad-masks/). The loss-mask rule for padded tails comes from there.
- Next: negative log-likelihood, which averages the loss over the non-ignored targets. In this track, document packing and cross-document masks come after this topic.
- Lineage: the factorization is from Bengio et al. (2003). Packed training with document masks is the next topic.
