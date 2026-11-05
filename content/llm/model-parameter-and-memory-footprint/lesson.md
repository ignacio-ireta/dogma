---
title: "Model parameter and memory footprint"
minutes: 12
min_viable: "Read The system and the worked example, then do p01 with the component list in front of you."
---

## The system

Two questions: how many numbers does the model hold, and how many bytes does each take, in which mode?
The count does not change with dtype or optimizer. Symbols: $V$ vocabulary, $D$ width, $N$ layers, $F$ MLP
width, $C$ context length (position rows, if learned), $H$ heads (only has to divide $D$).

**Parameter count, by component.** Each block has four $(D,D)$ attention projections, and an MLP. The
MLP is either gelu, with matrices $(D,F)$ and $(F,D)$, or swiglu, with three matrices. Each block has two
norms. The model also has a token table, an optional position table, a final norm and an optional head.

| Component | Count | Present when |
|---|---|---|
| Token table | $VD$ | always |
| Position table | $CD$ | learned positions |
| Attention, per block | $4D^2$ (+ $4D$ with bias) | always |
| MLP, per block | $2DF$ gelu, $3DF$ swiglu (+ biases) | always |
| Norms, per block | $2\cdot 2D$ layernorm, $2\cdot D$ rmsnorm | always |
| Final norm | $2D$ or $D$ | always |
| LM head | $VD$ | untied only |

One gelu block with biases and layernorm has $4D^2+2DF+9D+F$ parameters, the block formula from
`dl/transformer-block`. The $N$ blocks contribute $N$ times that.

**Bytes per parameter.** The dtype sets the size of one number: fp32 4, fp16 and bf16 2, fp8 and int8 1.

**Inference.** Memory is the weights, $N_\text{params}\cdot b$, plus activations and the KV cache.

**Training, the rule of thumb.** With mixed precision and Adam, a parameter costs:

| Term | Bytes per parameter | Why |
|---|---|---|
| Weights (working dtype, bf16 or fp16) | 2 | used in the forward and backward passes |
| Gradients (working dtype) | 2 | one per weight, same dtype |
| fp32 master copy | 4 | the optimizer keeps full-precision weights |
| Adam first and second moments, fp32 | 8 | two fp32 states of 4 bytes each |
| **Total** | **16** | |

The same 16 bytes come out of fp32 Adam with no master copy: $4 + 4 + 8$.

The 16 bytes assume a dense model with every parameter trainable, bf16 or fp16 weights and gradients, an
fp32 master copy, two fp32 Adam states, and no sharding. Activations, buffers and fragmentation are excluded.

## Worked example

Take the GPT-2 small configuration: $V=50{,}257$, $D=768$, $N=12$, $F=3072$, $C=1024$, learned positions,
biases, gelu, layernorm and a tied head.

| Component | Count |
|---|---|
| Token table $50{,}257\cdot768$ | 38,597,376 |
| Position table $1024\cdot768$ | 786,432 |
| Attention, 12 blocks: $12\cdot(4\cdot768^2+4\cdot768)$ | 28,348,416 |
| MLP, 12 blocks: $12\cdot(2\cdot768\cdot3072+3072+768)$ | 56,669,184 |
| Norms, 12 blocks: $12\cdot 2\cdot 1536$ | 36,864 |
| Final norm | 1,536 |
| LM head (tied) | 0 |
| **Total** | **124,439,808** |

Now the bytes. In fp32 the weights take $124{,}439{,}808\cdot4 = 497{,}759{,}232$ bytes, about 0.50 GB. Training
with Adam at 16 bytes per parameter takes $124{,}439{,}808\cdot16 = 1{,}991{,}036{,}928$ bytes, about 2.0 GB,
before activations.

A LLaMA-7B-style configuration (no biases, swiglu, rmsnorm, untied head) has 6,738,415,616 parameters. In
bf16 its weights take about 13.5 GB. At 16 bytes per parameter, training takes about 108 GB before activations.

## Why it works

Each component is a sum of tensor sizes, so the count is exact once you know which tensors exist. The
mixed-precision rule is a sum too: one term per tensor that must exist during training. The fp32 master copy keeps small updates that a 16-bit weight would round away; Micikevicius et al. (2018)
describe this mixed-precision recipe [paper: Micikevicius et al. 2018, arXiv:1710.03740]. Adam keeps two moving averages per parameter, the first
and second moments [corpus: library-papers/1412.6980.pdf.md, section 2], so it stores two fp32 states.

**The 117M label.** The GPT-2 paper labels this model 117M, with 12 layers and width 768
[corpus: library-papers/radford-gpt2-2019.pdf.md]. Counted term by term above, the released configuration
gives 124,439,808. I could not reproduce 117M from the stated architecture, so the two do not reconcile.
Use 124,439,808 for exact counts, and read 117M as the paper's label.

## Signal vs. noise

Signal: the component list, the per-block formula, the bytes per dtype and the named 16-byte breakdown.
Noise: the head count, and the learning rate, which changes nothing in memory.

## Where it breaks

- **Counting the tied head twice.** The head reuses the token table. Adding $VD$ again overstates the count
  by one full token table.
- **Assuming the config fits the table.** Grouped-query attention shares key and value heads, so its
  attention projections are smaller. This count assumes full-width projections.
- **Using 16 bytes as a ceiling.** Sharded optimizers (ZeRO-style) and 8-bit optimizers shrink the states.
  The rule describes one unsharded replica.
- **Forgetting activations.** At long context they can exceed the state term.

## Interview angle

Answer in two steps: the count (token table, blocks, head), then the bytes (2 weights, 2 gradients, 4 master,
8 Adam states, so 16 per parameter). Name what the rule leaves out: activations, buffers, sharding. Follow-ups:
a tied head adds nothing; inference is weights plus KV cache and activations.

## Connections

Builds on [`dl/counting-network-parameters`](../../dl/counting-network-parameters/lesson.md),
[`dl/transformer-block`](../../dl/transformer-block/lesson.md) and [`dl/adam`](../../dl/adam/lesson.md).
The same components appear as tensors in [`llm/gpt-forward-pass`](../gpt-forward-pass/lesson.md).
