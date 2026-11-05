# Interview notes — Model parameter and memory footprint

## Q1. How many parameters does a decoder-only transformer have, and where do they live?

**60-second answer.** Count by component. The token table is $VD$. A learned position table is $CD$, and a
rotary model has none. Each block has four $(D,D)$ attention projections, an MLP with $2DF$ weights for gelu
or $3DF$ for swiglu, and two norms. The final norm adds one more norm. An untied head adds $VD$, and a tied
head adds nothing. For GPT-2 small this gives 124,439,808 parameters, and the blocks hold most of them.

**Follow-ups.** How does the count change with heads? It does not, since heads only split $D$. What does a
bias add? A $(D,)$ vector per projection, which is small next to the $D^2$ matrices.

## Q2. How much memory does training need per parameter?

**60-second answer.** With mixed precision and Adam, about 16 bytes per parameter. Weights in bf16 take 2,
gradients in bf16 take 2, the fp32 master copy takes 4, and the two Adam moments in fp32 take 8. The same 16
comes from fp32 Adam with no master copy. I would state those assumptions first, then add activations, which
depend on batch and sequence length and are not in the rule.

**Follow-ups.** Why keep a master copy? Small updates are lost when weights are rounded to bf16. Does SGD
without momentum save memory? Yes, it stores no optimizer state, so fp32 SGD costs 8 bytes per parameter: 4 for weights and 4 for gradients.

## Q3. How much memory does inference need?

**60-second answer.** The weights, $N\cdot b$, where $b$ is the bytes per number. For a 7-billion-parameter model
in bf16 that is about 14 GB. Activations are small at batch size one, but the KV cache grows with the context
and the batch, so it is the term to budget for. Quantising the weights to int8 halves the weight term again.

**Follow-ups.** Why does training need so much more than inference? It keeps gradients and optimizer states,
which are several times the weights. Can sharding help? Yes, optimizer state can be split across devices, which
changes the per-device count but not the total.

## Common mistakes

- Counting the tied head as an extra $VD$.
- Using the working dtype for the Adam states. They are fp32.
- Quoting 16 bytes per parameter as the total, without activations.
- Forgetting the bias vectors and the two norms per block when checking GPT-2's count.
