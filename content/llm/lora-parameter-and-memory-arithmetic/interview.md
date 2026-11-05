# Interview notes — LoRA parameter and memory arithmetic

## Q1. How many trainable parameters does LoRA add, and what does it cost in memory?
**60-second answer.** Each adapted matrix holds rank times the sum of its input and output widths. A 4096 by 4096
matrix with rank 16 holds 131 thousand numbers, against about 16.8 million for a full update. For a model with
L blocks, multiply by the adapted matrices per block and by L. For memory, the frozen base costs its storage bits.
The adapter costs its weights, one gradient per weight, and 8 bytes of fp32 Adam state per number.
**Follow-ups.** Why is the optimizer the big saving? Its cost scales with the trainable count, and that is small.
Does LoRA shrink the model in memory? Not by itself. The base still has to be stored, and only quantization shrinks it.

## Q2. What does the estimate leave out?
**60-second answer.** Activations, the framework overhead, and fragmentation. Activations depend on batch size and
sequence length, and for long sequences they can exceed the whole adapter term. Checkpointing trades them for
recomputation.
**Follow-ups.** How would grouped-query attention change the count? Its k and v matrices are smaller, so their terms
use the key and value width rather than the model width. What is the fraction relative to? Say which base you mean. In the worked example it is 0.57% of a
7e9-number base, and 0.62% of full fine-tuning of the same matrices.

## Common mistakes
- Counting d_in times d_out, which is full fine-tuning.
- Forgetting the factor L, or applying it once per target.
- Treating bits as bytes, or rounding the total instead of each tensor.
