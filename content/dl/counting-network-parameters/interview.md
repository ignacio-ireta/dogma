# Interview notes — Counting network parameters

## Q1. How many parameters does a dense network have?

**60-second answer.** For each distinct dense layer, I count the entries of its input-by-output weight matrix and add one bias per output unit if enabled. Then I sum the layers, including the output layer. I make the sharing assumption explicit: using one layer twice does not create a second set of parameters. Batch size is irrelevant to this count.

**Follow-ups.**

- Do fixed activations add parameters? No; a learned slope would add its own parameter slots.
- How do you audit the total? Return or write the per-layer counts before summing.

## Q2. Can parameter count predict deployment memory?

**60-second answer.** It gives a raw parameter-storage baseline once the storage dtype is fixed. It does not include activations, gradients, optimizer state, temporary buffers or allocator overhead. Inference and training therefore have different memory budgets even for the same stored model. I would first clarify the workload and what memory is being measured.

**Follow-ups.**

- Does a larger batch add parameters? No; it adds workload-dependent activation storage.
- Why avoid allocating matrices to count them? Integer shape arithmetic is exact and works before a model fits in memory.

## Common mistakes

- Omitting the output layer.
- Adding one bias per input feature.
- Treating activation storage or repeated use as new learned storage.
