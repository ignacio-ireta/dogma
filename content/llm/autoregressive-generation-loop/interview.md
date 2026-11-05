# Interview notes: Autoregressive generation loop

## Q1. Walk me through greedy generation for a batch of prompts.

**60-second answer.** Run the prompts once as a prefill pass, which gives the logits for the first new token of every row. For each step, pick the argmax of each unfinished row's last-position logits, append it, and run the model again on the grown prefix. A row that emits EOS is finished, and it appends pad tokens from then on. The loop stops when every row is finished or when the step cap is reached. Output is rectangular, with the prompt, the generated tokens and padding on the right.

**Follow-ups.**

- Is greedy the most likely output? No. It chooses the most likely token at each step, which is a local choice. Beam search keeps several candidate prefixes to approximate the best full sequence.
- Why not remove finished rows from the batch? It keeps the tensor rectangular and simple. Continuous batching schedulers remove finished rows and admit new requests, which is a systems change.

## Q2. Why is token-by-token decoding slow, and what does batching change?

**60-second answer.** Each decode step reads all the weights from memory to produce one token per row. The arithmetic per byte read is about the batch size, so a small batch leaves the compute units idle. Batching multiplies the work done per weight read, so throughput rises almost linearly until the step becomes compute-bound. The cost of batching is that a finished row waits for the slowest row in its batch.

**Follow-ups.**

- What does the KV cache change? It stops the model from reprocessing the prefix, so each step touches one new token. It adds memory that grows with the context.
- Why does an EOS in one row not stop the others? Each row has its own finished flag, and the model is row-independent, so the loop only stops once every flag is set.

## Common mistakes

- Stopping the whole batch at the first EOS.
- Letting finished rows keep generating instead of appending `pad_id`.
- Counting the prompt toward `max_new_tokens`.
- Calling the model with the prompt only, so the generated tokens are never seen.
