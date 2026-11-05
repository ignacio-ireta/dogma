# Interview notes — Prefill vs decode bottlenecks

## Q1. What is the bottleneck of prefill, and what is the bottleneck of decode?
**60-second answer.** Prefill processes the whole prompt in one parallel pass. The weights are read once and used for every
prompt token, so the intensity grows with the prompt length, and a long prompt is compute-bound. Decode makes one token per
sequence per step. It reads the weights and each sequence's cache every step, so its intensity is about $2B/b$ at small
batch and it is memory-bound. Batching shares the weight read across sequences, which is the lever that moves decode.
**Follow-ups.** Why is a one-token prefill memory-bound? The FLOPs are tiny, and the weight read dominates the bytes. Which
term limits decode at long context? The cache: it is not shared, so with many KV heads it can keep decode memory-bound at
every batch size.

## Q2. How would you estimate the decode throughput of a model on one device?
**60-second answer.** Compute the step's bytes: the weights, plus the cache of every sequence in the batch. Compute its
FLOPs: $2P$ per sequence. Take the larger of bytes over bandwidth and FLOPs over peak. Throughput is the batch divided by
that step time. State the units, and say the estimate is a floor, because real kernels reach a fraction of peak.
**Follow-ups.** What changes the answer most? The batch size, until the cache traffic or memory capacity limits it, and
the KV-head count, which sets the cache per sequence. How would you check it? Measure the step time at two batch sizes and
compare with the model.

## Common mistakes
- Reading the weights once per sequence instead of once per step.
- Treating the prefill time as TTFT and leaving out queue wait.
- Quoting a decode throughput without the batch size.
- Ignoring the KV term at long context, where it can dominate the weight read.
- Assuming a batch can grow without a memory check.
