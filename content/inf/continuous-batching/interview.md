# Interview notes: Continuous batching

## Q1. Why does continuous batching beat static batching for generative serving?

**60-second answer.** Static batching admits a batch only when the batch is empty, so every slot freed early sits idle until the longest request finishes. Output lengths vary a lot, so that waste is large. Continuous batching decides at every step. A slot freed at the end of one step is refilled at the next, so utilisation stays close to the batch limit while work is waiting, and queued requests start sooner. The cost is per-step scheduling and memory bookkeeping.

**Follow-ups.** *What do you measure to prove it?* Utilisation as tokens per slot-step, and first-token latency under load. *Does it change the outputs?* No; it changes when requests run, not what they compute.

## Q2. What can still go wrong once you admit requests every step?

**60-second answer.** Three things. A large prompt at the head of a FCFS queue blocks smaller requests behind it, so the scheduler needs a prefill token budget. A big prefill also slows the decode tokens of requests already running in that step. And admission must check KV-cache memory, not only slots; a slot without blocks is useless. I would bound prefill per step and tie admission to the block allocator.

**Follow-ups.** *Why not skip ahead to small requests?* That breaks FCFS fairness and can starve large requests. *What would you tune first?* The maximum batch and the prefill budget, against the TTFT and ITL targets.

## Common mistakes

- Counting the admission step as a separate step. The prefill step produces the first token, so the finish step is first + output - 1.
- Letting a freed slot wait a full step, or a full batch, before reuse.
- Skipping the blocked head of the queue when a smaller request would fit.
- Treating a fixed batch size as a throughput guarantee when lengths vary.
