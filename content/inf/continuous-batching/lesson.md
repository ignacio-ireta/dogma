---
title: "Continuous batching"
minutes: 12
min_viable: "Read The system and Worked example, then solve p01 by tracing the worked example on paper."
---

## The system

A server runs one forward pass per **step**. Every running request produces exactly one token per
step. The scheduler decides which requests are in the batch. Two rules matter.

- **Static batching.** A batch is formed only when no request is running. It runs until its last
  request finishes. Freed slots stay idle until then.
- **Continuous (iteration-level) batching.** The scheduler re-decides at every step. A request is
  admitted in the step after a slot frees.

Both policies use the same limits and the same FCFS queue. Let $M$ be the maximum batch size (the
number of running requests) and $P$ the prefill token budget per step (the total prompt tokens
admitted in one step). Admission stops at the first queue head that breaks either limit. A later,
smaller request does not jump ahead.

Step arithmetic, for a request admitted at step $s$ with $O$ output tokens:

$$
\text{first token step} = s, \qquad \text{finish step} = s + O - 1.
$$

The prefill of step $s$ produces the first token, and each later step produces one more.

## Worked example

Four requests, all with prompt length 4, $M = 2$, and $P$ large enough not to bind. Arrivals are
$[0, 0, 1, 2]$ and outputs are $[3, 1, 2, 2]$.

| Step | Static batch | Continuous batch |
|---|---|---|
| 0 | start R0, R1. R1 finishes | start R0, R1. R1 finishes |
| 1 | R0 running, R2 waits | R0 running, **start R2** (slot freed) |
| 2 | R0 finishes, R2 waits | R0 and R2 finish |
| 3 | batch empty, **start R2, R3** | **start R3** |
| 4 | R2 and R3 finish | R3 finishes |

Results as (first, finish): static $[(0,2),(0,0),(3,4),(3,4)]$, continuous
$[(0,2),(0,0),(1,2),(3,4)]$. R2 gets its first token at step 1 instead of step 3.

## Why it works

Throughput is the number of tokens produced per step, which equals the number of running requests.
A static batch loses slots as its members finish. With outputs $[9, 1]$ and $M = 2$, the batch runs
9 steps and produces 10 tokens, out of 18 slot-steps: utilisation $10/18 \approx 56\%$. Continuous
batching fills the freed slot from the queue, so utilisation stays near $M$ while work is waiting.

Queueing delay is the other side. A request waiting for a static batch to drain waits for the longest
member, not for a free slot. Continuous batching shortens that wait, so the first-token step moves
earlier. That is the TTFT part of the trade-off in [TTFT, ITL and throughput](../ttft-itl-and-throughput/).

## Signal vs. noise

- **Signal:** the two admission rules and the timing rule. A slot freed in step $t$ is usable in
  step $t+1$.
- **Signal:** FCFS with no skipping. It makes head-of-line blocking visible.
- **Noise for this session:** wall-clock time per step, GPU kernels and scheduler overhead. The step
  model counts steps, not milliseconds.

## Where it breaks

- **Head-of-line blocking.** A large prompt at the head blocks smaller ones behind it, even when they
  would fit. The budget makes this visible in the p01 example.
- **Static batches with mixed lengths.** A batch of one 500-token output and seven 10-token outputs
  leaves seven slots idle for most of its life.
- **Prefill interference.** A big prefill in a step delays the decode tokens of running requests in
  that step. The step model ignores this; real systems bound prefill with a token budget.
- **Memory.** A slot is only useful if the KV cache can hold the request. Admission must also check
  blocks, which [paged KV cache](../paged-kv-cache/) manages.

## Interview angle

**60-second answer.** Static batching admits a batch only when it is empty, so every slot freed
early sits idle until the longest request ends. Continuous batching makes the admission decision at
every step, so a freed slot is refilled on the next step. Utilisation rises, and queued requests start
sooner. The cost is per-step scheduling work and memory bookkeeping. I would check the prefill budget
too, because one long prompt can stall decode for everyone.

**Follow-ups.** *Why does FCFS hurt here?* It preserves order and fairness, but a large head request
blocks the queue. *Does continuous batching change the output?* No. It changes when each request runs,
not what it computes.

## Connections

This topic uses the FCFS queue from [stacks and queues](../../algo/stacks-and-queues/). Its metrics,
TTFT and inter-token latency, come from [TTFT, ITL and throughput](../ttft-itl-and-throughput/). The
memory that makes a new slot usable is managed by [paged KV cache](../paged-kv-cache/). The practical
systems papers are listed in the topic sources.
