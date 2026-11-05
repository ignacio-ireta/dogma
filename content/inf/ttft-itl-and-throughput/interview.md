# Interview notes — TTFT, ITL and throughput

## Q1. A chat product feels slow. How do you measure it?
**60-second answer.** Split the latency into the phases the user waits through. TTFT is arrival to first
token, covering queue and prefill. ITL is the gap between tokens, which is decode speed. End-to-end is TTFT plus
the gaps, and it matters for long answers. I would report p50 and p95 for each, with the percentile method named,
and check which phase moved. Slow TTFT points at queueing or prefill. Slow ITL points at decode and batching.
**Follow-ups.** Why not the mean? One slow request can move it while most users are fine, and the tail is
what users remember. Which percentile? p95 is a common SLO target; p99 needs many more samples to be stable.

## Q2. Capacity planning: how many requests per second can one replica serve?
**60-second answer.** Report throughput in tokens per second over a window, and goodput under the SLO, which
counts only requests that meet the TTFT and ITL targets. Throughput can look healthy while goodput collapses,
because the batch got large enough to push some requests past the SLO. Plan around goodput.
**Follow-ups.** What defines the window? From the first arrival to the last completion, so idle time counts
against throughput. What about a single-token response? It has no ITL, so it can only fail on TTFT.

## Common mistakes
- Averaging ITL over all gaps across requests, which lets long responses dominate.
- Dividing the ITL sum by n instead of n - 1.
- Measuring end-to-end latency from the first token, which hides queueing.
- Quoting a throughput number with no latency SLO beside it.
