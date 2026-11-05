---
title: "TTFT, ITL and throughput"
minutes: 12
min_viable: "Read The system and Worked example, then solve p01 by writing one request's three numbers on paper first."
---

## The system

A streamed response has three latencies that answer three different questions. For request $i$ with arrival
time $a_i$, first-token time $f_i$ and token times $t_{i,1} < t_{i,2} < \dots < t_{i,n_i}$, where $t_{i,1} = f_i$:

| Metric | Definition | What it contains |
|---|---|---|
| TTFT | $f_i - a_i$ | queue wait plus prefill, the time before anything is visible |
| ITL, mean | $\dfrac{t_{i,n_i} - t_{i,1}}{n_i - 1}$ | average gap between consecutive tokens, decode speed |
| End-to-end | $t_{i,n_i} - a_i$ | the whole response |

The three are linked by an exact identity. The total of the $n_i - 1$ gaps is $t_{i,n_i} - t_{i,1}$, so

$$
\text{E2E}_i = \text{TTFT}_i + (n_i - 1)\cdot \text{ITL}_i .
$$

This holds for any gap pattern, not only constant gaps. A single-token response has no gaps and no ITL.

**Throughput** counts work per second. Over a window from the first arrival to the last completion, with
$T$ total generated tokens, throughput is $T / \text{window}$ in tokens per second. **Goodput** keeps only the
requests that meet a service-level objective (SLO), such as TTFT at most 0.5 s and mean ITL at most 0.05 s.

## Worked example

Three requests, times in seconds:

| Request | Arrival | Token times | TTFT | Mean ITL | E2E |
|---|---|---|---|---|---|
| A | 0.0 | 0.5, 0.6, 0.8 | 0.5 | $(0.8-0.5)/2 = 0.15$ | 0.8 |
| B | 0.2 | 1.2 | 1.0 | none | 1.0 |
| C | 1.0 | 1.1, 1.3, 1.6, 2.0 | 0.1 | $(2.0-1.1)/3 = 0.3$ | 1.0 |

Check A: $0.5 + 2 	imes 0.15 = 0.8$.

Percentiles use the TTFT list $[0.5, 1.0, 0.1]$, sorted to $[0.1, 0.5, 1.0]$. Under the definition below,
p50 is $0.5$ and p95 is $0.95$. For p95, $m=3$ and $h = 2 \cdot 0.95 = 1.9$, so the value is $x_1 + 0.9\,(x_2 - x_1) = 0.5 + 0.9 \cdot 0.5 = 0.95$, using the zero-based sorted entries $x_1 = 0.5$ and $x_2 = 1.0$. For p50, $h = 1$ and the value is $x_1 = 0.5$. p95 shows the tail that the mean of $0.53$ softens.

The ITL list covers only requests with at least two tokens: A and C, giving $[0.15, 0.3]$. Its p50 is
$0.225$. B contributes nothing, not a zero.

## Why it works

**Percentile definition.** This topic uses linear interpolation between order statistics, which is the default
of NumPy and Hyndman and Fan's type 7. For sorted values $x_0 \le \dots \le x_{m-1}$ and $p \in [0, 100]$:

$$
h = (m-1)\frac{p}{100}, \qquad \text{value} = x_{\lfloor h \rfloor} + (h - \lfloor h \rfloor)\,(x_{\lfloor h \rfloor + 1} - x_{\lfloor h \rfloor}).
$$

For $m = 1$ the value is $x_0$. Nearest-rank is another valid definition, but it gives different numbers on small
samples. State which one you used.

**Why means mislead.** A mean moves with one slow request, but a percentile names the experience of a
fraction of users.  Hyndman and Fan (1996, not in the corpus, so this is a general statement) survey several sample-quantile definitions used by statistical packages, so the exact rule varies; name yours.

## Signal vs. noise

- **Signal:** TTFT and ITL are different systems. TTFT is queue plus prefill; ITL is decode speed.
- **Signal:** report p50 and p95 per metric. A mean of ITL across all gaps weights long answers more.
- **Noise for now:** the exact percentile rule, provided you state it. The conventions differ by a small amount.

## Where it breaks

- **Pooling gaps.** Averaging all gaps across requests lets long responses dominate. Compute ITL per request, then
  take percentiles over requests.
- **Dividing by $n$ instead of $n-1$.** The mean ITL has $n-1$ gaps.
- **E2E from the first token.** E2E starts at arrival, so queueing counts.
- **Throughput as a latency claim.** Batching raises aggregate tokens per second and can raise each request's ITL.
  Illustrative numbers: at batch 1 one stream runs at 40 tokens/s; at batch 8 each stream drops to 15 tokens/s,
  and the aggregate reaches 120 tokens/s. Both numbers are true, and neither answers the other question.
- **Goodput without an SLO.** Goodput needs a stated bound. A request that is fast on average can still miss on TTFT.

## Interview angle

**60-second answer.** Name the metric by the phase it measures: TTFT includes queue and prefill, ITL is decode
speed, and E2E equals TTFT plus $(n-1)$ times ITL. Report p50 and p95 with the percentile method stated.
For capacity, report throughput next to goodput under the SLO, because a system can be fast in aggregate and
still miss most deadlines.

**Likely follow-ups.** Why not the mean? A few slow requests move the mean, and users feel the tail. What does
batching do? It raises throughput and typically raises per-stream ITL, so it is a trade-off to measure, not a free win.

## Connections

This topic needs the descriptive statistics of quantiles and means (`cml/descriptive-statistics`). It leads to
tail latency and goodput, continuous batching, chunked prefill, and the prefill and decode bottlenecks that set
TTFT and ITL. 
