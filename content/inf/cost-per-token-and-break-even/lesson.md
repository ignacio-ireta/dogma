---
title: "Cost per token and break-even"
minutes: 12
min_viable: "Read The system and Worked example, then solve p01 by writing the daily cost and the tokens per day before coding."
---

## The system

A self-hosted replica costs money every hour, whether or not it serves anything. Let $c$ be the price of one GPU-hour in
USD, $G$ the GPUs in the replica, $\tau$ the sustained output tokens per second at full load, and $u$ the utilization,
the fraction of time the replica serves at that rate.

| Quantity | Formula | Unit |
|---|---|---|
| Daily cost | $F = 24\,c\,G$ | USD per day |
| Tokens served per day | $S = 86{,}400\,\tau\,u$ | tokens per day |
| Self-hosted cost per million tokens | $F \cdot 10^6 / S$ | USD per million |
| Break-even volume | $V^* = F \cdot 10^6 / p$ | tokens per day |

Here $p$ is the API price in USD per million output tokens. The cost per token is a fixed bill divided by the tokens it
covers, so utilization sits in the denominator. The break-even volume $V^*$ is where the self-hosted bill equals the API bill; above it, self-hosting is cheaper. It does not depend on $u$. Utilization decides whether the
replica can reach $V^*$ at all.

**Input and output.** Providers often price input tokens below output tokens. Output tokens come one per decode step, the
memory-bound phase. Input tokens go through the parallel prefill pass, at a much higher tokens per second (see the prefill
topic). Price each side with its own $\tau$.

## Worked example

One GPU at $c = 2.50$ USD per hour, so $F = 60.00$ USD per day, with $\tau = 1000$ tokens per second.

| Utilization $u$ | Tokens per day | Cost per million | Rounded to the cent |
|---|---|---|---|
| 100 % | 86.4 M | $60 \cdot 10^6 / 86.4\text{M} = 0.694$ | 0.69 |
| 50 % | 43.2 M | $1.389$ | 1.39 |
| 25 % | 21.6 M | $2.778$ | 2.78 |

Against an API at $p = 0.60$ per million, $V^* = 60 \cdot 10^6 / 0.60 = 100$ M tokens per day. That is $1157$ tokens per
second on average, more than this replica can serve even at full load. Self-hosting loses at every utilization.

Change one number. At $\tau = 2000$ the full-load cost is $0.347$, and the break-even utilization is
$u^* = V^* / (86{,}400\,\tau) = 57.9$ percent. The replica must be busy more than that to beat the API.

## Why it works

**Cost per token is a division.** Write the self-hosted cost as $c\,G / (3600\,\tau\,u)$ per token, the hourly bill over the
tokens per hour. Halving $u$ doubles the cost, so an idle replica is expensive per token even when it is cheap per hour.

**Break-even is an intersection.** The API bill grows linearly with volume, $p\,V/10^6$. The self-hosted bill is flat at
$F$ per day as long as the replica keeps up. The two lines cross at $V^* = F \cdot 10^6 / p$. Below it the API is
cheaper; above it the replica is cheaper, if the capacity $86{,}400\,\tau$ reaches that volume.

## Signal vs. noise

- **Signal:** utilization in the denominator, and the daily fixed bill. Those two decide most of the answer.
- **Signal:** $\tau$ must be the sustained rate at the batch and context you run, not a benchmark peak.

## Where it breaks

- **Peak throughput as sustained.** A benchmark at one batch and short context overstates $\tau$, and the cost estimate
  falls with it.
- **Forgetting idle hours.** A replica that is quiet overnight still bills for those hours. Count them in $u$.
- **Mixing input and output prices.** Output is usually the expensive side of self-hosting. Do not compare an input
  price with an output cost.
- **Per-GPU price for a per-replica need.** Multiply by the GPUs of one replica, which the capacity topic sets.
- **Rounding early.** Round only the reported cost; intermediate rounding can move a cent.
- **No latency cost.** Headroom for bursts and latency targets means running below full load, which raises cost per
  token. The break-even figure assumes none.
- **Ignoring people and operations.** Engineering time and standby capacity are real costs left out here, so the model
  sets a floor on the self-hosted cost.

## Interview angle

**60-second answer.** Cost per million tokens equals the hourly bill divided by tokens per hour, times a million. Utilization
sits in the denominator, so an idle replica is expensive per token. The break-even volume is the daily fixed bill divided by
the API price per token. Self-hosting wins above that volume, and only if the replica can reach it at the utilization the
traffic really gives. Use the sustained throughput at the batch you serve, not the peak.

**Likely follow-ups.** What moves the answer most? Utilization and sustained throughput, usually more than the GPU price.
Why are input tokens cheaper to serve? Their prefill pass is parallel and compute-bound, so each token costs less time. How
would you cut the cost? Raise the batch (more throughput per GPU), scale the replicas to the traffic (higher utilization),
or quantise to fit fewer GPUs.

## Connections

This topic needs the throughput and latency definitions (`inf/ttft-itl-and-throughput`), where the tokens per second and
their percentiles are defined. It uses the decode and prefill rates from the prefill topic, and the replica size from the
GPU capacity topic. Serving systems schedule batches to keep GPUs busy for this reason.
