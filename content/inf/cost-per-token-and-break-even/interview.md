# Interview notes — Cost per token and break-even

## Q1. What does it cost to serve a million output tokens on our own GPUs?
**60-second answer.** Take the daily bill of the replica: GPU-hour price times GPUs times 24. Divide by the tokens it serves
per day, which is the sustained tokens per second times 86,400 times the utilization. Multiply by a million. Utilization
sits in the denominator, so an idle replica is expensive per token. State the units and the rounding rule.
**Follow-ups.** Why not use the peak tokens per second? A benchmark at one batch overstates what the traffic gets. Why do
output tokens cost more than input tokens? Output tokens are decode steps, which are memory-bound, while input tokens go
through a parallel prefill pass.

## Q2. At what volume does self-hosting beat an API?
**60-second answer.** The break-even volume is the daily bill times a million, divided by the API price per million tokens.
Above it, self-hosting is cheaper, provided the replica can serve that volume. Check the utilization that volume implies,
since a replica that is idle for part of the day raises the real cost.
**Follow-ups.** What moves the break-even most? The throughput, which sets how much of the volume one GPU can take, and the
API price. What else would you count? Engineering time, standby capacity and the latency headroom the service needs.

## Common mistakes
- Using peak throughput instead of the sustained rate at the batch served.
- Ignoring idle hours in utilization.
- Comparing an input price with an output cost.
- Pricing a replica by one GPU when it needs several.
- Rounding intermediate values, so the reported cent is off.
