# Interview notes — Arithmetic intensity and the roofline

## Q1. Is batch-1 LLM decode compute-bound or memory-bound? Show the numbers.
**60-second answer.** Decode multiplies one token's activation vector by each weight matrix. A product of
shape 1 by $k$ by $n$ costs $2kn$ FLOPs and reads about $kn$ elements, so intensity is about $2/b$ FLOP per
byte, which is 1 for fp16. A modern accelerator has a ridge point in the low hundreds of FLOP per byte. Decode
is therefore memory-bound by roughly two orders of magnitude, and its speed is set by how fast the weights stream
from memory. Name the units on every number.
**Follow-ups.** What raises the intensity? Batching: with $m$ sequences the intensity is about $2m/b$ until it
reaches the ridge. What does quantising the weights to fp8 do? It halves the bytes and doubles the intensity, but
the product stays memory-bound at practical batch sizes.

## Q2. How do you decide whether an operation is compute-bound or memory-bound?
**60-second answer.** Count the FLOPs and the bytes the operation must move, then compute intensity as the
ratio. Compare it with the ridge point, peak FLOP/s over bandwidth. Below the ridge, the time is bytes over
bandwidth. Above it, the time is FLOPs over peak. The time floor is the larger of the two.
**Follow-ups.** Why is the time a lower bound? Launch overhead, imperfect overlap and cache misses add time. Why
use the dtype's peak? A tensor-core fp16 peak and an fp32 peak are different ridge points, so the verdict depends
on the dtype.

## Common mistakes
- Counting multiply-accumulate pairs as one FLOP. A multiply and an add are two FLOPs.
- Leaving out the output matrix from the bytes.
- Using measured run time as if it were intensity. Intensity is a property of the operation.
- Comparing intensity with the ridge in mixed units, such as TB/s against TFLOP/s without converting.
- Quoting a peak for one dtype and bandwidth for another.
