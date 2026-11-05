# Interview notes — Floating-point formats

## Q1. Why does fp16 training need loss scaling, while bf16 usually does not?
**60-second answer.** Both formats store a sign, an exponent and a mantissa. fp16 has five exponent bits, so its range tops out at 65504 and its smallest subnormal is about $6\times10^{-8}$. Small gradients round to zero. bf16 has the fp32 eight-bit exponent, so it keeps fp32's range and small gradients survive. Each bf16 rounding costs at most about $0.4\%$ of relative error, against about $0.05\%$ for fp16, so the trade is precision against range, and bf16 buys range.
**Follow-ups.** 
- *Does bf16 still need fp32 master weights?* Yes. An update of $10^{-4}$ to a weight near $1$ is smaller than bf16's spacing, so the update is lost. Keep an fp32 copy for the optimizer.
- *What about fp8?* E4M3 has three mantissa bits and a maximum of 448, so it needs per-tensor or per-block scales to stay in range.

## Q2. Why can rounding break a softmax even when no value overflows?
**60-second answer.** The softmax is exponential, so a small change in an input produces a large change in its output. In bf16, logits $128$ and $128.5$ both round to $128$, and the softmax moves by about $36\%$ in the ST-MoE example. Subtracting the maximum helps with overflow but not with this sensitivity. The fix is to keep the softmax and the logit reductions in fp32, even when matrix multiplies run in bf16.
**Follow-ups.**
- *Where would you check?* Compare the fp32 and low-precision outputs on the largest logits.
- *Is this only bf16?* No. fp16 has 10 mantissa bits and is more precise, but the effect appears at large magnitudes in every format, and fp16 overflows sooner.

## Q3. How do you check that a number is representable in a format?
**60-second answer.** Round it to the format and compare. The format is defined by its exponent width, mantissa width and bias, so every representable value is a point on a grid. Machine epsilon $2^{-m}$ is the spacing at $1$. The largest value comes from the reserved all-ones exponent. A rounding test should check ties to even, the subnormal spacing and the overflow threshold, because those are where implementations differ.

## Common mistakes
- Adding an implicit leading 1 to subnormal numbers.
- Treating bf16 as fp16 with a larger range. The mantissa is only 7 bits.
- Quoting the largest fp16 value as $2^{16}$. It is $65504$.
- Quoting machine epsilon as the rounding error. The unit roundoff, the largest relative error after rounding to nearest, is half of epsilon.
- Forgetting that E4M3 has no infinity.
