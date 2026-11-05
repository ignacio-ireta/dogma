# Interview notes: Absmax and zero-point int8 quantization

## Q1. How would you quantize int8 weights, and what error can a reader expect?

**60-second answer.** I would use symmetric per-output-channel scales, $s_c = A_c / 127$, where $A_c$ is the absmax of channel $c$. Rounding is half to even, and codes are clipped to $[-127, 127]$. Each element is then within $s_c / 2$ of its original value, and zero stays exact. I would state the rule explicitly, because half-away rounding biases ties in a different direction. The scale is the only place the outlier problem enters, so the cost is easy to check.

**Follow-ups.** *Why per channel rather than per tensor?* A per-tensor scale is set by the largest channel, so small channels lose most of their codes. *Why not keep the scale in fp32?* You can, and it is cheap for the scale itself; the savings come from the codes.

## Q2. When would you use an asymmetric zero point, and what can go wrong?

**60-second answer.** Asymmetric quantization pays off when the data is one-sided, such as activations after a ReLU. The range is $[lo, hi]$ with $lo \le 0 \le hi$, so 256 codes cover the data instead of 127 per side. The zero point is an integer, so zero is exact, which matters for padding. The usual mistakes are leaving zero out of the range, which breaks exactness for all-positive data, and forgetting that dequantization subtracts the zero point.

**Follow-ups.** *Why does the range have to contain zero?* Otherwise $0.0$ falls between two codes, and a zero-padded value no longer round-trips. *What does the zero point cost at runtime?* An integer subtraction per element or a correction term in a matrix product, which is why symmetric weights are common.

## Common mistakes

- Computing codes in int8 arithmetic, so $127 + 1$ wraps to $-128$. Do the arithmetic in a wider type, then clip.
- Dividing by an all-zero absmax. Use scale 1.0 for that group.
- Using the global absmax for every channel and then reporting per-channel results.
- Using half-away rounding and comparing against round-half-to-even codes at exact ties.
