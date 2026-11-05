## Hint 1 — concept

Memory for a frozen base and a trainable adapter is four separate bills. The frozen base is only
stored. The adapter is stored, and it also has one gradient per number and two fp32 Adam moments per
number. Activations are not part of this bill, so leave them out.

## Hint 2 — formula / approach

Let `P` be the base count and `b` its bit width. The stored bytes are `ceil(P * b / 8)`. Use the same
rule for the adapter weights, with the adapter bit width. The gradients equal the adapter weights.
The Adam term is `2 * 4 * lora_params` bytes whatever the adapter bit width is. The total is the sum.
Round up with integer arithmetic, because floats can drift on large counts.

## Hint 3 — code shape

Check the arguments first: each count is a non-bool `int` and not negative, `base_bits` is in its set,
and `adapter_bits` is in its set. Define one helper that takes a count and a bit width and returns
`(count * bits + 7) // 8`. Compute the four terms, sum them into `total`, and return a dictionary whose
keys are inserted in the stated order.
