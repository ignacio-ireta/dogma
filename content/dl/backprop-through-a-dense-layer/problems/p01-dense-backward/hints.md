## Hint 1 — concept

Treat each output entry as a sum over features. Ask which axis reuses each parameter.

## Hint 2 — formula / approach

The input gradient multiplies upstream by transposed weights. The weight gradient contracts the batch axis. Bias adds contributions along that same batch axis. Mean scales the entire scalar objective.

## Hint 3 — code shape

Validate matrix dimensions first, choose one upstream scaling factor, and return three arrays with the documented shapes. Compare one entry against a scalar loop before writing a broader formula.
