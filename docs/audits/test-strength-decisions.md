# Test strength: decisions and notes

*Increment I-32. Written by the coordinator from the ledgers at the end of each wave. A `question`,
`overreach` or `reference-doubt` entry needs a person. An `inspect` entry is information.*

## 2026-11-10 · Platform phase and the first 170 blind accepts

Nothing here is waiting for a decision.

### question

- `dl/smooth-activations-tanh-gelu-swish/p01-gelu-and-swish` (and `p02`): the blind solver needed
  two submissions. It returned NumPy scalars for scalar input, and the hidden tests require
  arrays. **Resolved by the coordinator:** the Contract says three NumPy `float64` arrays "each
  exactly the shape of `x`", scalar inputs included. NumPy ufuncs return scalars for 0-d input,
  the solver missed that, the hidden message named it, and the second submission passed. No
  statement defect.

### inspect

- `math/reservoir-sampling/p01-reservoir-sample`: the blind solution is 0.96 similar to the
  reference when whole files are compared. The starter carries the docstring and the Contract
  fixes the draw protocol step by step, so the body can hardly differ. Judged independent. The
  tool's similarity check now ignores lines the starter already has.
- Two blind solvers listed the whole bundle directory while looking for their own bundles. They
  saw other problems' folder names and opened no file contents. Their solves stand.
