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

## 2026-11-10 · Pilots of the hygiene and reject sweeps

Nothing here blocks the sweeps. One item is a judgement call a person may want to look at.

### For a person to look at

- `inf/kv-cache-size-arithmetic/p02-max-batch-after-weights`: the new hidden test requires
  `ValueError` for the dtype name `"FP16"`. The statement lists four lower-case names and says an
  unknown dtype raises, and the companion problem's example rejects `"int4"` as "not one of the
  four names". The coordinator read that as promised. A learner who normalises case before the
  lookup now fails this test. If that is too strict, the statement should say names are matched
  exactly, or the test should go.

### question (resolved)

- `sql/sql-top-n-and-latest-row-per-group/p01-latest-events`: a query with the wrong type for
  one output column fails every test on the shared type check, so the tool calls it broken, not
  killed. **Resolved:** the bullet is recorded as `uncovered` with that reason, and the runbook
  now says so.
- `dl/layer-normalization/p01-layer-norm-forward`: do the Constraints bullets (allowed tools, a
  ban) need a wrong solution? **Resolved:** no, unless one promises a complexity. The runbook
  now says so.

### blocked (resolved)

- `inf/kv-cache-size-arithmetic` (both problems): the runbook pointed at a harvest folder the
  protocol forbade reading. **Resolved:** the protocol now allows it. Nothing was lost; these
  problems have no harvested files.

### message-defect

- `inf/kv-cache-size-arithmetic/p01-kv-cache-bytes`: for a solution that wrongly rejects a zero
  sequence length, `test_zero_context_has_no_cache_for_any_batch` shows a bare exception instead
  of a sentence. The test needs a wrapper around the learner call.
- `cml/vector-operations/p01-vector-projection`: for a solution that truncates the scale to an
  integer, `test_matches_independent_reference_on_random_vectors` shows a bare exception.
