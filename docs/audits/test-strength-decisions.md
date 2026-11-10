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

## 2026-11-10 · Hygiene sweep (phase 1) and reject sweep, weeks 4 to 8 and part of week 9

Scope: the 170 problems outside the 48 I-31 topics. Rows below are every note written in this
session up to the commit of `dl/multi-head-attention`. Two items wait for the owner.

### Waiting for the owner

- `dl/adamw-decoupled-weight-decay` is held and not committed. Its gate passes for both problems.
  On `p01-adamw-step`: (1) four new hidden tests were added, and the reject runbook caps a problem
  at three; (2) one of them, `test_moments_keep_the_bfloat16_dtype_of_their_parameter`, checks
  "with the input shapes and dtypes" using `bfloat16`, which the reference and both accepts pass
  here but which a correct solution could fail on a torch build without that operation; (3) the
  manifest keeps `uncovered` entries for two bullets that now have a killing test, and the tool
  has no command that drops an uncovered entry or takes back an added test. The choices are to
  commit it as it is, or to undo the problem and run it once more, which needs an approved revert.
  The work is saved as a patch under `var/audit-i32/coord/held/`.
- The per-wave merge to `main` (coordinator runbook section 7) has not been done in this session.
  The main checkout has untracked plan files and an uncommitted roadmap edit that would block a
  fast-forward, and merging `main` in brings the I-31c content while the `i32-start` tag must stay
  where it is until the first 170 problems are swept.

### For a person to look at

- `cml/shuffling-splitting-and-batching-data` (both problems): the new test
  `test_seed_is_keyword_only_in_the_signature` reads the signature with `inspect` and does not
  call the function with a positional seed. The Contract says the seed is keyword-only and the
  starter has it. A learner who rewrites the signature with `**kwargs` would see a bare `KeyError`.
- `dl/xavier-and-he-initialization/p01`: the new test requires `ValueError` for a scheme name in
  another case. The Contract says the scheme is "exactly" one of two lowercase names.
- `dl/transformer-block/p01`: the Contract writes the key projection with a bias, but dropping
  that bias cannot change any output (it shifts every score of a query row equally and the row
  softmax cancels it). Bullet C3 is recorded as uncovered for that reason.
- `dl/scaled-dot-product-attention/p01`: a solution that detaches its output passes every test,
  and no Contract sentence mentions gradients.
- `dl/learning-rate-decay-schedules`: a test helper `_close` with `abs_tol=1e-12` hid a
  difference near 1e-46 during this sweep. Helpers in other topics with tiny expected values may
  share that blind spot.
- Two manifest entries name a neighbouring bullet (the tests are sound):
  `dl/positional-encodings/p01` files the empty-table dtype reject under C5 where C3 promises the
  dtype, and `dl/a-complete-training-loop/p01` files the extra-target-columns reject under C2
  where procedure step 3 defines the loss.
- 23 `message-defect` rows below: existing tests that fail a wrong
  solution with a bare exception or a bare assert. No runbook of this program fixes them.
- 9 `unpromised` and 4 `inspect` rows are not listed here.
  `scripts/exercise_gate.py ledger --all --kind unpromised --kind inspect` prints them.

### Resolved by the coordinator

- Hygiene `question` rows for `cml/shuffling-splitting-and-batching-data` (both problems): a lint
  false positive. A module-level alias of a clean `settings(...)` call, used as a bare decorator,
  was reported as `HIDDEN_GIVEN_SETTINGS` (16 sites in 6 problems). Fixed in the lint (`6338733`)
  with unit tests; an alias that lacks a flag is still reported.
- Hygiene `question` row for `llm/model-parameter-and-memory-footprint/p02`: ids holding the
  precision names `fp8`, `fp16`, `fp32` and `bf16` were counted as value leaks. They are dtype
  names; the rule was extended (`aa02b18`) and the problem fixed on its second dispatch.
- Hygiene `question` rows for `dl/one-training-step` and `dl/sgd-with-momentum` (both problems):
  which id form applies. Answered from the runbook's own rule (an id names the kind of case and
  holds no number); the second dispatch fixed all three.
- `overreach` row for `dl/gradients-with-framework-autograd/p01`: not an overreach. The refused
  accept passed `f` a detached view that shares storage with `x`, and bullet C6 says `x` must be
  left unchanged. The existing test enforces that bullet.
- `no-witness` row for `dl/transformer-block/p01`: the key-bias case under "For a person to look
  at"; no output can differ, so there is no test to write.
- `no-witness` rows for `dl/adamw-decoupled-weight-decay/p01`: superseded. Their executor added a
  test for each on a second attempt. They belong to the held topic above.
- Four messages that stated a test input or an expected value were reworded before commit: three
  by their executors (temperature sampling p02, gradient descent p01, cosine annealing p01) and
  one by the coordinator (`cml/sigmoid-and-log-odds/p02`, a message-only edit gated in mode add).

### Checked

`make check` exit 0 on `ed7e65c` in a clean worktree: Ruff and format, strict mypy (180 files),
import contracts, content types, 2,192 Python tests, 175 topics verified, 187 browser tests and
the build, privacy audit clean. Commits after `ed7e65c` are not yet under a full gate.

### question

- `cml/shuffling-splitting-and-batching-data/p01-stratified-train-test-split`: HIDDEN_GIVEN_SETTINGS flagged 5 given sites at lines 90 96 104 120 137. Each has a SETTINGS alias decorator above it, and module line 17 defines SETTINGS with max_examples 60, deadline None and report_multiple_bugs False. Looks like a lint false positive on the alias. Not in the runbook decision table. No edits made.
- `cml/shuffling-splitting-and-batching-data/p02-minibatch-index-schedule`: HIDDEN_GIVEN_SETTINGS flagged 3 given sites at lines 81 96 110. Each has a SETTINGS alias decorator above it, and module line 14 defines SETTINGS with max_examples 60, deadline None and report_multiple_bugs False. Looks like a lint false positive on the alias. Not in the runbook decision table. No edits made.
- `dl/one-training-step/p01-manual-sgd-train-step`: HIDDEN_PARAM_IDS at line 219 on parametrize steps with cases 1 and 3. Cases are small step counts, not seeds and not named cases, so the runbook table does not cover them. Ids would need a kind word per case and the runbook forbids numbers in ids. No edits made.
- `dl/sgd-with-momentum/p01-momentum-step`: HIDDEN_PARAM_IDS at line 144 on parametrize case with 7 descriptive string cases. Runbook has no rule that fits both a named list which uses letter ids and short names that say the kind of case which must keep their meaning. Ids not chosen. No edits made.
- `dl/sgd-with-momentum/p02-nesterov-weight-decay`: HIDDEN_PARAM_IDS at line 150 on parametrize case with 6 descriptive string cases. Runbook has no rule that fits both a named list which uses letter ids and short names that say the kind of case which must keep their meaning. Ids not chosen. No edits made.
- `llm/model-parameter-and-memory-footprint/p02-training-and-inference-memory`: hygiene: lint 2 findings (HIDDEN_PARAM_IDS at the sgd parametrize and the INVALID_CASES parametrize) fixed in test but gate names FAIL on the unreported site test_total_matches_the_hand_worked_bytes_per_parameter (ids=lambda c: join of values, dtype ids fp8/fp16/fp32 contain digits); edits undone; left for coordinator
- `dl/adamw-decoupled-weight-decay/p01-adamw-step`: held uncommitted by the coordinator, gate passes: four new tests where the runbook caps a problem at three, one of them checks moment dtype with bfloat16, and the manifest keeps two uncovered entries for bullets that now have a killing test, the tool cannot drop an uncovered entry or take back a test

### overreach

- `dl/gradients-with-framework-autograd/p01-gradient-at-a-point`: accept_detached_view_leaf.py: test_function_writing_into_its_argument_leaves_the_caller_tensor_unchanged, no clause fixes this choice

### no-witness

- `dl/adamw-decoupled-weight-decay/p01-adamw-step`: reject_zero_dim_parameter_rejected.py survives and breaks C1, the one test that rejects it raises a bare exception so the gate does not count it, no test added
- `dl/adamw-decoupled-weight-decay/p01-adamw-step`: reject_lr_zero_rejected.py survives and breaks C3, the one test that rejects it raises a bare exception so the gate does not count it, no test added
- `dl/transformer-block/p01-pre-norm-causal-block`: reject_drops_key_bias.py survives and breaks C3, no test found, dropping b_k adds the same q dot b_k to every score in a row which softmax cancels so no output changes for any input

### reference-doubt

None.

### blocked

None.

### message-defect

- `algo/prefix-sums/p02-rectangle-range-sums`: reject_rejects_negative_cells.py: test_negative_values_and_repeated_queries shows kind=bare_exception
- `algo/prefix-sums/p02-rectangle-range-sums`: reject_single_row_query_rejected.py: test_inclusive_rectangles_match_the_worked_table shows kind=bare_exception
- `algo/prefix-sums/p02-rectangle-range-sums`: reject_drops_repeated_answers.py: test_negative_values_and_repeated_queries shows kind=raw_assert
- `algo/prefix-sums/p02-rectangle-range-sums`: reject_empty_grid_raises.py: test_empty_query_collection_and_reversed_bounds shows kind=bare_exception
- `cml/bias-variance-decomposition/p01-bias-variance-from-predictions`: reject_list_input_rejected.py: test_spread_without_bias_gives_variance_and_the_noise_term shows kind=bare_exception
- `cml/k-fold-cross-validation/p01-kfold-indices`: reject_default_shuffle_on.py: test_ten_positions_in_three_folds_give_consecutive_blocks_with_the_remainder_first shows kind=bare_exception
- `cml/shuffling-splitting-and-batching-data/p02-minibatch-index-schedule`: reject_zero_epochs_raises.py: test_zero_epochs_gives_an_empty_schedule shows kind=bare_exception
- `cml/xgboost-second-order-objective/p01-leaf-weight-and-split-gain`: reject_rejects_zero_lambda.py: test_opposite_gradients_give_opposite_leaf_weights shows kind=bare_exception
- `cml/xgboost-second-order-objective/p01-leaf-weight-and-split-gain`: reject_type_error_on_negative_lambda.py: test_negative_reg_lambda_raises_value_error shows kind=bare_exception
- `dl/counting-network-parameters/p01-count-dense-parameters`: reject_widths_rejects_tuple.py: test_no_bias_example shows kind=bare_exception
- `dl/gradients-with-framework-autograd/p01-gradient-at-a-point`: reject_result_forced_to_default_dtype.py: test_caller_tensor_keeps_requires_grad_flag_and_no_grad_field shows kind=bare_exception
- `dl/gradients-with-framework-autograd/p02-straight-through-round`: reject_float_step_refused.py: test_forward_rounds_each_entry_to_the_nearest_step_multiple shows kind=bare_exception
- `dl/learning-rate-warmup/p02-inverse-sqrt-lr`: reject_d_model_one_refused.py: test_peak_is_at_warmup_steps_with_value_d_to_minus_half_times_w_to_minus_half shows kind=bare_exception
- `dl/learning-rate-warmup/p02-inverse-sqrt-lr`: reject_one_step_warmup_refused.py: test_peak_is_at_warmup_steps_with_value_d_to_minus_half_times_w_to_minus_half shows kind=bare_exception
- `dl/position-wise-feed-forward-block/p01-gelu-ffn`: reject_w1_without_transpose.py: test_hidden_layer_bias_and_output_bias_are_both_added shows kind=bare_exception
- `dl/position-wise-feed-forward-block/p01-gelu-ffn`: reject_w2_without_transpose.py: test_hidden_layer_bias_and_output_bias_are_both_added shows kind=bare_exception
- `dl/position-wise-feed-forward-block/p02-swiglu-ffn`: reject_missing_gate_up_shape_check.py: test_gate_and_up_shapes_must_match shows kind=bare_exception
- `dl/residual-connections/p01-residual-block`: reject_transposed_second_weight.py: test_projection_shortcut_example shows kind=bare_exception
- `dl/reverse-mode-autograd-engine/p01-scalar-autograd`: reject_no_visited_set.py: test_shared_path_example shows kind=raw_assert
- `dl/reverse-mode-autograd-engine/p01-scalar-autograd`: reject_single_edge_for_square.py: test_square_example shows kind=raw_assert
- `dl/smooth-activations-tanh-gelu-swish/p02-smooth-activation-derivatives`: reject_empty_input_raises.py: test_empty_preserves_shape shows kind=bare_exception
- `dl/tensors-in-pytorch-and-tinygrad/p02-patchify-images-row-major`: reject_rejects_patch_one.py: test_channels_are_flattened_before_pixels_within_each_patch shows kind=bare_exception
- `dl/training-an-mlp-from-scratch/p01-train-two-layer-mlp`: reject_losses_drop_last.py: test_one_step_zero_input_example shows kind=raw_assert
