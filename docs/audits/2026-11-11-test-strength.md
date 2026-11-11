# Test strength

*Increment I-32, 2026-11-11. Every number below is pasted from a command. The commands, run from
the repository root on the final content of the branch:*

```bash
uv run --frozen --group torch python scripts/exercise_audit.py baseline  --out var/audit-i32/final
uv run --frozen --group torch python scripts/exercise_audit.py mutate    --out var/audit-i32/final
uv run --frozen --group torch python scripts/exercise_audit.py seeds     --out var/audit-i32/final
uv run --frozen --group torch python scripts/exercise_audit.py precision --out var/audit-i32/final
uv run --frozen --group torch python scripts/exercise_audit.py summary   --out var/audit-i32/final
uv run --frozen --group torch python scripts/exercise_gate.py lint       --out var/audit-i32/final
uv run --frozen --group torch python scripts/exercise_gate.py variants   --out var/audit-i32/final
uv run --frozen --group torch python scripts/exercise_gate.py scorecard  --out var/audit-i32/final
uv run --frozen --group torch python scripts/exercise_gate.py ledger     --out var/audit-i32/final --all
git diff --name-only <start>..HEAD -- 'content/*/*/problems/*/meta.yaml' | wc -l
git diff --shortstat <start>..HEAD -- 'content/*/*/problems/*/test_hidden.py'
```

`baseline`, `mutate`, `seeds`, `precision` and `summary` ran at commit `05559f0`. The two commits
after it (`b5edd2e`, `6c39158`) change only two `variants/manifest.yaml` files, which those five
commands do not read. `lint`, `variants`, `scorecard`, `ledger` and the `git diff` lines ran at
`6c39158`.

I-32 ran in two phases, so there are two start points. `i32-start-phase1` is the commit before
any I-32 content change. `i32-start` is the merge that brought in the 48 topics of I-31, after
which the second phase covered their 81 problems.

## Before and after

| | Before | After |
|---|---|---|
| Hidden-test lint | 214 lint findings in 73 problems | `lint: 0 findings in 0 problems` |
| Committed variants | 0 problems with committed variants | `with a blind accept: 251`, `problems: 251` |
| Correct solutions kept as accepts | 0 | `accepts: 461` |
| Wrong solutions kept as rejects | 0 | `rejects: 1847` |
| Contract bullets with a reject or an `uncovered` entry | not measured | `every bullet covered: 251` |
| Reject sweep and mutant triage | not run | `both sweeps: 251` |
| References pass | | `baseline: 251/251 references pass` |
| Variants behave as recorded | | `variants: 2308 ok, 0 mismatches in 251 problems` |

Files changed, whole increment (`i32-start-phase1..HEAD`):

```text
meta.yaml files changed: 162
test_hidden.py: 167 files changed, 3033 insertions(+), 320 deletions(-)
```

Of which the second phase (`i32-start..HEAD`):

```text
meta.yaml files changed: 35
test_hidden.py: 43 files changed, 814 insertions(+), 37 deletions(-)
```

## Mutation and seeds

Output of `summary`:

```text
mutants 10551: {'survived': 665, 'killed_hidden': 3322, 'killed_public': 6503, 'oom': 15, 'timeout': 46}
killed 93.7%
hidden-only kills 3322
  algo      665 killed  94.4%
  cml      4141 killed  91.6%
  dl       2004 killed  95.6%
  inf      1020 killed  95.7%
  llm      1878 killed  94.7%
  math      132 killed  95.5%
  rl        370 killed  93.0%
  safe      112 killed  96.4%
  sql       229 killed  94.3%
  op aggregate           21 survived     0
  op arithmetic        1544 survived    24
  op array_copy          33 survived    30
  op axis               115 survived     2
  op bool               115 survived    36
  op boolean            542 survived    33
  op call               637 survived    22
  op compare           1994 survived    74
  op constant          2839 survived   228
  op distinct             4 survived     0
  op drop_call          185 survived    38
  op drop_negation      750 survived    10
  op drop_statement     160 survived     2
  op dtype              223 survived    30
  op flow                30 survived     7
  op frame                5 survived     3
  op guard             1205 survived   119
  op join                 7 survived     2
  op null                11 survived     0
  op order               41 survived     0
  op transpose           51 survived     0
  op window              39 survived     5
```

Output of `seeds`, first and last line:

```text
seeds: 5020 runs
seeds: 3 failing runs on 1 problems
```

The three failing runs, from `seeds.jsonl`:

```text
cml/lagrange-multipliers/p01-constrained-quadratic-min seed 1: test_hidden.py::test_solution_is_feasible_and_satisfies_stationarity
cml/lagrange-multipliers/p01-constrained-quadratic-min seed 15: test_hidden.py::test_solution_is_feasible_and_satisfies_stationarity
cml/lagrange-multipliers/p01-constrained-quadratic-min seed 13: test_hidden.py::test_solution_is_feasible_and_satisfies_stationarity
```

The grader pins the Hypothesis seed to 0, and seed 0 passes, so a Submit does not see this. The
failing test is a property test that I-32 did not write or tighten. Not changed. A person decides
whether its tolerance or its input strategy should change.

## What stays open

Output of `ledger --all`, grouped as printed. The decisions taken on these rows, and the reasons,
are in [test-strength-decisions.md](test-strength-decisions.md). Rows in a group other than `done`
are open for the owner; a row records what the executor wrote at the time, so a few describe a
state that a later commit resolved (the decisions document says which).

### blocked
- `algo/binary-tree-recursion-and-traversal/p02-tree-diameter`: reject_quadratic_rescan.py probe timed out twice (VERDICT infra timeout): it recomputes subtree heights at each node, which is exponential in depth. Replaced by reject_full_height_rescan.py for C5
- `llm/retrieval-augmented-generation-pipeline/p01-chunk-words`: reject_zero_overlap_steps_by_size_minus_one: probe status oom twice in a row. With size 1 and overlap 0 its step is 0, so the loop never ends. Replaced by reject_zero_overlap_terminating (same bullet, step 1 when size is 1).
- `inf/kv-cache-size-arithmetic/p01-kv-cache-bytes`: runbook step 2 says to copy up to three files from var/audit-i32/harvest/ if that folder exists, but protocol section 2 forbids reading var outside OUT. Followed the protocol: harvest files not used. Eight wrong solutions were written from the catalogue instead
- `inf/kv-cache-size-arithmetic/p02-max-batch-after-weights`: runbook step 2 says to copy up to three files from var/audit-i32/harvest/ if that folder exists, but protocol section 2 forbids reading var outside OUT. Followed the protocol: harvest files not used

### question
- `cml/shuffling-splitting-and-batching-data/p01-stratified-train-test-split`: HIDDEN_GIVEN_SETTINGS flagged 5 given sites at lines 90 96 104 120 137. Each has a SETTINGS alias decorator above it, and module line 17 defines SETTINGS with max_examples 60, deadline None and report_multiple_bugs False. Looks like a lint false positive on the alias. Not in the runbook decision table. No edits made.
- `cml/shuffling-splitting-and-batching-data/p02-minibatch-index-schedule`: HIDDEN_GIVEN_SETTINGS flagged 3 given sites at lines 81 96 110. Each has a SETTINGS alias decorator above it, and module line 14 defines SETTINGS with max_examples 60, deadline None and report_multiple_bugs False. Looks like a lint false positive on the alias. Not in the runbook decision table. No edits made.
- `dl/one-training-step/p01-manual-sgd-train-step`: HIDDEN_PARAM_IDS at line 219 on parametrize steps with cases 1 and 3. Cases are small step counts, not seeds and not named cases, so the runbook table does not cover them. Ids would need a kind word per case and the runbook forbids numbers in ids. No edits made.
- `dl/sgd-with-momentum/p01-momentum-step`: HIDDEN_PARAM_IDS at line 144 on parametrize case with 7 descriptive string cases. Runbook has no rule that fits both a named list which uses letter ids and short names that say the kind of case which must keep their meaning. Ids not chosen. No edits made.
- `dl/sgd-with-momentum/p02-nesterov-weight-decay`: HIDDEN_PARAM_IDS at line 150 on parametrize case with 6 descriptive string cases. Runbook has no rule that fits both a named list which uses letter ids and short names that say the kind of case which must keep their meaning. Ids not chosen. No edits made.
- `llm/model-parameter-and-memory-footprint/p02-training-and-inference-memory`: hygiene: lint 2 findings (HIDDEN_PARAM_IDS at the sgd parametrize and the INVALID_CASES parametrize) fixed in test but gate names FAIL on the unreported site test_total_matches_the_hand_worked_bytes_per_parameter (ids=lambda c: join of values, dtype ids fp8/fp16/fp32 contain digits); edits undone; left for coordinator
- `dl/smooth-activations-tanh-gelu-swish/p01-gelu-and-swish`: blind solver needed 2 submissions: returned NumPy scalars for scalar input, hidden tests require arrays; check the Contract states the return type for scalar input
- `dl/adamw-decoupled-weight-decay/p01-adamw-step`: held uncommitted by the coordinator, gate passes: four new tests where the runbook caps a problem at three, one of them checks moment dtype with bfloat16, and the manifest keeps two uncovered entries for bullets that now have a killing test, the tool cannot drop an uncovered entry or take back a test
- `algo/binary-search/p01-lower-bound`: problem.md intro says the search must take a logarithmic number of steps, not a scan, but no K or C bullet states a complexity. A linear-scan wrong solution was not written. Decide whether the intro sentence counts as a Constraints bullet
- `algo/topological-sort/p01-lexicographic-topological-order`: C8 promises O of V plus E log V time. It is a Contract bullet, not a K bullet, and the runbook gives complexity wrong solutions only for K bullets. Not decided, no wrong solution written for C8
- `dl/layer-normalization/p01-layer-norm-forward`: Step 2 table row for constraints covers only complexity promises. K1 and K2 are a list of allowed tools and a ban, with no complexity. I recorded both as add-uncovered with that reason. Please confirm bans need no wrong solution.
- `sql/sql-top-n-and-latest-row-per-group/p01-latest-events`: C4 wrong query reject_rank_as_integer.sql (CAST rank to INTEGER) probes as VERDICT broken (no test passed), not killed: every test fails on the schema-type check before its own assertion, so the probe table has no row for this case. Not recorded, C4 has no reject or uncovered entry, mark-swept not run. Any C4 mistake fails the same way. Needs a decision on how a schema-only wrong query should be handled
- `cml/t-tests-and-anova/p01-welch-t-test`: mutant 8f7023199a84ff1e (no guard on se_squared equal to zero): on a equal to 1 and 1 plus tiny, b equal to 1e16 twice, the reference raises ValueError saying both samples are constant, although C4 says one constant sample with a varying other is valid; the mutant raises ZeroDivisionError. Reference behaviour looks off against C4; not changed

### unpromised
- `algo/heaps-and-priority-queues/p01-merge-sorted-streams`: reject_reads_one_ahead.py refills before yield and reads t plus 1 times, which the Contract allows, so no Contract sentence forbids it. No test added
- `cml/adaboost-and-gradient-boosting/p01-boost-stumps-mse`: reject_tie_within_tolerance.py: a later candidate replaces the incumbent on any larger gain, not only by more than 1e-9 as the procedure text says; the Contract bullet C4 only says equal to F_M and names no tie rule, so the Contract does not forbid it in its own words. No test added
- `cml/bayes-theorem/p01-bayes-posterior`: reject_likelihood_columns_normalised.py: rescales each observation column before use, which multiplies every hypothesis by the same factor and cancels in the normaliser, so the output is mathematically the same. No Contract sentence forbids it
- `cml/chain-rule/p01-mse-sigmoid-gradient`: reject_w_rounded_to_float32.py: w is rounded to float32 and survives, no Contract sentence forbids reduced precision for w
- `cml/eigenvalues-and-eigenvectors/p01-power-iteration`: reject_zero_matrix_zero_vector.py survives: its zero vector branch is unreachable, because when A times v is zero the residual is already zero and the loop stops first, so it behaves like the reference; no Contract sentence forbids it
- `cml/k-means/p02-kmeans-plus-plus-seeding`: reject_dtype_follows_input.py returns int64 centres for integer X, values are exact, only the dtype differs, no tolerance is missed by ten times, so no test added
- `cml/logistic-regression/p01-l2-logistic-fit`: reject rejects single row file: a one row X is refused, but one row cannot hold both classes, so the Contract already raises ValueError for it, no observable difference, no Contract sentence forbids the refusal of a single row by itself, no test added
- `cml/logistic-regression/p01-l2-logistic-fit`: loose stop at ten times tol: the file stops when the largest gradient entry is at most ten times tol, so it misses tol by at most ten times, precision only, measured over 6300 fits: worst ratio exactly 10 at the boundary, all other misses below 10, rule b so no test added
- `cml/matrix-inverse-and-rank/p01-gauss-jordan-inverse`: reject_nan_only_guard.py survives: for an infinite entry it raises the ValueError singular message instead of a finite-check message, and no Contract sentence fixes that message, so no test added
- `cml/precision-recall-and-f1/p02-best-f1-threshold`: reject_harmonic_mean_f1.py: F1 via precision and recall differs by at most 3e-16 from the reference, far below tolerance; it only changes which of two exactly tied F1 candidates wins through rounding, so no test is written
- `cml/principal-component-analysis/p01-pca-via-svd`: reject_inf_not_rejected.py: an infinite entry makes the centred data NaN, so numpy linalg raises LinAlgError which is a ValueError, the Contract asks for ValueError, no Contract sentence forbids the result as a whole, so no test added
- `cml/random-forests-and-feature-importance/p01-permutation-importance`: reject_column_vector_predictions_pass.py: accepts predictions shaped as a column, the Raises bullet only asks for one prediction per row and no Contract sentence forbids a column shape
- `cml/ranking-metrics/p01-ndcg-at-k`: reject_natural_log_discount survives: a natural log discount is a constant multiple of the log2 discount, so DCG and IDCG scale by the same factor and NDCG is identical for every input; an equivalent mutant, not a wrong solution
- `cml/roc-pr-curves-and-auc/p01-roc-auc-rank-statistic`: reject_accepts_column_labels.py: accepts labels shaped as a column of n rows by 1, the Raises bullet does not name a wrong dimension, C1 describes valid inputs only
- `cml/shuffling-splitting-and-batching-data/p01-stratified-train-test-split`: reject_float_n_not_checked.py accepts a float n such as 10.0, no Contract sentence forbids it: n is stated as integer but no Raises clause covers a non-integer n
- `cml/xgboost-second-order-objective/p01-leaf-weight-and-split-gain`: reject_clips_large_gamma survives, it caps gamma at 1e6, the Contract only says gamma is finite and zero or positive, and no Contract sentence forbids the cap, the statement formula subtracts gamma in full, a question for the coordinator, no test added
- `dl/adam/p01-adam-step`: reject_empty_list_rejected.py: rejects an empty params list; the Contract gives no sentence that forbids an empty list, so no test added (C1)
- `dl/adam/p01-adam-step`: reject_tuple_return.py: returns new params as a tuple; the Contract does not say the returned params must be a list, so no test added (C5)
- `dl/residual-connections/p01-residual-block`: reshape_flatten_fails_empty_batch: flattens to 2-D and back; survives all tests, no Contract sentence forbids it, it is a correct solution
- `dl/scaled-dot-product-attention/p01-scaled-dot-product-attention`: reject_output_detached.py: output is detached from the autograd graph, no Contract sentence forbids it (Contract says nothing about gradients). Survived all tests. Add no test
- `dl/tensors-in-pytorch-and-tinygrad/p02-patchify-images-row-major`: view on the split reshape is valid for non-contiguous input and gives the right tokens, so the file is not wrong and no Contract sentence forbids it
- `inf/data-quality-and-drift-detection/p01-population-stability-index`: reject_ties_collapse_edges.py survives, merging repeated edges drops only zero width bins, output identical to reference (max abs diff 3.6e-15 over 20000 random cases), no Contract sentence forbids it
- `llm/chat-templates-and-tool-call-formats/p02-parse-tool-call`: writing an empty required list into the tool shows only for a schema without that key, the statement does not say the key may be absent, test withdrawn
- `sql/pandas-groupby-aggregate-and-transform/p01-customer-order-features`: reject_swallows_errors.py catches an unreadable date and returns an empty frame. The Raises bullet says the function raises nothing, that bad input is the caller's problem and that it does not catch errors. A test that required an exception for an unreadable date was withdrawn by the coordinator before commit: a solution that parses with errors coerce raises nothing either and is not clearly wrong under that bullet
- `sql/sql-subqueries-and-ctes/p02-orders-above-their-customers-average`: reject_null_customer_grouped.sql: NULL customer orders share one average, but the outer filter drops them, so no output changes. No Contract sentence forbids the internal average
- `algo/bfs-and-dfs-on-graphs-and-grids/p01-grid-shortest-path`: mutant c08cb5d5f81fede5: removes the empty-grid check; an empty grid with a valid start still raises ValueError (message differs only), but with a non-tuple start it raises TypeError instead of ValueError. C1 fixes ValueError for an empty grid and does not rank errors when two inputs are wrong, so no clause covers it
- `algo/bfs-and-dfs-on-graphs-and-grids/p02-count-components`: mutant 1d4ce3697fcb01df: width check 0 becomes -1, so a grid of one empty row is accepted and returns 0 instead of ValueError; C1 requires rows of at least 1 and lists empty grid as ValueError, but does not say a grid of empty rows raises, so the input is outside the stated domain and no clause covers it
- `algo/binary-search/p02-min-ship-capacity`: mutant 331611cb830228da: the empty weights check is removed, so an empty list fails later in max with a ValueError and a different message, no clause covers the message
- `algo/binary-search/p02-min-ship-capacity`: mutant 673b597991955b91: the empty weights check compares length to minus 1, so an empty list fails later in max with a ValueError and a different message, no clause covers the message
- `algo/dynamic-programming/p02-longest-common-subsequence`: mutant 3ee744c4d0e6b5cf: the reconstruction tie-break (>= to >) can return a different witness of the same length, which is still a valid longest common subsequence; C2 accepts any valid witness, so no clause is broken
- `algo/hash-maps-and-sets/p01-top-k-frequent-tokens`: mutant ddb58b57848c61ee: strict less-than instead of less-or-equal in the k guard, differs only for a float k of zero (it raises TypeError instead of returning an empty list); the Contract says k is an int and no clause covers floats
- `algo/hash-maps-and-sets/p01-top-k-frequent-tokens`: mutant dbf119aca4e820d0: guard against minus one instead of zero, differs only for a float k of zero (it raises TypeError instead of returning an empty list); the Contract says k is an int and no clause covers floats
- `algo/heaps-and-priority-queues/p02-running-median`: mutant 4dc61d1837ee0046: the zero-sign test in the overflow guard changes; the two results differ only when the lower middle is a tiny negative subnormal and the upper middle is zero, where the mean rounds to the other side of a tie. Both results are equally correct roundings of the mean, and no clause fixes the rounding of a float mean
- `algo/heaps-and-priority-queues/p02-running-median`: mutant 1860c1cbac756b59: the sign threshold for the upper middle moves from 0 to 1; for middle values from 0 up to 1 the sum formula and the direct formula differ by at most last-digit rounding. No clause fixes that rounding, and no overflow case is reached
- `algo/heaps-and-priority-queues/p02-running-median`: mutant 1e1bd5b3139e077c: the sign threshold for the lower middle moves from 0 to minus 1; for lower middle values from minus 1 up to 0 the two formulas differ by at most last-digit rounding. No clause fixes that rounding, and no overflow case is reached
- `algo/heaps-and-priority-queues/p02-running-median`: mutant c350fac8b9778134: the sign threshold for the upper middle moves from 0 to minus 1; for negative middle values from minus 1 up to 0 the two formulas differ by at most last-digit rounding. No clause fixes that rounding, and no overflow case is reached
- `algo/heaps-and-priority-queues/p02-running-median`: mutant ee81ad1cc056858c: the sign threshold for the lower middle moves from 0 to 1; for lower middle values from 0 up to 1 the two formulas differ by at most last-digit rounding. No clause fixes that rounding, and no overflow case is reached
- `algo/prefix-sums/p02-rectangle-range-sums`: mutant 0aa0ddfaf92a900e: the query length check is removed, so a query of wrong length with a bad element type raises TypeError instead of ValueError; the Contract lists both errors without saying which wins, so no clause is broken
- `algo/topological-sort/p01-lexicographic-topological-order`: mutant 573c07a56737f83a: _find_cycle starts from the largest unplaced name instead of the smallest; it still returns a real cycle of the graph, and C5 promises only one cycle, so no clause is broken
- `algo/topological-sort/p01-lexicographic-topological-order`: mutant beab75521069f5dd: _find_cycle takes the largest unplaced predecessor instead of the smallest; it still walks a real cycle, and C5 promises only one cycle, so no clause is broken
- `cml/adaboost-and-gradient-boosting/p02-adaboost-round`: mutant 96a2084920b824aa: removing the y ndim check accepts a 2-D y instead of raising, no clause covers it (C1 says 1-D)
- `cml/adaboost-and-gradient-boosting/p02-adaboost-round`: mutant 8fb622d7d7f5afd0: 'and' in the y check accepts a 2-D non-empty y instead of raising, no clause covers it (C1 says 1-D)
- `cml/bayes-theorem/p01-bayes-posterior`: mutant 7a4166b23e8c4c8b: the 1-D check on observations is removed. The contract says observations is 1-D, so a 2-D input is outside the stated domain, and the Raises bullet only names shape mistakes in general. No clause covers it
- `cml/bayes-theorem/p01-bayes-posterior`: mutant a0f88364567f2555: the 1-D and non-empty check on the prior is removed. A 2-D prior is outside the 1-D contract, and the Raises bullet only names shape mistakes in general. No clause covers it
- `cml/bayes-theorem/p01-bayes-posterior`: mutant 0beca3de34c85f58: the likelihood row-sum check uses >= instead of >, so it differs only when a row sum is off by exactly 1e-9, a float tie at the threshold; no clause covers an exact tie
- `cml/bayes-theorem/p01-bayes-posterior`: mutant 70f6914388dcde9e: the prior check uses and instead of or. A 2-D non-empty prior is outside the 1-D contract, and the Raises bullet only names shape mistakes in general. An empty prior is still rejected by the sum check
- `cml/bayes-theorem/p01-bayes-posterior`: mutant 5e926dba9fd95194: the empty-sequence return multiplies the prior by its sum instead of dividing. Identical for priors summing to exactly 1. Measured: the output sum is off by 1.8e-9 for a prior off by 9e-10, against the test rtol of 1e-9, a factor under 2 and below the ten-times rule for precision-scale mutants
- `cml/bernoulli-binomial-and-poisson/p01-binomial-pmf`: mutant b5a15519106ff043 changes isfinite all to any so a mixed array with one infinite entry and integral entries returns values instead of raising ValueError, the Contract does not say whether infinity is a non-integer under C5 or an entry outside the range that gets probability zero under C1, no clause fixes it
- `cml/bias-variance-decomposition/p01-bias-variance-from-predictions`: mutant 7f1a873dbff071e6: in the overflow branch the scale is max(column) without abs; results differ only by rounding of the mean near the float64 limit, where bias2 is infinite in both. No clause covers the last digits
- `cml/bias-variance-decomposition/p01-bias-variance-from-predictions`: mutant c558bd6d26c66903: _mean_square scale is max(values) without abs; the variance differs only by rounding, and the scaled values stay bounded by B. No clause covers the last digits
- `cml/choosing-a-tree-split/p01-best-split-threshold`: mutant 2c5f1cef5f76619c (line 81): tie loop '>' becomes '>=' so an exact 1e-12 gap changes the incumbent; C4 says 'more than 1e-12', an exact-threshold strict vs non-strict tie depends on the last float digit, so no test
- `cml/choosing-a-tree-split/p01-best-split-threshold`: mutant 43b9b009be885be4 (line 20): entropy zero-proportion guard p > 0 becomes p >= 0 (NaN on pure side); the 0 log 0 = 0 rule is in the impurity formula of the body, not in any labelled clause from the clauses output, so no test
- `cml/covariance-and-correlation/p01-covariance-matrix`: mutant 6f5b29b065ad959f: guard on x.ndim removed; non-2-D input still raises ValueError (unpacking error), only the message text differs; no clause covers the message text, C4 holds
- `cml/covariance-and-correlation/p01-covariance-matrix`: mutant 45058b2cde8353ef: n==0 changed to n==-1; zero-row input still raises ValueError from the n<=ddof check, only the message differs; no clause covers the message text, C4 holds
- `cml/covariance-and-correlation/p02-correlation-matrix`: mutant 1875e7b78aa67818: x.ndim guard removed; non-2-D input still raises ValueError (unpacking error), only the message text differs; no clause covers the message text
- `cml/covariance-and-correlation/p02-correlation-matrix`: mutant 3b38bd4b6777a85c: covariance divisor n-1 becomes multiplier n-1; correlation scale cancels, so only rounding differs (max 3.3e-16 vs tolerance atol 1e-12, rtol 1e-9); precision only, not tested
- `cml/covariance-and-correlation/p02-correlation-matrix`: mutant 679c994252961e21: n==0 check changed to n==-1; zero-row input still raises ValueError from the n<2 check, only the message differs; no clause covers the message text
- `cml/covariance-and-correlation/p02-correlation-matrix`: mutant 8fd4bbd023062f7f: covariance divisor n-1 becomes n+1; correlation scale cancels, so only rounding differs (max 3.3e-16 vs tolerance atol 1e-12, rtol 1e-9); precision only, not tested
- `cml/covariance-and-correlation/p02-correlation-matrix`: mutant c6813259e401c8dc: n==0 check changed to n==1; one-row input still raises ValueError from the n<2 check, only the message differs; no clause covers the message text
- `cml/covariance-and-correlation/p02-correlation-matrix`: mutant e93f919a259ee8a0: covariance divisor n-1 becomes n-0; correlation scale cancels, so only rounding differs (max 2.2e-16 vs tolerance atol 1e-12, rtol 1e-9); precision only, not tested
- `cml/eigenvalues-and-eigenvectors/p01-power-iteration`: mutant 662b6cf696b7d1fc: strict < instead of <= in the residual stop, differs only when the residual equals the threshold exactly in floating point; a test would depend on the last digit
- `cml/eigenvalues-and-eigenvectors/p01-power-iteration`: mutant fc485f8c45d844ba: >= instead of > in the symmetry check, differs only when the asymmetry equals the threshold exactly in floating point; a test would depend on the last digit
- `cml/entropy-and-mutual-information/p01-distribution-entropy`: mutant 33fb3c058aa5a391: sum check uses >= instead of >, differs only when abs(sum-1) is exactly 1e-9 (a float tie at the threshold), no clause decides that boundary
- `cml/expectation-and-variance/p01-discrete-moments`: mutant 05ad52ab78a4ebab: empty check removed. Empty input still raises ValueError (from the sum check), so only the message differs; no clause covers the message text
- `cml/expectation-and-variance/p01-discrete-moments`: mutant d065f3024ee52105: sum tolerance comparison changed from > to >=. Differs only when the sum error is exactly 1e-9 in float, a last-digit tie at the threshold, so no test
- `cml/expectation-and-variance/p01-discrete-moments`: mutant 2e73c1fdbf5399d2: ndim check changed from or to and. Input with a wrong ndim still raises ValueError from the length-mismatch check, so only the message differs; no clause covers the message text
- `cml/expectation-and-variance/p01-discrete-moments`: mutant e4eefa05aadf1fb2: empty check changed from size == 0 to size == -1. Empty input still raises ValueError (from the sum check), so only the message differs; no clause covers the message text
- `cml/feature-scaling/p02-robust-median-iqr-scaler`: mutant 1d66417df6a27130: strict-to-nonstrict sign test in the interpolation changes only the last bits, measured worst relative difference 6e-16 over 4000 random cases, far below ten times the tolerance, so precision only and no clause is broken
- `cml/feature-scaling/p02-robust-median-iqr-scaler`: mutant ee02029307559a25: scale rule uses less-or-equal instead of strictly below at exactly ten machine epsilons, a tie at an exact float threshold that depends on the last digit, so no test is written
- `cml/feature-scaling/p01-standard-scaler`: mutant 824047c788e564e7: the peak is the largest signed deviation instead of the largest absolute one. It differs from the reference only when rounding puts the mean on the top value of a column, as in a spread of one subnormal step, where the true scale is half the smallest subnormal and is not representable. C10 says accuracy tests use representable results. A test built on that case was withdrawn by the coordinator before commit
- `cml/gaussian-mixtures-and-em/p01-gmm-em-full-covariance`: mutant 50f7acba8554aad6: no finite check on X. Non-finite X then fails inside cholesky with LinAlgError, a ValueError subclass, not with the reference message; C7 names ValueError only, so no clause covers the different error class
- `cml/gaussian-mixtures-and-em/p01-gmm-em-full-covariance`: mutant 59ca836ec1371c61: no finite check on reg. Infinite or NaN reg then fails inside cholesky with LinAlgError, a ValueError subclass, not with the reference message; C7 names ValueError only, so no clause covers the different error class
- `cml/gaussian-mixtures-and-em/p01-gmm-em-full-covariance`: mutant 99552133393aeda7: accepts a boolean reg as 1.0. C5 says reg is a finite real number and only k and n_iter are stated as not booleans; a bool is an int subclass, so the Contract does not promise a rejection, and rule 6c applies
- `cml/gradient-descent/p01-gradient-descent`: mutant bf2f4a9745d6fdc8: '<= tol' becomes '< tol' in the gradient-norm stop rule (line 41), so the result differs only when the norm equals tol exactly; that is a last-digit float tie and no Contract clause (C5 names only the tolerance) fixes the boundary, so no test
- `cml/handling-class-imbalance/p02-prior-shift-adjust`: mutant 1f15129ddfb64c33: equal priors return the caller's own array instead of a copy; values are identical, and no clause promises a fresh array
- `cml/k-fold-cross-validation/p02-cross-validate`: mutant 321a3e7f466f558c: removes the non-empty and 1-D guard in _checked_indices. Only non-1-D index arrays (outside the stated domain, C4) take a different path; empty 1-D int arrays still raise ValueError from the min() call with a different message. No clause covers the message text or non-1-D input
- `cml/k-fold-cross-validation/p02-cross-validate`: mutant 6bc39e4dc5af1754: guard uses 'and' instead of 'or'. Differs only for non-1-D index arrays (outside C4 domain); empty 1-D arrays still raise ValueError. No clause covers non-1-D index input
- `cml/k-fold-cross-validation/p02-cross-validate`: mutant 7c7de4b3694d58a1: guard compares idx.size == -1, which is never true. Empty 1-D arrays still raise ValueError from min() (or from the integer check for float lists), so C4/C7 raise-type clauses still hold; only the message text differs
- `cml/k-means/p01-lloyd-kmeans`: mutant 990a07135766be92: labels returned as int32 instead of int64, still an integer array of shape (n,), no clause fixes the width
- `cml/lagrange-multipliers/p01-constrained-quadratic-min`: mutant 7f894e4ebea6e2ed: symmetry check uses >= instead of >, differs only when the asymmetry equals exactly 1e-10 times the scale; no clause states that threshold, C1 says only 'symmetric up to rounding'
- `cml/lagrange-multipliers/p01-constrained-quadratic-min`: mutant 68bb47c946e74115: symmetry tolerance scale uses max(Q) instead of max(abs(Q)); outputs differ only for a near-symmetric Q at the 1e-10 relative rounding boundary, and the Contract gives no numeric tolerance
- `cml/lasso-and-elastic-net/p01-lasso-coordinate-descent`: mutant aa2bdde9f8858386: stop test '<' changed to '<=' (line 68, C7 'below tol'); differs only when the largest change equals tol exactly, a float tie at an exact threshold, so no test
- `cml/leakage-and-walk-forward-validation/p02-expanding-target-mean`: mutant 5d7b686fc90666c8 takes the other of two equal update formulas when a target is zero, only rounding changes, measured at most 1.6e-16 of the largest target per case against tolerance 1e-9, no clause covers rounding
- `cml/logistic-regression/p01-l2-logistic-fit`: mutant b8b8e1ee7528f2e1: stop test '<=' changed to '<' (line 80); differs only when the largest gradient entry equals tol exactly, a float tie at an exact threshold, so no test
- `cml/logistic-regression/p01-l2-logistic-fit`: mutant 93b7199842c23cb4: line-search objective uses a sum instead of a mean (line 71); it changes only the step path, and the Contract leaves the solver free, so no clause fixes the path
- `cml/logistic-regression/p01-l2-logistic-fit`: mutant c22e84c205ef1095: line-search rounding slack uses float32 eps (line 98); measured on 5 problems (random, scaled, separable) against the reference: coefficient difference 0 in every case, so no test can miss a tolerance by 10x
- `cml/logistic-regression/p01-l2-logistic-fit`: mutant 13782b159c0f1251: line-search rounding slack constant 32 instead of 16 eps (line 98); measured on the same 5 problems: coefficient difference 0 in every case, so no test can miss a tolerance by 10x
- `cml/matrix-multiplication/p01-matrix-multiply`: mutant 3e6b3e2ae1c1c132: on ragged input the ValueError text names the wrong row length for row 0; no clause covers the message text, clause C3 only promises ValueError
- `cml/matrix-shapes-and-transpose/p01-transpose-matrix`: mutant 7463f3e26f7c9878: on ragged input the ValueError text names the wrong row length for row 0; no clause covers the message text, clause C3 only promises ValueError
- `cml/maximum-likelihood-and-map/p02-beta-bernoulli-map`: mutant 5d83d5852b9e58d5: the b > 1 becomes b >= 1; at b equal to 1 both branches give 1.0 in exact arithmetic, and the change shows only as a last-digit float difference, so no test added
- `cml/maximum-likelihood-and-map/p02-beta-bernoulli-map`: mutant eab24d35a6734886: the a > 1 becomes a >= 1; at a equal to 1 the mutant divides by a value that can round to zero for a very small prior and raises ZeroDivisionError. This is a strict vs non-strict comparison at an exact threshold depending on the last digit of a float, which the protocol excludes, so no test added
- `cml/missing-values-and-outliers/p02-modified-zscore-outliers`: mutant 471dd62d8940c2e7 uses a strict greater on the half-scale overflow test in the median, only the last bit of the midpoint changes, measured 0 differences in 40002 samples, no clause covers rounding
- `cml/missing-values-and-outliers/p02-modified-zscore-outliers`: mutant 15350fb0289893c8 uses the float32 maximum for the half-scale overflow test in the median, only the last bit of the midpoint changes, max relative error 2.2e-16 against tolerance 1e-9, no clause covers rounding
- `cml/missing-values-and-outliers/p02-modified-zscore-outliers`: mutant 844fc0dcc5b8c14f uses a quarter of the maximum for the half-scale overflow test in the median, only the last bit of the midpoint changes, max relative error 2.2e-16 against tolerance 1e-9, no clause covers rounding
- `cml/nearest-neighbour-search-and-knn/p02-knn-regress`: mutant 2c6ae4877ee698c1: weight 2.0/d instead of 1.0/d, scale cancels in the ratio and differs only when 2/d overflows (d near 1e-308), no clause covers it
- `cml/normal-and-multivariate-gaussian/p01-mvn-logpdf`: mutant e2060aa4821ca335: the symmetry check's > becomes >=; differs only when the asymmetry equals the tolerance exactly (strict vs non-strict at an exact float threshold), so no test added
- `cml/normal-equation/p01-normal-equation-fit`: mutant b99b507ab4be5ffa: a NaN or inf in X alongside finite entries reaches the solver and raises LinAlgError, which is a subclass of ValueError, so the Raises bullet is met by type; no clause names the message, so no test
- `cml/precision-recall-and-f1/p01-precision-recall-averaging`: mutant 5ea82b72c31846bc (classes left as an unsorted set): only the order of float sums in macro and weighted changes, in the last bits, which the Contract leaves free, no clause covers it
- `cml/precision-recall-and-f1/p02-best-f1-threshold`: mutant 89110e0163867286 (labels cast to int32): counts could wrap only past 2 to the 31st samples, and the Contract states no count size or exactness clause, no clause covers it
- `cml/principal-component-analysis/p01-pca-via-svd`: mutant 121dc7072167746f: the non-finite guard is removed, so NaN or inf reaches the SVD and raises LinAlgError (a ValueError subclass) with another message, no clause covers the message or the exact class
- `cml/principal-component-analysis/p01-pca-via-svd`: mutant 38913a2c89a31dbd: the non-finite guard fires only when every entry is non-finite, so a partly NaN matrix raises LinAlgError (a ValueError subclass) from the SVD, no clause covers it
- `cml/principal-component-analysis/p01-pca-via-svd`: mutant 97c9a51c44f56752: the tie band test becomes strict, so only an entry exactly at the band edge (a float-exact tie) changes sign choice, no clause covers it
- `cml/principal-component-analysis/p02-components-for-variance`: mutant 354fc28ebccfdfb1: the non-finite guard is removed, so NaN or inf reaches the SVD and raises LinAlgError (a ValueError subclass) with another message, no clause covers the message or the exact class
- `cml/principal-component-analysis/p02-components-for-variance`: mutant 0bb1dadfff103194: the non-finite guard fires only when every entry is non-finite, so a partly NaN matrix raises LinAlgError (a ValueError subclass) from the SVD, no clause covers it
- `cml/principal-component-analysis/p02-components-for-variance`: mutant 64ca065542ad2446: the margin comparison becomes strict, so only a cumulative ratio exactly at threshold minus the margin changes count (a float-exact tie), no clause covers it
- `cml/probability-basics/p01-marginals-and-conditional`: mutant 2aa97a69361fc4b6: the sum check uses >= instead of >, so it differs only when the sum is off by exactly 1e-9, a float tie at the threshold; no clause covers an exact tie
- `cml/random-forests-and-feature-importance/p02-bootstrap-oob-masks`: mutant 6269163967256986 (line 22): in_bag dtype int64 becomes int32; values are identical for any realistic n, and C5 only says 'integer array', so no clause covers the dtype, no test
- `cml/ranking-metrics/p01-ndcg-at-k`: mutant 461e10beb815e26f: empty-input guard removed; empty input still raises ValueError (numpy bincount length error), only the message differs; clause C8 holds
- `cml/ridge-regression-and-weight-decay/p01-ridge-fit`: mutant a096325643ab2f73: yc = y + y_mean instead of y - y_mean; the centred columns sum to zero, so only rounding differs (max relative coef diff 3e-9 over 3000 random fits, from ill-conditioned cases); no clause covers rounding
- `cml/rolling-statistics-stationarity-and-autocorrelation/p01-rolling-mean-std`: mutant 6b5947e3a9e4fcb2: line 49 raw_new + shift changes only the internal scale; results agree with the reference up to rounding (max rel 5.7e-10, inside the test tolerance of 1e-9); no clause covers rounding
- `cml/rolling-statistics-stationarity-and-autocorrelation/p01-rolling-mean-std`: mutant d5f62648e68565ac: line 51 max becomes min in the overflow branch changes only the internal scale; results agree up to rounding (max rel 1.6e-16 over 22 cases); no clause covers rounding
- `cml/rolling-statistics-stationarity-and-autocorrelation/p01-rolling-mean-std`: mutant efa3c8e1e05ef243: line 51 drops abs on shift in the overflow branch changes only the internal scale; results agree up to rounding (max rel 1.6e-16 over 22 cases); no clause covers rounding
- `cml/rolling-statistics-stationarity-and-autocorrelation/p01-rolling-mean-std`: mutant 69d532dc9d9dc949: line 51 drops abs on raw_new in the overflow branch changes only the internal scale; results agree up to rounding (max rel 0 over 22 cases); no clause covers rounding
- `cml/rolling-statistics-stationarity-and-autocorrelation/p02-acf`: mutant 974efe1fbfe76f66: the fallback divides by max(values) instead of max(abs(values)); the later normalisation removes the scale and the outputs agree with the reference (worst abs error 0 over 25 cases); no clause covers it
- `cml/rolling-statistics-stationarity-and-autocorrelation/p02-acf`: mutant 5663658aeb9346de: the anchor is the second value instead of the first; the centring removes the constant, so outputs differ by rounding only (worst abs error 2.2e-16); no clause covers rounding
- `cml/rolling-statistics-stationarity-and-autocorrelation/p02-acf`: mutant 8d9eb66e372675b3: the anchor is the last value instead of the first; outputs differ by rounding only (worst abs error 1.7e-16); no clause covers rounding
- `cml/rolling-statistics-stationarity-and-autocorrelation/p02-acf`: mutant 58f6d4433c496dd9: adding the first value instead of subtracting it leaves a centring error at offset 1e9 (worst abs error 1.7e-7); a plain correct centring has 3.7e-8 there, so no test at a tolerance a correct solution passes can separate them (rule f)
- `cml/shuffling-splitting-and-batching-data/p01-stratified-train-test-split`: mutant c23bc3809c3bafec: the n type and minimum guard is removed; a string n then raises TypeError instead of ValueError, and n outside the integer domain has no clause, no clause covers it
- `cml/shuffling-splitting-and-batching-data/p01-stratified-train-test-split`: mutant b0a314e141df1eb3: the n guard is narrowed to bool and non-integer n; a string n then raises TypeError instead of ValueError, and n outside the integer domain has no clause, no clause covers it
- `cml/shuffling-splitting-and-batching-data/p02-minibatch-index-schedule`: mutant 7f1be2f61110a6e4: the n type and minimum guard is removed; n=True returns a schedule and a non-integer n raises AxisError, and bool or non-integer n is outside the integer domain, no clause covers it
- `cml/shuffling-splitting-and-batching-data/p02-minibatch-index-schedule`: mutant 92725bf93e7a758e: the n guard is narrowed to bool and non-integer n; n=True returns a schedule instead of ValueError, and bool or non-integer n is outside the integer domain, no clause covers it
- `cml/sigmoid-and-log-odds/p02-logit`: mutant f157e71a288f3fd4: m < 0.25 became m <= 0.25; differs by one last bit at m = 0.25 only; no clause label covers last-bit differences
- `cml/sigmoid-and-log-odds/p02-logit`: mutant 3c62e2236341175d: p < 0.5 became p < 1.0; loses relative precision for p near 1; the relative-precision sentence is in problem.md body text, not in clauses output, and no clause label is broken, so no test added
- `cml/sigmoid-and-log-odds/p02-logit`: mutant 987f09dac70f2af0: m < 0.25 became m < 0.5; loses relative precision near one half (about 1e-13 relative error versus 1e-16 for the reference); the relative-precision sentence is in problem.md body text, not in clauses output, and no clause label is broken, so no test added
- `cml/singular-value-decomposition/p01-best-rank-k`: mutant 03b9380776484b1c: non-2-D input skips the explicit check; it still raises ValueError (unpacking shape), only the message text differs; clause C5 holds, no clause covers the message
- `cml/singular-value-decomposition/p01-best-rank-k`: mutant 1b64019c15de0fe4: finiteness check uses any instead of all; NaN input still raises ValueError (numpy LinAlgError, a ValueError subclass), only the message differs; clause C5 holds
- `cml/singular-value-decomposition/p01-best-rank-k`: mutant c091d88c8a77ee01: finiteness check removed; NaN or inf input still raises ValueError (numpy LinAlgError), only the message differs; clause C5 holds
- `cml/singular-value-decomposition/p02-pseudo-inverse`: mutant 4379dfd4e1bed54b: finiteness guard removed; NaN or inf input still raises ValueError (numpy LinAlgError, a ValueError subclass), only the message differs; clause C5 holds
- `cml/singular-value-decomposition/p02-pseudo-inverse`: mutant 74474c9ed062357b: 2-D guard removed; non-2-D input still raises ValueError (unpacking the shape), only the message differs; clause C5 holds
- `cml/singular-value-decomposition/p02-pseudo-inverse`: mutant 99d2422943cd3515: empty guard reduced to a check on columns only; empty input still raises ValueError (zero-size reduction in numpy), only the message differs; clause C5 holds
- `cml/singular-value-decomposition/p02-pseudo-inverse`: mutant 8e80fef6867c5324: empty guard changed from or to and; empty input still raises ValueError (zero-size reduction in numpy), only the message differs; clause C5 holds
- `cml/singular-value-decomposition/p02-pseudo-inverse`: mutant 9fa80ccf2510eb86: finiteness check uses any instead of all; NaN input still raises ValueError (numpy LinAlgError, a ValueError subclass), only the message differs; clause C5 holds
- `cml/singular-value-decomposition/p02-pseudo-inverse`: mutant 5e272cbc9e8789d9: empty guard constant changed to m < 0; empty input still raises ValueError (zero-size reduction in numpy), only the message differs; clause C5 holds
- `cml/singular-value-decomposition/p02-pseudo-inverse`: mutant ef245cd84d959cd8: empty guard constant changed to n < 0; empty input still raises ValueError (zero-size reduction in numpy), only the message differs; clause C5 holds
- `cml/stochastic-and-mini-batch-gradient-descent/p01-minibatch-sgd`: mutant 373a59942d898e86: the explicit 'X must be 2-D' check is removed. A 1-D or 3-D X still raises ValueError from the shape unpacking on the next line, so only the message text changes; the Contract (C8) promises the ValueError type, not its text, so no clause is broken and no test
- `cml/t-tests-and-anova/p01-welch-t-test`: mutant 55a530979abefdfc: centres on the sum instead of the mean, which changes only rounding (vs scipy 8e-7 at offset 1e8, reference 1.5e-8, existing offset test rtol 1e-6, so not 10x); no clause covers it
- `cml/t-tests-and-anova/p01-welch-t-test`: mutant 4e1f274dd9b06cfc: scale drops abs on the second sample, changes only rounding (max relative diff from reference 4.6e-14 on random inputs, far below rtol 1e-9); no clause covers it
- `cml/t-tests-and-anova/p01-welch-t-test`: mutant 5f8387137a661a6b: scale drops abs on the first sample, changes only rounding (max relative diff from reference 8.4e-13 on random inputs, far below rtol 1e-9); no clause covers it
- `cml/vector-operations/p01-vector-projection`: mutant ce2f03caf5b632a8 changes or to and in the 1-D check, so a scalar a with a 1-D b raises IndexError instead of ValueError, the Raises bullet only lists shape conditions and a scalar is a wrong kind of argument that the bullet does not name, so no test
- `dl/a-complete-training-loop/p01-fit-with-seeded-batches`: mutant 33a03eb1f8826936: the row check looks at the feature count instead of the row count, so an empty input is not rejected and zero features are rejected; the contract only states n >= 1 and no raise clause covers either case
- `dl/a-complete-training-loop/p01-fit-with-seeded-batches`: mutant defc43a369940476: the row check is written so it can never fire, so an empty input is accepted and fails later with a different error; the contract only states n >= 1 and no raise clause covers it
- `dl/adam/p01-adam-step`: mutant 59a3c0fd0ed4333a: removes the torch-tensor type check on params and grads, so non-tensor inputs reach the shape check and fail differently; the stated domain is lists of floating-point tensors, so no clause covers it
- `dl/adam/p01-adam-step`: mutant 4da7fd75e6f0b5e3: the tensor type check on params and grads becomes an and, so a mix of tensor and non-tensor inputs gets past it; the stated domain is lists of floating-point tensors, so no clause covers it
- `dl/adam/p02-amsgrad-step`: mutant 70c8a48a81eb72e0: removes the torch-tensor type check on params and grads, so non-tensor inputs fail differently; the stated domain is lists of floating-point tensors and no clause covers non-tensor inputs
- `dl/adam/p02-amsgrad-step`: mutant 5a193bd63c99cb6d: the tensor type check on params and grads becomes an and, so a mix of tensor and non-tensor inputs passes; the stated domain is lists of floating-point tensors and no clause covers it
- `dl/adamw-decoupled-weight-decay/p01-adamw-step`: mutant 19505aa15767699d: a non-tensor param or grad no longer raises ValueError early (AttributeError or other error instead), and only non-tensor inputs differ, which the Contract does not cover; no clause covers it
- `dl/adamw-decoupled-weight-decay/p01-adamw-step`: mutant bd6a01400e372b22: a mixed tensor and non-tensor pair no longer raises ValueError early, and only non-tensor inputs differ, which the Contract does not cover; no clause covers it
- `dl/batch-normalization/p01-batch-norm-forward`: mutant 089f0a20a898c2da: the zero-feature check (n_features == 0) becomes a check against -1, so a batch with zero features is accepted instead of rejected; the stated domain is C >= 1 and no clause covers a zero-feature input
- `dl/convolution-forward-pass/p01-conv2d-forward`: mutant dc720fe5f9662960: removing the zero-size weight guard changes behaviour only for a weight with a zero axis (C_out, kH or kW = 0), which the Contract's weight shape requires to be >= 1 and which no raise clause lists; no clause covers it
- `dl/layer-normalization/p01-layer-norm-forward`: mutant 07d2c0078a5a3149: the default value of eps in the signature changes, so a call that omits eps gives slightly different outputs; the default appears only in the signature line of problem.md, and no labelled clause (C1 to C7, K1, K2) covers the default
- `dl/multi-head-attention/p01-self-attention-heads`: mutant 1ee6306d3f570529: key bias subtracted instead of added; in exact arithmetic the output does not depend on b_k (a per-row constant shift cancels in softmax); measured max abs diff 4.5e-12 vs reference at bias scale 10, below tolerance 1e-9; rounding only, no test
- `dl/positional-encodings/p01-sinusoidal-table`: mutant 9c2631a248e4b31b: pos arange in float32 instead of float64 (precision only); measured max abs diff 0.0 vs reference over max_len 0..4096 and d_model 2..128, dtype and shape equal; misses tolerance by factor 0; no test
- `dl/positional-encodings/p01-sinusoidal-table`: mutant 7cc46c1965ecc86a: d_model guard <=1 instead of <=0; only differs at d_model 1, which the odd-width check already rejects with the same message, so same behaviour; recorded here as unpromised, no test
- `dl/positional-encodings/p02-rope-rotations`: mutant 14b2a3f0c5cd2977: position arange in float32 instead of float64 (precision only); measured max abs diff 0.0 vs reference over float32/float64 inputs, L 0..4096, d 2..128; dtype and shape equal; misses tolerance by factor 0; no test
- `dl/rmsnorm/p01-rmsnorm-forward`: mutant 64b8ae9f66834b62: default eps in the signature changes from 1e-6 to 2e-6, which changes outputs only for default-eps calls; no clause covers the default value (C3 says eps > 0 only)
- `dl/scaled-dot-product-attention/p01-scaled-dot-product-attention`: mutant 802af07f36180ac7: the tensor-type check on q, k, v is skipped, so a non-tensor input fails with another exception type instead of ValueError; the contract only promises ValueError for non-floating tensors (C11), not for non-tensor inputs, which are outside the stated domain (C1-C3), no clause covers it
- `dl/scaled-dot-product-attention/p01-scaled-dot-product-attention`: mutant efb036c541fc3970: the check that k and v share the number of positions S is skipped, so mismatched S fails later with another error; S-mismatch is outside the stated domain (C3) and C11 does not list it as a ValueError case, no clause covers it
- `dl/scaled-dot-product-attention/p01-scaled-dot-product-attention`: mutant 56779ae40fdc6d0d: the tensor-type check uses any instead of all, so a mixed list and tensor input passes the check and fails later with AttributeError; non-tensor input is outside the stated domain and C11 does not promise ValueError for it, no clause covers it
- `dl/sgd-with-momentum/p01-momentum-step`: mutant 79c77e05e38526c9: the tensor type guard is removed, so a non-tensor param or grad now fails with an AttributeError instead of a ValueError; the contract types params and grads as tensors (C1) and C6 lists no raise for non-tensors, so no clause covers it
- `dl/sgd-with-momentum/p01-momentum-step`: mutant 99a0fb66f2e800b9: the tensor type guard uses and in place of or, so a mixed non-tensor param or grad passes the guard and fails with an AttributeError; the contract types params and grads as tensors (C1) and C6 lists no raise for non-tensors, so no clause covers it
- `dl/sgd-with-momentum/p02-nesterov-weight-decay`: mutant eb8d142789978d13: the tensor type guard is removed, so a non-tensor param or grad now fails with an AttributeError instead of a ValueError; the contract types params and grads as tensors (C1) and C6 lists no raise for non-tensors, so no clause covers it
- `dl/sgd-with-momentum/p02-nesterov-weight-decay`: mutant b4890a3ad2bb574f: the tensor type guard uses and in place of or, so a mixed non-tensor param or grad passes the guard and fails with an AttributeError; the contract types params and grads as tensors (C1) and C6 lists no raise for non-tensors, so no clause covers it
- `dl/training-an-mlp-from-scratch/p01-train-two-layer-mlp`: mutant e593f8cb6978289e: inputs cast to float32 instead of float64. The rounding moves every result by about 1e-7 relative, inside the tolerance of the tests (atol 1e-8, rtol 1e-6). A test written for it rejected the mutant with a 4 percent margin only, so the coordinator withdrew it before commit: no robust test exists at this tolerance
- `dl/transformer-block/p01-pre-norm-causal-block`: mutant 2d67cc571ff7ed2c: removing the check that the feed-forward width F is at least 1 lets a zero-width W_1 through instead of raising. F >= 1 is a domain rule; no clause promises a ValueError for it, and the Raises list in the contract does not name F
- `inf/data-quality-and-drift-detection/p02-ks-drift-test`: mutant 66eb879036d519d5: the lambda < 1 test becomes <=, so only lambda exactly 1 changes branch and the two series agree to float rounding, no clause covers it
- `inf/data-quality-and-drift-detection/p02-ks-drift-test`: mutant 81a24a7f6b04c125: the series stop test becomes <=, so only a term exactly equal to 1e-17 changes and the skipped term is below float resolution, no clause covers it
- `inf/data-quality-and-drift-detection/p02-ks-drift-test`: mutant 85c6321585bb30e8: break becomes continue in the alternating loop, so only terms below 1e-17 are added and the result changes below float resolution, no clause covers it
- `inf/kv-cache-size-arithmetic/p01-kv-cache-bytes`: mutant fe1a53291b935069: removes the dtype str type check, so a non-str dtype raises ValueError instead of TypeError; the Contract (C4) only says dtype is a str and C6 lists no dtype error, and the change is outside the stated domain, no clause covers it
- `inf/kv-cache-size-arithmetic/p01-kv-cache-bytes`: mutant 041c60f58d91a121: head_dim minimum 1 becomes 0, so head_dim 0 returns 0 instead of ValueError; the Contract gives no numeric minimum for head_dim (C7 says only its minimum), no clause covers it
- `inf/kv-cache-size-arithmetic/p01-kv-cache-bytes`: mutant cf8c033d834b0c15: num_kv_heads minimum 1 becomes 0, so num_kv_heads 0 returns 0 instead of ValueError; the Contract gives no numeric minimum for num_kv_heads (C7 says only its minimum), no clause covers it
- `inf/kv-cache-size-arithmetic/p02-max-batch-after-weights`: mutant 67c8b628cf2ad39a: num_layers minimum 1 becomes 0, so num_layers 0 divides by zero instead of ValueError; the Contract gives no numeric minimum for num_layers (C1 says only its minimum), no clause covers it
- `inf/kv-cache-size-arithmetic/p02-max-batch-after-weights`: mutant c88660d0971d0d94: head_dim minimum 1 becomes 0, so head_dim 0 divides by zero instead of ValueError; the Contract gives no numeric minimum for head_dim (C1 says only its minimum), no clause covers it
- `inf/kv-cache-size-arithmetic/p02-max-batch-after-weights`: mutant dc2ca9fbf904e4e1: num_kv_heads minimum 1 becomes 0, so num_kv_heads 0 divides by zero instead of ValueError; the Contract gives no numeric minimum for num_kv_heads (C1 says only its minimum), no clause covers it
- `inf/speculative-decoding-speedup-math/p02-accept-or-resample`: mutant a3aa5bce4158a9c6: dtype guard removed, so bool, complex and numeric-string arrays are no longer rejected. Only non-real inputs differ; no clause covers them as a real numeric array-like with a required ValueError, and Raises names no such case in words
- `inf/speculative-decoding-speedup-math/p02-accept-or-resample`: mutant 55942108aef84be8: empty input check changed from < 1 to < 0. Empty input still raises ValueError (now from the sum check), so only the message differs; no clause covers the message text
- `inf/speculative-decoding-speedup-math/p02-accept-or-resample`: mutant 984582e5ea5d19ac: sum tolerance 1e-9 widened to 2e-9. A precision threshold that differs by a factor of 2, under the factor-of-ten rule for precision changes, so no test
- `inf/ttft-itl-and-throughput/p01-latency-percentiles`: mutant 4523a312869c05e8: the type check on requests is removed, so a tuple of valid requests is accepted and returns a result; the Contract names a list and states no error for other top-level types, so no clause covers it
- `inf/ttft-itl-and-throughput/p02-throughput-and-goodput`: mutant 15ccf7152f57d0c9: the type check on requests is removed, so a tuple of valid requests is accepted; the wrong-type rule names request fields and SLO values, not a top-level non-list, so no clause covers it
- `llm/bradley-terry-reward-models/p01-reward-pair-loss-and-accuracy`: mutant 7a9a0e08aded7f3e: dtype guard uses 'and' where 'or' stood, so for a float/int pair the ValueError text changes from 'must be floating-point' to 'must share one dtype'. Still ValueError for every input; no clause names the message; no test added
- `llm/bradley-terry-reward-models/p01-reward-pair-loss-and-accuracy`: mutant 86acac8075e29823: ndim guard uses 'and' where 'or' stood, so for 1-D vs 2-D or 0-D inputs the ValueError text changes to 'must have the same shape'. Still ValueError for every input; C5 promises the type only; no test added
- `llm/chat-templates-and-tool-call-formats/p02-parse-tool-call`: mutant ea6a248b7993d919: drops the exactly-one-tag count, so a call tag quoted inside a JSON string is accepted; the format is one JSON object between the outer tags and no sentence forbids a tag inside a string, so no clause is broken
- `llm/chat-templates-and-tool-call-formats/p02-parse-tool-call`: mutant c52556c410207db5: count guard with and instead of or, so a call tag quoted inside a JSON string is accepted; no clause forbids a tag inside a string, so no clause is broken (listed twice in survivors.json, one key)
- `llm/chat-templates-and-tool-call-formats/p01-render-chat-template`: mutants 0192afa3c8aafee4 and 5794c9d638a59a8e: without the isinstance check a role object that is not a str but compares equal to a role name (collections.UserString) is accepted and rendered like the str. Ordinary non-str roles are already rejected by the existing tests. Two tests built on that look-alike were withdrawn by the coordinator before commit: an adversarial input, not a realistic mistake
- `llm/direct-preference-optimization-dpo/p01-sequence-log-probs-with-response-mask`: mutant 38d6671f63e8c4b4: the zero-length guard tests seq_len == -1, so a zero-length sequence fails with IndexError instead of ValueError. Zero length is outside the stated domain (T >= 1) and no Raises bullet names it; no test added
- `llm/direct-preference-optimization-dpo/p01-sequence-log-probs-with-response-mask`: mutant 8fee835fe4cb79cf: the zero-vocabulary guard tests vocab == -1, so an empty vocabulary gets a different ValueError message. Vocabulary 0 is outside the stated domain (V >= 1) and the Raises bullet names only the type; no test added
- `llm/direct-preference-optimization-dpo/p02-dpo-loss-and-implicit-rewards`: mutant ca5778b3db0e0a4e: the shape and dtype reference becomes the second argument instead of the first. Every argument is still compared with that reference, so the same inputs raise ValueError and only the message can differ. Clause C5 promises the type only, so no clause is broken. No test added. Random probes showed no change in raise or return.
- `llm/direct-preference-optimization-dpo/p02-dpo-loss-and-implicit-rewards`: mutant 097237094d7fcb6b: the shape and dtype reference becomes the last argument instead of the first. Same argument as ca5778b3db0e0a4e: raise-or-not is unchanged and only the message can differ. Clause C5 promises the type only. No test added. Random probes showed no change in raise or return.
- `llm/direct-preference-optimization-dpo/p01-sequence-log-probs-with-response-mask`: mutants ad96f7bf9301426b and 16e8dd57f15c7c33: without the type check a logits argument that is not a tensor crashes with AttributeError. C6 lists logits that is not 3-D or not floating-point, and names a tensor only for input_ids and response_mask. A test for a nested list was withdrawn by the coordinator before commit, in line with the other optimizer and attention problems where a non-tensor argument is outside the stated domain
- `llm/lora-adapters/p01-lora-linear-forward-and-merge`: mutant 6772c159fee0abbc: default merge flipped from False to True. Only rounding differs. Measured in float64 with a cancelling bias: thin-path error 1.7e-8 and merged-path error 6.1e-8 at W scale 1e8, but the reference already misses the 1e-9 tolerance there; at W scale 1e6 the merged path is 0.2x tolerance, so no input where the reference passes and the mutant misses by 10x. No test added.
- `llm/pass-k-and-majority-voting/p01-unbiased-pass-at-k`: mutant 7e985542fc42235d: for c = 0 the return value becomes -0.0 instead of 0.0 (sign bit only, equal under ==), no clause covers the sign of zero
- `llm/temperature-top-k-and-top-p-sampling/p01-filtered-next-token-distribution`: mutant f1c4a2f921143e33 (line 67 fallback V - 1 to V - 2): drops the least likely token only when the float cumulative sum falls short of top_p. No clause gives a rule for that case (C5 is silent when no prefix reaches top_p, and in exact arithmetic it never occurs). Only float rounding reaches it, so no test.
- `llm/temperature-top-k-and-top-p-sampling/p02-sample-with-a-generator`: mutant 2f88fcf4a114e31d: the index check in the rounding guard becomes an inclusive check, so it returns the length when the cdf total falls short of the draw. The Contract pins this rule in C4 but says it cannot be tested by draws. Reaching it needs the draw above the cdf total, which depends on how the draw is taken, so no test.
- `llm/temperature-top-k-and-top-p-sampling/p02-sample-with-a-generator`: mutant 889834845597b6f2: the sum tolerance check becomes inclusive, so it differs only when the sum error is exactly the tolerance in float. A float threshold tie, so no test.
- `llm/temperature-top-k-and-top-p-sampling/p02-sample-with-a-generator`: mutant c8e55639643ea14a: the rounding guard returns a token with zero probability. It is reached only when the draw exceeds the cdf total, which depends on how the draw is taken, so no test.
- `llm/temperature-top-k-and-top-p-sampling/p02-sample-with-a-generator`: mutant ee4fd46a4ac2bbff: the rounding guard accepts zero-probability tokens, so it can return the last index even when its probability is zero. Reached only when the draw exceeds the cdf total, which depends on how the draw is taken, so no test.
- `llm/temperature-top-k-and-top-p-sampling/p02-sample-with-a-generator`: mutant b4d717fec434ee90: the search becomes left-sided, so it differs only when the draw equals a cdf value exactly. A float tie that also depends on the draw, so no test.
- `llm/temperature-top-k-and-top-p-sampling/p02-sample-with-a-generator`: mutant 6d240b4ab8f2a761: the draw is cast to float32, so it changes the draw only by float32 rounding and flips a sample only when the draw lies within about one part in ten million of a cdf boundary. A float tie that depends on the draw, not reachable by a seeded test, so no test.
- `llm/temperature-top-k-and-top-p-sampling/p02-sample-with-a-generator`: mutant 1df4478289b4b0fd: the rounding guard looks for a probability above one, finds none and raises an error. Reached only when the draw exceeds the cdf total, which depends on how the draw is taken, so no test.
- `llm/temperature-top-k-and-top-p-sampling/p02-sample-with-a-generator`: mutant a7d6ebeff69ee9d3: the rounding guard returns the second-to-last positive-probability token instead of the last one. Reached only when the draw exceeds the cdf total, which depends on how the draw is taken, so no test.
- `math/inverse-transform-sampling/p02-sample-discrete`: mutant e2cdcaab9d32482c: the sum tolerance check becomes greater or equal, so a sum error of exactly 1e-9 is rejected, which is a float tie at the exact threshold, no clause fixes the tie
- `math/inverse-transform-sampling/p02-sample-discrete`: mutant 009202062e89a7f4: the empty check compares size to minus 1, so empty probs are rejected later by the sum check with ValueError and a different message, no clause covers the message
- `math/monte-carlo-simulation/p01-mc-estimate`: mutant 8857f1319957d738: the upper bound of the confidence check becomes less than or equal to 1, so confidence 1.0 gets past the check and fails later with a StatisticsError that is a subclass of ValueError with a different message, no clause covers the message or the exact error class
- `math/monte-carlo-simulation/p01-mc-estimate`: mutant b0bb789a294cb9a9: the upper bound of the confidence check becomes 2, so confidence 1.0 and 1.5 get past the check and fail later with a StatisticsError that is a subclass of ValueError with a different message, no clause covers the message or the exact error class
- `rl/action-value-estimates/p01-incremental-action-values`: mutant 249e01b1a834e70b keeps counts as int32 so the sample average goes wrong after 2 to the 31 pulls of one action, no clause covers it
- `rl/epsilon-greedy-exploration/p01-epsilon-greedy-action`: mutant edcf1f4f836870ab lets an empty q_values array through the check, so one rng random draw happens before a ValueError with a different message, C7 only promises ValueError and no clause says whether an invalid call draws, no clause covers it
- `rl/mdps-and-discounted-returns/p02-iterative-policy-evaluation`: mutant d2cccf897f3750ac: ndim guard on P/R/pi removed; still ValueError (unpack or shape check), only the message differs; no clause covers the message
- `rl/mdps-and-discounted-returns/p02-iterative-policy-evaluation`: mutant ecaa1eca9e0a13a9: shape guard on S, A, last removed; still ValueError (numpy max-of-empty, matmul or row-sum check), only the message differs; no clause covers the message
- `rl/mdps-and-discounted-returns/p02-iterative-policy-evaluation`: mutant 6ef43bc42055744f: guard changed from or to and; still ValueError on every invalid input, only the message differs; no clause covers the message
- `rl/mdps-and-discounted-returns/p02-iterative-policy-evaluation`: mutant 55fb58ed26fe0b7a: S < 1 guard changed to S < 0; S=0 still ValueError (numpy max of empty), only the message differs; no clause covers the message
- `rl/mdps-and-discounted-returns/p02-iterative-policy-evaluation`: mutant aebb997e1e19cbb0: default tol changed from 1e-10 to 2e-10; default-call results differ by about 5e-10; no clause states the default value, and the difference is a stopping-precision effect
- `rl/reinforce/p01-reinforce-loss`: mutant add7ce762d203964: logits ndim guard removed; still ValueError from the tuple unpack, only the message differs; no clause covers the message
- `rl/sarsa-and-expected-sarsa/p01-sarsa-and-expected-sarsa-updates`: mutant 5bba5a20dca67c3c: alpha/gamma/epsilon of a non-real type (for example a numeric string) is accepted instead of rejected. C11 lists only out-of-range values for these, and C6 does not say non-real values are refused, so no clause covers it
- `rl/sarsa-and-expected-sarsa/p01-sarsa-and-expected-sarsa-updates`: mutant eaa3fa3a7a59e8e3: reward of a non-real type (for example a numeric string) is accepted instead of rejected. C11 lists only a non-finite reward, and C3 does not say non-real values are refused, so no clause covers it
- `rl/sarsa-and-expected-sarsa/p01-sarsa-and-expected-sarsa-updates`: mutant 376e656e03bfa37e: a bool alpha/gamma/epsilon is accepted instead of rejected. The Contract does not say bool is refused, so no clause covers it
- `rl/sarsa-and-expected-sarsa/p01-sarsa-and-expected-sarsa-updates`: mutant d4eb9cbcdff47a92: a bool reward is accepted instead of rejected. The Contract does not say bool is refused, so no clause covers it
- `safe/split-conformal-prediction/p02-conformal-prediction-sets`: mutant be5e9e42d8fef9f2: a row whose sum deviates from 1 by exactly 1e-9 is accepted instead of rejected; this is a float boundary tie and the Contract's 'within 1e-9' does not say whether equality is inside, so no test
- `sql/pandas-groupby-aggregate-and-transform/p01-customer-order-features`: mutant 6463698b53cd7646: reset_index drop=False adds an index column, so only an input that already has a column named index raises, but the Contract lists the input columns and says nothing about extra columns, no clause covers it

### no-witness
- `cml/bootstrap-and-permutation-resampling/p02-permutation-test-mean-diff`: reject_group_sizes_swapped.py survives and breaks C4, no test found, the first group takes len b values instead of len a. Both group sizes give the same distribution of the statistic, since the absolute mean difference is symmetric under swapping complementary groups, so no test on the p-value can separate them. An exact-seed test would pin the draw method, which the Contract leaves free
- `cml/choosing-a-tree-split/p01-best-split-threshold`: reject_pairwise_rescan.py survives and breaks K2 (per-candidate recount with vectorised sums, O(n squared)): it passes the 30k-row time budget, and no deterministic assertion can see its cost, so no test added. The K2 bullet is covered by reject_python_pairwise_loop.py
- `dl/adamw-decoupled-weight-decay/p01-adamw-step`: reject_zero_dim_parameter_rejected.py survives and breaks C1, the one test that rejects it raises a bare exception so the gate does not count it, no test added
- `dl/adamw-decoupled-weight-decay/p01-adamw-step`: reject_lr_zero_rejected.py survives and breaks C3, the one test that rejects it raises a bare exception so the gate does not count it, no test added
- `dl/kl-divergence/p01-kl-from-logits`: reject_support_from_exp_underflow.py survives and breaks C4, no test found
- `dl/transformer-block/p01-pre-norm-causal-block`: reject_drops_key_bias.py survives and breaks C3, no test found, dropping b_k adds the same q dot b_k to every score in a row which softmax cancels so no output changes for any input
- `inf/prefill-vs-decode-bottlenecks/p01-serving-step-times`: reject_config_gets_a_scratch_key.py survives and breaks C2, no test found: two add-test runs, the gate reports wrong FAIL (not rejected by the new tests) even though the file does write a key into the config dict when run directly; the existing config-mutation test also passes on it
- `cml/bias-variance-decomposition/p01-bias-variance-from-predictions`: mutant b9e4ad8b0b93d4ee breaks C4 (sum instead of mean in the overflow branch); the only nonzero-mean witness tried failed accept_blind, so no test added
- `cml/confusion-matrix-and-count-based-scores/p01-confusion-matrix-from-labels`: mutant 19fdb3144a4059ec (int64 to int32 counts) breaks C3 only for n above 2 to the 31st; a witness needs over two billion samples, and asserting a dtype would reject valid int32 solutions, so no test added
- `rl/action-value-estimates/p01-incremental-action-values`: mutant 15fa6334e1fd822f casts action indices to int32 and differs from the reference only for indices from 2 to the 31 up, breaks C6, a test would need an array over 16 GB, no test added

### overreach
- `cml/a-b-test-design-and-analysis/p01-sample-size-two-proportions`: tiny nonzero effect test with mde 3e-13 and reference equality, fails accept_blind, reject_zero_effect_tolerance survives
- `cml/handling-class-imbalance/p01-balanced-class-weights`: bare scalar input must raise ValueError, fails accept_blind
- `cml/leakage-and-walk-forward-validation/p02-expanding-target-mean`: accept_sum_then_divide.py: test_finite_repeated_large_targets_do_not_overflow_the_historical_mean, no clause fixes this choice
- `cml/rolling-statistics-stationarity-and-autocorrelation/p01-rolling-mean-std`: accept_plain_welford.py: test_large_and_small_readings_keep_a_representable_sample_spread and test_large_common_offset_over_long_series_keeps_the_spread, no Contract clause fixes the centring or scaling choice (the Contract only says finite reals)
- `dl/gradients-with-framework-autograd/p01-gradient-at-a-point`: accept_detached_view_leaf.py: test_function_writing_into_its_argument_leaves_the_caller_tensor_unchanged, no clause fixes this choice
- `dl/kl-divergence/p01-kl-from-logits`: underflowed probability exp of minus 800 under q zero must give inf, fails accept blind
- `llm/temperature-top-k-and-top-p-sampling/p02-sample-with-a-generator`: zero draw test mocks torch rand, a correct solution that draws with uniform fails it, test withdrawn, wrong solution right false search survives
- `llm/temperature-top-k-and-top-p-sampling/p02-sample-with-a-generator`: one draw test compares with a float64 draw stream, a correct solution that draws in float32 fails it, test withdrawn, wrong solution two draws per call survives
- `cml/bias-variance-decomposition/p01-bias-variance-from-predictions`: mean of models 1e308, -1e308 and 3 (true mean 1) with truth 1: the exact small contribution is lost in a plain sum, fails accept_blind. No retry

### message-defect
- `algo/binary-search/p01-lower-bound`: reject_empty_list_raises.py: test_target_outside_the_range_and_an_empty_list shows kind=bare_exception
- `algo/dynamic-programming/p02-longest-common-subsequence`: reject_memoised_recursion_depth.py: test_long_strings_finish_within_the_time_limit shows kind=bare_exception
- `algo/hash-maps-and-sets/p01-top-k-frequent-tokens`: reject_raises_on_empty.py: test_k_zero_and_empty_input_both_return_an_empty_list shows kind=bare_exception
- `algo/prefix-sums/p02-rectangle-range-sums`: reject_rejects_negative_cells.py: test_negative_values_and_repeated_queries shows kind=bare_exception
- `algo/prefix-sums/p02-rectangle-range-sums`: reject_single_row_query_rejected.py: test_inclusive_rectangles_match_the_worked_table shows kind=bare_exception
- `algo/prefix-sums/p02-rectangle-range-sums`: reject_drops_repeated_answers.py: test_negative_values_and_repeated_queries shows kind=raw_assert
- `algo/prefix-sums/p02-rectangle-range-sums`: reject_empty_grid_raises.py: test_empty_query_collection_and_reversed_bounds shows kind=bare_exception
- `cml/bias-variance-decomposition/p01-bias-variance-from-predictions`: reject_list_input_rejected.py: test_spread_without_bias_gives_variance_and_the_noise_term shows kind=bare_exception
- `cml/covariance-and-correlation/p01-covariance-matrix`: reject_refuses_single_observation.py: test_single_observation_with_ddof_zero_is_a_zero_matrix shows kind=bare_exception
- `cml/descriptive-statistics/p01-welford-mean-variance`: reject_needs_length.py: test_generator_input_gives_the_same_answer_as_a_list shows kind=bare_exception
- `cml/descriptive-statistics/p02-iqr-outlier-fences`: reject_zero_k_rejected.py: test_k_zero_gives_the_quartiles_themselves shows kind=bare_exception
- `cml/descriptive-statistics/p02-iqr-outlier-fences`: reject_rejects_duplicates.py: test_property_agrees_with_numpy_on_any_finite_list shows kind=bare_exception
- `cml/eigenvalues-and-eigenvectors/p01-power-iteration`: reject_rejects_size_one.py: test_matches_eigh_dominant_pair_on_random_symmetric_matrices shows kind=bare_exception
- `cml/eigenvalues-and-eigenvectors/p01-power-iteration`: reject_unscaled_symmetry_tolerance.py: test_symmetry_tolerance_scales_with_large_entries shows kind=bare_exception
- `cml/eigenvalues-and-eigenvectors/p01-power-iteration`: reject_num_iters_one_refused.py: test_a_single_iteration_is_an_allowed_budget shows kind=bare_exception
- `cml/feature-scaling/p01-standard-scaler`: reject_single_training_row_refused.py: test_empty_transform_preserves_feature_axis shows kind=bare_exception
- `cml/feature-scaling/p01-standard-scaler`: reject_empty_values_refused.py: test_empty_transform_preserves_feature_axis shows kind=bare_exception
- `cml/feature-scaling/p02-robust-median-iqr-scaler`: reject_empty_values_refused.py: test_empty_values_keep_the_feature_axis shows kind=bare_exception
- `cml/feature-scaling/p02-robust-median-iqr-scaler`: reject_integer_input_refused.py: test_median_and_iqr_come_from_the_training_rows shows kind=bare_exception
- `cml/gaussian-mixtures-and-em/p01-gmm-em-full-covariance`: reject_k_equal_n_rejected.py: test_initial_state_of_the_two_point_fixture_has_equal_weights_and_closed_form_loglik shows kind=bare_exception
- `cml/gradient-descent/p01-gradient-descent`: reject_list_start_returned_as_is.py: test_one_dimensional_quadratic_reaches_its_minimum shows kind=bare_exception
- `cml/k-fold-cross-validation/p01-kfold-indices`: reject_default_shuffle_on.py: test_ten_positions_in_three_folds_give_consecutive_blocks_with_the_remainder_first shows kind=bare_exception
- `cml/k-means/p01-lloyd-kmeans`: reject_column_mismatch_type_error.py: test_init_with_wrong_number_of_columns_raises_value_error shows kind=bare_exception
- `cml/lagrange-multipliers/p01-constrained-quadratic-min`: reject_list_c_refused.py: test_equal_split_of_a_sum_constraint_with_unit_multiplier shows kind=bare_exception
- `cml/lasso-and-elastic-net/p02-elastic-net-coordinate-descent`: reject_l1_ratio_one_refused.py: test_l1_ratio_one_reduces_to_the_lasso_solution shows kind=bare_exception
- `cml/matrix-inverse-and-rank/p02-matrix-rank`: reject_rejects_zero_tol_explicitly.py: test_zero_matrix_has_rank_zero_and_wide_shape_is_handled shows kind=bare_exception
- `cml/matrix-multiplication/p01-matrix-multiply`: reject_empty_matrix_refused.py: test_zero_width_inputs_give_empty_results shows kind=bare_exception
- `cml/maximum-likelihood-and-map/p01-gaussian-mle`: reject_rejects_integer_input.py: test_mean_and_biased_covariance_of_three_points shows kind=bare_exception
- `cml/nearest-neighbour-search-and-knn/p01-knn-classify`: reject_xtrain_rejects_lists.py: test_majority_vote_among_the_three_nearest_rows shows kind=bare_exception
- `cml/nearest-neighbour-search-and-knn/p02-knn-regress`: reject_xtrain_rejects_lists.py: test_uniform_weights_average_the_k_nearest_targets shows kind=bare_exception
- `cml/normal-and-multivariate-gaussian/p01-mvn-logpdf`: reject_list_input_needs_astype.py: test_standard_normal_at_zero_is_minus_half_log_two_pi shows kind=bare_exception
- `cml/normal-equation/p01-normal-equation-fit`: reject_list_input_only.py: test_exact_line_recovers_intercept_and_slope shows kind=bare_exception
- `cml/normal-equation/p01-normal-equation-fit`: reject_coef_scalar_for_one_column.py: test_exact_line_recovers_intercept_and_slope shows kind=bare_exception
- `cml/sampling-distributions-standard-errors-and-cis/p01-mean-confidence-interval`: reject_integer_input.py: test_example_interval_for_one_to_five_uses_t_with_four_degrees_of_freedom shows kind=bare_exception
- `cml/sampling-distributions-standard-errors-and-cis/p01-mean-confidence-interval`: reject_n_below_three.py: test_matches_student_t_reference_on_seeded_samples shows kind=bare_exception
- `cml/shuffling-splitting-and-batching-data/p02-minibatch-index-schedule`: reject_zero_epochs_raises.py: test_zero_epochs_gives_an_empty_schedule shows kind=bare_exception
- `cml/stochastic-and-mini-batch-gradient-descent/p01-minibatch-sgd`: reject_rejects_single_row.py: test_any_batch_size_matches_the_reference shows kind=bare_exception
- `cml/stochastic-and-mini-batch-gradient-descent/p01-minibatch-sgd`: reject_excludes_full_batch.py: test_single_full_batch_step_matches_hand_computation shows kind=bare_exception
- `cml/stochastic-and-mini-batch-gradient-descent/p01-minibatch-sgd`: reject_rejects_zero_epochs.py: test_zero_epochs_returns_zero_weights shows kind=bare_exception
- `cml/t-tests-and-anova/p01-welch-t-test`: reject_one_constant_invalid.py: test_one_constant_sample_is_valid_and_matches_scipy shows kind=bare_exception
- `cml/vector-and-matrix-norms/p01-vector-norm`: two add-test failure or helper messages added in test_hidden.py state the hidden inputs and expected values; coordinator asked for hand rewording, but the runbook forbids typing into test_hidden.py, so not changed by hand; needs a tool-path fix or a coordinator decision
- `cml/xgboost-second-order-objective/p01-leaf-weight-and-split-gain`: reject_rejects_zero_lambda.py: test_opposite_gradients_give_opposite_leaf_weights shows kind=bare_exception
- `cml/xgboost-second-order-objective/p01-leaf-weight-and-split-gain`: reject_type_error_on_negative_lambda.py: test_negative_reg_lambda_raises_value_error shows kind=bare_exception
- `dl/building-tensors-by-broadcasting/p01-pairwise-squared-distances`: reject_empty_input_refused.py: test_empty_point_sets_give_matrices_with_the_right_shape shows kind=bare_exception
- `dl/convolution-forward-pass/p01-conv2d-forward`: reject_single_example_batch_rejected.py: test_ones_kernel_sums_each_two_by_two_window shows kind=bare_exception
- `dl/convolution-forward-pass/p01-conv2d-forward`: reject_output_size_plus_one_before_floor.py: test_ones_kernel_sums_each_two_by_two_window shows kind=bare_exception
- `dl/convolution-forward-pass/p01-conv2d-forward`: reject_kernel_exactly_fitting_rejected.py: test_kernel_is_not_flipped shows kind=bare_exception
- `dl/counting-network-parameters/p01-count-dense-parameters`: reject_widths_rejects_tuple.py: test_no_bias_example shows kind=bare_exception
- `dl/gradients-with-framework-autograd/p01-gradient-at-a-point`: reject_result_forced_to_default_dtype.py: test_caller_tensor_keeps_requires_grad_flag_and_no_grad_field shows kind=bare_exception
- `dl/gradients-with-framework-autograd/p02-straight-through-round`: reject_float_step_refused.py: test_forward_rounds_each_entry_to_the_nearest_step_multiple shows kind=bare_exception
- `dl/kl-divergence/p01-kl-from-logits`: reject_nan_check_drops_posinf.py: test_invalid_inputs_raise_value_error shows kind=no_detail
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
- `inf/absmax-and-zero-point-int8-quantization/p01-absmax-quantize`: reject_per_tensor_scale_as_vector.py: test_ties_round_to_even_codes_with_unit_scale shows kind=bare_exception
- `inf/data-quality-and-drift-detection/p01-population-stability-index`: reject_expected_list_rejected.py: test_identical_samples_have_zero_psi shows kind=bare_exception
- `inf/floating-point-formats/p02-round-to-nearest-even`: reject_largest_finite_overflows.py: test_fp16_just_below_overflow_midpoint_rounds_down shows kind=raw_assert
- `inf/floating-point-formats/p02-round-to-nearest-even`: reject_exponent_bits_upper_seven.py: test_bf16_tie_between_one_and_next_value_rounds_to_even_one shows kind=bare_exception
- `inf/floating-point-formats/p02-round-to-nearest-even`: reject_infinite_input_rejected.py: test_nan_and_infinities_pass_through_and_zero_keeps_its_sign shows kind=bare_exception
- `inf/prefill-vs-decode-bottlenecks/p02-compute-bound-decode-batch`: reject_context_minimum_is_one.py: test_model_already_compute_bound_at_batch_one_returns_one shows kind=bare_exception
- `inf/speculative-decoding-speedup-math/p01-speculative-speedup`: reject_alpha_one_only_gamma_zero.py: test_full_acceptance_emits_gamma_plus_one_tokens shows kind=bare_exception
- `inf/ttft-itl-and-throughput/p02-throughput-and-goodput`: reject_first_token_at_arrival_rejected.py: test_inclusive_bound_on_itl_is_met_exactly shows kind=bare_exception
- `llm/bradley-terry-reward-models/p01-reward-pair-loss-and-accuracy`: reject_rejected_detached.py: test_gradient_matches_the_analytic_form_in_float64 shows kind=bare_exception
- `llm/bradley-terry-reward-models/p01-reward-pair-loss-and-accuracy`: reject_inplace_clamp_rejected.py: test_gradient_matches_the_analytic_form_in_float64 shows kind=bare_exception
- `llm/direct-preference-optimization-dpo/p02-dpo-loss-and-implicit-rewards`: reject_rewards_float32.py: test_implicit_rewards_and_loss_for_a_pair_the_policy_prefers shows kind=bare_exception
- `llm/direct-preference-optimization-dpo/p02-dpo-loss-and-implicit-rewards`: reject_detaches_chosen_policy.py: test_gradient_wrt_policy_matches_analytic_form_in_float64 shows kind=bare_exception
- `llm/grpo-and-its-variants/p01-group-normalised-advantages`: reject_float64_cast_to_float32.py: test_binary_group_gets_plus_and_minus_one_with_population_std shows kind=bare_exception
- `llm/grpo-and-its-variants/p02-clipped-grpo-token-loss`: reject_kl_coef_zero_refused.py: test_single_unclipped_token_gives_negative_advantage shows kind=bare_exception
- `llm/llm-as-judge-and-agreement-with-humans/p01-cohens-kappa`: reject_unused_label_rejected.py: test_perfect_agreement_over_a_larger_label_set_is_one shows kind=bare_exception
- `llm/llm-as-judge-and-agreement-with-humans/p01-cohens-kappa`: reject_zero_division_not_guarded.py: test_one_label_for_every_item_is_undefined_not_nan shows kind=bare_exception
- `llm/lora-adapters/p01-lora-linear-forward-and-merge`: reject_b_not_transposed.py: test_worked_example_adds_scaled_low_rank_update_to_base_output shows kind=bare_exception
- `llm/lora-parameter-and-memory-arithmetic/p01-lora-trainable-parameter-budget`: reject_rank_one_refused.py: test_all_seven_targets_match_the_hand_computed_budget shows kind=bare_exception
- `llm/lora-parameter-and-memory-arithmetic/p01-lora-trainable-parameter-budget`: reject_targets_tuple_refused.py: test_two_attention_targets_across_two_layers_give_128_trainable_parameters shows kind=bare_exception
- `llm/lora-parameter-and-memory-arithmetic/p02-finetune-memory-in-bytes`: reject_zero_lora_params_refused.py: test_four_bit_base_rounds_a_fractional_byte_up shows kind=bare_exception
- `llm/lora-parameter-and-memory-arithmetic/p02-finetune-memory-in-bytes`: reject_adapter_16_bits_refused.py: test_bf16_base_with_a_thousand_adapter_numbers_totals_2012000_bytes shows kind=bare_exception
- `llm/model-parameter-and-memory-footprint/p01-count-decoder-parameters`: reject_n_layers_minimum_one.py: test_zero_layers_have_no_per_block_parameters shows kind=bare_exception
- `llm/rotary-position-embeddings-rope/p02-linear-position-interpolation`: reject_int_scale_refused.py: test_scale_one_on_position_zero_is_unchanged shows kind=bare_exception
- `llm/temperature-top-k-and-top-p-sampling/p02-sample-with-a-generator`: reject_single_token_vocab_rejected.py: test_single_entry_vocabulary_returns_zero shows kind=bare_exception
- `llm/temperature-top-k-and-top-p-sampling/p02-sample-with-a-generator`: reject_inplace_cumsum_mutates_probs.py: test_zero_probability_token_is_never_returned shows kind=bare_exception
- `llm/word-and-character-tokenizers/p01-word-level-vocabulary`: reject_decode_pad_id_rejected.py: test_reserved_special_strings_in_the_corpus_do_not_take_ids shows kind=bare_exception
- `math/determinants-and-trace/p01-determinant`: reject_empty_matrix_refused.py: test_zero_by_zero_matrix_has_determinant_one shows kind=bare_exception
- `math/inverse-transform-sampling/p01-sample-exponential`: reject_zero_count_rejected.py: test_zero_draws_give_an_empty_float_array shows kind=bare_exception
- `math/inverse-transform-sampling/p02-sample-discrete`: reject_float_indices.py: test_chi_square_goodness_of_fit_at_a_fixed_seed shows kind=bare_exception
- `inf/kv-cache-size-arithmetic/p01-kv-cache-bytes`: reject_zero_seq_len_rejected.py: test_zero_context_has_no_cache_for_any_batch shows kind=bare_exception
- `cml/vector-operations/p01-vector-projection`: reject_integer_scale_truncated.py: test_matches_independent_reference_on_random_vectors shows kind=bare_exception
- `rl/mdps-and-discounted-returns/p02-iterative-policy-evaluation`: reject_zero_gamma_rejected.py: test_zero_discount_returns_expected_reward_for_one_state shows kind=bare_exception
- `rl/reinforce/p01-reinforce-loss`: reject_squeezed_actions.py: test_worked_example_matches_minus_two_log_half shows kind=bare_exception
- `rl/sarsa-and-expected-sarsa/p01-sarsa-and-expected-sarsa-updates`: reject_rejects_integer_table.py: test_outputs_are_float_tables_of_the_input_shape shows kind=bare_exception
- `rl/sarsa-and-expected-sarsa/p01-sarsa-and-expected-sarsa-updates`: reject_rejects_integer_reward.py: test_outputs_are_float_tables_of_the_input_shape shows kind=bare_exception
- `rl/sarsa-and-expected-sarsa/p01-sarsa-and-expected-sarsa-updates`: reject_checks_a_next_when_terminal.py: test_terminal_next_state_targets_the_reward_alone shows kind=bare_exception
- `rl/sarsa-and-expected-sarsa/p01-sarsa-and-expected-sarsa-updates`: reject_alpha_upper_bound_exclusive.py: test_sarsa_target_uses_the_action_taken_next shows kind=bare_exception
- `safe/split-conformal-prediction/p01-split-conformal-intervals`: reject_integer_inputs_refused.py: test_threshold_is_the_eighth_smallest_residual_at_alpha_point_two shows kind=bare_exception
- `sql/pandas-merge-and-reshape/p01-monthly-revenue-by-region`: reject_datetime_only_dates.py: test_worked_example_matches_the_table_in_problem_md shows kind=bare_exception
- `sql/sql-conditional-aggregation/p01-channel-conversion-and-device-revenue`: probe output pairs a FAILED test name with a message from a different check, for example reject_null_revenue_rows_dropped reports test_rate_is_a_fraction_not_a_percentage with a sessions message; the verdict and killed_by lines were unaffected

### inspect
- `inf/floating-point-formats/p02-round-to-nearest-even`: test_bfloat16_and_fp16_overflow_boundaries_are_exact compares against the float literal 3.4028234663852886e38 (float32 largest value); a comment on the line above says where it comes from. No pass, global random or np.random use in this file.
- `llm/grpo-and-its-variants/p01-group-normalised-advantages`: test_sample_standard_deviation_is_not_used compares against the literal 0.7071067811865476 for the sample-std variant with atol 1e-4. The value is the sample-std answer for rewards 0 and 1, so the literal is derived and no comment says where it comes from
- `inf/speculative-decoding-speedup-math/p02-accept-or-resample`: test_zero_residual_mass_falls_back_to_target_distribution uses the literal 0.9999999995 (u1, lines 99 and 102) and q = 0.5 + 5e-10 to build a zero-residual case; the 5e-10 slack is stated in the comment above, the u1 value has no source comment
- `cml/cosine-similarity/p01-cosine-similarity-matrix`: test_similarities_are_clipped_to_exactly_minus_one_and_one (lines 125-126): the literals 1.0000000000000002 and -1.0000000000000002 appear only inside a comment that explains float round-off of unit-vector dot products. They are not compared against; no hidden check depends on them. No pass, global np.random call, or other hit.
- `llm/retrieval-augmented-generation-pipeline/p02-cited-context-prompt`: test_scores_equal_to_nine_decimal_places_tie_and_go_to_the_smaller_id (line 301): literals 0.5000000004 and 0.5000000001 are chosen so both round to 0.5 at nine decimals; a comment explains the tie rule, not where the numbers come from.
- `llm/retrieval-augmented-generation-pipeline/p02-cited-context-prompt`: test_scores_that_differ_in_the_ninth_decimal_are_not_ties (line 314, comment): mentions 0.500000002 and 0.500000004 as the pair under test; the comment states the intent, not a derivation.
- `llm/retrieval-augmented-generation-pipeline/p02-cited-context-prompt`: test_scores_that_differ_in_the_ninth_decimal_are_not_ties (line 316): literals 0.5000000020 and 0.5000000040 are the ninth-decimal pair; no source is given for them beyond the comment above.
- `math/reservoir-sampling/p01-reservoir-sample`: blind solution is 0.96 similar to the reference by whole-file lines; the starter carries the docstring and the Contract fixes the draw protocol, so the body can hardly differ. Judged independent by the coordinator. The similarity check should ignore lines the starter already has.
- `dl/smooth-activations-tanh-gelu-swish/p01-gelu-and-swish`: resolved by the coordinator: the Contract says three NumPy float64 arrays each exactly the shape of x, scalar inputs included. NumPy ufuncs return scalars for 0-d input, the solver missed that, the hidden message named it and the second submission passed. No statement defect.
- `cml/bootstrap-and-permutation-resampling/p01-bootstrap-percentile-ci`: the no-mutation test uses an input that rounding to 12 places does not change; an in-place rounding is caught only by the same-seed test. A new test was refused as already rejected by base tests
- `cml/matrix-multiplication/p01-matrix-multiply`: reject_rounds_inputs_in_place.py is killed only by test_flat_list_input_raises_value_error through a TypeError, not by test_inputs_are_not_mutated; the in-place rounding is not exercised by any test
- `cml/probability-basics/p01-marginals-and-conditional`: correction to the done line: survivors 3 were 1 unpromised and 2 same behaviour (returns identical, RuntimeWarning only); correct counts are same behaviour 2, unpromised 1, tests added 0
- `llm/direct-preference-optimization-dpo/p01-sequence-log-probs-with-response-mask`: mutant 16e8dd57f15c7c33 ('and' for 'or' in the logits guard): same non-tensor input as ad96 raises AttributeError instead of ValueError, so the test added for ad96 also fails it (probed). No separate test added

### done
- `dl/one-training-step/p01-manual-sgd-train-step`: hygiene: 1 findings fixed, mode text
- `dl/sgd-with-momentum/p01-momentum-step`: hygiene: 1 findings fixed, mode text
- `dl/sgd-with-momentum/p02-nesterov-weight-decay`: hygiene: 1 findings fixed, mode text
- `dl/adam/p01-adam-step`: hygiene: 1 findings fixed, mode text
- `dl/adam/p02-amsgrad-step`: hygiene: 2 findings fixed, mode text
- `dl/adamw-decoupled-weight-decay/p01-adamw-step`: hygiene: 2 findings fixed, mode text
- `dl/transformer-block/p01-pre-norm-causal-block`: hygiene: 1 findings fixed, mode text
- `dl/convolution-forward-pass/p01-conv2d-forward`: hygiene: 1 findings fixed, mode text
- `algo/topological-sort/p01-lexicographic-topological-order`: hygiene: 3 findings fixed, mode refactor
- `llm/gpt-forward-pass/p01-decoder-only-gpt-forward`: hygiene: 1 findings fixed, mode text
- `llm/rotary-position-embeddings-rope/p01-half-split-rope-with-offset`: hygiene: 1 findings fixed, mode text
- `llm/rotary-position-embeddings-rope/p02-linear-position-interpolation`: hygiene: 1 findings fixed, mode text
- `llm/temperature-top-k-and-top-p-sampling/p02-sample-with-a-generator`: hygiene: 1 findings fixed, mode text
- `llm/model-parameter-and-memory-footprint/p01-count-decoder-parameters`: hygiene: 1 findings fixed, mode text
- `llm/gqa-mqa-and-cross-layer-kv-sharing/p01-grouped-query-attention`: hygiene: 1 findings fixed, mode refactor
- `dl/mixed-precision/p01-dynamic-loss-scaler`: hygiene: 9 findings fixed, mode text
- `inf/floating-point-formats/p02-round-to-nearest-even`: hygiene: 5 findings fixed, mode text
- `inf/paged-kv-cache/p01-block-allocator`: hygiene: 5 findings fixed, mode refactor
- `inf/continuous-batching/p01-batch-scheduler-steps`: hygiene: 3 findings fixed, mode text
- `llm/chat-templates-and-tool-call-formats/p02-parse-tool-call`: hygiene: 1 findings fixed, mode text
- `llm/grpo-and-its-variants/p01-group-normalised-advantages`: hygiene: 1 findings fixed, mode text
- `llm/grpo-and-its-variants/p02-clipped-grpo-token-loss`: hygiene: 1 findings fixed, mode text
- `llm/llm-as-judge-and-agreement-with-humans/p02-debiased-win-rate`: hygiene: 1 findings fixed, mode text
- `inf/speculative-decoding-speedup-math/p02-accept-or-resample`: hygiene: 1 findings fixed, mode text
- `cml/derivatives/p01-numerical-derivative`: hygiene: 2 findings fixed, mode text
- `cml/partial-derivatives-and-gradients/p01-numerical-gradient`: hygiene: 1 findings fixed, mode text
- `cml/gradient-descent/p01-gradient-descent`: hygiene: 4 findings fixed, mode text
- `cml/stochastic-and-mini-batch-gradient-descent/p01-minibatch-sgd`: hygiene: 7 findings fixed, mode text
- `cml/descriptive-statistics/p01-welford-mean-variance`: hygiene: 11 findings fixed, mode text
- `cml/descriptive-statistics/p02-iqr-outlier-fences`: hygiene: 10 findings fixed, mode text
- `cml/probability-basics/p01-marginals-and-conditional`: hygiene: 5 findings fixed, mode text
- `cml/bayes-theorem/p01-bayes-posterior`: hygiene: 5 findings fixed, mode text
- `cml/bernoulli-binomial-and-poisson/p01-binomial-pmf`: hygiene: 4 findings fixed, mode text
- `cml/bernoulli-binomial-and-poisson/p02-poisson-pmf`: hygiene: 5 findings fixed, mode text
- `cml/vector-operations/p01-vector-projection`: hygiene: 4 findings fixed, mode text
- `cml/matrix-shapes-and-transpose/p01-transpose-matrix`: hygiene: 3 findings fixed, mode text
- `cml/matrix-multiplication/p01-matrix-multiply`: hygiene: 5 findings fixed, mode text
- `dl/building-tensors-by-broadcasting/p01-pairwise-squared-distances`: hygiene: 2 findings fixed, mode text
- `cml/vector-and-matrix-norms/p01-vector-norm`: hygiene: 4 findings fixed, mode text
- `cml/cosine-similarity/p01-cosine-similarity-matrix`: hygiene: 6 findings fixed, mode text
- `cml/matrix-inverse-and-rank/p01-gauss-jordan-inverse`: hygiene: 5 findings fixed, mode text
- `cml/matrix-inverse-and-rank/p02-matrix-rank`: hygiene: 7 findings fixed, mode text
- `math/determinants-and-trace/p01-determinant`: hygiene: 9 findings fixed, mode text
- `cml/eigenvalues-and-eigenvectors/p01-power-iteration`: hygiene: 5 findings fixed, mode text
- `algo/hash-maps-and-sets/p01-top-k-frequent-tokens`: hygiene: 1 findings fixed, mode text
- `cml/lagrange-multipliers/p01-constrained-quadratic-min`: hygiene: 2 findings fixed, mode text
- `llm/document-packing-and-cross-document-masks/p01-packed-causal-mask`: hygiene: 2 findings fixed, mode text
- `rl/sarsa-and-expected-sarsa/p01-sarsa-and-expected-sarsa-updates`: hygiene: 2 findings fixed, mode text
- `llm/model-parameter-and-memory-footprint/p02-training-and-inference-memory`: hygiene: 2 findings fixed, mode text
- `sql/sql-conditional-aggregation/p01-channel-conversion-and-device-revenue`: hygiene: 3 findings fixed, mode text
- `cml/choosing-a-tree-split/p01-best-split-threshold`: hygiene: 1 findings fixed, mode text
- `cml/principal-component-analysis/p01-pca-via-svd`: hygiene: 1 findings fixed, mode text
- `inf/absmax-and-zero-point-int8-quantization/p01-absmax-quantize`: hygiene: 3 findings fixed, mode text
- `inf/absmax-and-zero-point-int8-quantization/p02-zero-point-quantize`: hygiene: 1 findings fixed, mode text
- `llm/retrieval-augmented-generation-pipeline/p02-cited-context-prompt`: hygiene: 1 findings fixed, mode text
- `llm/pass-k-and-majority-voting/p02-majority-vote`: hygiene: 1 findings fixed, mode refactor
- `cml/t-tests-and-anova/p01-welch-t-test`: hygiene: 3 findings fixed, mode text
- `cml/normal-equation/p01-normal-equation-fit`: hygiene: 14 findings fixed, mode text
- `cml/gaussian-mixtures-and-em/p01-gmm-em-full-covariance`: hygiene: 2 findings fixed, mode text
- `sql/sql-joins-and-unions/p01-customer-order-totals`: hygiene: 1 findings fixed, mode text
- `sql/sql-joins-and-unions/p02-warehouses-with-a-shipment-or-a-return`: hygiene: 1 findings fixed, mode text
- `sql/pandas-merge-and-reshape/p01-monthly-revenue-by-region`: hygiene: 1 findings fixed, mode text
- `sql/sql-select-filter-and-sort/p01-top-five-completed-orders-2024`: hygiene: 1 findings fixed, mode text
- `sql/sql-group-by-and-having/p01-customers-with-three-completed-orders`: hygiene: 1 findings fixed, mode text
- `dl/causal-masking/p01-causal-padding-mask`: hygiene: 1 findings fixed, mode text
- `cml/chain-rule/p01-mse-sigmoid-gradient`: hygiene: 2 findings fixed, mode text
- `inf/floating-point-formats/p01-decode-a-bit-pattern`: hygiene: 1 findings fixed, mode text
- `algo/two-pointers-and-sliding-windows/p01-longest-window-within-budget`: hygiene: 2 findings fixed, mode text
- `llm/llm-as-judge-and-agreement-with-humans/p01-cohens-kappa`: hygiene: 1 findings fixed, mode refactor
- `algo/bfs-and-dfs-on-graphs-and-grids/p01-grid-shortest-path`: mutants: survivors 3, same behaviour 2, unpromised 1, no-witness 0, tests added 0
- `algo/bfs-and-dfs-on-graphs-and-grids/p02-count-components`: mutants: survivors 5, same behaviour 4, unpromised 1, no-witness 0, tests added 0
- `algo/binary-search/p01-lower-bound`: mutants: survivors 1, same behaviour 1, unpromised 0, no-witness 0, tests added 0
- `algo/binary-search/p02-min-ship-capacity`: mutants: survivors 3, same behaviour 1, unpromised 2, no-witness 0, tests added 0
- `algo/binary-tree-recursion-and-traversal/p01-inorder-traversal-iterative`: mutants: survivors 0, same behaviour 0, unpromised 0, no-witness 0, tests added 0
- `algo/binary-tree-recursion-and-traversal/p02-tree-diameter`: mutants: survivors 0, same behaviour 0, unpromised 0, no-witness 0, tests added 0
- `algo/dynamic-programming/p01-minimum-coins`: mutants: survivors 4, same behaviour 4, unpromised 0, no-witness 0, tests added 0
- `algo/dynamic-programming/p02-longest-common-subsequence`: mutants: survivors 1, same behaviour 0, unpromised 1, no-witness 0, tests added 0
- `algo/hash-maps-and-sets/p01-top-k-frequent-tokens`: mutants: survivors 2, same behaviour 0, unpromised 2, no-witness 0, tests added 0
- `algo/heaps-and-priority-queues/p01-merge-sorted-streams`: mutants: survivors 0, same behaviour 0, unpromised 0, no-witness 0, tests added 0
- `algo/heaps-and-priority-queues/p02-running-median`: mutants: survivors 7, same behaviour 2, unpromised 5, no-witness 0, tests added 0
- `algo/lru-cache/p01-lru-cache`: mutants: survivors 5, same behaviour 5, unpromised 0, no-witness 0, tests added 0
- `algo/prefix-sums/p02-rectangle-range-sums`: mutants: survivors 2, same behaviour 1, unpromised 1, no-witness 0, tests added 0
- `algo/prefix-sums/p01-subarrays-with-sum-k`: mutants: survivors 0, same behaviour 0, unpromised 0, no-witness 0, tests added 0
- `algo/sort-then-sweep/p01-merge-intervals`: mutants: survivors 0, same behaviour 0, unpromised 0, no-witness 0, tests added 0
- `algo/sort-then-sweep/p02-peak-concurrent-sessions`: mutants: survivors 0, same behaviour 0, unpromised 0, no-witness 0, tests added 0
- `algo/stacks-and-queues/p01-days-until-higher`: mutants: survivors 0, same behaviour 0, unpromised 0, no-witness 0, tests added 0
- `algo/stacks-and-queues/p02-sliding-window-maximum`: mutants: survivors 1, same behaviour 1, unpromised 0, no-witness 0, tests added 0
- `algo/topological-sort/p01-lexicographic-topological-order`: mutants: survivors 2, same behaviour 0, unpromised 2, no-witness 0, tests added 0
- `algo/two-pointers-and-sliding-windows/p01-longest-window-within-budget`: mutants: survivors 0, same behaviour 0, unpromised 0, no-witness 0, tests added 0
- `cml/a-b-test-design-and-analysis/p01-sample-size-two-proportions`: mutants: survivors 0, same behaviour 0, unpromised 0, no-witness 0, tests added 0
- `cml/a-b-test-design-and-analysis/p02-two-proportion-z-test`: mutants: survivors 0, same behaviour 0, unpromised 0, no-witness 0, tests added 0
- `cml/adaboost-and-gradient-boosting/p02-adaboost-round`: mutants: survivors 9, same behaviour 7, unpromised 2, no-witness 0, tests added 0
- `cml/adaboost-and-gradient-boosting/p01-boost-stumps-mse`: mutants: survivors 4, same behaviour 4, unpromised 0, no-witness 0, tests added 0
- `cml/bayes-theorem/p01-bayes-posterior`: mutants: survivors 5, same behaviour 0, unpromised 5, no-witness 0, tests added 0
- `cml/bernoulli-binomial-and-poisson/p01-binomial-pmf`: mutants: survivors 3, same behaviour 2, unpromised 1, no-witness 0, tests added 0
- `cml/bernoulli-binomial-and-poisson/p02-poisson-pmf`: mutants: survivors 2, same behaviour 1, unpromised 0, no-witness 0, tests added 1
- `cml/bias-variance-decomposition/p01-bias-variance-from-predictions`: mutants: survivors 10, same behaviour 3, unpromised 2, no-witness 1, tests added 2
- `cml/bootstrap-and-permutation-resampling/p01-bootstrap-percentile-ci`: mutants: survivors 0, same behaviour 0, unpromised 0, no-witness 0, tests added 0
- `cml/bootstrap-and-permutation-resampling/p02-permutation-test-mean-diff`: mutants: survivors 0, same behaviour 0, unpromised 0, no-witness 0, tests added 0
- `cml/chain-rule/p01-mse-sigmoid-gradient`: mutants: survivors 7, same behaviour 4, unpromised 0, no-witness 0, tests added 3
- `cml/choosing-a-tree-split/p01-best-split-threshold`: mutants: survivors 8, same behaviour 6, unpromised 2, no-witness 0, tests added 0
- `cml/confusion-matrix-and-count-based-scores/p01-confusion-matrix-from-labels`: mutants: survivors 3, same behaviour 2, unpromised 0, no-witness 1, tests added 0
- `cml/confusion-matrix-and-count-based-scores/p02-binary-count-scores`: mutants: survivors 0, same behaviour 0, unpromised 0, no-witness 0, tests added 0
- `cml/cosine-similarity/p01-cosine-similarity-matrix`: mutants: survivors 4, same behaviour 4, unpromised 0, no-witness 0, tests added 0
- `cml/covariance-and-correlation/p01-covariance-matrix`: mutants: survivors 2, same behaviour 0, unpromised 2, no-witness 0, tests added 0
- `cml/covariance-and-correlation/p02-correlation-matrix`: mutants: survivors 10, same behaviour 4, unpromised 6, no-witness 0, tests added 0
- `cml/cross-entropy/p01-cross-entropy-from-logits`: mutants: survivors 7, same behaviour 7, unpromised 0, no-witness 0, tests added 0
- `cml/cross-entropy/p02-binary-cross-entropy-with-logits`: mutants: survivors 7, same behaviour 7, unpromised 0, no-witness 0, tests added 0
- `cml/derivatives/p01-numerical-derivative`: mutants: survivors 0, same behaviour 0, unpromised 0, no-witness 0, tests added 0
- `cml/descriptive-statistics/p02-iqr-outlier-fences`: mutants: survivors 2, same behaviour 2, unpromised 0, no-witness 0, tests added 0
- `cml/descriptive-statistics/p01-welford-mean-variance`: mutants: survivors 0, same behaviour 0, unpromised 0, no-witness 0, tests added 0
- `cml/eigenvalues-and-eigenvectors/p01-power-iteration`: mutants: survivors 10, same behaviour 6, unpromised 2, no-witness 0, tests added 2
- `cml/eigenvalues-and-eigenvectors/p01-power-iteration`: mutants: survivors 10, same behaviour 6, unpromised 2, no-witness 0, tests added 2 (second test replaced on coordinator review)
- `cml/entropy-and-mutual-information/p01-distribution-entropy`: mutants: survivors 3, same behaviour 2, unpromised 1, no-witness 0, tests added 0
- `cml/entropy-and-mutual-information/p02-mutual-information-from-labels`: mutants: survivors 4, same behaviour 3, unpromised 0, no-witness 0, tests added 1
- `cml/expectation-and-variance/p01-discrete-moments`: mutants: survivors 4, same behaviour 0, unpromised 4, no-witness 0, tests added 0
- `cml/feature-scaling/p02-robust-median-iqr-scaler`: mutants: survivors 3, same behaviour 1, unpromised 2, no-witness 0, tests added 0
- `cml/feature-scaling/p01-standard-scaler`: mutants: survivors 6, same behaviour 5, unpromised 0, no-witness 0, tests added 1
- `cml/feature-scaling/p01-standard-scaler`: mutants (coordinator, after withdrawing one test): survivors 6, same behaviour 5, unpromised 1, no-witness 0, tests added 0
- `cml/gaussian-mixtures-and-em/p01-gmm-em-full-covariance`: mutants: survivors 5, same behaviour 2, unpromised 3, no-witness 0, tests added 0
- `cml/gradient-descent/p01-gradient-descent`: mutants: survivors 3, same behaviour 2, unpromised 1, no-witness 0, tests added 0
- `cml/handling-class-imbalance/p01-balanced-class-weights`: mutants: survivors 1, same behaviour 1, unpromised 0, no-witness 0, tests added 0
- `cml/handling-class-imbalance/p02-prior-shift-adjust`: mutants: survivors 5, same behaviour 4, unpromised 1, no-witness 0, tests added 0
- `cml/k-fold-cross-validation/p02-cross-validate`: mutants: survivors 4, same behaviour 1, unpromised 3, no-witness 0, tests added 0
- `cml/k-fold-cross-validation/p01-kfold-indices`: mutants: survivors 0, same behaviour 0, unpromised 0, no-witness 0, tests added 0
- `cml/k-means/p01-lloyd-kmeans`: mutants: survivors 4, same behaviour 3, unpromised 1, no-witness 0, tests added 0
- `cml/k-means/p02-kmeans-plus-plus-seeding`: mutants: survivors 7, same behaviour 6, unpromised 0, no-witness 0, tests added 1
- `cml/lagrange-multipliers/p01-constrained-quadratic-min`: mutants: survivors 10, same behaviour 7, unpromised 2, no-witness 0, tests added 0
- `cml/lasso-and-elastic-net/p01-lasso-coordinate-descent`: mutants: survivors 3, same behaviour 2, unpromised 1, no-witness 0, tests added 0
- `cml/lasso-and-elastic-net/p02-elastic-net-coordinate-descent`: mutants: survivors 2, same behaviour 2, unpromised 0, no-witness 0, tests added 0
- `cml/leakage-and-walk-forward-validation/p01-walk-forward-splits`: mutants: survivors 0, same behaviour 0, unpromised 0, no-witness 0, tests added 0
- `cml/leakage-and-walk-forward-validation/p02-expanding-target-mean`: mutants: survivors 3, same behaviour 2, unpromised 1, no-witness 0, tests added 0
- `cml/logistic-regression/p01-l2-logistic-fit`: mutants: survivors 10, same behaviour 5, unpromised 4, no-witness 0, tests added 1
- `cml/matrix-inverse-and-rank/p01-gauss-jordan-inverse`: mutants: survivors 5, same behaviour 5, unpromised 0, no-witness 0, tests added 0
- `cml/matrix-inverse-and-rank/p02-matrix-rank`: mutants: survivors 1, same behaviour 1, unpromised 0, no-witness 0, tests added 0
- `cml/matrix-multiplication/p01-matrix-multiply`: mutants: survivors 1, same behaviour 0, unpromised 1, no-witness 0, tests added 0
- `cml/matrix-shapes-and-transpose/p01-transpose-matrix`: mutants: survivors 1, same behaviour 1, unpromised 0, no-witness 0, tests added 0
- `cml/matrix-shapes-and-transpose/p01-transpose-matrix`: mutants: survivors 1, same behaviour 0, unpromised 1, no-witness 0, tests added 0 (supersedes the earlier done note: survivor 7463f3e26f7c9878 is unpromised, not same behaviour)
- `cml/maximum-likelihood-and-map/p01-gaussian-mle`: mutants: survivors 1, same behaviour 1, unpromised 0, no-witness 0, tests added 0
- `cml/maximum-likelihood-and-map/p02-beta-bernoulli-map`: mutants: survivors 2, same behaviour 0, unpromised 2, no-witness 0, tests added 0
- `cml/missing-values-and-outliers/p01-impute-with-missing-indicators`: mutants: survivors 0, same behaviour 0, unpromised 0, no-witness 0, tests added 0
- `cml/missing-values-and-outliers/p02-modified-zscore-outliers`: mutants: survivors 6, same behaviour 3, unpromised 3, no-witness 0, tests added 0
- `cml/nearest-neighbour-search-and-knn/p01-knn-classify`: mutants: survivors 1, same behaviour 1, unpromised 0, no-witness 0, tests added 0
- `cml/nearest-neighbour-search-and-knn/p02-knn-regress`: mutants: survivors 2, same behaviour 1, unpromised 1, no-witness 0, tests added 0
- `cml/normal-and-multivariate-gaussian/p01-mvn-logpdf`: mutants: survivors 6, same behaviour 5, unpromised 1, no-witness 0, tests added 0
- `cml/normal-equation/p01-normal-equation-fit`: mutants: survivors 3, same behaviour 2, unpromised 1, no-witness 0, tests added 0
- `cml/over-and-underfitting-early-stopping/p01-early-stopping`: mutants: survivors 4, same behaviour 4, unpromised 0, no-witness 0, tests added 0
- `cml/partial-derivatives-and-gradients/p01-numerical-gradient`: mutants: survivors 1, same behaviour 0, unpromised 0, no-witness 0, tests added 1
- `cml/precision-recall-and-f1/p01-precision-recall-averaging`: mutants: survivors 7, same behaviour 6, unpromised 1, no-witness 0, tests added 0
- `cml/precision-recall-and-f1/p02-best-f1-threshold`: mutants: survivors 1, same behaviour 0, unpromised 1, no-witness 0, tests added 0
- `cml/preprocessing-pipelines/p01-numeric-categorical-pipeline`: mutants: survivors 7, same behaviour 7, unpromised 0, no-witness 0, tests added 0
- `cml/principal-component-analysis/p01-pca-via-svd`: mutants: survivors 6, same behaviour 3, unpromised 3, no-witness 0, tests added 0
- `cml/principal-component-analysis/p02-components-for-variance`: mutants: survivors 6, same behaviour 3, unpromised 3, no-witness 0, tests added 0
- `cml/probability-basics/p01-marginals-and-conditional`: mutants: survivors 3, same behaviour 0, unpromised 1, no-witness 0, tests added 0
- `cml/random-forests-and-feature-importance/p01-permutation-importance`: mutants: survivors 3, same behaviour 2, unpromised 0, no-witness 0, tests added 1
- `cml/random-forests-and-feature-importance/p02-bootstrap-oob-masks`: mutants: survivors 2, same behaviour 1, unpromised 1, no-witness 0, tests added 0
- `cml/ranking-metrics/p01-ndcg-at-k`: mutants: survivors 4, same behaviour 2, unpromised 1, no-witness 0, tests added 1
- `cml/ranking-metrics/p02-retrieval-metrics-at-k`: mutants: survivors 0, same behaviour 0, unpromised 0, no-witness 0, tests added 0
- `cml/regression-metrics-and-robust-losses/p01-regression-report`: mutants: survivors 2, same behaviour 2, unpromised 0, no-witness 0, tests added 0
- `cml/regression-metrics-and-robust-losses/p02-huber-loss-gradient`: mutants: survivors 4, same behaviour 4, unpromised 0, no-witness 0, tests added 0
- `cml/ridge-regression-and-weight-decay/p01-ridge-fit`: mutants: survivors 2, same behaviour 1, unpromised 1, no-witness 0, tests added 0
- `cml/roc-pr-curves-and-auc/p01-roc-auc-rank-statistic`: mutants: survivors 4, same behaviour 4, unpromised 0, no-witness 0, tests added 0
- `cml/roc-pr-curves-and-auc/p02-average-precision-step`: mutants: survivors 3, same behaviour 3, unpromised 0, no-witness 0, tests added 0
- `cml/rolling-statistics-stationarity-and-autocorrelation/p01-rolling-mean-std`: mutants: survivors 7, same behaviour 0, unpromised 4, no-witness 0, tests added 2 (6b59 and d5f6 and efa3 and 69d5 are unpromised; 1846 and a5872 are killed by the first new test, fde8 by the second)
- `cml/rolling-statistics-stationarity-and-autocorrelation/p02-acf`: mutants: survivors 10, same behaviour 3, unpromised 4, no-witness 0, tests added 1 (1811, 19a1 and a094 are killed by the new test; 9166, 5068 and c12c are the same behaviour)
- `cml/rolling-statistics-stationarity-and-autocorrelation/p03-seasonal-naive-forecast-and-mase`: mutants: survivors 0, same behaviour 0, unpromised 0, no-witness 0, tests added 0
- `cml/sampling-distributions-standard-errors-and-cis/p01-mean-confidence-interval`: mutants: survivors 0, same behaviour 0, unpromised 0, no-witness 0, tests added 0
- `cml/shuffling-splitting-and-batching-data/p01-stratified-train-test-split`: mutants: survivors 9, same behaviour 7, unpromised 2, no-witness 0, tests added 0
- `cml/shuffling-splitting-and-batching-data/p02-minibatch-index-schedule`: mutants: survivors 4, same behaviour 2, unpromised 2, no-witness 0, tests added 0
- `cml/sigmoid-and-log-odds/p01-stable-sigmoid`: mutants: survivors 1, same behaviour 1, unpromised 0, no-witness 0, tests added 0
- `cml/sigmoid-and-log-odds/p02-logit`: mutants: survivors 4, same behaviour 1, unpromised 3, no-witness 0, tests added 0
- `cml/singular-value-decomposition/p01-best-rank-k`: mutants: survivors 4, same behaviour 1, unpromised 3, no-witness 0, tests added 0
- `cml/singular-value-decomposition/p02-pseudo-inverse`: mutants: survivors 7, same behaviour 0, unpromised 7, no-witness 0, tests added 0
- `cml/softmax/p01-stable-softmax`: mutants: survivors 4, same behaviour 4, unpromised 0, no-witness 0, tests added 0
- `cml/softmax/p02-log-softmax`: mutants: survivors 4, same behaviour 4, unpromised 0, no-witness 0, tests added 0
- `cml/stochastic-and-mini-batch-gradient-descent/p01-minibatch-sgd`: mutants: survivors 2, same behaviour 1, unpromised 1, no-witness 0, tests added 0
- `cml/t-tests-and-anova/p01-welch-t-test`: mutants: survivors 10, same behaviour 3, unpromised 3, no-witness 0, tests added 3, question 1
- `cml/t-tests-and-anova/p02-one-way-anova`: mutants: survivors 3, same behaviour 3, unpromised 0, no-witness 0, tests added 0
- `cml/tf-idf-and-bm25/p01-tfidf-matrix`: mutants: survivors 1, same behaviour 1, unpromised 0, no-witness 0, tests added 0 (c79de is the same behaviour: every nonzero row has norm at least 1, so dividing by 1 or not changes nothing)
- `cml/tf-idf-and-bm25/p02-bm25-scores`: mutants: survivors 0, same behaviour 0, unpromised 0, no-witness 0, tests added 0
- `cml/vector-and-matrix-norms/p01-vector-norm`: mutants: survivors 2, same behaviour 2, unpromised 0, no-witness 0, tests added 0
- `cml/vector-operations/p01-vector-projection`: mutants: survivors 2, same behaviour 1, unpromised 1, no-witness 0, tests added 0
- `cml/vector-operations/p01-vector-projection`: mutants: survivors 2, same behaviour 1, unpromised 0, no-witness 0, tests added 1 (supersedes the earlier done note)
- `cml/xgboost-second-order-objective/p01-leaf-weight-and-split-gain`: mutants: survivors 0, same behaviour 0, unpromised 0, no-witness 0, tests added 0
- `cml/xgboost-second-order-objective/p02-logloss-grad-hess`: mutants: survivors 2, same behaviour 2, unpromised 0, no-witness 0, tests added 0
- `dl/a-complete-training-loop/p01-fit-with-seeded-batches`: mutants: survivors 4, same behaviour 2, unpromised 2, no-witness 0, tests added 0
- `dl/adam/p01-adam-step`: mutants: survivors 10, same behaviour 7, unpromised 2, no-witness 0, tests added 1
- `dl/adam/p02-amsgrad-step`: mutants: survivors 9, same behaviour 6, unpromised 2, no-witness 0, tests added 1
- `dl/adamw-decoupled-weight-decay/p01-adamw-step`: mutants: survivors 9, same behaviour 7, unpromised 2, no-witness 0, tests added 0
- `dl/adamw-decoupled-weight-decay/p02-decay-parameter-groups`: mutants: survivors 1, same behaviour 1, unpromised 0, no-witness 0, tests added 0
- `dl/backprop-through-a-chain-of-scalars/p01-scalar-graph-forward-backward`: mutants: survivors 0, same behaviour 0, unpromised 0, no-witness 0, tests added 0
- `dl/backprop-through-a-dense-layer/p01-dense-backward`: mutants: survivors 0, same behaviour 0, unpromised 0, no-witness 0, tests added 0
- `dl/batch-normalization/p01-batch-norm-forward`: mutants: survivors 1, same behaviour 0, unpromised 1, no-witness 0, tests added 0
- `dl/batch-normalization/p02-batch-norm-backward`: mutants: survivors 0, same behaviour 0, unpromised 0, no-witness 0, tests added 0
- `dl/building-tensors-by-broadcasting/p01-pairwise-squared-distances`: mutants: survivors 0, same behaviour 0, unpromised 0, no-witness 0, tests added 0
- `dl/causal-masking/p01-causal-padding-mask`: mutants: survivors 1, same behaviour 1, unpromised 0, no-witness 0, tests added 0
- `dl/convolution-forward-pass/p01-conv2d-forward`: mutants: survivors 2, same behaviour 1, unpromised 1, no-witness 0, tests added 0
- `dl/counting-network-parameters/p01-count-dense-parameters`: mutants: survivors 0, same behaviour 0, unpromised 0, no-witness 0, tests added 0
- `dl/dense-layer-forward-pass/p01-dense-layer-forward`: mutants: survivors 2, same behaviour 2, unpromised 0, no-witness 0, tests added 0
- `dl/dropout/p01-inverted-dropout`: mutants: survivors 1, same behaviour 0, unpromised 0, no-witness 0, tests added 1
- `dl/gradients-with-framework-autograd/p01-gradient-at-a-point`: mutants: survivors 0, same behaviour 0, unpromised 0, no-witness 0, tests added 0
- `dl/gradients-with-framework-autograd/p02-straight-through-round`: mutants: survivors 0, same behaviour 0, unpromised 0, no-witness 0, tests added 0
- `dl/infonce-contrastive-loss/p01-symmetric-infonce-loss`: mutants: survivors 1, same behaviour 1, unpromised 0, no-witness 0, tests added 0
- `dl/kl-divergence/p01-kl-from-logits`: mutants: survivors 2, same behaviour 2, unpromised 0, no-witness 0, tests added 0
- `dl/kl-divergence/p02-diagonal-gaussian-kl`: mutants: survivors 0, same behaviour 0, unpromised 0, no-witness 0, tests added 0
- `dl/layer-normalization/p01-layer-norm-forward`: mutants: survivors 1, same behaviour 0, unpromised 1, no-witness 0, tests added 0
- `dl/learning-rate-decay-schedules/p01-cosine-annealing-lr`: mutants: survivors 1, same behaviour 1, unpromised 0, no-witness 0, tests added 0
- `dl/learning-rate-decay-schedules/p02-step-decay-lr`: mutants: survivors 0, same behaviour 0, unpromised 0, no-witness 0, tests added 0
- `dl/learning-rate-warmup/p01-warmup-then-cosine-lr`: mutants: survivors 1, same behaviour 1, unpromised 0, no-witness 0, tests added 0
- `dl/learning-rate-warmup/p02-inverse-sqrt-lr`: mutants: survivors 0, same behaviour 0, unpromised 0, no-witness 0, tests added 0
- `dl/mixed-precision/p01-dynamic-loss-scaler`: mutants: survivors 0, same behaviour 0, unpromised 0, no-witness 0, tests added 0
- `dl/multi-head-attention/p01-self-attention-heads`: mutants: survivors 3, same behaviour 2, unpromised 1, no-witness 0, tests added 0
- `dl/one-training-step/p01-manual-sgd-train-step`: mutants: survivors 0, same behaviour 0, unpromised 0, no-witness 0, tests added 0
- `dl/position-wise-feed-forward-block/p01-gelu-ffn`: mutants: survivors 1, same behaviour 1, unpromised 0, no-witness 0, tests added 0
- `dl/position-wise-feed-forward-block/p02-swiglu-ffn`: mutants: survivors 2, same behaviour 2, unpromised 0, no-witness 0, tests added 0
- `dl/positional-encodings/p01-sinusoidal-table`: mutants: survivors 2, same behaviour 0, unpromised 2, no-witness 0, tests added 0
- `dl/positional-encodings/p02-rope-rotations`: mutants: survivors 1, same behaviour 0, unpromised 1, no-witness 0, tests added 0
- `dl/relu-family-activations/p01-relu-forward-and-gradient`: mutants: survivors 1, same behaviour 1, unpromised 0, no-witness 0, tests added 0
- `dl/residual-connections/p01-residual-block`: mutants: survivors 0, same behaviour 0, unpromised 0, no-witness 0, tests added 0
- `dl/reverse-mode-autograd-engine/p01-scalar-autograd`: mutants: survivors 0, same behaviour 0, unpromised 0, no-witness 0, tests added 0
- `dl/rmsnorm/p01-rmsnorm-forward`: mutants: survivors 1, same behaviour 0, unpromised 1, no-witness 0, tests added 0
- `dl/scaled-dot-product-attention/p01-scaled-dot-product-attention`: mutants: survivors 4, same behaviour 1, unpromised 3, no-witness 0, tests added 0
- `dl/sgd-with-momentum/p01-momentum-step`: mutants: survivors 5, same behaviour 3, unpromised 2, no-witness 0, tests added 0
- `dl/sgd-with-momentum/p02-nesterov-weight-decay`: mutants: survivors 6, same behaviour 4, unpromised 2, no-witness 0, tests added 0
- `dl/smooth-activations-tanh-gelu-swish/p01-gelu-and-swish`: mutants: survivors 1, same behaviour 1, unpromised 0, no-witness 0, tests added 0
- `dl/smooth-activations-tanh-gelu-swish/p02-smooth-activation-derivatives`: mutants: survivors 1, same behaviour 1, unpromised 0, no-witness 0, tests added 0
- `dl/tensors-in-pytorch-and-tinygrad/p01-batched-pairwise-distances`: mutants: survivors 0, same behaviour 0, unpromised 0, no-witness 0, tests added 0
- `dl/tensors-in-pytorch-and-tinygrad/p02-patchify-images-row-major`: mutants: survivors 0, same behaviour 0, unpromised 0, no-witness 0, tests added 0
- `dl/training-an-mlp-from-scratch/p01-train-two-layer-mlp`: mutants: survivors 8, same behaviour 7, unpromised 0, no-witness 0, tests added 1
- `dl/training-an-mlp-from-scratch/p01-train-two-layer-mlp`: mutants (coordinator, after withdrawing one test): survivors 8, same behaviour 7, unpromised 1, no-witness 0, tests added 0
- `dl/transformer-block/p01-pre-norm-causal-block`: mutants: survivors 2, same behaviour 1, unpromised 1, no-witness 0, tests added 0
- `dl/xavier-and-he-initialization/p01-initialize-dense-weights`: mutants: survivors 1, same behaviour 1, unpromised 0, no-witness 0, tests added 0
- `inf/absmax-and-zero-point-int8-quantization/p01-absmax-quantize`: mutants: survivors 2, same behaviour 2, unpromised 0, no-witness 0, tests added 0
- `inf/absmax-and-zero-point-int8-quantization/p02-zero-point-quantize`: mutants: survivors 4, same behaviour 4, unpromised 0, no-witness 0, tests added 0
- `inf/arithmetic-intensity-and-the-roofline/p01-matmul-roofline`: mutants: survivors 0, same behaviour 0, unpromised 0, no-witness 0, tests added 0
- `inf/continuous-batching/p01-batch-scheduler-steps`: mutants: survivors 6 (10 entries in survivors.json; 0782991c51b85e74, 2dd5eedf5440a19e, 480a62865cf7a1db, 5073912e2cefb266 each listed twice), same behaviour 6, unpromised 0, no-witness 0, tests added 0
- `inf/cost-per-token-and-break-even/p01-cost-per-million-and-break-even`: mutants: survivors 1, same behaviour 1, unpromised 0, no-witness 0, tests added 0
- `inf/data-quality-and-drift-detection/p01-population-stability-index`: mutants: survivors 1, same behaviour 1, unpromised 0, no-witness 0, tests added 0
- `inf/data-quality-and-drift-detection/p02-ks-drift-test`: mutants: survivors 7, same behaviour 3, unpromised 3, no-witness 0, tests added 1
- `inf/floating-point-formats/p01-decode-a-bit-pattern`: mutants: survivors 0, same behaviour 0, unpromised 0, no-witness 0, tests added 0
- `inf/floating-point-formats/p02-round-to-nearest-even`: mutants: survivors 1 (survivors.json lists the same key twice), same behaviour 1, unpromised 0, no-witness 0, tests added 0
- `inf/gpu-memory-and-capacity-sizing/p01-min-gpu-count`: mutants: survivors 0, same behaviour 0, unpromised 0, no-witness 0, tests added 0
- `inf/kv-cache-size-arithmetic/p01-kv-cache-bytes`: mutants: survivors 3, same behaviour 0, unpromised 3, no-witness 0, tests added 0
- `inf/kv-cache-size-arithmetic/p02-max-batch-after-weights`: mutants: survivors 5, same behaviour 2, unpromised 3, no-witness 0, tests added 0
- `inf/paged-kv-cache/p01-block-allocator`: mutants: survivors 2, same behaviour 2, unpromised 0, no-witness 0, tests added 0
- `inf/prefill-vs-decode-bottlenecks/p02-compute-bound-decode-batch`: mutants: survivors 2, same behaviour 2, unpromised 0, no-witness 0, tests added 0
- `inf/prefill-vs-decode-bottlenecks/p01-serving-step-times`: mutants: survivors 0, same behaviour 0, unpromised 0, no-witness 0, tests added 0
- `inf/speculative-decoding-speedup-math/p02-accept-or-resample`: mutants: survivors 7, same behaviour 4, unpromised 3, no-witness 0, tests added 0
- `inf/speculative-decoding-speedup-math/p01-speculative-speedup`: mutants: survivors 0, same behaviour 0, unpromised 0, no-witness 0, tests added 0
- `inf/ttft-itl-and-throughput/p01-latency-percentiles`: mutants: survivors 1, same behaviour 0, unpromised 1, no-witness 0, tests added 0
- `inf/ttft-itl-and-throughput/p02-throughput-and-goodput`: mutants: survivors 2, same behaviour 1, unpromised 1, no-witness 0, tests added 0 (8a4d raises the same ValueError for an empty list, only the message differs)
- `llm/autoregressive-generation-loop/p01-greedy-generation-with-eos`: mutants: survivors 1, same behaviour 1, unpromised 0, no-witness 0, tests added 0
- `llm/bpe-encode-and-decode/p01-apply-bpe-merges`: mutants: survivors 3, same behaviour 3, unpromised 0, no-witness 0, tests added 0 (9f48, 4fa6 and d248 change only the scan bound; the scan reaches the last symbol with a left that equals it only if the left contains the marker, which only the last symbol can contain, so they never differ; 4000 random cases agree)
- `llm/bpe-merge-training/p01-learn-bpe-merges`: mutants: survivors 6, same behaviour 6, unpromised 0, no-witness 0, tests added 0
- `llm/bradley-terry-reward-models/p01-reward-pair-loss-and-accuracy`: mutants: survivors 2, same behaviour 0, unpromised 2, no-witness 0, tests added 0
- `llm/chat-templates-and-tool-call-formats/p01-render-chat-template`: mutants: survivors 3, same behaviour 1, unpromised 0, no-witness 0, tests added 2
- `llm/chat-templates-and-tool-call-formats/p02-parse-tool-call`: mutants: survivors 4, same behaviour 0, unpromised 2, no-witness 0, tests added 2
- `llm/chat-templates-and-tool-call-formats/p01-render-chat-template`: mutants (coordinator, after withdrawing two tests): survivors 3, same behaviour 1, unpromised 2, no-witness 0, tests added 0
- `llm/direct-preference-optimization-dpo/p01-sequence-log-probs-with-response-mask`: mutants: survivors 4, same behaviour 0, unpromised 2, no-witness 0, tests added 1
- `llm/direct-preference-optimization-dpo/p02-dpo-loss-and-implicit-rewards`: mutants: survivors 2, same behaviour 0, unpromised 2, no-witness 0, tests added 0
- `llm/direct-preference-optimization-dpo/p01-sequence-log-probs-with-response-mask`: mutants (coordinator, after withdrawing one test): survivors 4, same behaviour 0, unpromised 4, no-witness 0, tests added 0
- `llm/document-packing-and-cross-document-masks/p01-packed-causal-mask`: mutants: survivors 7, same behaviour 7, unpromised 0, no-witness 0, tests added 0
- `llm/gpt-forward-pass/p01-decoder-only-gpt-forward`: mutants: survivors 0, same behaviour 0, unpromised 0, no-witness 0, tests added 0
- `llm/gqa-mqa-and-cross-layer-kv-sharing/p01-grouped-query-attention`: mutants: survivors 2, same behaviour 2, unpromised 0, no-witness 0, tests added 0
- `llm/grpo-and-its-variants/p01-group-normalised-advantages`: mutants: survivors 0, same behaviour 0, unpromised 0, no-witness 0, tests added 0
- `llm/grpo-and-its-variants/p02-clipped-grpo-token-loss`: mutants: survivors 1, same behaviour 1, unpromised 0, no-witness 0, tests added 0
- `llm/incremental-decoding-with-a-kv-cache/p01-cached-causal-attention-step`: mutants: survivors 2, same behaviour 2, unpromised 0, no-witness 0, tests added 0
- `llm/instruction-tuning-batches-and-loss-masks/p01-sft-batch-with-loss-masks`: mutants: survivors 0, same behaviour 0, unpromised 0, no-witness 0, tests added 0
- `llm/llm-as-judge-and-agreement-with-humans/p01-cohens-kappa`: mutants: survivors 4, same behaviour 3, unpromised 0, no-witness 0, tests added 1
- `llm/llm-as-judge-and-agreement-with-humans/p02-debiased-win-rate`: mutants: survivors 1, same behaviour 1, unpromised 0, no-witness 0, tests added 0
- `llm/lora-adapters/p01-lora-linear-forward-and-merge`: mutants: survivors 3, same behaviour 2, unpromised 1, no-witness 0, tests added 0
- `llm/lora-parameter-and-memory-arithmetic/p01-lora-trainable-parameter-budget`: mutants: survivors 4, same behaviour 4, unpromised 0, no-witness 0, tests added 0
- `llm/lora-parameter-and-memory-arithmetic/p02-finetune-memory-in-bytes`: mutants: survivors 1, same behaviour 1, unpromised 0, no-witness 0, tests added 0
- `llm/model-parameter-and-memory-footprint/p01-count-decoder-parameters`: mutants: survivors 0, same behaviour 0, unpromised 0, no-witness 0, tests added 0
- `llm/model-parameter-and-memory-footprint/p02-training-and-inference-memory`: mutants: survivors 2, same behaviour 2, unpromised 0, no-witness 0, tests added 0
- `llm/moe-top-k-routing/p01-top-k-router-gates`: mutants: survivors 1, same behaviour 0, unpromised 0, no-witness 0, tests added 1
- `llm/moe-top-k-routing/p02-load-balancing-loss`: mutants: survivors 2, same behaviour 1, unpromised 0, no-witness 0, tests added 1
- `llm/negative-log-likelihood-and-perplexity/p01-masked-nll-and-perplexity`: mutants: survivors 2, same behaviour 2, unpromised 0, no-witness 0, tests added 0
- `llm/next-token-targets-and-batches/p01-pack-next-token-blocks`: mutants: survivors 4, same behaviour 4, unpromised 0, no-witness 0, tests added 0
- `llm/padding-truncation-and-pad-masks/p01-pad-and-truncate-token-ids`: mutants: survivors 3, same behaviour 3, unpromised 0, no-witness 0, tests added 0
- `llm/pass-k-and-majority-voting/p01-unbiased-pass-at-k`: mutants: survivors 3, same behaviour 2, unpromised 1, no-witness 0, tests added 0
- `llm/pass-k-and-majority-voting/p02-majority-vote`: mutants: survivors 4, same behaviour 4, unpromised 0, no-witness 0, tests added 0
- `llm/retrieval-augmented-generation-pipeline/p01-chunk-words`: mutants: survivors 2, same behaviour 2, unpromised 0, no-witness 0, tests added 0
- `llm/retrieval-augmented-generation-pipeline/p02-cited-context-prompt`: mutants: survivors 2, same behaviour 2, unpromised 0, no-witness 0, tests added 0
- `llm/rotary-position-embeddings-rope/p01-half-split-rope-with-offset`: mutants: survivors 3, same behaviour 3, unpromised 0, no-witness 0, tests added 0
- `llm/rotary-position-embeddings-rope/p02-linear-position-interpolation`: mutants: survivors 3, same behaviour 3, unpromised 0, no-witness 0, tests added 0
- `llm/scaling-laws-and-compute-budgets/p01-compute-optimal-allocation`: mutants: survivors 4, same behaviour 4, unpromised 0, no-witness 0, tests added 0
- `llm/scaling-laws-and-compute-budgets/p02-fit-power-law-log-log`: mutants: survivors 3, same behaviour 3, unpromised 0, no-witness 0, tests added 0
- `llm/temperature-top-k-and-top-p-sampling/p01-filtered-next-token-distribution`: mutants: survivors 3, same behaviour 2, unpromised 1, no-witness 0, tests added 0
- `llm/temperature-top-k-and-top-p-sampling/p02-sample-with-a-generator`: mutants: survivors 10, same behaviour 2, unpromised 8, no-witness 0, tests added 0
- `llm/word-and-character-tokenizers/p01-word-level-vocabulary`: mutants: survivors 0, same behaviour 0, unpromised 0, no-witness 0, tests added 0
- `math/determinants-and-trace/p01-determinant`: mutants: survivors 2, same behaviour 2, unpromised 0, no-witness 0, tests added 0
- `math/inverse-transform-sampling/p01-sample-exponential`: mutants: survivors 0, same behaviour 0, unpromised 0, no-witness 0, tests added 0
- `math/inverse-transform-sampling/p02-sample-discrete`: mutants: survivors 2, same behaviour 0, unpromised 2, no-witness 0, tests added 0
- `math/monte-carlo-simulation/p01-mc-estimate`: mutants: survivors 2, same behaviour 0, unpromised 2, no-witness 0, tests added 0
- `math/reservoir-sampling/p01-reservoir-sample`: mutants: survivors 0, same behaviour 0, unpromised 0, no-witness 0, tests added 0
- `rl/action-value-estimates/p01-incremental-action-values`: mutants: survivors 3, same behaviour 1, unpromised 1, no-witness 1, tests added 0
- `rl/epsilon-greedy-exploration/p01-epsilon-greedy-action`: mutants: survivors 3, same behaviour 2, unpromised 1, no-witness 0, tests added 0
- `rl/mdps-and-discounted-returns/p01-discounted-returns`: mutants: survivors 2, same behaviour 2, unpromised 0, no-witness 0, tests added 0
- `rl/mdps-and-discounted-returns/p02-iterative-policy-evaluation`: mutants: survivors 6, same behaviour 1, unpromised 5, no-witness 0, tests added 0
- `rl/q-learning/p01-q-learning-gridworld`: mutants: survivors 0, same behaviour 0, unpromised 0, no-witness 0, tests added 0
- `rl/reinforce/p01-reinforce-loss`: mutants: survivors 2, same behaviour 1, unpromised 1, no-witness 0, tests added 0
- `rl/sarsa-and-expected-sarsa/p01-sarsa-and-expected-sarsa-updates`: mutants: survivors 10, same behaviour 6, unpromised 4, no-witness 0, tests added 0
- `safe/split-conformal-prediction/p01-split-conformal-intervals`: mutants: survivors 3, same behaviour 3, unpromised 0, no-witness 0, tests added 0
- `safe/split-conformal-prediction/p02-conformal-prediction-sets`: mutants: survivors 1, same behaviour 0, unpromised 1, no-witness 0, tests added 0
- `sql/pandas-groupby-aggregate-and-transform/p01-customer-order-features`: mutants: survivors 1, same behaviour 0, unpromised 1, no-witness 0, tests added 0
- `sql/pandas-merge-and-reshape/p01-monthly-revenue-by-region`: mutants: survivors 1, same behaviour 1, unpromised 0, no-witness 0, tests added 0
- `sql/sql-cohort-retention/p01-cohort-retention-by-month`: mutants: survivors 0, same behaviour 0, unpromised 0, no-witness 0, tests added 0
- `sql/sql-conditional-aggregation/p01-channel-conversion-and-device-revenue`: mutants: survivors 0, same behaviour 0, unpromised 0, no-witness 0, tests added 0
- `sql/sql-date-handling/p01-monthly-active-users`: mutants: survivors 0, same behaviour 0, unpromised 0, no-witness 0, tests added 0
- `sql/sql-date-handling/p02-days-from-first-to-second-event`: mutants: survivors 0, same behaviour 0, unpromised 0, no-witness 0, tests added 0
- `sql/sql-first-and-last-activity-per-user/p01-first-and-last-event`: mutants: survivors 0, same behaviour 0, unpromised 0, no-witness 0, tests added 0
- `sql/sql-group-by-and-having/p01-customers-with-three-completed-orders`: mutants: survivors 0, same behaviour 0, unpromised 0, no-witness 0, tests added 0
- `sql/sql-joins-and-unions/p01-customer-order-totals`: mutants: survivors 0, same behaviour 0, unpromised 0, no-witness 0, tests added 0
- `sql/sql-joins-and-unions/p02-warehouses-with-a-shipment-or-a-return`: mutants: survivors 0, same behaviour 0, unpromised 0, no-witness 0, tests added 0
- `sql/sql-lag-and-lead/p01-day-over-day-change`: mutants: survivors 0, same behaviour 0, unpromised 0, no-witness 0, tests added 0
- `sql/sql-lag-and-lead/p02-sessions-by-inactivity-gap`: mutants: survivors 0, same behaviour 0, unpromised 0, no-witness 0, tests added 0
- `sql/sql-ranking-window-functions/p01-rank-within-category`: mutants: survivors 0, same behaviour 0, unpromised 0, no-witness 0, tests added 0
- `sql/sql-ranking-window-functions/p02-top-two-with-ties`: mutants: survivors 0, same behaviour 0, unpromised 0, no-witness 0, tests added 0
- `sql/sql-running-totals-and-frames/p01-running-balance-with-ties`: mutants: survivors 0, same behaviour 0, unpromised 0, no-witness 0, tests added 0
- `sql/sql-running-totals-and-frames/p02-trailing-seven-day-average`: mutants: survivors 0, same behaviour 0, unpromised 0, no-witness 0, tests added 0
- `sql/sql-select-filter-and-sort/p01-top-five-completed-orders-2024`: mutants: survivors 0, same behaviour 0, unpromised 0, no-witness 0, tests added 0
- `sql/sql-streaks-gaps-and-islands/p01-longest-streak-per-user`: mutants: survivors 0, same behaviour 0, unpromised 0, no-witness 0, tests added 0
- `sql/sql-streaks-gaps-and-islands/p02-all-islands-per-user`: mutants: survivors 0, same behaviour 0, unpromised 0, no-witness 0, tests added 0
- `sql/sql-subqueries-and-ctes/p01-customers-above-average-spend`: mutants: survivors 0, same behaviour 0, unpromised 0, no-witness 0, tests added 0
- `sql/sql-subqueries-and-ctes/p02-orders-above-their-customers-average`: mutants: survivors 0, same behaviour 0, unpromised 0, no-witness 0, tests added 0
- `sql/sql-top-n-and-latest-row-per-group/p01-latest-events`: mutants: survivors 0, same behaviour 0, unpromised 0, no-witness 0, tests added 0

ledger: 695 rows

## Declared against enforced tolerance

| Problem | Declared rtol | Smallest error the tests catch | Reading |
|---|---:|---:|---|
| `cml/bernoulli-binomial-and-poisson/p01-binomial-pmf` | 1e-07 | 1e-08 | tests are stricter than declared |
| `cml/gaussian-mixtures-and-em/p01-gmm-em-full-covariance` | 1e-07 | 1e-08 | tests are stricter than declared |
| `cml/gradient-descent/p01-gradient-descent` | 1e-06 | 1e-08 | tests are stricter than declared |
| `cml/lagrange-multipliers/p01-constrained-quadratic-min` | 1e-07 | 1e-08 | tests are stricter than declared |
| `cml/logistic-regression/p01-l2-logistic-fit` | 1e-06 | 1e-08 | tests are stricter than declared |
| `dl/dropout/p01-inverted-dropout` | 1e-06 | 1e-08 | tests are stricter than declared |
| `dl/gradients-with-framework-autograd/p01-gradient-at-a-point` | 1e-06 | 1e-03 | tests are looser than declared |
| `dl/training-an-mlp-from-scratch/p01-train-two-layer-mlp` | 1e-06 | 1e-08 | tests are stricter than declared |

Reported, not changed. A person decides.

## Enforcement

- **A1, lint in validation: on.** `dogma content validate` reports each hidden-test lint finding
  under the rule's name.
- **A2, id-leak check in `content verify`: off.** With the check in place,
  `dogma content verify --all` ended with `25 of 175 topic(s) failed`: 35 problems have
  parametrized hidden tests whose ids are neutral but numbered (for example `case-0`, `grid0`,
  `random-2`), and the rule refuses any digit in an id. The change was undone as the runbook
  says. The rule or the ids have to change before it can be turned on. The 35 problems:

  - `algo/heaps-and-priority-queues/p01-merge-sorted-streams`
  - `algo/heaps-and-priority-queues/p02-running-median`
  - `cml/a-b-test-design-and-analysis/p01-sample-size-two-proportions`
  - `cml/a-b-test-design-and-analysis/p02-two-proportion-z-test`
  - `cml/bootstrap-and-permutation-resampling/p01-bootstrap-percentile-ci`
  - `cml/bootstrap-and-permutation-resampling/p02-permutation-test-mean-diff`
  - `cml/confusion-matrix-and-count-based-scores/p01-confusion-matrix-from-labels`
  - `cml/cross-entropy/p02-binary-cross-entropy-with-logits`
  - `cml/handling-class-imbalance/p01-balanced-class-weights`
  - `cml/handling-class-imbalance/p02-prior-shift-adjust`
  - `cml/lasso-and-elastic-net/p01-lasso-coordinate-descent`
  - `cml/logistic-regression/p01-l2-logistic-fit`
  - `cml/maximum-likelihood-and-map/p01-gaussian-mle`
  - `cml/precision-recall-and-f1/p01-precision-recall-averaging`
  - `cml/precision-recall-and-f1/p02-best-f1-threshold`
  - `cml/ranking-metrics/p01-ndcg-at-k`
  - `cml/ranking-metrics/p02-retrieval-metrics-at-k`
  - `cml/ridge-regression-and-weight-decay/p01-ridge-fit`
  - `cml/roc-pr-curves-and-auc/p01-roc-auc-rank-statistic`
  - `cml/roc-pr-curves-and-auc/p02-average-precision-step`
  - `cml/sampling-distributions-standard-errors-and-cis/p01-mean-confidence-interval`
  - `cml/sigmoid-and-log-odds/p01-stable-sigmoid`
  - `cml/sigmoid-and-log-odds/p02-logit`
  - `cml/t-tests-and-anova/p02-one-way-anova`
  - `cml/tf-idf-and-bm25/p01-tfidf-matrix`
  - `cml/tf-idf-and-bm25/p02-bm25-scores`
  - `dl/dense-layer-forward-pass/p01-dense-layer-forward`
  - `inf/cost-per-token-and-break-even/p01-cost-per-million-and-break-even`
  - `inf/gpu-memory-and-capacity-sizing/p01-min-gpu-count`
  - `inf/kv-cache-size-arithmetic/p01-kv-cache-bytes`
  - `llm/lora-adapters/p01-lora-linear-forward-and-merge`
  - `llm/lora-parameter-and-memory-arithmetic/p02-finetune-memory-in-bytes`
  - `math/inverse-transform-sampling/p02-sample-discrete`
  - `safe/split-conformal-prediction/p01-split-conformal-intervals`
  - `safe/split-conformal-prediction/p02-conformal-prediction-sets`

- **A3, variants required: on.** `VARIANTS_REQUIRED = True`: a `verified` or `reviewed` problem
  needs a blind accept and, for its Contract, at least one reject or `uncovered` entry.

## Scorecard

Output of `scorecard`:

| Problem | Version | Lint | Blind | Accepts | Rejects | Bullets | Sweeps | Notes |
|---|---|---|---|---|---|---|---|---|
| algo/bfs-and-dfs-on-graphs-and-grids/p01-grid-shortest-path | 4 | 0 | yes | 2 | 7 | 5/5 | reject,mutants | unpromised 1, done 1 |
| algo/bfs-and-dfs-on-graphs-and-grids/p02-count-components | 4 | 0 | yes | 2 | 5 | 3/3 | reject,mutants | unpromised 1, done 1 |
| algo/binary-search/p01-lower-bound | 2 | 0 | yes | 2 | 6 | 6/6 | reject,mutants | question 1, message-defect 1, done 1 |
| algo/binary-search/p02-min-ship-capacity | 1 | 0 | yes | 2 | 6 | 5/5 | reject,mutants | unpromised 2, done 1 |
| algo/binary-tree-recursion-and-traversal/p01-inorder-traversal-iterative | 2 | 0 | yes | 1 | 6 | 6/6 | reject,mutants | done 1 |
| algo/binary-tree-recursion-and-traversal/p02-tree-diameter | 3 | 0 | yes | 2 | 6 | 6/6 | reject,mutants | blocked 1, done 1 |
| algo/dynamic-programming/p01-minimum-coins | 3 | 0 | yes | 1 | 6 | 6/6 | reject,mutants | done 1 |
| algo/dynamic-programming/p02-longest-common-subsequence | 3 | 0 | yes | 2 | 5 | 5/5 | reject,mutants | unpromised 1, message-defect 1, done 1 |
| algo/hash-maps-and-sets/p01-top-k-frequent-tokens | 1 | 0 | yes | 1 | 7 | 6/6 | reject,mutants | unpromised 2, message-defect 1, done 2 |
| algo/heaps-and-priority-queues/p01-merge-sorted-streams | 2 | 0 | yes | 1 | 7 | 7/7 | reject,mutants | unpromised 1, done 1 |
| algo/heaps-and-priority-queues/p02-running-median | 2 | 0 | yes | 1 | 6 | 6/6 | reject,mutants | unpromised 5, done 1 |
| algo/lru-cache/p01-lru-cache | 3 | 0 | yes | 2 | 8 | 7/7 | reject,mutants | done 1 |
| algo/prefix-sums/p01-subarrays-with-sum-k | 1 | 0 | yes | 1 | 7 | 6/6 | reject,mutants | done 1 |
| algo/prefix-sums/p02-rectangle-range-sums | 2 | 0 | yes | 1 | 8 | 8/8 | reject,mutants | unpromised 1, message-defect 4, done 1 |
| algo/sort-then-sweep/p01-merge-intervals | 2 | 0 | yes | 1 | 8 | 6/6 | reject,mutants | done 1 |
| algo/sort-then-sweep/p02-peak-concurrent-sessions | 2 | 0 | yes | 1 | 7 | 6/6 | reject,mutants | done 1 |
| algo/stacks-and-queues/p01-days-until-higher | 2 | 0 | yes | 1 | 7 | 6/6 | reject,mutants | done 1 |
| algo/stacks-and-queues/p02-sliding-window-maximum | 3 | 0 | yes | 1 | 6 | 6/6 | reject,mutants | done 1 |
| algo/topological-sort/p01-lexicographic-topological-order | 4 | 0 | yes | 1 | 7 | 8/8 | reject,mutants | question 1, unpromised 2, done 2 |
| algo/two-pointers-and-sliding-windows/p01-longest-window-within-budget | 3 | 0 | yes | 1 | 8 | 6/6 | reject,mutants | done 2 |
| cml/a-b-test-design-and-analysis/p01-sample-size-two-proportions | 4 | 0 | yes | 1 | 7 | 7/7 | reject,mutants | overreach 1, done 1 |
| cml/a-b-test-design-and-analysis/p02-two-proportion-z-test | 4 | 0 | yes | 1 | 6 | 6/6 | reject,mutants | done 1 |
| cml/adaboost-and-gradient-boosting/p01-boost-stumps-mse | 4 | 0 | yes | 2 | 7 | 7/7 | reject,mutants | unpromised 1, done 1 |
| cml/adaboost-and-gradient-boosting/p02-adaboost-round | 4 | 0 | yes | 2 | 9 | 7/7 | reject,mutants | unpromised 2, done 1 |
| cml/bayes-theorem/p01-bayes-posterior | 5 | 0 | yes | 2 | 7 | 8/8 | reject,mutants | unpromised 6, done 2 |
| cml/bernoulli-binomial-and-poisson/p01-binomial-pmf | 4 | 0 | yes | 2 | 6 | 6/6 | reject,mutants | unpromised 1, done 2 |
| cml/bernoulli-binomial-and-poisson/p02-poisson-pmf | 4 | 0 | yes | 2 | 6 | 5/5 | reject,mutants | done 2 |
| cml/bias-variance-decomposition/p01-bias-variance-from-predictions | 4 | 0 | yes | 2 | 10 | 8/8 | reject,mutants | unpromised 2, no-witness 1, overreach 1, message-defect 1, done 1 |
| cml/bootstrap-and-permutation-resampling/p01-bootstrap-percentile-ci | 5 | 0 | yes | 2 | 8 | 9/9 | reject,mutants | inspect 1, done 1 |
| cml/bootstrap-and-permutation-resampling/p02-permutation-test-mean-diff | 4 | 0 | yes | 2 | 7 | 7/7 | reject,mutants | no-witness 1, done 1 |
| cml/chain-rule/p01-mse-sigmoid-gradient | 3 | 0 | yes | 2 | 10 | 8/8 | reject,mutants | unpromised 1, done 2 |
| cml/choosing-a-tree-split/p01-best-split-threshold | 2 | 0 | yes | 2 | 8 | 7/7 | reject,mutants | unpromised 2, no-witness 1, done 2 |
| cml/confusion-matrix-and-count-based-scores/p01-confusion-matrix-from-labels | 5 | 0 | yes | 1 | 9 | 7/7 | reject,mutants | no-witness 1, done 1 |
| cml/confusion-matrix-and-count-based-scores/p02-binary-count-scores | 3 | 0 | yes | 3 | 8 | 6/6 | reject,mutants | done 1 |
| cml/cosine-similarity/p01-cosine-similarity-matrix | 4 | 0 | yes | 3 | 7 | 6/6 | reject,mutants | inspect 1, done 2 |
| cml/covariance-and-correlation/p01-covariance-matrix | 3 | 0 | yes | 2 | 6 | 5/5 | reject,mutants | unpromised 2, message-defect 1, done 1 |
| cml/covariance-and-correlation/p02-correlation-matrix | 3 | 0 | yes | 2 | 7 | 5/5 | reject,mutants | unpromised 6, done 1 |
| cml/cross-entropy/p01-cross-entropy-from-logits | 4 | 0 | yes | 2 | 6 | 6/6 | reject,mutants | done 1 |
| cml/cross-entropy/p02-binary-cross-entropy-with-logits | 5 | 0 | yes | 2 | 7 | 7/7 | reject,mutants | done 1 |
| cml/derivatives/p01-numerical-derivative | 4 | 0 | yes | 2 | 8 | 6/6 | reject,mutants | done 2 |
| cml/descriptive-statistics/p01-welford-mean-variance | 1 | 0 | yes | 1 | 6 | 4/4 | reject,mutants | message-defect 1, done 2 |
| cml/descriptive-statistics/p02-iqr-outlier-fences | 1 | 0 | yes | 2 | 8 | 5/5 | reject,mutants | message-defect 2, done 2 |
| cml/eigenvalues-and-eigenvectors/p01-power-iteration | 4 | 0 | yes | 2 | 10 | 9/9 | reject,mutants | unpromised 3, message-defect 3, done 3 |
| cml/entropy-and-mutual-information/p01-distribution-entropy | 4 | 0 | yes | 2 | 7 | 5/5 | reject,mutants | unpromised 1, done 1 |
| cml/entropy-and-mutual-information/p02-mutual-information-from-labels | 3 | 0 | yes | 2 | 6 | 4/4 | reject,mutants | done 1 |
| cml/expectation-and-variance/p01-discrete-moments | 4 | 0 | yes | 2 | 6 | 5/5 | reject,mutants | unpromised 4, done 1 |
| cml/feature-scaling/p01-standard-scaler | 3 | 0 | yes | 2 | 8 | 10/10 | reject,mutants | unpromised 1, message-defect 2, done 2 |
| cml/feature-scaling/p02-robust-median-iqr-scaler | 5 | 0 | yes | 2 | 8 | 10/10 | reject,mutants | unpromised 2, message-defect 2, done 1 |
| cml/gaussian-mixtures-and-em/p01-gmm-em-full-covariance | 4 | 0 | yes | 3 | 8 | 8/8 | reject,mutants | unpromised 3, message-defect 1, done 2 |
| cml/gradient-descent/p01-gradient-descent | 2 | 0 | yes | 2 | 8 | 8/8 | reject,mutants | unpromised 1, message-defect 1, done 2 |
| cml/handling-class-imbalance/p01-balanced-class-weights | 1 | 0 | yes | 2 | 5 | 5/5 | reject,mutants | overreach 1, done 1 |
| cml/handling-class-imbalance/p02-prior-shift-adjust | 2 | 0 | yes | 2 | 6 | 6/6 | reject,mutants | unpromised 1, done 1 |
| cml/k-fold-cross-validation/p01-kfold-indices | 1 | 0 | yes | 1 | 8 | 10/10 | reject,mutants | message-defect 1, done 1 |
| cml/k-fold-cross-validation/p02-cross-validate | 3 | 0 | yes | 1 | 6 | 8/8 | reject,mutants | unpromised 3, done 1 |
| cml/k-means/p01-lloyd-kmeans | 4 | 0 | yes | 2 | 6 | 6/6 | reject,mutants | unpromised 1, message-defect 1, done 1 |
| cml/k-means/p02-kmeans-plus-plus-seeding | 4 | 0 | yes | 2 | 9 | 8/8 | reject,mutants | unpromised 1, done 1 |
| cml/lagrange-multipliers/p01-constrained-quadratic-min | 4 | 0 | yes | 2 | 7 | 7/7 | reject,mutants | unpromised 2, message-defect 1, done 2 |
| cml/lasso-and-elastic-net/p01-lasso-coordinate-descent | 5 | 0 | yes | 2 | 9 | 9/9 | reject,mutants | unpromised 1, done 1 |
| cml/lasso-and-elastic-net/p02-elastic-net-coordinate-descent | 4 | 0 | yes | 2 | 9 | 9/9 | reject,mutants | message-defect 1, done 1 |
| cml/leakage-and-walk-forward-validation/p01-walk-forward-splits | 1 | 0 | yes | 2 | 8 | 11/11 | reject,mutants | done 1 |
| cml/leakage-and-walk-forward-validation/p02-expanding-target-mean | 4 | 0 | yes | 1 | 8 | 8/8 | reject,mutants | unpromised 1, overreach 1, done 1 |
| cml/logistic-regression/p01-l2-logistic-fit | 5 | 0 | yes | 2 | 11 | 10/10 | reject,mutants | unpromised 6, done 1 |
| cml/matrix-inverse-and-rank/p01-gauss-jordan-inverse | 5 | 0 | yes | 2 | 6 | 6/6 | reject,mutants | unpromised 1, done 2 |
| cml/matrix-inverse-and-rank/p02-matrix-rank | 5 | 0 | yes | 2 | 8 | 6/6 | reject,mutants | message-defect 1, done 2 |
| cml/matrix-multiplication/p01-matrix-multiply | 3 | 0 | yes | 2 | 7 | 4/4 | reject,mutants | unpromised 1, message-defect 1, inspect 1, done 2 |
| cml/matrix-shapes-and-transpose/p01-transpose-matrix | 2 | 0 | yes | 1 | 5 | 4/4 | reject,mutants | unpromised 1, done 3 |
| cml/maximum-likelihood-and-map/p01-gaussian-mle | 3 | 0 | yes | 2 | 4 | 4/4 | reject,mutants | message-defect 1, done 1 |
| cml/maximum-likelihood-and-map/p02-beta-bernoulli-map | 2 | 0 | yes | 2 | 3 | 4/4 | reject,mutants | unpromised 2, done 1 |
| cml/missing-values-and-outliers/p01-impute-with-missing-indicators | 5 | 0 | yes | 1 | 10 | 7/7 | reject,mutants | done 1 |
| cml/missing-values-and-outliers/p02-modified-zscore-outliers | 5 | 0 | yes | 2 | 10 | 8/8 | reject,mutants | unpromised 3, done 1 |
| cml/nearest-neighbour-search-and-knn/p01-knn-classify | 4 | 0 | yes | 2 | 8 | 8/8 | reject,mutants | message-defect 1, done 1 |
| cml/nearest-neighbour-search-and-knn/p02-knn-regress | 4 | 0 | yes | 2 | 8 | 8/8 | reject,mutants | unpromised 1, message-defect 1, done 1 |
| cml/normal-and-multivariate-gaussian/p01-mvn-logpdf | 3 | 0 | yes | 2 | 8 | 6/6 | reject,mutants | unpromised 1, message-defect 1, done 1 |
| cml/normal-equation/p01-normal-equation-fit | 5 | 0 | yes | 1 | 8 | 6/6 | reject,mutants | unpromised 1, message-defect 2, done 2 |
| cml/over-and-underfitting-early-stopping/p01-early-stopping | 4 | 0 | yes | 1 | 8 | 9/9 | reject,mutants | done 1 |
| cml/partial-derivatives-and-gradients/p01-numerical-gradient | 3 | 0 | yes | 3 | 8 | 5/5 | reject,mutants | done 2 |
| cml/precision-recall-and-f1/p01-precision-recall-averaging | 4 | 0 | yes | 1 | 8 | 12/12 | reject,mutants | unpromised 1, done 1 |
| cml/precision-recall-and-f1/p02-best-f1-threshold | 5 | 0 | yes | 2 | 7 | 10/10 | reject,mutants | unpromised 2, done 1 |
| cml/preprocessing-pipelines/p01-numeric-categorical-pipeline | 6 | 0 | yes | 1 | 9 | 7/7 | reject,mutants | done 1 |
| cml/principal-component-analysis/p01-pca-via-svd | 4 | 0 | yes | 1 | 9 | 7/7 | reject,mutants | unpromised 4, done 2 |
| cml/principal-component-analysis/p02-components-for-variance | 5 | 0 | yes | 1 | 8 | 6/6 | reject,mutants | unpromised 3, done 1 |
| cml/probability-basics/p01-marginals-and-conditional | 3 | 0 | yes | 2 | 8 | 4/4 | reject,mutants | unpromised 1, inspect 1, done 2 |
| cml/random-forests-and-feature-importance/p01-permutation-importance | 3 | 0 | yes | 1 | 8 | 9/9 | reject,mutants | unpromised 1, done 1 |
| cml/random-forests-and-feature-importance/p02-bootstrap-oob-masks | 3 | 0 | yes | 1 | 6 | 6/6 | reject,mutants | unpromised 1, done 1 |
| cml/ranking-metrics/p01-ndcg-at-k | 3 | 0 | yes | 2 | 9 | 10/10 | reject,mutants | unpromised 2, done 1 |
| cml/ranking-metrics/p02-retrieval-metrics-at-k | 2 | 0 | yes | 1 | 8 | 11/11 | reject,mutants | done 1 |
| cml/regression-metrics-and-robust-losses/p01-regression-report | 4 | 0 | yes | 2 | 7 | 6/6 | reject,mutants | done 1 |
| cml/regression-metrics-and-robust-losses/p02-huber-loss-gradient | 2 | 0 | yes | 2 | 7 | 7/7 | reject,mutants | done 1 |
| cml/ridge-regression-and-weight-decay/p01-ridge-fit | 3 | 0 | yes | 2 | 8 | 9/9 | reject,mutants | unpromised 1, done 1 |
| cml/roc-pr-curves-and-auc/p01-roc-auc-rank-statistic | 4 | 0 | yes | 2 | 8 | 6/6 | reject,mutants | unpromised 1, done 1 |
| cml/roc-pr-curves-and-auc/p02-average-precision-step | 4 | 0 | yes | 2 | 10 | 8/8 | reject,mutants | done 1 |
| cml/rolling-statistics-stationarity-and-autocorrelation/p01-rolling-mean-std | 6 | 0 | yes | 1 | 7 | 5/5 | reject,mutants | unpromised 4, overreach 1, done 1 |
| cml/rolling-statistics-stationarity-and-autocorrelation/p02-acf | 6 | 0 | yes | 2 | 7 | 4/4 | reject,mutants | unpromised 4, done 1 |
| cml/rolling-statistics-stationarity-and-autocorrelation/p03-seasonal-naive-forecast-and-mase | 3 | 0 | yes | 3 | 8 | 9/9 | reject,mutants | done 1 |
| cml/sampling-distributions-standard-errors-and-cis/p01-mean-confidence-interval | 3 | 0 | yes | 3 | 6 | 5/5 | reject,mutants | message-defect 2, done 1 |
| cml/shuffling-splitting-and-batching-data/p01-stratified-train-test-split | 4 | 0 | yes | 1 | 8 | 7/7 | reject,mutants | question 1, unpromised 3, done 1 |
| cml/shuffling-splitting-and-batching-data/p02-minibatch-index-schedule | 4 | 0 | yes | 1 | 8 | 7/7 | reject,mutants | question 1, unpromised 2, message-defect 1, done 1 |
| cml/sigmoid-and-log-odds/p01-stable-sigmoid | 3 | 0 | yes | 2 | 6 | 6/6 | reject,mutants | done 1 |
| cml/sigmoid-and-log-odds/p02-logit | 3 | 0 | yes | 2 | 5 | 5/5 | reject,mutants | unpromised 3, done 1 |
| cml/singular-value-decomposition/p01-best-rank-k | 2 | 0 | yes | 1 | 5 | 6/6 | reject,mutants | unpromised 3, done 1 |
| cml/singular-value-decomposition/p02-pseudo-inverse | 4 | 0 | yes | 1 | 6 | 6/6 | reject,mutants | unpromised 7, done 1 |
| cml/softmax/p01-stable-softmax | 2 | 0 | yes | 3 | 7 | 6/6 | reject,mutants | done 1 |
| cml/softmax/p02-log-softmax | 2 | 0 | yes | 3 | 8 | 7/7 | reject,mutants | done 1 |
| cml/stochastic-and-mini-batch-gradient-descent/p01-minibatch-sgd | 3 | 0 | yes | 3 | 9 | 9/9 | reject,mutants | unpromised 1, message-defect 3, done 2 |
| cml/t-tests-and-anova/p01-welch-t-test | 5 | 0 | yes | 1 | 12 | 5/5 | reject,mutants | question 1, unpromised 3, message-defect 1, done 2 |
| cml/t-tests-and-anova/p02-one-way-anova | 4 | 0 | yes | 1 | 7 | 5/5 | reject,mutants | done 1 |
| cml/tf-idf-and-bm25/p01-tfidf-matrix | 5 | 0 | yes | 2 | 10 | 10/10 | reject,mutants | done 1 |
| cml/tf-idf-and-bm25/p02-bm25-scores | 3 | 0 | yes | 2 | 10 | 10/10 | reject,mutants | done 1 |
| cml/vector-and-matrix-norms/p01-vector-norm | 2 | 0 | yes | 2 | 7 | 6/6 | reject,mutants | message-defect 1, done 2 |
| cml/vector-operations/p01-vector-projection | 3 | 0 | yes | 3 | 7 | 4/4 | reject,mutants | unpromised 1, message-defect 1, done 3 |
| cml/xgboost-second-order-objective/p01-leaf-weight-and-split-gain | 2 | 0 | yes | 2 | 7 | 8/8 | reject,mutants | unpromised 1, message-defect 2, done 1 |
| cml/xgboost-second-order-objective/p02-logloss-grad-hess | 2 | 0 | yes | 2 | 7 | 6/6 | reject,mutants | done 1 |
| dl/a-complete-training-loop/p01-fit-with-seeded-batches | 4 | 0 | yes | 3 | 8 | 8/8 | reject,mutants | unpromised 2, done 1 |
| dl/adam/p01-adam-step | 3 | 0 | yes | 3 | 7 | 7/7 | reject,mutants | unpromised 4, done 2 |
| dl/adam/p02-amsgrad-step | 4 | 0 | yes | 3 | 7 | 7/7 | reject,mutants | unpromised 2, done 2 |
| dl/adamw-decoupled-weight-decay/p01-adamw-step | 3 | 0 | yes | 2 | 8 | 7/7 | reject,mutants | question 1, unpromised 2, no-witness 2, done 2 |
| dl/adamw-decoupled-weight-decay/p02-decay-parameter-groups | 4 | 0 | yes | 1 | 7 | 7/7 | reject,mutants | done 1 |
| dl/backprop-through-a-chain-of-scalars/p01-scalar-graph-forward-backward | 4 | 0 | yes | 1 | 7 | 5/5 | reject,mutants | done 1 |
| dl/backprop-through-a-dense-layer/p01-dense-backward | 3 | 0 | yes | 2 | 7 | 4/4 | reject,mutants | done 1 |
| dl/batch-normalization/p01-batch-norm-forward | 4 | 0 | yes | 2 | 8 | 10/10 | reject,mutants | unpromised 1, done 1 |
| dl/batch-normalization/p02-batch-norm-backward | 4 | 0 | yes | 2 | 8 | 6/6 | reject,mutants | done 1 |
| dl/building-tensors-by-broadcasting/p01-pairwise-squared-distances | 2 | 0 | yes | 2 | 6 | 6/6 | reject,mutants | message-defect 1, done 2 |
| dl/causal-masking/p01-causal-padding-mask | 2 | 0 | yes | 1 | 8 | 8/8 | reject,mutants | done 2 |
| dl/convolution-forward-pass/p01-conv2d-forward | 4 | 0 | yes | 2 | 8 | 9/9 | reject,mutants | unpromised 1, message-defect 3, done 2 |
| dl/counting-network-parameters/p01-count-dense-parameters | 2 | 0 | yes | 1 | 8 | 5/5 | reject,mutants | message-defect 1, done 1 |
| dl/dense-layer-forward-pass/p01-dense-layer-forward | 4 | 0 | yes | 3 | 7 | 8/8 | reject,mutants | done 1 |
| dl/dropout/p01-inverted-dropout | 4 | 0 | yes | 2 | 9 | 6/6 | reject,mutants | done 1 |
| dl/gradients-with-framework-autograd/p01-gradient-at-a-point | 3 | 0 | yes | 1 | 6 | 7/7 | reject,mutants | overreach 1, message-defect 1, done 1 |
| dl/gradients-with-framework-autograd/p02-straight-through-round | 1 | 0 | yes | 2 | 8 | 7/7 | reject,mutants | message-defect 1, done 1 |
| dl/infonce-contrastive-loss/p01-symmetric-infonce-loss | 3 | 0 | yes | 2 | 8 | 8/8 | reject,mutants | done 1 |
| dl/kl-divergence/p01-kl-from-logits | 3 | 0 | yes | 2 | 6 | 6/6 | reject,mutants | no-witness 1, overreach 1, message-defect 1, done 1 |
| dl/kl-divergence/p02-diagonal-gaussian-kl | 3 | 0 | yes | 2 | 7 | 6/6 | reject,mutants | done 1 |
| dl/layer-normalization/p01-layer-norm-forward | 3 | 0 | yes | 2 | 8 | 7/7 | reject,mutants | question 1, unpromised 1, done 1 |
| dl/learning-rate-decay-schedules/p01-cosine-annealing-lr | 2 | 0 | yes | 3 | 6 | 7/7 | reject,mutants | done 1 |
| dl/learning-rate-decay-schedules/p02-step-decay-lr | 4 | 0 | yes | 3 | 7 | 7/7 | reject,mutants | done 1 |
| dl/learning-rate-warmup/p01-warmup-then-cosine-lr | 2 | 0 | yes | 3 | 7 | 8/8 | reject,mutants | done 1 |
| dl/learning-rate-warmup/p02-inverse-sqrt-lr | 1 | 0 | yes | 3 | 5 | 6/6 | reject,mutants | message-defect 2, done 1 |
| dl/mixed-precision/p01-dynamic-loss-scaler | 3 | 0 | yes | 2 | 6 | 5/5 | reject,mutants | done 2 |
| dl/multi-head-attention/p01-self-attention-heads | 2 | 0 | yes | 2 | 8 | 8/8 | reject,mutants | unpromised 1, done 1 |
| dl/one-training-step/p01-manual-sgd-train-step | 3 | 0 | yes | 2 | 8 | 8/8 | reject,mutants | question 1, done 2 |
| dl/position-wise-feed-forward-block/p01-gelu-ffn | 3 | 0 | yes | 2 | 8 | 9/9 | reject,mutants | message-defect 2, done 1 |
| dl/position-wise-feed-forward-block/p02-swiglu-ffn | 2 | 0 | yes | 2 | 8 | 8/8 | reject,mutants | message-defect 1, done 1 |
| dl/positional-encodings/p01-sinusoidal-table | 3 | 0 | yes | 2 | 6 | 6/6 | reject,mutants | unpromised 2, done 1 |
| dl/positional-encodings/p02-rope-rotations | 1 | 0 | yes | 2 | 6 | 7/7 | reject,mutants | unpromised 1, done 1 |
| dl/relu-family-activations/p01-relu-forward-and-gradient | 2 | 0 | yes | 2 | 7 | 5/5 | reject,mutants | done 1 |
| dl/residual-connections/p01-residual-block | 3 | 0 | yes | 2 | 8 | 8/8 | reject,mutants | unpromised 1, message-defect 1, done 1 |
| dl/reverse-mode-autograd-engine/p01-scalar-autograd | 3 | 0 | yes | 1 | 7 | 5/5 | reject,mutants | message-defect 2, done 1 |
| dl/rmsnorm/p01-rmsnorm-forward | 2 | 0 | yes | 2 | 7 | 7/7 | reject,mutants | unpromised 1, done 1 |
| dl/scaled-dot-product-attention/p01-scaled-dot-product-attention | 3 | 0 | yes | 3 | 7 | 12/12 | reject,mutants | unpromised 4, done 1 |
| dl/sgd-with-momentum/p01-momentum-step | 3 | 0 | yes | 2 | 6 | 7/7 | reject,mutants | question 1, unpromised 2, done 2 |
| dl/sgd-with-momentum/p02-nesterov-weight-decay | 2 | 0 | yes | 2 | 7 | 7/7 | reject,mutants | question 1, unpromised 2, done 2 |
| dl/smooth-activations-tanh-gelu-swish/p01-gelu-and-swish | 2 | 0 | yes | 2 | 5 | 4/4 | reject,mutants | question 1, inspect 1, done 1 |
| dl/smooth-activations-tanh-gelu-swish/p02-smooth-activation-derivatives | 2 | 0 | yes | 2 | 5 | 5/5 | reject,mutants | message-defect 1, done 1 |
| dl/tensors-in-pytorch-and-tinygrad/p01-batched-pairwise-distances | 4 | 0 | yes | 2 | 6 | 6/6 | reject,mutants | done 1 |
| dl/tensors-in-pytorch-and-tinygrad/p02-patchify-images-row-major | 2 | 0 | yes | 2 | 7 | 7/7 | reject,mutants | unpromised 1, message-defect 1, done 1 |
| dl/training-an-mlp-from-scratch/p01-train-two-layer-mlp | 4 | 0 | yes | 2 | 7 | 6/6 | reject,mutants | unpromised 1, message-defect 1, done 2 |
| dl/transformer-block/p01-pre-norm-causal-block | 3 | 0 | yes | 2 | 7 | 8/8 | reject,mutants | unpromised 1, no-witness 1, done 2 |
| dl/xavier-and-he-initialization/p01-initialize-dense-weights | 2 | 0 | yes | 2 | 8 | 8/8 | reject,mutants | done 1 |
| inf/absmax-and-zero-point-int8-quantization/p01-absmax-quantize | 3 | 0 | yes | 1 | 8 | 12/12 | reject,mutants | message-defect 1, done 2 |
| inf/absmax-and-zero-point-int8-quantization/p02-zero-point-quantize | 4 | 0 | yes | 1 | 7 | 8/8 | reject,mutants | done 2 |
| inf/arithmetic-intensity-and-the-roofline/p01-matmul-roofline | 1 | 0 | yes | 1 | 8 | 18/18 | reject,mutants | done 1 |
| inf/continuous-batching/p01-batch-scheduler-steps | 3 | 0 | yes | 1 | 7 | 6/6 | reject,mutants | done 2 |
| inf/cost-per-token-and-break-even/p01-cost-per-million-and-break-even | 3 | 0 | yes | 1 | 8 | 7/7 | reject,mutants | done 1 |
| inf/data-quality-and-drift-detection/p01-population-stability-index | 3 | 0 | yes | 1 | 7 | 11/11 | reject,mutants | unpromised 1, message-defect 1, done 1 |
| inf/data-quality-and-drift-detection/p02-ks-drift-test | 4 | 0 | yes | 1 | 8 | 7/7 | reject,mutants | unpromised 3, done 1 |
| inf/floating-point-formats/p01-decode-a-bit-pattern | 3 | 0 | yes | 2 | 8 | 7/7 | reject,mutants | done 2 |
| inf/floating-point-formats/p02-round-to-nearest-even | 2 | 0 | yes | 2 | 8 | 7/7 | reject,mutants | message-defect 3, inspect 1, done 2 |
| inf/gpu-memory-and-capacity-sizing/p01-min-gpu-count | 4 | 0 | yes | 1 | 8 | 10/10 | reject,mutants | done 1 |
| inf/kv-cache-size-arithmetic/p01-kv-cache-bytes | 2 | 0 | yes | 1 | 8 | 8/8 | reject,mutants | blocked 1, unpromised 3, message-defect 1, done 1 |
| inf/kv-cache-size-arithmetic/p02-max-batch-after-weights | 3 | 0 | yes | 1 | 7 | 7/7 | reject,mutants | blocked 1, unpromised 3, done 1 |
| inf/paged-kv-cache/p01-block-allocator | 1 | 0 | yes | 2 | 8 | 9/9 | reject,mutants | done 2 |
| inf/prefill-vs-decode-bottlenecks/p01-serving-step-times | 4 | 0 | yes | 1 | 8 | 9/9 | reject,mutants | no-witness 1, done 1 |
| inf/prefill-vs-decode-bottlenecks/p02-compute-bound-decode-batch | 4 | 0 | yes | 1 | 7 | 7/7 | reject,mutants | message-defect 1, done 1 |
| inf/speculative-decoding-speedup-math/p01-speculative-speedup | 2 | 0 | yes | 2 | 8 | 7/7 | reject,mutants | message-defect 1, done 1 |
| inf/speculative-decoding-speedup-math/p02-accept-or-resample | 4 | 0 | yes | 3 | 7 | 7/7 | reject,mutants | unpromised 3, inspect 1, done 2 |
| inf/ttft-itl-and-throughput/p01-latency-percentiles | 2 | 0 | yes | 2 | 9 | 7/7 | reject,mutants | unpromised 1, done 1 |
| inf/ttft-itl-and-throughput/p02-throughput-and-goodput | 2 | 0 | yes | 2 | 7 | 5/5 | reject,mutants | unpromised 1, message-defect 1, done 1 |
| llm/autoregressive-generation-loop/p01-greedy-generation-with-eos | 4 | 0 | yes | 2 | 11 | 8/8 | reject,mutants | done 1 |
| llm/bpe-encode-and-decode/p01-apply-bpe-merges | 4 | 0 | yes | 2 | 8 | 4/4 | reject,mutants | done 1 |
| llm/bpe-merge-training/p01-learn-bpe-merges | 3 | 0 | yes | 1 | 11 | 8/8 | reject,mutants | done 1 |
| llm/bradley-terry-reward-models/p01-reward-pair-loss-and-accuracy | 4 | 0 | yes | 3 | 8 | 6/6 | reject,mutants | unpromised 2, message-defect 2, done 1 |
| llm/chat-templates-and-tool-call-formats/p01-render-chat-template | 3 | 0 | yes | 1 | 8 | 8/8 | reject,mutants | unpromised 1, done 2 |
| llm/chat-templates-and-tool-call-formats/p02-parse-tool-call | 4 | 0 | yes | 2 | 6 | 4/4 | reject,mutants | unpromised 3, done 2 |
| llm/direct-preference-optimization-dpo/p01-sequence-log-probs-with-response-mask | 3 | 0 | yes | 2 | 7 | 7/7 | reject,mutants | unpromised 3, inspect 1, done 2 |
| llm/direct-preference-optimization-dpo/p02-dpo-loss-and-implicit-rewards | 3 | 0 | yes | 2 | 6 | 6/6 | reject,mutants | unpromised 2, message-defect 2, done 1 |
| llm/document-packing-and-cross-document-masks/p01-packed-causal-mask | 3 | 0 | yes | 2 | 7 | 5/5 | reject,mutants | done 2 |
| llm/gpt-forward-pass/p01-decoder-only-gpt-forward | 3 | 0 | yes | 3 | 7 | 7/7 | reject,mutants | done 2 |
| llm/gqa-mqa-and-cross-layer-kv-sharing/p01-grouped-query-attention | 4 | 0 | yes | 2 | 8 | 9/9 | reject,mutants | done 2 |
| llm/grpo-and-its-variants/p01-group-normalised-advantages | 3 | 0 | yes | 3 | 7 | 6/6 | reject,mutants | message-defect 1, inspect 1, done 2 |
| llm/grpo-and-its-variants/p02-clipped-grpo-token-loss | 4 | 0 | yes | 3 | 8 | 8/8 | reject,mutants | message-defect 1, done 2 |
| llm/incremental-decoding-with-a-kv-cache/p01-cached-causal-attention-step | 2 | 0 | yes | 2 | 9 | 8/8 | reject,mutants | done 1 |
| llm/instruction-tuning-batches-and-loss-masks/p01-sft-batch-with-loss-masks | 4 | 0 | yes | 2 | 8 | 8/8 | reject,mutants | done 1 |
| llm/llm-as-judge-and-agreement-with-humans/p01-cohens-kappa | 3 | 0 | yes | 2 | 9 | 8/8 | reject,mutants | message-defect 2, done 2 |
| llm/llm-as-judge-and-agreement-with-humans/p02-debiased-win-rate | 2 | 0 | yes | 1 | 7 | 7/7 | reject,mutants | done 2 |
| llm/lora-adapters/p01-lora-linear-forward-and-merge | 3 | 0 | yes | 2 | 8 | 12/12 | reject,mutants | unpromised 1, message-defect 1, done 1 |
| llm/lora-parameter-and-memory-arithmetic/p01-lora-trainable-parameter-budget | 3 | 0 | yes | 1 | 8 | 10/10 | reject,mutants | message-defect 2, done 1 |
| llm/lora-parameter-and-memory-arithmetic/p02-finetune-memory-in-bytes | 1 | 0 | yes | 1 | 7 | 7/7 | reject,mutants | message-defect 2, done 1 |
| llm/model-parameter-and-memory-footprint/p01-count-decoder-parameters | 2 | 0 | yes | 1 | 5 | 5/5 | reject,mutants | message-defect 1, done 2 |
| llm/model-parameter-and-memory-footprint/p02-training-and-inference-memory | 1 | 0 | yes | 1 | 8 | 10/10 | reject,mutants | question 1, done 2 |
| llm/moe-top-k-routing/p01-top-k-router-gates | 2 | 0 | yes | 3 | 8 | 7/7 | reject,mutants | done 1 |
| llm/moe-top-k-routing/p02-load-balancing-loss | 4 | 0 | yes | 3 | 7 | 6/6 | reject,mutants | done 1 |
| llm/negative-log-likelihood-and-perplexity/p01-masked-nll-and-perplexity | 3 | 0 | yes | 2 | 8 | 9/9 | reject,mutants | done 1 |
| llm/next-token-targets-and-batches/p01-pack-next-token-blocks | 3 | 0 | yes | 3 | 8 | 9/9 | reject,mutants | done 1 |
| llm/padding-truncation-and-pad-masks/p01-pad-and-truncate-token-ids | 3 | 0 | yes | 1 | 8 | 9/9 | reject,mutants | done 1 |
| llm/pass-k-and-majority-voting/p01-unbiased-pass-at-k | 1 | 0 | yes | 1 | 8 | 9/9 | reject,mutants | unpromised 1, done 1 |
| llm/pass-k-and-majority-voting/p02-majority-vote | 3 | 0 | yes | 1 | 8 | 9/9 | reject,mutants | done 2 |
| llm/retrieval-augmented-generation-pipeline/p01-chunk-words | 3 | 0 | yes | 1 | 8 | 9/9 | reject,mutants | blocked 1, done 1 |
| llm/retrieval-augmented-generation-pipeline/p02-cited-context-prompt | 5 | 0 | yes | 2 | 8 | 7/7 | reject,mutants | inspect 3, done 2 |
| llm/rotary-position-embeddings-rope/p01-half-split-rope-with-offset | 3 | 0 | yes | 3 | 7 | 8/8 | reject,mutants | done 2 |
| llm/rotary-position-embeddings-rope/p02-linear-position-interpolation | 3 | 0 | yes | 3 | 7 | 10/10 | reject,mutants | message-defect 1, done 2 |
| llm/scaling-laws-and-compute-budgets/p01-compute-optimal-allocation | 1 | 0 | yes | 1 | 7 | 6/6 | reject,mutants | done 1 |
| llm/scaling-laws-and-compute-budgets/p02-fit-power-law-log-log | 3 | 0 | yes | 2 | 7 | 4/4 | reject,mutants | done 1 |
| llm/temperature-top-k-and-top-p-sampling/p01-filtered-next-token-distribution | 4 | 0 | yes | 2 | 10 | 8/8 | reject,mutants | unpromised 1, done 1 |
| llm/temperature-top-k-and-top-p-sampling/p02-sample-with-a-generator | 3 | 0 | yes | 3 | 6 | 7/7 | reject,mutants | unpromised 8, overreach 2, message-defect 2, done 2 |
| llm/word-and-character-tokenizers/p01-word-level-vocabulary | 3 | 0 | yes | 1 | 9 | 8/8 | reject,mutants | message-defect 1, done 1 |
| math/determinants-and-trace/p01-determinant | 3 | 0 | yes | 3 | 8 | 6/6 | reject,mutants | message-defect 1, done 2 |
| math/inverse-transform-sampling/p01-sample-exponential | 3 | 0 | yes | 2 | 7 | 6/6 | reject,mutants | message-defect 1, done 1 |
| math/inverse-transform-sampling/p02-sample-discrete | 4 | 0 | yes | 2 | 8 | 7/7 | reject,mutants | unpromised 2, message-defect 1, done 1 |
| math/monte-carlo-simulation/p01-mc-estimate | 5 | 0 | yes | 2 | 8 | 8/8 | reject,mutants | unpromised 2, done 1 |
| math/reservoir-sampling/p01-reservoir-sample | 2 | 0 | yes | 1 | 5 | 6/6 | reject,mutants | inspect 1, done 1 |
| rl/action-value-estimates/p01-incremental-action-values | 4 | 0 | yes | 2 | 8 | 8/8 | reject,mutants | unpromised 1, no-witness 1, done 1 |
| rl/epsilon-greedy-exploration/p01-epsilon-greedy-action | 2 | 0 | yes | 2 | 8 | 7/7 | reject,mutants | unpromised 1, done 1 |
| rl/mdps-and-discounted-returns/p01-discounted-returns | 1 | 0 | yes | 1 | 6 | 5/5 | reject,mutants | done 1 |
| rl/mdps-and-discounted-returns/p02-iterative-policy-evaluation | 3 | 0 | yes | 2 | 8 | 8/8 | reject,mutants | unpromised 5, message-defect 1, done 1 |
| rl/q-learning/p01-q-learning-gridworld | 4 | 0 | yes | 2 | 8 | 12/12 | reject,mutants | done 1 |
| rl/reinforce/p01-reinforce-loss | 3 | 0 | yes | 1 | 8 | 8/8 | reject,mutants | unpromised 1, message-defect 1, done 1 |
| rl/sarsa-and-expected-sarsa/p01-sarsa-and-expected-sarsa-updates | 4 | 0 | yes | 2 | 8 | 12/12 | reject,mutants | unpromised 4, message-defect 4, done 2 |
| safe/split-conformal-prediction/p01-split-conformal-intervals | 2 | 0 | yes | 2 | 8 | 9/9 | reject,mutants | message-defect 1, done 1 |
| safe/split-conformal-prediction/p02-conformal-prediction-sets | 5 | 0 | yes | 2 | 8 | 10/10 | reject,mutants | unpromised 1, done 1 |
| sql/pandas-groupby-aggregate-and-transform/p01-customer-order-features | 3 | 0 | yes | 2 | 6 | 5/5 | reject,mutants | unpromised 2, done 1 |
| sql/pandas-merge-and-reshape/p01-monthly-revenue-by-region | 4 | 0 | yes | 2 | 8 | 7/7 | reject,mutants | message-defect 1, done 2 |
| sql/sql-cohort-retention/p01-cohort-retention-by-month | 2 | 0 | yes | 2 | 7 | 11/11 | reject,mutants | done 1 |
| sql/sql-conditional-aggregation/p01-channel-conversion-and-device-revenue | 3 | 0 | yes | 3 | 8 | 6/6 | reject,mutants | message-defect 1, done 2 |
| sql/sql-date-handling/p01-monthly-active-users | 5 | 0 | yes | 2 | 9 | 8/8 | reject,mutants | done 1 |
| sql/sql-date-handling/p02-days-from-first-to-second-event | 3 | 0 | yes | 2 | 6 | 7/7 | reject,mutants | done 1 |
| sql/sql-first-and-last-activity-per-user/p01-first-and-last-event | 3 | 0 | yes | 2 | 7 | 9/9 | reject,mutants | done 1 |
| sql/sql-group-by-and-having/p01-customers-with-three-completed-orders | 3 | 0 | yes | 2 | 7 | 9/9 | reject,mutants | done 2 |
| sql/sql-joins-and-unions/p01-customer-order-totals | 3 | 0 | yes | 2 | 7 | 6/6 | reject,mutants | done 2 |
| sql/sql-joins-and-unions/p02-warehouses-with-a-shipment-or-a-return | 3 | 0 | yes | 2 | 6 | 7/7 | reject,mutants | done 2 |
| sql/sql-lag-and-lead/p01-day-over-day-change | 1 | 0 | yes | 2 | 6 | 6/6 | reject,mutants | done 1 |
| sql/sql-lag-and-lead/p02-sessions-by-inactivity-gap | 3 | 0 | yes | 2 | 7 | 8/8 | reject,mutants | done 1 |
| sql/sql-ranking-window-functions/p01-rank-within-category | 3 | 0 | yes | 2 | 8 | 11/11 | reject,mutants | done 1 |
| sql/sql-ranking-window-functions/p02-top-two-with-ties | 3 | 0 | yes | 2 | 7 | 8/8 | reject,mutants | done 1 |
| sql/sql-running-totals-and-frames/p01-running-balance-with-ties | 2 | 0 | yes | 2 | 5 | 8/8 | reject,mutants | done 1 |
| sql/sql-running-totals-and-frames/p02-trailing-seven-day-average | 1 | 0 | yes | 2 | 5 | 4/4 | reject,mutants | done 1 |
| sql/sql-select-filter-and-sort/p01-top-five-completed-orders-2024 | 2 | 0 | yes | 2 | 7 | 6/6 | reject,mutants | done 2 |
| sql/sql-streaks-gaps-and-islands/p01-longest-streak-per-user | 2 | 0 | yes | 2 | 7 | 9/9 | reject,mutants | done 1 |
| sql/sql-streaks-gaps-and-islands/p02-all-islands-per-user | 3 | 0 | yes | 1 | 6 | 7/7 | reject,mutants | done 1 |
| sql/sql-subqueries-and-ctes/p01-customers-above-average-spend | 3 | 0 | yes | 2 | 7 | 11/11 | reject,mutants | done 1 |
| sql/sql-subqueries-and-ctes/p02-orders-above-their-customers-average | 2 | 0 | yes | 2 | 5 | 9/9 | reject,mutants | unpromised 1, done 1 |
| sql/sql-top-n-and-latest-row-per-group/p01-latest-events | 1 | 0 | yes | 2 | 7 | 7/7 | reject,mutants | question 1, done 1 |

problems: 251
with a blind accept: 251
accepts: 461
rejects: 1847
every bullet covered: 251
both sweeps: 251
lint findings: 0
