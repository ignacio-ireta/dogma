# Exercise audit

*Increment I-28, 2026-10-31. How well do the grader, the edge cases and the tests hold up across all
251 problems in 168 topics? Every number comes from `scripts/exercise_audit.py`, run through the real
grading path (Seatbelt sandbox, fixed Hypothesis seed).*

## Verdict

The exercises are sound. They accept correct work and they reject wrong work on the core behaviour.
The weak spot was input validation in the newer torch and LLM topics, and it is now mostly closed.

| Question | Before | After |
|---|---:|---:|
| Can a problem be solved from its statement alone? | 251/251, 235 first try | same, 15 statements clearer |
| Do tests reject wrong answers? (mutants killed, raw) | 88.1% | 93.3% |
| Same, excluding mutants that cannot be told apart | 93.2% | 98.6% |
| Does a correct independent solution pass? | 251/251 | 251/251 |
| Can a misbehaving submission read as passed? | no (35 probes) | no |
| Does Submit depend on luck? | no: the seed is fixed | no |

## What was measured

| Measure | How | Result |
|---|---|---|
| Mutation | 10,551 deliberate bugs, up to 60 per problem | 9,297 killed, 2,734 of them only by hidden tests |
| Blind solves | Agents solved every problem from the statement and public tests | 251 solved, 235 first try, 16 after one resubmit, 0 unsolved |
| Blindness check | Scanned all 1,115 agent tool calls | 0 reads of solutions, hidden tests or hints |
| Branch coverage | Each reference under its own tests | median 100%, lowest 82.1%, 162 of 231 at 100% |
| Effective precision | Scaled every returned float by 1 + ε | 161 problems catch ε = 1e-8; none that returns floats is loose |
| Seed robustness | 5,020 runs on 20 other seeds | 2 problems fail on some seed, neither at the pinned seed |
| Grader probes | 35 misbehaving submissions | 1 crash found and fixed |
| Failure messages | Learner-visible text of 2,734 hidden-only kills | 96.0% name a mistake |

## Findings

**1. Statements left numeric limits unstated (fixed).** 13 of the 16 blind retries came from tests
that require survival near `1e308` or `1e-300`, such as `(a + b) / 2` overflowing, while the statement
said only "finite real numbers". Statements that do state such a limit, such as `gelu-ffn` and
`logloss-grad-hess`, passed first try. 14 of the 16 retries were in `cml`. Every one was repaired from
the failure message alone, so none was a dead end. 15 statements now say what their tests require.
Tests and versions did not change.

**2. Real test gaps (fixed).** Of 1,254 surviving mutants, 1,199 were triaged. 543 are equivalent, so
no test can see them. 107 differ only on behaviour no statement promises, so testing them would invent
a requirement. **549 are real gaps** in 144 problems. 544 are now closed by 425 new hidden tests.
By a rough keyword split, about 57% are validation that was never exercised, and the rest change
behaviour on valid input. Examples:

- A documented default changed from 3.5 to 7.0 and nothing noticed (`modified-zscore-outliers`).
- PCA wrongly rejected its smallest valid inputs, `n = 2` and `d = 1`.
- F1 returned `0.0` instead of `0.5` on one count pattern (`precision-recall-averaging`).
- Two AdaBoost tests passed by accident, because a perfect classifier never reached the code under test.
- GQA tests were confounded by unrelated shape errors, so divisibility was never really tested.

`residual-block` had the lowest branch coverage and also 15 of 15 survivors as real gaps. Two
independent measures found the same weakness. SQL had none: all 13 survivors were equivalent.

**3. One grader crash (fixed).** A source deep enough to overflow the parser raised `MemoryError`
through `run()`. It is now a syntax finding. Everything else held: skips, xfails, exit tricks,
`KeyboardInterrupt`, NaN returns and hostile SQL never read as passed.

**4. Two failure messages misdirected learners (fixed).** `adaboost-round` and `kmeans-plus-plus`
blamed the wrong cause on an overflow. The check that this changed text only was an AST comparison.

**5. `meta.yaml` tolerances are decorative (open).** Grading never reads them. They are stored and
shown by the API, and each test hard-codes its own tolerance. Effective precision is good anyway, but
the declared value can drift from the real one.

## How the new tests were kept honest

A test was accepted only if the tool saw all of this through the real grader:

1. It passes the reference.
2. It passes an independent solution written blind. This rejected several proposed tests as over-reach.
3. It fails the mutant it targets.

Then, on the final content: 251/251 references pass, 251/251 blind solutions pass, and 544/549 gap
mutants die. Across 144 test files the diff is +4,302 lines and no existing line was changed or
removed. Each changed problem's version is bumped once.

Five tests were dropped on purpose. They sat exactly on a float threshold, so a correct solution that
rounds one ulp differently could fail them, and a strict-versus-non-strict tie teaches little.

## Decisions for you

- **`rolling-mean-std` tolerance.** Its property test uses `atol = 1e-8 * scale`, which sits at the
  rounding floor of an incremental update. The reference misses it by 0.1% on 10 of 20 other seeds, for
  example `[19.0, 0.0, 0.0]`, window 2. Seed 0 passes, and so did the blind solution, but another
  correct rounding path could fail. Fixing it means loosening a hidden tolerance, which the hard rules
  reserve, so it was left alone.
- **`lagrange-multipliers` feasibility.** 2 of 5,001 possible instances fail for the reference, all
  with square `A`. Seed 0 draws neither. This is narrow enough to leave as it is.
- **Unstated minimums.** `kv-cache-size-arithmetic` p01 and p02 state their minimums only in the starter,
  and `training-an-mlp-from-scratch` is ambiguous on whether its bound covers `y`. Their survivors
  stay untested until the statements say so.

## Known limits

- **Deliberate cheating is not stopped.** `__import__`, `getattr` and `eval` get past the banned-import
  check, and a submission can replace `pytest.fail`. These are pinned as known limits in
  `tests/integration/runner/test_grader_probes.py`. The sandbox is not a security boundary (§11).
- **Agents are a floor, not a learner model.** Blind solves show that a capable reader can succeed.
  They do not measure how hard a problem is for a person.
- **Triage is judgement.** Equivalent and unpromised verdicts came from agents and were spot-checked,
  not proven. One tie-order survivor is recorded as "judged equivalent, not proven". The witness gate
  proves gaps, not their absence.
- **"Named" is a low bar for messages.** "valid input must be accepted; the call raised ValueError"
  (173 kills) names a kind of mistake, not the specific one.
- **Mutants are not exhaustive.** Each problem had at most 60, and 25 survivors over the 15-per-problem
  triage cap were not reviewed. 137 mutants remain alive: 107 unpromised, 25 over the cap, 5 dropped.

## Scorecard

Regenerate with `uv run --frozen --group torch python scripts/exercise_audit.py report --out DIR`.
*Gaps* are real gaps, now closed unless listed above. *Blind* is the number of submissions needed.
*Raw killed* is before fixes.

| Track | Problems | Mutants | Raw killed % | Real gaps | First-try blind solves |
|---|---:|---:|---:|---:|---:|
| algo | 20 | 665 | 92.5 | 13 | 19/20 |
| cml | 92 | 4141 | 86.1 | 204 | 78/92 |
| dl | 46 | 2004 | 89.7 | 111 | 45/46 |
| inf | 19 | 1020 | 90.3 | 54 | 19/19 |
| llm | 38 | 1878 | 88.2 | 117 | 38/38 |
| math | 5 | 132 | 93.9 | 2 | 5/5 |
| rl | 7 | 370 | 83.2 | 36 | 7/7 |
| safe | 2 | 112 | 85.7 | 12 | 2/2 |
| sql | 22 | 229 | 94.3 | 0 | 22/22 |

| Problem | Mutants | Raw killed % | Hidden-only | Gaps | Equiv. | Coverage % | Blind | Seeds |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `algo/bfs-and-dfs-on-graphs-and-grids/p01-grid-shortest-path` | 60 | 91.7 | 18 | 2 | 2 | 97.7 | 1 | 0/20 |
| `algo/bfs-and-dfs-on-graphs-and-grids/p02-count-components` | 60 | 90.0 | 14 | 0 | 5 | 100.0 | 1 | 0/20 |
| `algo/binary-search/p01-lower-bound` | 14 | 92.9 | 0 | 0 | 1 | 100.0 | 1 | 0/20 |
| `algo/binary-search/p02-min-ship-capacity` | 45 | 93.3 | 9 | 0 | 3 | 100.0 | 1 | 0/20 |
| `algo/binary-tree-recursion-and-traversal/p01-inorder-traversal-iterative` | 10 | 90.0 | 1 | 1 | 0 | 88.5 | 1 | 0/20 |
| `algo/binary-tree-recursion-and-traversal/p02-tree-diameter` | 21 | 95.2 | 1 | 1 | 0 | 87.0 | 1 | 0/20 |
| `algo/dynamic-programming/p01-minimum-coins` | 46 | 87.0 | 7 | 2 | 4 | 100.0 | 1 | 0/20 |
| `algo/dynamic-programming/p02-longest-common-subsequence` | 60 | 98.3 | 5 | 0 | 1 | 100.0 | 1 | 0/20 |
| `algo/hash-maps-and-sets/p01-top-k-frequent-tokens` | 14 | 85.7 | 1 | 0 | 2 | 100.0 | 1 | 0/20 |
| `algo/heaps-and-priority-queues/p01-merge-sorted-streams` | 4 | 100.0 | 0 | 0 | 0 | 100.0 | 1 | 0/20 |
| `algo/heaps-and-priority-queues/p02-running-median` | 52 | 86.5 | 9 | 0 | 7 | 100.0 | 2 | 0/20 |
| `algo/lru-cache/p01-lru-cache` | 28 | 82.1 | 4 | 0 | 5 | 92.1 | 1 | 0/20 |
| `algo/prefix-sums/p01-subarrays-with-sum-k` | 22 | 100.0 | 1 | 0 | 0 | 100.0 | 1 | 0/20 |
| `algo/prefix-sums/p02-rectangle-range-sums` | 60 | 93.3 | 10 | 2 | 2 | 100.0 | 1 | 0/20 |
| `algo/sort-then-sweep/p01-merge-intervals` | 20 | 100.0 | 3 | 0 | 0 | 100.0 | 1 | 0/20 |
| `algo/sort-then-sweep/p02-peak-concurrent-sessions` | 17 | 100.0 | 3 | 0 | 0 | 100.0 | 1 | 0/20 |
| `algo/stacks-and-queues/p01-days-until-higher` | 11 | 100.0 | 1 | 0 | 0 | 100.0 | 1 | 0/20 |
| `algo/stacks-and-queues/p02-sliding-window-maximum` | 31 | 96.8 | 2 | 0 | 1 | 100.0 | 1 | 0/20 |
| `algo/topological-sort/p01-lexicographic-topological-order` | 58 | 87.9 | 5 | 5 | 0 | 92.2 | 1 | 0/20 |
| `algo/two-pointers-and-sliding-windows/p01-longest-window-within-budget` | 32 | 100.0 | 2 | 0 | 0 | 100.0 | 1 | 0/20 |
| `cml/a-b-test-design-and-analysis/p01-sample-size-two-proportions` | 60 | 90.0 | 12 | 6 | 0 | 100.0 | 1 | 0/20 |
| `cml/a-b-test-design-and-analysis/p02-two-proportion-z-test` | 51 | 78.4 | 10 | 11 | 0 | 100.0 | 1 | 0/20 |
| `cml/adaboost-and-gradient-boosting/p01-boost-stumps-mse` | 60 | 85.0 | 23 | 5 | 4 | 98.0 | 2 | 0/20 |
| `cml/adaboost-and-gradient-boosting/p02-adaboost-round` | 60 | 80.0 | 14 | 3 | 7 | 100.0 | 2 | 0/20 |
| `cml/bayes-theorem/p01-bayes-posterior` | 60 | 78.3 | 15 | 6 | 2 | 93.4 | 1 | 0/20 |
| `cml/bernoulli-binomial-and-poisson/p01-binomial-pmf` | 60 | 88.3 | 12 | 4 | 2 | 95.6 | 1 | 0/20 |
| `cml/bernoulli-binomial-and-poisson/p02-poisson-pmf` | 35 | 85.7 | 8 | 3 | 1 | 94.3 | 1 | 0/20 |
| `cml/bias-variance-decomposition/p01-bias-variance-from-predictions` | 60 | 80.0 | 20 | 1 | 5 | 92.3 | 1 | 0/20 |
| `cml/bootstrap-and-permutation-resampling/p01-bootstrap-percentile-ci` | 39 | 92.3 | 14 | 2 | 0 | 100.0 | 1 | 0/20 |
| `cml/bootstrap-and-permutation-resampling/p02-permutation-test-mean-diff` | 46 | 87.0 | 6 | 6 | 0 | 100.0 | 1 | 0/20 |
| `cml/chain-rule/p01-mse-sigmoid-gradient` | 60 | 88.3 | 20 | 0 | 4 | 100.0 | 2 | 0/20 |
| `cml/choosing-a-tree-split/p01-best-split-threshold` | 60 | 83.3 | 16 | 0 | 8 | 100.0 | 2 | 0/20 |
| `cml/confusion-matrix-and-count-based-scores/p01-confusion-matrix-from-labels` | 16 | 75.0 | 2 | 1 | 2 | 100.0 | 1 | 0/20 |
| `cml/confusion-matrix-and-count-based-scores/p02-binary-count-scores` | 37 | 100.0 | 4 | 0 | 0 | 100.0 | 1 | 0/20 |
| `cml/cosine-similarity/p01-cosine-similarity-matrix` | 50 | 88.0 | 14 | 2 | 4 | 100.0 | 1 | 0/20 |
| `cml/covariance-and-correlation/p01-covariance-matrix` | 35 | 91.4 | 10 | 1 | 2 | 100.0 | 1 | 0/20 |
| `cml/covariance-and-correlation/p02-correlation-matrix` | 50 | 78.0 | 16 | 1 | 10 | 100.0 | 1 | 0/20 |
| `cml/cross-entropy/p01-cross-entropy-from-logits` | 60 | 83.3 | 16 | 3 | 7 | 100.0 | 1 | 0/20 |
| `cml/cross-entropy/p02-binary-cross-entropy-with-logits` | 51 | 84.3 | 8 | 1 | 7 | 100.0 | 2 | 0/20 |
| `cml/derivatives/p01-numerical-derivative` | 31 | 96.8 | 12 | 1 | 0 | 100.0 | 1 | 0/20 |
| `cml/descriptive-statistics/p01-welford-mean-variance` | 23 | 100.0 | 1 | 0 | 0 | 100.0 | 1 | 0/20 |
| `cml/descriptive-statistics/p02-iqr-outlier-fences` | 46 | 93.5 | 10 | 0 | 2 | 100.0 | 1 | 0/20 |
| `cml/eigenvalues-and-eigenvectors/p01-power-iteration` | 60 | 61.7 | 9 | 7 | 8 | 93.2 | 1 | 0/20 |
| `cml/entropy-and-mutual-information/p01-distribution-entropy` | 40 | 90.0 | 9 | 1 | 3 | 100.0 | 1 | 0/20 |
| `cml/entropy-and-mutual-information/p02-mutual-information-from-labels` | 21 | 57.1 | 1 | 5 | 3 | 94.7 | 1 | 0/20 |
| `cml/expectation-and-variance/p01-discrete-moments` | 42 | 83.3 | 10 | 3 | 4 | 100.0 | 1 | 0/20 |
| `cml/feature-scaling/p01-standard-scaler` | 60 | 88.3 | 19 | 1 | 6 | 100.0 | 2 | 0/20 |
| `cml/feature-scaling/p02-robust-median-iqr-scaler` | 60 | 91.7 | 24 | 3 | 2 | 100.0 | 1 | 0/20 |
| `cml/gaussian-mixtures-and-em/p01-gmm-em-full-covariance` | 60 | 73.3 | 13 | 7 | 4 | 93.4 | 1 | 0/20 |
| `cml/gradient-descent/p01-gradient-descent` | 27 | 66.7 | 4 | 7 | 2 | 100.0 | 1 | 0/20 |
| `cml/handling-class-imbalance/p01-balanced-class-weights` | 12 | 91.7 | 1 | 0 | 1 | 100.0 | 1 | 0/20 |
| `cml/handling-class-imbalance/p02-prior-shift-adjust` | 60 | 91.7 | 8 | 0 | 4 | 100.0 | 2 | 0/20 |
| `cml/k-fold-cross-validation/p01-kfold-indices` | 23 | 100.0 | 6 | 0 | 0 | 100.0 | 1 | 0/20 |
| `cml/k-fold-cross-validation/p02-cross-validate` | 42 | 81.0 | 7 | 4 | 2 | 100.0 | 1 | 0/20 |
| `cml/k-means/p01-lloyd-kmeans` | 60 | 86.7 | 21 | 4 | 3 | 100.0 | 1 | 0/20 |
| `cml/k-means/p02-kmeans-plus-plus-seeding` | 60 | 83.3 | 22 | 3 | 7 | 100.0 | 2 | 0/20 |
| `cml/lagrange-multipliers/p01-constrained-quadratic-min` | 57 | 73.7 | 14 | 1 | 11 | 100.0 | 1 | 3/20 |
| `cml/lasso-and-elastic-net/p01-lasso-coordinate-descent` | 60 | 81.7 | 21 | 7 | 2 | 100.0 | 1 | 0/20 |
| `cml/lasso-and-elastic-net/p02-elastic-net-coordinate-descent` | 60 | 86.7 | 24 | 4 | 2 | 100.0 | 1 | 0/20 |
| `cml/leakage-and-walk-forward-validation/p01-walk-forward-splits` | 40 | 100.0 | 14 | 0 | 0 | 100.0 | 1 | 0/20 |
| `cml/leakage-and-walk-forward-validation/p02-expanding-target-mean` | 50 | 88.0 | 16 | 3 | 3 | 100.0 | 1 | 0/20 |
| `cml/logistic-regression/p01-l2-logistic-fit` | 60 | 70.0 | 14 | 5 | 6 | 93.2 | 2 | 0/20 |
| `cml/matrix-inverse-and-rank/p01-gauss-jordan-inverse` | 36 | 72.2 | 8 | 4 | 5 | 100.0 | 1 | 0/20 |
| `cml/matrix-inverse-and-rank/p02-matrix-rank` | 50 | 90.0 | 16 | 4 | 1 | 100.0 | 1 | 0/20 |
| `cml/matrix-multiplication/p01-matrix-multiply` | 19 | 94.7 | 6 | 0 | 1 | 100.0 | 1 | 0/20 |
| `cml/matrix-shapes-and-transpose/p01-transpose-matrix` | 7 | 85.7 | 2 | 0 | 1 | 100.0 | 1 | 0/20 |
| `cml/maximum-likelihood-and-map/p01-gaussian-mle` | 22 | 95.5 | 5 | 0 | 1 | 100.0 | 1 | 0/20 |
| `cml/maximum-likelihood-and-map/p02-beta-bernoulli-map` | 49 | 95.9 | 23 | 0 | 2 | 95.7 | 1 | 0/20 |
| `cml/missing-values-and-outliers/p01-impute-with-missing-indicators` | 20 | 100.0 | 10 | 0 | 0 | 100.0 | 1 | 0/20 |
| `cml/missing-values-and-outliers/p02-modified-zscore-outliers` | 60 | 85.0 | 33 | 3 | 6 | 100.0 | 2 | 0/20 |
| `cml/nearest-neighbour-search-and-knn/p01-knn-classify` | 54 | 94.4 | 14 | 2 | 1 | 98.3 | 2 | 0/20 |
| `cml/nearest-neighbour-search-and-knn/p02-knn-regress` | 60 | 95.0 | 21 | 1 | 2 | 100.0 | 1 | 0/20 |
| `cml/normal-and-multivariate-gaussian/p01-mvn-logpdf` | 60 | 80.0 | 19 | 7 | 5 | 100.0 | 1 | 0/20 |
| `cml/normal-equation/p01-normal-equation-fit` | 38 | 89.5 | 7 | 1 | 3 | 100.0 | 1 | 0/20 |
| `cml/over-and-underfitting-early-stopping/p01-early-stopping` | 39 | 87.2 | 13 | 1 | 4 | 100.0 | 1 | 0/20 |
| `cml/partial-derivatives-and-gradients/p01-numerical-gradient` | 30 | 96.7 | 11 | 0 | 1 | 100.0 | 1 | 0/20 |
| `cml/precision-recall-and-f1/p01-precision-recall-averaging` | 60 | 86.7 | 9 | 1 | 7 | 100.0 | 1 | 0/20 |
| `cml/precision-recall-and-f1/p02-best-f1-threshold` | 54 | 94.4 | 8 | 2 | 1 | 100.0 | 1 | 0/20 |
| `cml/preprocessing-pipelines/p01-numeric-categorical-pipeline` | 60 | 85.0 | 9 | 2 | 7 | 100.0 | 2 | 0/20 |
| `cml/principal-component-analysis/p01-pca-via-svd` | 60 | 80.0 | 15 | 5 | 5 | 100.0 | 1 | 0/20 |
| `cml/principal-component-analysis/p02-components-for-variance` | 56 | 78.6 | 9 | 5 | 6 | 100.0 | 1 | 0/20 |
| `cml/probability-basics/p01-marginals-and-conditional` | 32 | 81.2 | 7 | 2 | 3 | 100.0 | 1 | 0/20 |
| `cml/random-forests-and-feature-importance/p01-permutation-importance` | 53 | 69.8 | 8 | 13 | 1 | 100.0 | 1 | 0/20 |
| `cml/random-forests-and-feature-importance/p02-bootstrap-oob-masks` | 27 | 85.2 | 2 | 2 | 1 | 100.0 | 1 | 0/20 |
| `cml/ranking-metrics/p01-ndcg-at-k` | 60 | 93.3 | 11 | 0 | 4 | 100.0 | 1 | 0/20 |
| `cml/ranking-metrics/p02-retrieval-metrics-at-k` | 32 | 100.0 | 6 | 0 | 0 | 100.0 | 1 | 0/20 |
| `cml/regression-metrics-and-robust-losses/p01-regression-report` | 53 | 88.7 | 11 | 2 | 2 | 100.0 | 2 | 0/20 |
| `cml/regression-metrics-and-robust-losses/p02-huber-loss-gradient` | 38 | 86.8 | 4 | 0 | 4 | 89.7 | 2 | 0/20 |
| `cml/ridge-regression-and-weight-decay/p01-ridge-fit` | 60 | 93.3 | 22 | 0 | 2 | 100.0 | 1 | 0/20 |
| `cml/roc-pr-curves-and-auc/p01-roc-auc-rank-statistic` | 60 | 90.0 | 7 | 2 | 4 | 100.0 | 1 | 0/20 |
| `cml/roc-pr-curves-and-auc/p02-average-precision-step` | 51 | 92.2 | 8 | 1 | 3 | 100.0 | 1 | 0/20 |
| `cml/rolling-statistics-stationarity-and-autocorrelation/p01-rolling-mean-std` | 60 | 88.3 | 5 | 0 | 2 | 97.6 | 1 | 10/20 |
| `cml/rolling-statistics-stationarity-and-autocorrelation/p02-acf` | 53 | 75.5 | 8 | 1 | 8 | 95.3 | 1 | 0/20 |
| `cml/rolling-statistics-stationarity-and-autocorrelation/p03-seasonal-naive-forecast-and-mase` | 45 | 97.8 | 13 | 1 | 0 | 96.2 | 1 | 0/20 |
| `cml/sampling-distributions-standard-errors-and-cis/p01-mean-confidence-interval` | 36 | 100.0 | 7 | 0 | 0 | 100.0 | 1 | 0/20 |
| `cml/shuffling-splitting-and-batching-data/p01-stratified-train-test-split` | 55 | 74.5 | 14 | 5 | 9 | 100.0 | 1 | 0/20 |
| `cml/shuffling-splitting-and-batching-data/p02-minibatch-index-schedule` | 36 | 83.3 | 11 | 2 | 3 | 100.0 | 1 | 0/20 |
| `cml/sigmoid-and-log-odds/p01-stable-sigmoid` | 17 | 94.1 | 4 | 0 | 1 | 100.0 | 1 | 0/20 |
| `cml/sigmoid-and-log-odds/p02-logit` | 29 | 86.2 | 8 | 0 | 4 | 100.0 | 1 | 0/20 |
| `cml/singular-value-decomposition/p01-best-rank-k` | 39 | 82.1 | 10 | 2 | 4 | 100.0 | 1 | 0/20 |
| `cml/singular-value-decomposition/p02-pseudo-inverse` | 39 | 76.9 | 11 | 1 | 7 | 100.0 | 1 | 0/20 |
| `cml/softmax/p01-stable-softmax` | 22 | 81.8 | 4 | 0 | 4 | 100.0 | 1 | 0/20 |
| `cml/softmax/p02-log-softmax` | 23 | 82.6 | 7 | 0 | 4 | 100.0 | 1 | 0/20 |
| `cml/stochastic-and-mini-batch-gradient-descent/p01-minibatch-sgd` | 43 | 93.0 | 11 | 1 | 2 | 100.0 | 1 | 0/20 |
| `cml/t-tests-and-anova/p01-welch-t-test` | 60 | 76.7 | 7 | 3 | 5 | 92.0 | 1 | 0/20 |
| `cml/t-tests-and-anova/p02-one-way-anova` | 47 | 85.1 | 3 | 4 | 3 | 100.0 | 1 | 0/20 |
| `cml/tf-idf-and-bm25/p01-tfidf-matrix` | 36 | 88.9 | 6 | 2 | 1 | 100.0 | 1 | 0/20 |
| `cml/tf-idf-and-bm25/p02-bm25-scores` | 49 | 100.0 | 8 | 0 | 0 | 100.0 | 1 | 0/20 |
| `cml/vector-and-matrix-norms/p01-vector-norm` | 25 | 92.0 | 3 | 0 | 2 | 100.0 | 1 | 0/20 |
| `cml/vector-operations/p01-vector-projection` | 21 | 90.5 | 5 | 0 | 2 | 100.0 | 1 | 0/20 |
| `cml/xgboost-second-order-objective/p01-leaf-weight-and-split-gain` | 60 | 100.0 | 12 | 0 | 0 | 100.0 | 1 | 0/20 |
| `cml/xgboost-second-order-objective/p02-logloss-grad-hess` | 42 | 95.2 | 9 | 0 | 2 | 100.0 | 1 | 0/20 |
| `dl/a-complete-training-loop/p01-fit-with-seeded-batches` | 60 | 90.0 | 28 | 2 | 2 | 100.0 | 1 | 0/20 |
| `dl/adam/p01-adam-step` | 60 | 71.7 | 8 | 6 | 7 | 89.5 | 1 | 0/20 |
| `dl/adam/p02-amsgrad-step` | 60 | 71.7 | 5 | 8 | 5 | 87.3 | 1 | 0/20 |
| `dl/adamw-decoupled-weight-decay/p01-adamw-step` | 60 | 68.3 | 11 | 8 | 5 | 83.3 | 1 | 0/20 |
| `dl/adamw-decoupled-weight-decay/p02-decay-parameter-groups` | 28 | 89.3 | 8 | 2 | 1 | 100.0 | 1 | 0/20 |
| `dl/backprop-through-a-chain-of-scalars/p01-scalar-graph-forward-backward` | 23 | 100.0 | 2 | 0 | 0 | 100.0 | 1 | 0/20 |
| `dl/backprop-through-a-dense-layer/p01-dense-backward` | 53 | 86.8 | 9 | 7 | 0 | 100.0 | 1 | 0/20 |
| `dl/batch-normalization/p01-batch-norm-forward` | 60 | 91.7 | 20 | 4 | 1 | 100.0 | 1 | 0/20 |
| `dl/batch-normalization/p02-batch-norm-backward` | 47 | 97.9 | 13 | 1 | 0 | 100.0 | 1 | 0/20 |
| `dl/building-tensors-by-broadcasting/p01-pairwise-squared-distances` | 22 | 100.0 | 5 | 0 | 0 | 87.5 | 1 | 0/20 |
| `dl/causal-masking/p01-causal-padding-mask` | 34 | 97.1 | 22 | 0 | 1 | 100.0 | 1 | 0/20 |
| `dl/convolution-forward-pass/p01-conv2d-forward` | 60 | 85.0 | 14 | 7 | 1 | 97.0 | 1 | 0/20 |
| `dl/counting-network-parameters/p01-count-dense-parameters` | 23 | 100.0 | 9 | 0 | 0 | 100.0 | 1 | 0/20 |
| `dl/dense-layer-forward-pass/p01-dense-layer-forward` | 29 | 93.1 | 7 | 0 | 2 | 100.0 | 1 | 0/20 |
| `dl/dropout/p01-inverted-dropout` | 60 | 71.7 | 13 | 15 | 0 | 100.0 | 1 | 0/20 |
| `dl/gradients-with-framework-autograd/p01-gradient-at-a-point` | 18 | 83.3 | 4 | 1 | 2 | 100.0 | 1 | 0/20 |
| `dl/gradients-with-framework-autograd/p02-straight-through-round` | 17 | 100.0 | 7 | 0 | 0 | 100.0 | 1 | 0/20 |
| `dl/infonce-contrastive-loss/p01-symmetric-infonce-loss` | 60 | 93.3 | 21 | 3 | 1 | 96.3 | 1 | 0/20 |
| `dl/kl-divergence/p01-kl-from-logits` | 46 | 87.0 | 11 | 4 | 2 | 100.0 | 1 | 0/20 |
| `dl/kl-divergence/p02-diagonal-gaussian-kl` | 53 | 94.3 | 10 | 3 | 0 | 100.0 | 1 | 0/20 |
| `dl/layer-normalization/p01-layer-norm-forward` | 40 | 97.5 | 13 | 0 | 0 | 100.0 | 1 | 0/20 |
| `dl/learning-rate-decay-schedules/p01-cosine-annealing-lr` | 44 | 97.7 | 18 | 0 | 1 | 100.0 | 1 | 0/20 |
| `dl/learning-rate-decay-schedules/p02-step-decay-lr` | 38 | 97.4 | 19 | 1 | 0 | 100.0 | 1 | 0/20 |
| `dl/learning-rate-warmup/p01-warmup-then-cosine-lr` | 55 | 98.2 | 23 | 0 | 1 | 100.0 | 1 | 0/20 |
| `dl/learning-rate-warmup/p02-inverse-sqrt-lr` | 35 | 100.0 | 11 | 0 | 0 | 100.0 | 1 | 0/20 |
| `dl/mixed-precision/p01-dynamic-loss-scaler` | 56 | 100.0 | 23 | 0 | 0 | 100.0 | 1 | 0/20 |
| `dl/multi-head-attention/p01-self-attention-heads` | 60 | 95.0 | 27 | 0 | 3 | 100.0 | 1 | 0/20 |
| `dl/one-training-step/p01-manual-sgd-train-step` | 18 | 94.4 | 1 | 0 | 1 | 100.0 | 1 | 0/20 |
| `dl/position-wise-feed-forward-block/p01-gelu-ffn` | 34 | 91.2 | 7 | 2 | 1 | 93.8 | 1 | 0/20 |
| `dl/position-wise-feed-forward-block/p02-swiglu-ffn` | 30 | 86.7 | 7 | 2 | 2 | 93.1 | 1 | 0/20 |
| `dl/positional-encodings/p01-sinusoidal-table` | 40 | 95.0 | 6 | 0 | 2 | 100.0 | 1 | 0/20 |
| `dl/positional-encodings/p02-rope-rotations` | 54 | 98.1 | 15 | 0 | 1 | 100.0 | 1 | 0/20 |
| `dl/relu-family-activations/p01-relu-forward-and-gradient` | 31 | 96.8 | 11 | 0 | 1 | 100.0 | 1 | 0/20 |
| `dl/residual-connections/p01-residual-block` | 60 | 73.3 | 14 | 15 | 0 | 82.1 | 1 | 0/20 |
| `dl/reverse-mode-autograd-engine/p01-scalar-autograd` | 27 | 100.0 | 11 | 0 | 0 | 100.0 | 2 | 0/20 |
| `dl/rmsnorm/p01-rmsnorm-forward` | 31 | 96.8 | 14 | 0 | 0 | 100.0 | 1 | 0/20 |
| `dl/scaled-dot-product-attention/p01-scaled-dot-product-attention` | 60 | 91.7 | 21 | 1 | 1 | 90.9 | 1 | 0/20 |
| `dl/sgd-with-momentum/p01-momentum-step` | 45 | 82.2 | 12 | 3 | 3 | 92.0 | 1 | 0/20 |
| `dl/sgd-with-momentum/p02-nesterov-weight-decay` | 56 | 82.1 | 10 | 4 | 4 | 85.7 | 1 | 0/20 |
| `dl/smooth-activations-tanh-gelu-swish/p01-gelu-and-swish` | 42 | 95.2 | 8 | 0 | 1 | 100.0 | 1 | 0/20 |
| `dl/smooth-activations-tanh-gelu-swish/p02-smooth-activation-derivatives` | 60 | 96.7 | 13 | 0 | 1 | 100.0 | 1 | 0/20 |
| `dl/tensors-in-pytorch-and-tinygrad/p01-batched-pairwise-distances` | 25 | 96.0 | 4 | 1 | 0 | 100.0 | 1 | 0/20 |
| `dl/tensors-in-pytorch-and-tinygrad/p02-patchify-images-row-major` | 36 | 100.0 | 6 | 0 | 0 | 100.0 | 1 | 0/20 |
| `dl/training-an-mlp-from-scratch/p01-train-two-layer-mlp` | 60 | 71.7 | 24 | 8 | 7 | 95.5 | 1 | 0/20 |
| `dl/transformer-block/p01-pre-norm-causal-block` | 60 | 91.7 | 20 | 3 | 1 | 95.2 | 1 | 0/20 |
| `dl/xavier-and-he-initialization/p01-initialize-dense-weights` | 34 | 97.1 | 10 | 0 | 1 | 100.0 | 1 | 0/20 |
| `inf/absmax-and-zero-point-int8-quantization/p01-absmax-quantize` | 60 | 85.0 | 28 | 7 | 1 | 97.6 | 1 | 0/20 |
| `inf/absmax-and-zero-point-int8-quantization/p02-zero-point-quantize` | 43 | 81.4 | 10 | 4 | 4 | 100.0 | 1 | 0/20 |
| `inf/arithmetic-intensity-and-the-roofline/p01-matmul-roofline` | 27 | 100.0 | 4 | 0 | 0 | 100.0 | 1 | 0/20 |
| `inf/continuous-batching/p01-batch-scheduler-steps` | 60 | 88.3 | 11 | 1 | 6 | 100.0 | 1 | 0/20 |
| `inf/cost-per-token-and-break-even/p01-cost-per-million-and-break-even` | 60 | 98.3 | 14 | 0 | 0 | 100.0 | 1 | 0/20 |
| `inf/data-quality-and-drift-detection/p01-population-stability-index` | 46 | 93.5 | 10 | 2 | 1 | 100.0 | 1 | 0/20 |
| `inf/data-quality-and-drift-detection/p02-ks-drift-test` | 60 | 88.3 | 22 | 0 | 6 | 98.3 | 1 | 0/20 |
| `inf/floating-point-formats/p01-decode-a-bit-pattern` | 60 | 93.3 | 18 | 4 | 0 | 100.0 | 1 | 0/20 |
| `inf/floating-point-formats/p02-round-to-nearest-even` | 60 | 85.0 | 14 | 8 | 1 | 100.0 | 1 | 0/20 |
| `inf/gpu-memory-and-capacity-sizing/p01-min-gpu-count` | 60 | 91.7 | 29 | 5 | 0 | 100.0 | 1 | 0/20 |
| `inf/kv-cache-size-arithmetic/p01-kv-cache-bytes` | 46 | 91.3 | 17 | 1 | 0 | 95.7 | 1 | 0/20 |
| `inf/kv-cache-size-arithmetic/p02-max-batch-after-weights` | 58 | 81.0 | 18 | 5 | 2 | 96.0 | 1 | 0/20 |
| `inf/paged-kv-cache/p01-block-allocator` | 37 | 94.6 | 12 | 0 | 2 | 100.0 | 1 | 0/20 |
| `inf/prefill-vs-decode-bottlenecks/p01-serving-step-times` | 60 | 95.0 | 25 | 3 | 0 | 100.0 | 1 | 0/20 |
| `inf/prefill-vs-decode-bottlenecks/p02-compute-bound-decode-batch` | 55 | 90.9 | 10 | 3 | 2 | 100.0 | 1 | 0/20 |
| `inf/speculative-decoding-speedup-math/p01-speculative-speedup` | 48 | 95.8 | 16 | 2 | 0 | 95.6 | 1 | 0/20 |
| `inf/speculative-decoding-speedup-math/p02-accept-or-resample` | 60 | 83.3 | 23 | 3 | 5 | 96.8 | 1 | 0/20 |
| `inf/ttft-itl-and-throughput/p01-latency-percentiles` | 60 | 96.7 | 13 | 1 | 0 | 97.5 | 1 | 0/20 |
| `inf/ttft-itl-and-throughput/p02-throughput-and-goodput` | 60 | 88.3 | 11 | 5 | 1 | 90.9 | 1 | 0/20 |
| `llm/autoregressive-generation-loop/p01-greedy-generation-with-eos` | 58 | 89.7 | 17 | 5 | 1 | 95.7 | 1 | 0/20 |
| `llm/bpe-encode-and-decode/p01-apply-bpe-merges` | 60 | 88.3 | 13 | 4 | 3 | 97.6 | 1 | 0/20 |
| `llm/bpe-merge-training/p01-learn-bpe-merges` | 60 | 88.3 | 14 | 1 | 6 | 97.5 | 1 | 0/20 |
| `llm/bradley-terry-reward-models/p01-reward-pair-loss-and-accuracy` | 42 | 88.1 | 8 | 3 | 2 | 94.9 | 1 | 0/20 |
| `llm/chat-templates-and-tool-call-formats/p01-render-chat-template` | 43 | 88.4 | 9 | 2 | 3 | 97.3 | 1 | 0/20 |
| `llm/chat-templates-and-tool-call-formats/p02-parse-tool-call` | 43 | 88.4 | 15 | 1 | 2 | 94.3 | 1 | 0/20 |
| `llm/direct-preference-optimization-dpo/p01-sequence-log-probs-with-response-mask` | 60 | 80.0 | 11 | 8 | 1 | 91.7 | 1 | 0/20 |
| `llm/direct-preference-optimization-dpo/p02-dpo-loss-and-implicit-rewards` | 48 | 85.4 | 14 | 5 | 2 | 93.2 | 1 | 0/20 |
| `llm/document-packing-and-cross-document-masks/p01-packed-causal-mask` | 43 | 81.4 | 8 | 1 | 7 | 100.0 | 1 | 0/20 |
| `llm/gpt-forward-pass/p01-decoder-only-gpt-forward` | 60 | 80.0 | 30 | 12 | 0 | 89.3 | 1 | 0/20 |
| `llm/gqa-mqa-and-cross-layer-kv-sharing/p01-grouped-query-attention` | 60 | 93.3 | 17 | 2 | 2 | 96.8 | 1 | 0/20 |
| `llm/grpo-and-its-variants/p01-group-normalised-advantages` | 44 | 90.9 | 14 | 4 | 0 | 95.1 | 1 | 0/20 |
| `llm/grpo-and-its-variants/p02-clipped-grpo-token-loss` | 60 | 78.3 | 27 | 12 | 1 | 86.5 | 1 | 0/20 |
| `llm/incremental-decoding-with-a-kv-cache/p01-cached-causal-attention-step` | 60 | 86.7 | 16 | 6 | 2 | 94.7 | 1 | 0/20 |
| `llm/instruction-tuning-batches-and-loss-masks/p01-sft-batch-with-loss-masks` | 60 | 95.0 | 24 | 3 | 0 | 96.8 | 1 | 0/20 |
| `llm/llm-as-judge-and-agreement-with-humans/p01-cohens-kappa` | 38 | 86.8 | 6 | 1 | 4 | 100.0 | 1 | 0/20 |
| `llm/llm-as-judge-and-agreement-with-humans/p02-debiased-win-rate` | 29 | 96.6 | 7 | 0 | 1 | 100.0 | 1 | 0/20 |
| `llm/lora-adapters/p01-lora-linear-forward-and-merge` | 60 | 93.3 | 18 | 1 | 3 | 100.0 | 1 | 0/20 |
| `llm/lora-parameter-and-memory-arithmetic/p01-lora-trainable-parameter-budget` | 34 | 88.2 | 9 | 0 | 4 | 100.0 | 1 | 0/20 |
| `llm/lora-parameter-and-memory-arithmetic/p02-finetune-memory-in-bytes` | 43 | 97.7 | 9 | 0 | 1 | 100.0 | 1 | 0/20 |
| `llm/model-parameter-and-memory-footprint/p01-count-decoder-parameters` | 60 | 95.0 | 13 | 3 | 0 | 100.0 | 1 | 0/20 |
| `llm/model-parameter-and-memory-footprint/p02-training-and-inference-memory` | 60 | 96.7 | 24 | 0 | 2 | 100.0 | 1 | 0/20 |
| `llm/moe-top-k-routing/p01-top-k-router-gates` | 38 | 97.4 | 11 | 0 | 1 | 100.0 | 1 | 0/20 |
| `llm/moe-top-k-routing/p02-load-balancing-loss` | 48 | 93.8 | 11 | 1 | 2 | 100.0 | 1 | 0/20 |
| `llm/negative-log-likelihood-and-perplexity/p01-masked-nll-and-perplexity` | 60 | 91.7 | 13 | 3 | 2 | 95.7 | 1 | 0/20 |
| `llm/next-token-targets-and-batches/p01-pack-next-token-blocks` | 59 | 74.6 | 15 | 11 | 3 | 93.2 | 1 | 0/20 |
| `llm/padding-truncation-and-pad-masks/p01-pad-and-truncate-token-ids` | 56 | 78.6 | 17 | 9 | 3 | 93.8 | 1 | 0/20 |
| `llm/pass-k-and-majority-voting/p01-unbiased-pass-at-k` | 38 | 92.1 | 12 | 0 | 3 | 100.0 | 1 | 0/20 |
| `llm/pass-k-and-majority-voting/p02-majority-vote` | 16 | 75.0 | 3 | 0 | 4 | 100.0 | 1 | 0/20 |
| `llm/retrieval-augmented-generation-pipeline/p01-chunk-words` | 31 | 93.5 | 9 | 0 | 2 | 100.0 | 1 | 0/20 |
| `llm/retrieval-augmented-generation-pipeline/p02-cited-context-prompt` | 60 | 88.3 | 25 | 5 | 2 | 96.3 | 1 | 0/20 |
| `llm/rotary-position-embeddings-rope/p01-half-split-rope-with-offset` | 60 | 93.3 | 20 | 1 | 3 | 100.0 | 1 | 0/20 |
| `llm/rotary-position-embeddings-rope/p02-linear-position-interpolation` | 60 | 88.3 | 17 | 4 | 3 | 95.8 | 1 | 0/20 |
| `llm/scaling-laws-and-compute-budgets/p01-compute-optimal-allocation` | 25 | 84.0 | 8 | 0 | 4 | 100.0 | 1 | 0/20 |
| `llm/scaling-laws-and-compute-budgets/p02-fit-power-law-log-log` | 52 | 84.6 | 14 | 5 | 3 | 100.0 | 1 | 0/20 |
| `llm/temperature-top-k-and-top-p-sampling/p01-filtered-next-token-distribution` | 60 | 93.3 | 17 | 1 | 2 | 100.0 | 1 | 0/20 |
| `llm/temperature-top-k-and-top-p-sampling/p02-sample-with-a-generator` | 50 | 70.0 | 12 | 1 | 4 | 90.9 | 1 | 0/20 |
| `llm/word-and-character-tokenizers/p01-word-level-vocabulary` | 40 | 95.0 | 13 | 2 | 0 | 97.0 | 1 | 0/20 |
| `math/determinants-and-trace/p01-determinant` | 35 | 94.3 | 6 | 0 | 2 | 100.0 | 1 | 0/20 |
| `math/inverse-transform-sampling/p01-sample-exponential` | 15 | 100.0 | 2 | 0 | 0 | 100.0 | 1 | 0/20 |
| `math/inverse-transform-sampling/p02-sample-discrete` | 34 | 91.2 | 15 | 1 | 2 | 100.0 | 1 | 0/20 |
| `math/monte-carlo-simulation/p01-mc-estimate` | 33 | 90.9 | 6 | 1 | 2 | 100.0 | 1 | 0/20 |
| `math/reservoir-sampling/p01-reservoir-sample` | 15 | 100.0 | 3 | 0 | 0 | 100.0 | 1 | 0/20 |
| `rl/action-value-estimates/p01-incremental-action-values` | 60 | 90.0 | 22 | 3 | 3 | 100.0 | 1 | 0/20 |
| `rl/epsilon-greedy-exploration/p01-epsilon-greedy-action` | 34 | 91.2 | 12 | 0 | 3 | 100.0 | 1 | 0/20 |
| `rl/mdps-and-discounted-returns/p01-discounted-returns` | 36 | 94.4 | 9 | 0 | 2 | 100.0 | 1 | 0/20 |
| `rl/mdps-and-discounted-returns/p02-iterative-policy-evaluation` | 60 | 75.0 | 15 | 9 | 6 | 97.6 | 1 | 0/20 |
| `rl/q-learning/p01-q-learning-gridworld` | 60 | 85.0 | 33 | 9 | 0 | 98.2 | 1 | 0/20 |
| `rl/reinforce/p01-reinforce-loss` | 60 | 81.7 | 16 | 9 | 2 | 91.8 | 1 | 0/20 |
| `rl/sarsa-and-expected-sarsa/p01-sarsa-and-expected-sarsa-updates` | 60 | 73.3 | 12 | 6 | 5 | 89.5 | 1 | 0/20 |
| `safe/split-conformal-prediction/p01-split-conformal-intervals` | 52 | 94.2 | 17 | 0 | 3 | 100.0 | 1 | 0/20 |
| `safe/split-conformal-prediction/p02-conformal-prediction-sets` | 60 | 78.3 | 19 | 12 | 0 | 93.8 | 1 | 0/20 |
| `sql/pandas-groupby-aggregate-and-transform/p01-customer-order-features` | 10 | 90.0 | 3 | 0 | 1 | 100.0 | 1 | 0/20 |
| `sql/pandas-merge-and-reshape/p01-monthly-revenue-by-region` | 2 | 50.0 | 0 | 0 | 1 | 100.0 | 1 | 0/20 |
| `sql/sql-cohort-retention/p01-cohort-retention-by-month` | 18 | 88.9 | 2 | 0 | 2 | sql | 1 | 0/20 |
| `sql/sql-conditional-aggregation/p01-channel-conversion-and-device-revenue` | 17 | 100.0 | 3 | 0 | 0 | sql | 1 | 0/20 |
| `sql/sql-date-handling/p01-monthly-active-users` | 9 | 100.0 | 0 | 0 | 0 | sql | 1 | 0/20 |
| `sql/sql-date-handling/p02-days-from-first-to-second-event` | 10 | 100.0 | 0 | 0 | 0 | sql | 1 | 0/20 |
| `sql/sql-first-and-last-activity-per-user/p01-first-and-last-event` | 6 | 100.0 | 0 | 0 | 0 | sql | 1 | 0/20 |
| `sql/sql-group-by-and-having/p01-customers-with-three-completed-orders` | 8 | 100.0 | 1 | 0 | 0 | sql | 1 | 0/20 |
| `sql/sql-joins-and-unions/p01-customer-order-totals` | 6 | 100.0 | 0 | 0 | 0 | sql | 1 | 0/20 |
| `sql/sql-joins-and-unions/p02-warehouses-with-a-shipment-or-a-return` | 3 | 100.0 | 0 | 0 | 0 | sql | 1 | 0/20 |
| `sql/sql-lag-and-lead/p01-day-over-day-change` | 4 | 100.0 | 0 | 0 | 0 | sql | 1 | 0/20 |
| `sql/sql-lag-and-lead/p02-sessions-by-inactivity-gap` | 11 | 90.9 | 2 | 0 | 1 | sql | 1 | 0/20 |
| `sql/sql-ranking-window-functions/p01-rank-within-category` | 19 | 94.7 | 1 | 0 | 1 | sql | 1 | 0/20 |
| `sql/sql-ranking-window-functions/p02-top-two-with-ties` | 14 | 100.0 | 0 | 0 | 0 | sql | 1 | 0/20 |
| `sql/sql-running-totals-and-frames/p01-running-balance-with-ties` | 12 | 83.3 | 3 | 0 | 2 | sql | 1 | 0/20 |
| `sql/sql-running-totals-and-frames/p02-trailing-seven-day-average` | 11 | 100.0 | 2 | 0 | 0 | sql | 1 | 0/20 |
| `sql/sql-select-filter-and-sort/p01-top-five-completed-orders-2024` | 11 | 100.0 | 0 | 0 | 0 | sql | 1 | 0/20 |
| `sql/sql-streaks-gaps-and-islands/p01-longest-streak-per-user` | 13 | 84.6 | 0 | 0 | 2 | sql | 1 | 0/20 |
| `sql/sql-streaks-gaps-and-islands/p02-all-islands-per-user` | 17 | 88.2 | 2 | 0 | 2 | sql | 1 | 0/20 |
| `sql/sql-subqueries-and-ctes/p01-customers-above-average-spend` | 8 | 100.0 | 1 | 0 | 0 | sql | 1 | 0/20 |
| `sql/sql-subqueries-and-ctes/p02-orders-above-their-customers-average` | 11 | 100.0 | 0 | 0 | 0 | sql | 1 | 0/20 |
| `sql/sql-top-n-and-latest-row-per-group/p01-latest-events` | 9 | 88.9 | 1 | 0 | 1 | sql | 1 | 0/20 |
