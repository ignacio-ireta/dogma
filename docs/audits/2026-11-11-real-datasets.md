# Real datasets in the problems (I-33)

*2026-11-11. Every number and the table are generated from the hidden tests in `content/`.*

## Summary

All 251 problems have hidden tests that read a real dataset. They fall into two groups, and the table marks each problem with its group:

- **Data (221).** The problem takes data (an array, a table, a series, text, a distribution, rewards, verdicts) and the arrays its hidden tests draw are real values, or are derived from real values by a stated rule. The draws that are still generated are listed, problem by problem and with the reason, in the last section. Hand-built edge cases with exact answers and Hypothesis properties over the stated domain are unchanged.
- **Sizes and rates (30).** The problem takes no data: its arguments are sizes, counts, lengths, rates, prices or settings. The numbers its tests feed it are read off real datasets (row and feature counts, passage and sentence lengths, rates counted from tables, real prices), where they were drawn at random before. The values of the dataset never enter the function, because the function has no argument that could take them.

| Track | Problems | Data | Sizes and rates |
|---|---|---|---|
| Algorithms (`algo`) | 20 | 20 | 0 |
| Classical ML (`cml`) | 92 | 85 | 7 |
| Deep learning (`dl`) | 46 | 39 | 7 |
| Inference (`inf`) | 19 | 10 | 9 |
| LLMs (`llm`) | 38 | 32 | 6 |
| Math (`math`) | 5 | 4 | 1 |
| Reinforcement learning (`rl`) | 7 | 7 | 0 |
| Safety (`safe`) | 2 | 2 | 0 |
| SQL and pandas (`sql`) | 22 | 22 | 0 |
| Total | 251 | 221 | 30 |

How to read "real": the values come from a vendored dataset (scikit-learn and statsmodels bundles, pinned HuggingFace snapshots, Online Retail II from UCI, MT-Bench human judgments). What a test still sets itself, and each helper says so in its docstring:

- **A known truth.** Lasso, ridge, logistic regression, the normal equation and permutation importance take real features and a target built from a known rule. One Monte Carlo test keeps a standard normal, because it checks a closed form, E[exp(Z)] = e^(1/2).
- **A property that must be exact.** Spectra in power iteration and gradient descent, the rank in matrix rank, gaps at exactly thirty minutes in the session problem, dates on a year boundary.
- **Dirt and gaps the source lacks.** Missing measurements, a missing region, a status typed with a capital, labels in another letter case, a missing timestamp.
- **Hand-built edge cases.** A zero probability at either end, a draw exactly on a cumulative sum, identical distributions, tied action values, the special words of a floating-point format (both zeros, subnormals with either sign, infinities, NaNs with a payload), exact rounding ties.
- **Hypothesis properties over the stated domain,** where a problem has them, and the parameters of a simulation: a policy, a step size, a batch limit, arrival steps, hardware figures.
- **A real value in place of a random one.** Where a secondary test drew a normal array, it now takes standardised `breast_cancer` measurements of the same shape (helper `_real_normal`): a vector is one real measurement over that many tumours, a matrix is real tumours by real measurements. Where it drew uniform numbers, whole numbers in a range, probability vectors or a random mask, it takes the same measurements passed through the normal distribution function (`_real_unit`), cut into bands (`_real_ints`), normalised (`_real_shares`) or thresholded. These keep the order, ties and skew of real measurements; they are derived values, and each helper says how.
- **A real value in a role no dataset records.** Tensors inside a network are standardised `breast_cancer` measurements in the shape asked for (helper `_real_values`, 29 torch problems): the inputs and hidden states are real feature vectors, and the weights and gradients, which are parameters, take real values too. Token timings in the two latency problems are real waits between invoices, rescaled to seconds. GPU and API prices are real prices in pounds. Overflow flags are runs of cancelled invoices. Graded samples in pass@k are the right and wrong calls of a simple rule on real wines.

The SQL problems map real retail rows onto each problem's own table (an invoice line stands for a session, a posting, a view or a shipment), and four of them (monthly active users, top five orders of 2024, both streak problems) move real dates forward by a fixed number of days so that the calendar window of the statement falls inside the data.

## The problems (251)

| Problem | Group | Datasets |
|---|---|---|
| `algo/bfs-and-dfs-on-graphs-and-grids/p01-grid-shortest-path` | data | `digits` |
| `algo/bfs-and-dfs-on-graphs-and-grids/p02-count-components` | data | `digits` |
| `algo/binary-search/p01-lower-bound` | data | `digits`, `iris` |
| `algo/binary-search/p02-min-ship-capacity` | data | `online_retail` |
| `algo/binary-tree-recursion-and-traversal/p01-inorder-traversal-iterative` | data | `wine` |
| `algo/binary-tree-recursion-and-traversal/p02-tree-diameter` | data | `wine` |
| `algo/dynamic-programming/p01-minimum-coins` | data | `online_retail` |
| `algo/dynamic-programming/p02-longest-common-subsequence` | data | `gutenberg_frankenstein`, `gutenberg_pride_prejudice`, `gutenberg_walden` |
| `algo/hash-maps-and-sets/p01-top-k-frequent-tokens` | data | `gutenberg_frankenstein`, `gutenberg_pride_prejudice`, `gutenberg_walden` |
| `algo/heaps-and-priority-queues/p01-merge-sorted-streams` | data | `breast_cancer`, `online_retail` |
| `algo/heaps-and-priority-queues/p02-running-median` | data | `breast_cancer`, `elnino`, `sunspots` |
| `algo/lru-cache/p01-lru-cache` | data | `online_retail` |
| `algo/prefix-sums/p01-subarrays-with-sum-k` | data | `breast_cancer`, `elnino` |
| `algo/prefix-sums/p02-rectangle-range-sums` | data | `breast_cancer`, `digits` |
| `algo/sort-then-sweep/p01-merge-intervals` | data | `online_retail` |
| `algo/sort-then-sweep/p02-peak-concurrent-sessions` | data | `online_retail` |
| `algo/stacks-and-queues/p01-days-until-higher` | data | `elnino` |
| `algo/stacks-and-queues/p02-sliding-window-maximum` | data | `elnino`, `sunspots` |
| `algo/topological-sort/p01-lexicographic-topological-order` | data | `online_retail` |
| `algo/two-pointers-and-sliding-windows/p01-longest-window-within-budget` | data | `co2`, `elnino` |
| `cml/a-b-test-design-and-analysis/p01-sample-size-two-proportions` | sizes and rates | `breast_cancer`, `online_retail`, `spector` |
| `cml/a-b-test-design-and-analysis/p02-two-proportion-z-test` | data | `adult` |
| `cml/adaboost-and-gradient-boosting/p01-boost-stumps-mse` | data | `breast_cancer`, `diabetes` |
| `cml/adaboost-and-gradient-boosting/p02-adaboost-round` | data | `breast_cancer` |
| `cml/bayes-theorem/p01-bayes-posterior` | data | `breast_cancer`, `california_housing`, `iris`, `wine` |
| `cml/bernoulli-binomial-and-poisson/p01-binomial-pmf` | sizes and rates | `online_retail` |
| `cml/bernoulli-binomial-and-poisson/p02-poisson-pmf` | sizes and rates | `online_retail` |
| `cml/bias-variance-decomposition/p01-bias-variance-from-predictions` | data | `breast_cancer`, `diabetes` |
| `cml/bootstrap-and-permutation-resampling/p01-bootstrap-percentile-ci` | data | `california_housing`, `iris` |
| `cml/bootstrap-and-permutation-resampling/p02-permutation-test-mean-diff` | data | `iris` |
| `cml/chain-rule/p01-mse-sigmoid-gradient` | data | `breast_cancer` |
| `cml/choosing-a-tree-split/p01-best-split-threshold` | data | `breast_cancer`, `iris` |
| `cml/confusion-matrix-and-count-based-scores/p01-confusion-matrix-from-labels` | data | `breast_cancer` |
| `cml/confusion-matrix-and-count-based-scores/p02-binary-count-scores` | data | `adult`, `breast_cancer` |
| `cml/cosine-similarity/p01-cosine-similarity-matrix` | data | `breast_cancer`, `digits` |
| `cml/covariance-and-correlation/p01-covariance-matrix` | data | `breast_cancer`, `wine` |
| `cml/covariance-and-correlation/p02-correlation-matrix` | data | `wine` |
| `cml/cross-entropy/p01-cross-entropy-from-logits` | data | `breast_cancer`, `digits` |
| `cml/cross-entropy/p02-binary-cross-entropy-with-logits` | data | `breast_cancer` |
| `cml/derivatives/p01-numerical-derivative` | data | `wine` |
| `cml/descriptive-statistics/p01-welford-mean-variance` | data | `breast_cancer`, `california_housing`, `nile`, `sunspots` |
| `cml/descriptive-statistics/p02-iqr-outlier-fences` | data | `engel`, `nile` |
| `cml/eigenvalues-and-eigenvectors/p01-power-iteration` | data | `breast_cancer`, `wine` |
| `cml/entropy-and-mutual-information/p01-distribution-entropy` | data | `adult`, `breast_cancer` |
| `cml/entropy-and-mutual-information/p02-mutual-information-from-labels` | data | `adult`, `breast_cancer` |
| `cml/expectation-and-variance/p01-discrete-moments` | data | `iris` |
| `cml/feature-scaling/p01-standard-scaler` | data | `wine` |
| `cml/feature-scaling/p02-robust-median-iqr-scaler` | data | `digits`, `wine` |
| `cml/gaussian-mixtures-and-em/p01-gmm-em-full-covariance` | data | `iris`, `wine` |
| `cml/gradient-descent/p01-gradient-descent` | data | `breast_cancer`, `wine` |
| `cml/handling-class-imbalance/p01-balanced-class-weights` | data | `adult` |
| `cml/handling-class-imbalance/p02-prior-shift-adjust` | data | `breast_cancer` |
| `cml/k-fold-cross-validation/p01-kfold-indices` | sizes and rates | `breast_cancer`, `diabetes`, `iris`, `wine` |
| `cml/k-fold-cross-validation/p02-cross-validate` | data | `diabetes` |
| `cml/k-means/p01-lloyd-kmeans` | data | `breast_cancer`, `california_housing` |
| `cml/k-means/p02-kmeans-plus-plus-seeding` | data | `breast_cancer`, `california_housing` |
| `cml/lagrange-multipliers/p01-constrained-quadratic-min` | data | `wine` |
| `cml/lasso-and-elastic-net/p01-lasso-coordinate-descent` | data | `breast_cancer`, `diabetes` |
| `cml/lasso-and-elastic-net/p02-elastic-net-coordinate-descent` | data | `breast_cancer`, `diabetes` |
| `cml/leakage-and-walk-forward-validation/p01-walk-forward-splits` | sizes and rates | `nile` |
| `cml/leakage-and-walk-forward-validation/p02-expanding-target-mean` | data | `adult`, `breast_cancer` |
| `cml/logistic-regression/p01-l2-logistic-fit` | data | `breast_cancer`, `spector` |
| `cml/matrix-inverse-and-rank/p01-gauss-jordan-inverse` | data | `breast_cancer`, `wine` |
| `cml/matrix-inverse-and-rank/p02-matrix-rank` | data | `breast_cancer`, `wine` |
| `cml/matrix-multiplication/p01-matrix-multiply` | data | `digits`, `wine` |
| `cml/matrix-shapes-and-transpose/p01-transpose-matrix` | data | `digits`, `wine` |
| `cml/maximum-likelihood-and-map/p01-gaussian-mle` | data | `breast_cancer`, `wine` |
| `cml/maximum-likelihood-and-map/p02-beta-bernoulli-map` | data | `breast_cancer` |
| `cml/missing-values-and-outliers/p01-impute-with-missing-indicators` | data | `breast_cancer`, `penguins` |
| `cml/missing-values-and-outliers/p02-modified-zscore-outliers` | data | `california_housing` |
| `cml/nearest-neighbour-search-and-knn/p01-knn-classify` | data | `breast_cancer`, `california_housing` |
| `cml/nearest-neighbour-search-and-knn/p02-knn-regress` | data | `breast_cancer`, `california_housing` |
| `cml/normal-and-multivariate-gaussian/p01-mvn-logpdf` | data | `breast_cancer`, `wine` |
| `cml/normal-equation/p01-normal-equation-fit` | data | `breast_cancer`, `diabetes`, `longley`, `stackloss` |
| `cml/over-and-underfitting-early-stopping/p01-early-stopping` | data | `breast_cancer` |
| `cml/partial-derivatives-and-gradients/p01-numerical-gradient` | data | `wine` |
| `cml/precision-recall-and-f1/p01-precision-recall-averaging` | data | `breast_cancer`, `digits` |
| `cml/precision-recall-and-f1/p02-best-f1-threshold` | data | `breast_cancer` |
| `cml/preprocessing-pipelines/p01-numeric-categorical-pipeline` | data | `adult`, `breast_cancer` |
| `cml/principal-component-analysis/p01-pca-via-svd` | data | `breast_cancer`, `diabetes`, `wine` |
| `cml/principal-component-analysis/p02-components-for-variance` | data | `breast_cancer`, `wine` |
| `cml/probability-basics/p01-marginals-and-conditional` | data | `adult`, `breast_cancer` |
| `cml/random-forests-and-feature-importance/p01-permutation-importance` | data | `breast_cancer` |
| `cml/random-forests-and-feature-importance/p02-bootstrap-oob-masks` | sizes and rates | `digits`, `iris`, `wine` |
| `cml/ranking-metrics/p01-ndcg-at-k` | data | `breast_cancer`, `california_housing` |
| `cml/ranking-metrics/p02-retrieval-metrics-at-k` | data | `california_housing` |
| `cml/regression-metrics-and-robust-losses/p01-regression-report` | data | `breast_cancer`, `california_housing` |
| `cml/regression-metrics-and-robust-losses/p02-huber-loss-gradient` | data | `california_housing` |
| `cml/ridge-regression-and-weight-decay/p01-ridge-fit` | data | `breast_cancer`, `diabetes` |
| `cml/roc-pr-curves-and-auc/p01-roc-auc-rank-statistic` | data | `breast_cancer` |
| `cml/roc-pr-curves-and-auc/p02-average-precision-step` | data | `breast_cancer` |
| `cml/rolling-statistics-stationarity-and-autocorrelation/p01-rolling-mean-std` | data | `breast_cancer`, `elnino`, `nile`, `sunspots` |
| `cml/rolling-statistics-stationarity-and-autocorrelation/p02-acf` | data | `elnino`, `nile`, `sunspots` |
| `cml/rolling-statistics-stationarity-and-autocorrelation/p03-seasonal-naive-forecast-and-mase` | data | `breast_cancer`, `co2`, `elnino`, `sunspots` |
| `cml/sampling-distributions-standard-errors-and-cis/p01-mean-confidence-interval` | data | `breast_cancer`, `california_housing` |
| `cml/shuffling-splitting-and-batching-data/p01-stratified-train-test-split` | data | `digits` |
| `cml/shuffling-splitting-and-batching-data/p02-minibatch-index-schedule` | sizes and rates | `iris`, `wine` |
| `cml/sigmoid-and-log-odds/p01-stable-sigmoid` | data | `breast_cancer` |
| `cml/sigmoid-and-log-odds/p02-logit` | data | `breast_cancer` |
| `cml/singular-value-decomposition/p01-best-rank-k` | data | `breast_cancer`, `wine` |
| `cml/singular-value-decomposition/p02-pseudo-inverse` | data | `breast_cancer`, `wine` |
| `cml/softmax/p01-stable-softmax` | data | `breast_cancer` |
| `cml/softmax/p02-log-softmax` | data | `breast_cancer` |
| `cml/stochastic-and-mini-batch-gradient-descent/p01-minibatch-sgd` | data | `diabetes` |
| `cml/t-tests-and-anova/p01-welch-t-test` | data | `breast_cancer`, `iris` |
| `cml/t-tests-and-anova/p02-one-way-anova` | data | `digits`, `iris` |
| `cml/tf-idf-and-bm25/p01-tfidf-matrix` | data | `gutenberg_frankenstein`, `gutenberg_pride_prejudice`, `gutenberg_walden` |
| `cml/tf-idf-and-bm25/p02-bm25-scores` | data | `gutenberg_frankenstein`, `gutenberg_pride_prejudice`, `gutenberg_walden` |
| `cml/vector-and-matrix-norms/p01-vector-norm` | data | `wine` |
| `cml/vector-operations/p01-vector-projection` | data | `digits`, `wine` |
| `cml/xgboost-second-order-objective/p01-leaf-weight-and-split-gain` | data | `breast_cancer` |
| `cml/xgboost-second-order-objective/p02-logloss-grad-hess` | data | `breast_cancer` |
| `dl/a-complete-training-loop/p01-fit-with-seeded-batches` | data | `breast_cancer`, `diabetes` |
| `dl/adam/p01-adam-step` | data | `breast_cancer` |
| `dl/adam/p02-amsgrad-step` | data | `breast_cancer` |
| `dl/adamw-decoupled-weight-decay/p01-adamw-step` | data | `breast_cancer` |
| `dl/adamw-decoupled-weight-decay/p02-decay-parameter-groups` | data | `breast_cancer` |
| `dl/backprop-through-a-chain-of-scalars/p01-scalar-graph-forward-backward` | data | `wine` |
| `dl/backprop-through-a-dense-layer/p01-dense-backward` | data | `breast_cancer`, `digits` |
| `dl/batch-normalization/p01-batch-norm-forward` | data | `breast_cancer` |
| `dl/batch-normalization/p02-batch-norm-backward` | data | `breast_cancer` |
| `dl/building-tensors-by-broadcasting/p01-pairwise-squared-distances` | data | `breast_cancer`, `digits` |
| `dl/causal-masking/p01-causal-padding-mask` | data | `breast_cancer` |
| `dl/convolution-forward-pass/p01-conv2d-forward` | data | `breast_cancer`, `digits` |
| `dl/counting-network-parameters/p01-count-dense-parameters` | sizes and rates | `breast_cancer`, `california_housing`, `diabetes`, `digits`, `iris`, `wine` |
| `dl/dense-layer-forward-pass/p01-dense-layer-forward` | data | `breast_cancer`, `digits` |
| `dl/dropout/p01-inverted-dropout` | data | `wine` |
| `dl/gradients-with-framework-autograd/p01-gradient-at-a-point` | data | `breast_cancer` |
| `dl/gradients-with-framework-autograd/p02-straight-through-round` | data | `breast_cancer` |
| `dl/infonce-contrastive-loss/p01-symmetric-infonce-loss` | data | `digits` |
| `dl/kl-divergence/p01-kl-from-logits` | data | `breast_cancer` |
| `dl/kl-divergence/p02-diagonal-gaussian-kl` | data | `iris` |
| `dl/layer-normalization/p01-layer-norm-forward` | data | `breast_cancer` |
| `dl/learning-rate-decay-schedules/p01-cosine-annealing-lr` | sizes and rates | `breast_cancer`, `diabetes`, `digits`, `iris`, `wine` |
| `dl/learning-rate-decay-schedules/p02-step-decay-lr` | sizes and rates | `breast_cancer`, `diabetes`, `digits`, `iris`, `wine` |
| `dl/learning-rate-warmup/p01-warmup-then-cosine-lr` | sizes and rates | `breast_cancer`, `diabetes`, `digits`, `iris`, `wine` |
| `dl/learning-rate-warmup/p02-inverse-sqrt-lr` | sizes and rates | `breast_cancer`, `diabetes`, `digits`, `iris`, `wine` |
| `dl/mixed-precision/p01-dynamic-loss-scaler` | data | `online_retail` |
| `dl/multi-head-attention/p01-self-attention-heads` | data | `breast_cancer` |
| `dl/one-training-step/p01-manual-sgd-train-step` | data | `diabetes` |
| `dl/position-wise-feed-forward-block/p01-gelu-ffn` | data | `breast_cancer` |
| `dl/position-wise-feed-forward-block/p02-swiglu-ffn` | data | `breast_cancer` |
| `dl/positional-encodings/p01-sinusoidal-table` | sizes and rates | `gsm8k` |
| `dl/positional-encodings/p02-rope-rotations` | data | `breast_cancer` |
| `dl/relu-family-activations/p01-relu-forward-and-gradient` | data | `breast_cancer` |
| `dl/residual-connections/p01-residual-block` | data | `breast_cancer` |
| `dl/reverse-mode-autograd-engine/p01-scalar-autograd` | data | `wine` |
| `dl/rmsnorm/p01-rmsnorm-forward` | data | `breast_cancer` |
| `dl/scaled-dot-product-attention/p01-scaled-dot-product-attention` | data | `breast_cancer` |
| `dl/sgd-with-momentum/p01-momentum-step` | data | `breast_cancer` |
| `dl/sgd-with-momentum/p02-nesterov-weight-decay` | data | `breast_cancer` |
| `dl/smooth-activations-tanh-gelu-swish/p01-gelu-and-swish` | data | `breast_cancer` |
| `dl/smooth-activations-tanh-gelu-swish/p02-smooth-activation-derivatives` | data | `breast_cancer` |
| `dl/tensors-in-pytorch-and-tinygrad/p01-batched-pairwise-distances` | data | `breast_cancer` |
| `dl/tensors-in-pytorch-and-tinygrad/p02-patchify-images-row-major` | data | `digits` |
| `dl/training-an-mlp-from-scratch/p01-train-two-layer-mlp` | data | `diabetes` |
| `dl/transformer-block/p01-pre-norm-causal-block` | data | `breast_cancer` |
| `dl/xavier-and-he-initialization/p01-initialize-dense-weights` | sizes and rates | `breast_cancer`, `diabetes`, `digits`, `iris`, `wine` |
| `inf/absmax-and-zero-point-int8-quantization/p01-absmax-quantize` | data | `breast_cancer` |
| `inf/absmax-and-zero-point-int8-quantization/p02-zero-point-quantize` | data | `breast_cancer` |
| `inf/arithmetic-intensity-and-the-roofline/p01-matmul-roofline` | sizes and rates | `breast_cancer`, `california_housing`, `diabetes`, `digits`, `iris`, `wine` |
| `inf/continuous-batching/p01-batch-scheduler-steps` | data | `gsm8k`, `online_retail` |
| `inf/cost-per-token-and-break-even/p01-cost-per-million-and-break-even` | sizes and rates | `online_retail` |
| `inf/data-quality-and-drift-detection/p01-population-stability-index` | data | `california_housing` |
| `inf/data-quality-and-drift-detection/p02-ks-drift-test` | data | `california_housing` |
| `inf/floating-point-formats/p01-decode-a-bit-pattern` | data | `adult`, `california_housing`, `co2`, `gutenberg_walden` |
| `inf/floating-point-formats/p02-round-to-nearest-even` | data | `adult`, `california_housing`, `gutenberg_walden` |
| `inf/gpu-memory-and-capacity-sizing/p01-min-gpu-count` | sizes and rates | `gutenberg_frankenstein`, `gutenberg_pride_prejudice`, `gutenberg_walden` |
| `inf/kv-cache-size-arithmetic/p01-kv-cache-bytes` | sizes and rates | `gutenberg_frankenstein`, `gutenberg_pride_prejudice`, `gutenberg_walden` |
| `inf/kv-cache-size-arithmetic/p02-max-batch-after-weights` | sizes and rates | `gutenberg_frankenstein`, `gutenberg_pride_prejudice`, `gutenberg_walden` |
| `inf/paged-kv-cache/p01-block-allocator` | sizes and rates | `gsm8k` |
| `inf/prefill-vs-decode-bottlenecks/p01-serving-step-times` | sizes and rates | `gutenberg_frankenstein`, `gutenberg_pride_prejudice`, `gutenberg_walden` |
| `inf/prefill-vs-decode-bottlenecks/p02-compute-bound-decode-batch` | sizes and rates | `gutenberg_frankenstein`, `gutenberg_pride_prejudice`, `gutenberg_walden` |
| `inf/speculative-decoding-speedup-math/p01-speculative-speedup` | sizes and rates | `gutenberg_pride_prejudice`, `gutenberg_walden` |
| `inf/speculative-decoding-speedup-math/p02-accept-or-resample` | data | `gutenberg_pride_prejudice`, `gutenberg_walden` |
| `inf/ttft-itl-and-throughput/p01-latency-percentiles` | data | `online_retail` |
| `inf/ttft-itl-and-throughput/p02-throughput-and-goodput` | data | `online_retail` |
| `llm/autoregressive-generation-loop/p01-greedy-generation-with-eos` | data | `gutenberg_frankenstein`, `gutenberg_pride_prejudice`, `gutenberg_walden` |
| `llm/bpe-encode-and-decode/p01-apply-bpe-merges` | data | `gutenberg_frankenstein` |
| `llm/bpe-merge-training/p01-learn-bpe-merges` | data | `gutenberg_frankenstein` |
| `llm/bradley-terry-reward-models/p01-reward-pair-loss-and-accuracy` | data | `breast_cancer` |
| `llm/chat-templates-and-tool-call-formats/p01-render-chat-template` | data | `gsm8k` |
| `llm/chat-templates-and-tool-call-formats/p02-parse-tool-call` | data | `gsm8k`, `online_retail` |
| `llm/direct-preference-optimization-dpo/p01-sequence-log-probs-with-response-mask` | data | `breast_cancer` |
| `llm/direct-preference-optimization-dpo/p02-dpo-loss-and-implicit-rewards` | data | `gutenberg_frankenstein`, `gutenberg_pride_prejudice`, `gutenberg_walden` |
| `llm/document-packing-and-cross-document-masks/p01-packed-causal-mask` | data | `gutenberg_walden` |
| `llm/gpt-forward-pass/p01-decoder-only-gpt-forward` | data | `breast_cancer`, `gutenberg_frankenstein`, `gutenberg_pride_prejudice`, `gutenberg_walden` |
| `llm/gqa-mqa-and-cross-layer-kv-sharing/p01-grouped-query-attention` | data | `breast_cancer` |
| `llm/grpo-and-its-variants/p01-group-normalised-advantages` | data | `breast_cancer` |
| `llm/grpo-and-its-variants/p02-clipped-grpo-token-loss` | data | `breast_cancer` |
| `llm/incremental-decoding-with-a-kv-cache/p01-cached-causal-attention-step` | data | `breast_cancer` |
| `llm/instruction-tuning-batches-and-loss-masks/p01-sft-batch-with-loss-masks` | data | `breast_cancer`, `gsm8k` |
| `llm/llm-as-judge-and-agreement-with-humans/p01-cohens-kappa` | data | `breast_cancer`, `iris` |
| `llm/llm-as-judge-and-agreement-with-humans/p02-debiased-win-rate` | data | `mt_bench_judgments` |
| `llm/lora-adapters/p01-lora-linear-forward-and-merge` | data | `breast_cancer` |
| `llm/lora-parameter-and-memory-arithmetic/p01-lora-trainable-parameter-budget` | sizes and rates | `breast_cancer`, `diabetes`, `digits`, `iris`, `wine` |
| `llm/lora-parameter-and-memory-arithmetic/p02-finetune-memory-in-bytes` | sizes and rates | `breast_cancer`, `diabetes`, `digits`, `iris`, `wine` |
| `llm/model-parameter-and-memory-footprint/p01-count-decoder-parameters` | sizes and rates | `gutenberg_frankenstein` |
| `llm/model-parameter-and-memory-footprint/p02-training-and-inference-memory` | sizes and rates | `digits` |
| `llm/moe-top-k-routing/p01-top-k-router-gates` | data | `breast_cancer` |
| `llm/moe-top-k-routing/p02-load-balancing-loss` | data | `breast_cancer` |
| `llm/negative-log-likelihood-and-perplexity/p01-masked-nll-and-perplexity` | data | `breast_cancer`, `gutenberg_frankenstein`, `gutenberg_pride_prejudice`, `gutenberg_walden` |
| `llm/next-token-targets-and-batches/p01-pack-next-token-blocks` | data | `gutenberg_frankenstein`, `gutenberg_pride_prejudice`, `gutenberg_walden` |
| `llm/padding-truncation-and-pad-masks/p01-pad-and-truncate-token-ids` | data | `gutenberg_frankenstein`, `gutenberg_pride_prejudice`, `gutenberg_walden` |
| `llm/pass-k-and-majority-voting/p01-unbiased-pass-at-k` | sizes and rates | `wine` |
| `llm/pass-k-and-majority-voting/p02-majority-vote` | data | `gsm8k` |
| `llm/retrieval-augmented-generation-pipeline/p01-chunk-words` | data | `gutenberg_walden` |
| `llm/retrieval-augmented-generation-pipeline/p02-cited-context-prompt` | data | `breast_cancer`, `gutenberg_pride_prejudice` |
| `llm/rotary-position-embeddings-rope/p01-half-split-rope-with-offset` | data | `breast_cancer` |
| `llm/rotary-position-embeddings-rope/p02-linear-position-interpolation` | data | `breast_cancer` |
| `llm/scaling-laws-and-compute-budgets/p01-compute-optimal-allocation` | sizes and rates | `gutenberg_frankenstein`, `gutenberg_pride_prejudice`, `gutenberg_walden` |
| `llm/scaling-laws-and-compute-budgets/p02-fit-power-law-log-log` | data | `gutenberg_frankenstein`, `gutenberg_pride_prejudice`, `gutenberg_walden` |
| `llm/temperature-top-k-and-top-p-sampling/p01-filtered-next-token-distribution` | data | `breast_cancer` |
| `llm/temperature-top-k-and-top-p-sampling/p02-sample-with-a-generator` | data | `gutenberg_frankenstein` |
| `llm/word-and-character-tokenizers/p01-word-level-vocabulary` | data | `gutenberg_frankenstein`, `gutenberg_pride_prejudice` |
| `math/determinants-and-trace/p01-determinant` | data | `breast_cancer`, `wine` |
| `math/inverse-transform-sampling/p01-sample-exponential` | sizes and rates | `online_retail` |
| `math/inverse-transform-sampling/p02-sample-discrete` | data | `online_retail` |
| `math/monte-carlo-simulation/p01-mc-estimate` | data | `california_housing`, `iris`, `wine` |
| `math/reservoir-sampling/p01-reservoir-sample` | data | `gutenberg_frankenstein` |
| `rl/action-value-estimates/p01-incremental-action-values` | data | `online_retail` |
| `rl/epsilon-greedy-exploration/p01-epsilon-greedy-action` | data | `digits`, `wine` |
| `rl/mdps-and-discounted-returns/p01-discounted-returns` | data | `elnino`, `nile` |
| `rl/mdps-and-discounted-returns/p02-iterative-policy-evaluation` | data | `breast_cancer`, `elnino` |
| `rl/q-learning/p01-q-learning-gridworld` | data | `wine` |
| `rl/reinforce/p01-reinforce-loss` | data | `breast_cancer` |
| `rl/sarsa-and-expected-sarsa/p01-sarsa-and-expected-sarsa-updates` | data | `breast_cancer`, `digits` |
| `safe/split-conformal-prediction/p01-split-conformal-intervals` | data | `california_housing` |
| `safe/split-conformal-prediction/p02-conformal-prediction-sets` | data | `breast_cancer`, `digits` |
| `sql/pandas-groupby-aggregate-and-transform/p01-customer-order-features` | data | `online_retail` |
| `sql/pandas-merge-and-reshape/p01-monthly-revenue-by-region` | data | `online_retail` |
| `sql/sql-cohort-retention/p01-cohort-retention-by-month` | data | `online_retail` |
| `sql/sql-conditional-aggregation/p01-channel-conversion-and-device-revenue` | data | `breast_cancer`, `online_retail` |
| `sql/sql-date-handling/p01-monthly-active-users` | data | `online_retail` |
| `sql/sql-date-handling/p02-days-from-first-to-second-event` | data | `online_retail` |
| `sql/sql-first-and-last-activity-per-user/p01-first-and-last-event` | data | `online_retail` |
| `sql/sql-group-by-and-having/p01-customers-with-three-completed-orders` | data | `breast_cancer`, `online_retail` |
| `sql/sql-joins-and-unions/p01-customer-order-totals` | data | `online_retail` |
| `sql/sql-joins-and-unions/p02-warehouses-with-a-shipment-or-a-return` | data | `online_retail` |
| `sql/sql-lag-and-lead/p01-day-over-day-change` | data | `online_retail` |
| `sql/sql-lag-and-lead/p02-sessions-by-inactivity-gap` | data | `online_retail` |
| `sql/sql-ranking-window-functions/p01-rank-within-category` | data | `breast_cancer`, `online_retail` |
| `sql/sql-ranking-window-functions/p02-top-two-with-ties` | data | `breast_cancer`, `online_retail` |
| `sql/sql-running-totals-and-frames/p01-running-balance-with-ties` | data | `online_retail` |
| `sql/sql-running-totals-and-frames/p02-trailing-seven-day-average` | data | `online_retail` |
| `sql/sql-select-filter-and-sort/p01-top-five-completed-orders-2024` | data | `online_retail` |
| `sql/sql-streaks-gaps-and-islands/p01-longest-streak-per-user` | data | `online_retail` |
| `sql/sql-streaks-gaps-and-islands/p02-all-islands-per-user` | data | `online_retail` |
| `sql/sql-subqueries-and-ctes/p01-customers-above-average-spend` | data | `online_retail` |
| `sql/sql-subqueries-and-ctes/p02-orders-above-their-customers-average` | data | `online_retail` |
| `sql/sql-top-n-and-latest-row-per-group/p01-latest-events` | data | `online_retail` |

## The 30 marked "sizes and rates"

- **Index generators** (`kfold-indices`, `walk-forward-splits`, `bootstrap-oob-masks`, `minibatch-index-schedule`): the number of rows is that of a real dataset or series (iris 150, wine 178, diabetes 442, breast_cancer 569, digits 1,797, the Nile 100).
- **Schedules and tables** (four learning-rate schedules, `sinusoidal-table`): a run lasts one epoch over a real dataset at batch sizes 16, 32 and 64; a table is as long as a real GSM8K question.
- **Network sizes** (`count-dense-parameters`, `initialize-dense-weights`, both LoRA problems, `count-decoder-parameters`, `training-and-inference-memory`): widths are the features and classes of real datasets; a vocabulary and a context are the distinct words and the longest line of a real passage; an activation term is the bytes of the digits images.
- **Serving arithmetic** (`kv-cache-bytes`, `max-batch-after-weights`, `min-gpu-count`, both prefill problems, `matmul-roofline`, `block-allocator`, `compute-optimal-allocation`): sequence, prompt and context lengths are the word counts of real passages of three novels; matrix shapes are real design matrices; request sizes are GSM8K sentence and step counts; a compute budget pays for one pass over a real corpus.
- **Rates and prices** (`sample-size-two-proportions`, `binomial-pmf`, `poisson-pmf`, `sample-exponential`, `speculative-speedup`, `cost-per-million-and-break-even`, `unbiased-pass-at-k`): rates counted from real tables (cancelled invoices, malignant tumours, improved grades, invoices per trading day), acceptance rates between the next-word distributions of two novels, real prices in pounds, and counts of right calls by a simple rule on real wines.

In these problems the Hypothesis properties over the whole parameter domain and the hand-worked edge cases are unchanged.

The functions of these problems, as the learner implements them. None has an argument that could hold dataset values:

| Problem | Function |
|---|---|
| `cml/a-b-test-design-and-analysis/p01-sample-size-two-proportions` | `sample_size_two_proportions(p_baseline, mde, alpha, power)` |
| `cml/bernoulli-binomial-and-poisson/p01-binomial-pmf` | `binomial_pmf(k, n, p)` |
| `cml/bernoulli-binomial-and-poisson/p02-poisson-pmf` | `poisson_pmf(k, lam)` |
| `cml/k-fold-cross-validation/p01-kfold-indices` | `kfold_indices(n, k, shuffle, seed)` |
| `cml/leakage-and-walk-forward-validation/p01-walk-forward-splits` | `walk_forward_splits(n, initial_train, horizon, step, gap, expanding)` |
| `cml/random-forests-and-feature-importance/p02-bootstrap-oob-masks` | `bootstrap_oob(n, n_trees, seed)` |
| `cml/shuffling-splitting-and-batching-data/p02-minibatch-index-schedule` | `minibatch_indices(n, batch_size, epochs, seed, drop_last)` |
| `dl/counting-network-parameters/p01-count-dense-parameters` | `count_dense_parameters(widths, use_bias)` |
| `dl/learning-rate-decay-schedules/p01-cosine-annealing-lr` | `cosine_annealing_lr(step, base_lr, t_max, eta_min)` |
| `dl/learning-rate-decay-schedules/p02-step-decay-lr` | `step_decay_lr(step, base_lr, step_size, gamma)` |
| `dl/learning-rate-warmup/p01-warmup-then-cosine-lr` | `warmup_cosine_lr(step, base_lr, warmup_steps, total_steps, min_lr)` |
| `dl/learning-rate-warmup/p02-inverse-sqrt-lr` | `inverse_sqrt_lr(step, d_model, warmup_steps)` |
| `dl/positional-encodings/p01-sinusoidal-table` | `sinusoidal_encoding(max_len, d_model)` |
| `dl/xavier-and-he-initialization/p01-initialize-dense-weights` | `initialize_dense(fan_in, fan_out, scheme, seed)` |
| `inf/arithmetic-intensity-and-the-roofline/p01-matmul-roofline` | `matmul_roofline(m, n, k, element_bytes, peak_flops, bandwidth)` |
| `inf/cost-per-token-and-break-even/p01-cost-per-million-and-break-even` | `cost_and_break_even(gpu_hour_price, gpus, tokens_per_second, utilization_percent, api_price_per_million)` |
| `inf/gpu-memory-and-capacity-sizing/p01-min-gpu-count` | `min_gpu_count(params, config, batch, context_len, element_bytes, gpu_memory_bytes, workspace_bytes, reserve_percent)` |
| `inf/kv-cache-size-arithmetic/p01-kv-cache-bytes` | `kv_cache_bytes(config, seq_len, batch, dtype)` |
| `inf/kv-cache-size-arithmetic/p02-max-batch-after-weights` | `max_batch_size(config, seq_len, dtype, weight_bytes, memory_bytes, reserve_bytes)` |
| `inf/paged-kv-cache/p01-block-allocator` | `class BlockAllocator(num_blocks, block_size) with num_free_blocks, allocate, append_token, num_tokens, block_table, free` |
| `inf/prefill-vs-decode-bottlenecks/p01-serving-step-times` | `serving_step_times(params, config, prompt_len, context_len, batch, element_bytes, peak_flops, bandwidth)` |
| `inf/prefill-vs-decode-bottlenecks/p02-compute-bound-decode-batch` | `compute_bound_decode_batch(params, config, context_len, element_bytes, peak_flops, bandwidth)` |
| `inf/speculative-decoding-speedup-math/p01-speculative-speedup` | `speculative_speedup(alpha, gamma, c)` |
| `llm/lora-parameter-and-memory-arithmetic/p01-lora-trainable-parameter-budget` | `lora_budget(n_layers, d_model, d_ff, rank, targets, base_params)` |
| `llm/lora-parameter-and-memory-arithmetic/p02-finetune-memory-in-bytes` | `finetune_memory_bytes(base_params, base_bits, lora_params, adapter_bits)` |
| `llm/model-parameter-and-memory-footprint/p01-count-decoder-parameters` | `count_parameters(config)` |
| `llm/model-parameter-and-memory-footprint/p02-training-and-inference-memory` | `estimate_memory(num_params, mode, dtype, optimizer, master_weights, activation_bytes)` |
| `llm/pass-k-and-majority-voting/p01-unbiased-pass-at-k` | `pass_at_k(n, c, k)` |
| `llm/scaling-laws-and-compute-budgets/p01-compute-optimal-allocation` | `compute_optimal_allocation(flop_budget, tokens_per_param)` |
| `math/inverse-transform-sampling/p01-sample-exponential` | `sample_exponential(rate, n, rng)` |

## Array draws that are still generated (10 problems)

Found by scanning every hidden test for calls that draw an array from a random distribution. Not listed: scalars, Hypothesis strategies, and draws that only pick positions of real items (which real row, word, image or invoice comes next, which cells of a real grid a query covers, which class a resampled real tumour is taken from), and one call that only advances the global random stream.

**Expected outputs: the function's own seeded draws, replayed by the oracle (7).**

| Problem | Lines | What it is |
|---|---|---|
| `cml/bootstrap-and-permutation-resampling/p01-bootstrap-percentile-ci` | 1 | Seeded resampling indices. |
| `cml/random-forests-and-feature-importance/p02-bootstrap-oob-masks` | 1 | Seeded bootstrap draws. |
| `dl/dropout/p01-inverted-dropout` | 2 | The seeded dropout mask. |
| `dl/training-an-mlp-from-scratch/p01-train-two-layer-mlp` | 2 | The seeded initial weights of the network being trained. |
| `dl/xavier-and-he-initialization/p01-initialize-dense-weights` | 5 | The seeded initial weights. |
| `math/inverse-transform-sampling/p01-sample-exponential` | 1 | The uniform draws behind the samples. |
| `math/inverse-transform-sampling/p02-sample-discrete` | 2 | The uniform draws behind the samples. |

**The test's own Monte Carlo check; the draw is never passed to the function (2).**

| Problem | Lines | What it is |
|---|---|---|
| `cml/a-b-test-design-and-analysis/p01-sample-size-two-proportions` | 2 | Binomial counts of a power simulation at a known true rate. The function receives only the two rates. |
| `dl/kl-divergence/p02-diagonal-gaussian-kl` | 1 | Normal samples for a Monte Carlo check of the closed form. The function receives only the real means and standard deviations. |

**Passed to the function (1).**

| Problem | Lines | What it is |
|---|---|---|
| `math/monte-carlo-simulation/p01-mc-estimate` | 1 | One test checks a closed form, E[exp(Z)] = e^(1/2), so its sampler draws a standard normal. The test is named after that closed form. |
