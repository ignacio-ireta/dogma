export type TopicScene =
  | "study"
  | "vectors"
  | "matrices"
  | "calculus"
  | "optimization"
  | "probability"
  | "regression"
  | "classification"
  | "data"
  | "trees"
  | "clusters"
  | "series"
  | "neural"
  | "attention"
  | "tokens"
  | "retrieval"
  | "tables"
  | "algorithms"
  | "reinforcement"
  | "systems"
  | "safety";

const sceneByTopicId: Readonly<Record<string, TopicScene>> = {
  // Classical machine learning: foundations, statistics, and model families.
  "cml/vector-operations": "vectors",
  "cml/matrix-shapes-and-transpose": "matrices",
  "cml/derivatives": "calculus",
  "cml/descriptive-statistics": "probability",
  "cml/shuffling-splitting-and-batching-data": "data",
  "cml/probability-basics": "probability",
  "cml/sigmoid-and-log-odds": "classification",
  "cml/confusion-matrix-and-count-based-scores": "classification",
  "cml/ranking-metrics": "classification",
  "cml/tf-idf-and-bm25": "retrieval",
  "cml/matrix-multiplication": "matrices",
  "cml/matrix-inverse-and-rank": "matrices",
  "cml/normal-equation": "regression",
  "cml/partial-derivatives-and-gradients": "calculus",
  "cml/chain-rule": "calculus",
  "cml/gradient-descent": "optimization",
  "cml/stochastic-and-mini-batch-gradient-descent": "optimization",
  "cml/feature-scaling": "data",
  "cml/preprocessing-pipelines": "data",
  "cml/k-fold-cross-validation": "data",
  "cml/missing-values-and-outliers": "data",
  "cml/leakage-and-walk-forward-validation": "series",
  "cml/regression-metrics-and-robust-losses": "regression",
  "cml/over-and-underfitting-early-stopping": "regression",
  "cml/vector-and-matrix-norms": "matrices",
  "cml/eigenvalues-and-eigenvectors": "matrices",
  "cml/lagrange-multipliers": "optimization",
  "cml/bias-variance-decomposition": "regression",
  "cml/normal-and-multivariate-gaussian": "probability",
  "cml/maximum-likelihood-and-map": "probability",
  "cml/lasso-and-elastic-net": "regression",
  "cml/ridge-regression-and-weight-decay": "regression",
  "cml/expectation-and-variance": "probability",
  "cml/sampling-distributions-standard-errors-and-cis": "probability",
  "cml/t-tests-and-anova": "probability",
  "cml/a-b-test-design-and-analysis": "probability",
  "cml/bootstrap-and-permutation-resampling": "probability",
  "cml/logistic-regression": "classification",
  "cml/covariance-and-correlation": "probability",
  "cml/bayes-theorem": "probability",
  "cml/bernoulli-binomial-and-poisson": "probability",
  "cml/softmax": "classification",
  "cml/cross-entropy": "classification",
  "cml/precision-recall-and-f1": "classification",
  "cml/roc-pr-curves-and-auc": "classification",
  "cml/handling-class-imbalance": "classification",
  "cml/nearest-neighbour-search-and-knn": "classification",
  "cml/entropy-and-mutual-information": "probability",
  "cml/choosing-a-tree-split": "trees",
  "cml/adaboost-and-gradient-boosting": "trees",
  "cml/xgboost-second-order-objective": "trees",
  "cml/random-forests-and-feature-importance": "trees",
  "cml/k-means": "clusters",
  "cml/gaussian-mixtures-and-em": "clusters",
  "cml/principal-component-analysis": "matrices",
  "cml/singular-value-decomposition": "matrices",
  "cml/cosine-similarity": "vectors",
  "cml/rolling-statistics-stationarity-and-autocorrelation": "series",

  // Deep learning: computational graphs, training, and attention architectures.
  "dl/relu-family-activations": "neural",
  "dl/dropout": "neural",
  "dl/positional-encodings": "attention",
  "dl/building-tensors-by-broadcasting": "matrices",
  "dl/dense-layer-forward-pass": "neural",
  "dl/smooth-activations-tanh-gelu-swish": "neural",
  "dl/counting-network-parameters": "neural",
  "dl/xavier-and-he-initialization": "neural",
  "dl/backprop-through-a-chain-of-scalars": "calculus",
  "dl/backprop-through-a-dense-layer": "calculus",
  "dl/reverse-mode-autograd-engine": "calculus",
  "dl/training-an-mlp-from-scratch": "neural",
  "dl/tensors-in-pytorch-and-tinygrad": "matrices",
  "dl/gradients-with-framework-autograd": "calculus",
  "dl/one-training-step": "neural",
  "dl/a-complete-training-loop": "neural",
  "dl/sgd-with-momentum": "optimization",
  "dl/adam": "optimization",
  "dl/adamw-decoupled-weight-decay": "optimization",
  "dl/scaled-dot-product-attention": "attention",
  "dl/learning-rate-decay-schedules": "optimization",
  "dl/learning-rate-warmup": "optimization",
  "dl/batch-normalization": "neural",
  "dl/layer-normalization": "neural",
  "dl/rmsnorm": "neural",
  "dl/residual-connections": "neural",
  "dl/mixed-precision": "systems",
  "dl/convolution-forward-pass": "neural",
  "dl/multi-head-attention": "attention",
  "dl/position-wise-feed-forward-block": "attention",
  "dl/transformer-block": "attention",
  "dl/causal-masking": "attention",
  "dl/kl-divergence": "probability",
  "dl/infonce-contrastive-loss": "neural",

  // Language models: token handling, model internals, retrieval, and evaluation.
  "llm/word-and-character-tokenizers": "tokens",
  "llm/scaling-laws-and-compute-budgets": "systems",
  "llm/bpe-merge-training": "tokens",
  "llm/bpe-encode-and-decode": "tokens",
  "llm/padding-truncation-and-pad-masks": "tokens",
  "llm/next-token-targets-and-batches": "tokens",
  "llm/document-packing-and-cross-document-masks": "data",
  "llm/negative-log-likelihood-and-perplexity": "probability",
  "llm/gpt-forward-pass": "attention",
  "llm/model-parameter-and-memory-footprint": "systems",
  "llm/autoregressive-generation-loop": "tokens",
  "llm/temperature-top-k-and-top-p-sampling": "probability",
  "llm/incremental-decoding-with-a-kv-cache": "systems",
  "llm/gqa-mqa-and-cross-layer-kv-sharing": "attention",
  "llm/moe-top-k-routing": "neural",
  "llm/chat-templates-and-tool-call-formats": "tokens",
  "llm/instruction-tuning-batches-and-loss-masks": "data",
  "llm/lora-adapters": "neural",
  "llm/lora-parameter-and-memory-arithmetic": "systems",
  "llm/bradley-terry-reward-models": "probability",
  "llm/direct-preference-optimization-dpo": "neural",
  "llm/pass-k-and-majority-voting": "classification",
  "llm/llm-as-judge-and-agreement-with-humans": "classification",
  "llm/retrieval-augmented-generation-pipeline": "retrieval",
  "llm/grpo-and-its-variants": "neural",
  "llm/rotary-position-embeddings-rope": "attention",

  // Reinforcement learning.
  "rl/action-value-estimates": "reinforcement",
  "rl/epsilon-greedy-exploration": "reinforcement",
  "rl/mdps-and-discounted-returns": "reinforcement",
  "rl/sarsa-and-expected-sarsa": "reinforcement",
  "rl/q-learning": "reinforcement",
  "rl/reinforce": "reinforcement",

  // Inference and model serving systems.
  "inf/ttft-itl-and-throughput": "systems",
  "inf/floating-point-formats": "systems",
  "inf/absmax-and-zero-point-int8-quantization": "systems",
  "inf/arithmetic-intensity-and-the-roofline": "systems",
  "inf/prefill-vs-decode-bottlenecks": "systems",
  "inf/kv-cache-size-arithmetic": "systems",
  "inf/gpu-memory-and-capacity-sizing": "systems",
  "inf/cost-per-token-and-break-even": "systems",
  "inf/paged-kv-cache": "systems",
  "inf/continuous-batching": "systems",
  "inf/speculative-decoding-speedup-math": "systems",
  "inf/data-quality-and-drift-detection": "data",

  // Safety and uncertainty.
  "safe/split-conformal-prediction": "safety",

  // SQL and pandas both transform tables, but retain their exact topic IDs.
  "sql/sql-select-filter-and-sort": "tables",
  "sql/pandas-groupby-aggregate-and-transform": "tables",
  "sql/sql-group-by-and-having": "tables",
  "sql/sql-joins-and-unions": "tables",
  "sql/sql-subqueries-and-ctes": "tables",
  "sql/sql-conditional-aggregation": "tables",
  "sql/sql-date-handling": "tables",
  "sql/sql-ranking-window-functions": "tables",
  "sql/sql-top-n-and-latest-row-per-group": "tables",
  "sql/sql-lag-and-lead": "tables",
  "sql/sql-running-totals-and-frames": "tables",
  "sql/sql-first-and-last-activity-per-user": "tables",
  "sql/sql-streaks-gaps-and-islands": "tables",
  "sql/sql-cohort-retention": "tables",
  "sql/pandas-merge-and-reshape": "tables",

  // Algorithms and data structures.
  "algo/hash-maps-and-sets": "algorithms",
  "algo/two-pointers-and-sliding-windows": "algorithms",
  "algo/binary-search": "algorithms",
  "algo/stacks-and-queues": "algorithms",
  "algo/heaps-and-priority-queues": "algorithms",
  "algo/binary-tree-recursion-and-traversal": "trees",
  "algo/sort-then-sweep": "algorithms",
  "algo/dynamic-programming": "algorithms",
  "algo/prefix-sums": "algorithms",
  "algo/lru-cache": "algorithms",
  "algo/bfs-and-dfs-on-graphs-and-grids": "trees",
  "algo/topological-sort": "trees",

  // Math drills.
  "math/reservoir-sampling": "probability",
  "math/determinants-and-trace": "matrices",
  "math/inverse-transform-sampling": "probability",
  "math/monte-carlo-simulation": "probability",

  // Local system-design case packages have no problem route, but can be Today topics.
  "sys/rass-case-framework": "systems",
  "sys/churn-prediction-case": "systems",
  "sys/fraud-detection-case": "systems",
  "sys/predictive-maintenance-case": "systems",
  "sys/recommendation-system-case": "systems",
  "sys/logistics-optimization-case": "systems",
  "sys/rag-system-design-case": "systems",
};

export function topicScene(topicId?: string | null): TopicScene {
  return topicId && Object.hasOwn(sceneByTopicId, topicId)
    ? (sceneByTopicId[topicId] ?? "study")
    : "study";
}
