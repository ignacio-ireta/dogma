# Test datasets (`dogma_data`)

`dogma_data` is the test-data layer for content. It holds real offline datasets, vendored
snapshots and seeded synthetic generators. Problems use it so that bulk random inputs have real
structure: scales, ties, skew, missing values, autocorrelation.

Design and decisions: [ADR-0020](decisions/0020-test-data-layer-and-hf-snapshots.md).
Test rules: [CONTENT_SPEC §6.6](CONTENT_SPEC.md).

## Scope

| Rule | Why |
|---|---|
| Test-only. Content tests import it; submissions may not. | Learners must not get the data or the generators for free. The static check bans `dogma_data`, `datasets` and `huggingface_hub` in every submission. |
| `import dogma_data` loads numpy only. | Other libraries load inside the functions that use them. |
| Never imports `dogma` or `dogma_harness`, and they never import it. | Import-linter contracts enforce this. |
| Tests read local files. Nothing reaches the network. | Runs are repeatable and offline. |

## How to load

```python
from dogma_data import available, fingerprint, load, load_frame, load_series, load_text, rng, synth
from dogma_data import retail_invoices, sample_rows, sample_window, sampleable

data = load("breast_cancer")  # RealData: X, y, feature_names, target_name, frame()
values = load_series("co2")  # 1-D float64 in time order; NaNs kept
table = load_frame("penguins")  # pandas DataFrame, as-is
book = load_text("gutenberg_walden")  # about 40 KB of real English prose
ids = available()  # every dataset id, sorted
digest = fingerprint("iris")  # sha256 of the CSV text, used by the unit tests
gen = rng("py-realism:cms:0")  # seeded numpy Generator from a string key
X, y = sample_rows("wine", gen, 12, 3)  # 12 real rows and 3 real columns, drawn by gen
run = sample_window("elnino", gen, 24)  # 24 consecutive real values of a series
orders = retail_invoices()  # online_retail, one row per invoice
```

| Function | Returns | Notes |
|---|---|---|
| `load(name)` | `RealData` (frozen dataclass) | Numeric tabular sets only. Raw and unscaled. |
| `load_series(name)` | `ndarray` | Series sets only. NaNs are kept. |
| `load_frame(name)` | pandas `DataFrame` | Any dataset. Vendored sets and HF tables are read without cleaning; a text is one row per line. |
| `load_text(name)` | `str` | The Gutenberg slices only. Each starts at the book's first sentence. |
| `fingerprint(name)` | `str` | Hash of `load_frame(name)`. Detects drift after a lock change. |
| `available()` | `tuple[str, ...]` | All ids. |
| `rng(key)` | `numpy.random.Generator` | Key is hashed with sha256. Keys are `<problem>:<purpose>:<i>`. |
| `sample_rows(name, gen, n, d=None, *, standardize=False)` | `(X, y)` | `n` distinct rows and `d` distinct columns of a numeric set (`sampleable()` lists them, `california_housing` included), drawn by `gen`. `d=None` keeps every column in dataset order. `standardize=True` scales with the whole dataset's mean and spread. `y` is the target of the drawn rows, with shape `(n, 0)` when the set has none. |
| `sample_window(name, gen, n, *, keep_nan=False)` | `ndarray` | `n` consecutive values of a series, starting at a position drawn by `gen`. NaNs are dropped first unless `keep_nan=True`. |
| `retail_invoices()` | pandas `DataFrame` | `online_retail` rolled up to one row per invoice: `invoice`, `customer_id`, `invoice_ts`, `country`, `n_lines`, `amount`, `cancelled`. |
| `retail_events()` | pandas `DataFrame` | `online_retail` as an activity log, one row per invoice line: `event_id`, `user_id`, `ts`, `label` (the description as typed), `code`, `country`, `quantity`, `amount`, `invoice`, `cancelled`. |

Load in module-scoped pytest fixtures, not at import. A collection error would otherwise look
like a hidden-test failure. `sample_rows` and `sample_window` read each dataset once per process
and may be called inside a test or a Hypothesis example.

### Real draws instead of random numbers

A test that needs many inputs of a given shape asks for a draw of real rows, not for
`rng.normal(size=(n, d))`:

| The test needs | Use | What the real values add |
|---|---|---|
| A feature matrix of any small shape | `sample_rows("wine" \| "breast_cancer" \| "diabetes" \| "digits", gen, n, d)` | Scales from hundredths to thousands, correlated columns, repeated values |
| Signed, comparable features | the same with `standardize=True` | Both signs, skew and outliers, without the scale gap |
| Features with no exact ties (distances, k-means, kNN) | the per-household averages of `california_housing` | Ratios of counts, heavy tails |
| Labels, groups, scores | the target that comes back with the rows; one feature as a score | Real class shares, real errors of a simple model |
| A sample of one variable | one column of `sample_rows(..., d=None)` | Skew, caps, outliers |
| A series | `sample_window("elnino" \| "sunspots" \| "nile" \| "co2", gen, n)` | Season, cycles, trend, repeated readings |
| Text, tokens, word counts | `load_text` (three novels), `load_frame("gsm8k")` | Zipf frequencies, punctuation, capitals, numerals |
| Orders, customers, activity logs | `retail_invoices()`, `retail_events()` | Refunds, duplicate lines, shared timestamps, one dominant country, quiet days, labels typed with stray spaces |
| A tensor inside a network (weights, gradients, logits, hidden states) | standardised `breast_cancer` rows reshaped to the tensor (the `_real_values` helper in the torch tests) | Skew, correlation along a row, a few entries far from the rest. A stand-in: no dataset holds such tensors |
| A probability vector, a prior, a likelihood table | counts from real rows: word shares in a novel, weekday shares of invoices, class-by-band tables | Unequal masses, exact zeros (no invoice on a Saturday), exact fractions |
| A language model or log-probabilities | bigram or unigram counts from the novels | Sparse rows, ties, a few dominant next words |
| Rewards, action values, a transition model | invoice amounts, series changes, pixel rows; transitions counted between bands of `elnino` | Refunds as negative rewards, ties, zero-probability transitions |
| Request sizes, arrival times, event flags | GSM8K sentence and step counts, invoice times and line counts, cancellation flags in order | Bursts, shared timestamps, long quiet stretches |
| A secondary array a test used to draw at random (normal, uniform, whole numbers, shares, a mask) | the helpers `_real_normal`, `_real_unit`, `_real_ints`, `_real_shares` in the hidden tests: standardised `breast_cancer` measurements, as they are or passed through the normal distribution function, banded or normalised | The order, ties and skew of real measurements. Derived values: the helper says how |
| A rate or a share for a formula | a rate counted from a real table (cancelled invoices, invoices per trading day) | A number that is not a round decimal |
| Verdicts of a pairwise judge | `load_frame("mt_bench_judgments")`: items judged with the two answers in both orders | Ties, disagreement between the two orders |
| A size, a length or a count (rows, widths, sequence lengths, steps per epoch) | the shape of a real dataset, the word count of a real passage or question | Sizes that are not round and not multiples of the batch or fold |

Rules that came out of converting the existing tests (I-33; the per-problem list is in
[audits/2026-11-11-real-datasets.md](audits/2026-11-11-real-datasets.md)):

- **Keep what the old data did by accident.** Random normals have both signs, full precision and no
  ties. Real columns are often all positive, rounded and tied. If a recorded wrong solution stops
  failing, pick the real column that restores the property (standardised values for signs, a
  ratio column for precision, a weak predictor rounded for mixed ties) before anything else.
- **Skip the draws the contract excludes, and say so.** A handful of real rows can share a value
  down a whole column, or a real stump can be perfect. Redraw in a loop and write the reason in
  the helper's docstring.
- **Real inputs, known truth.** Where a test needs a known answer (a sparse coefficient vector, a
  rule the model follows), draw the features from real rows and build the target from the known
  rule. Say "semi-synthetic" in the docstring.
- **Scalar parameters are not data.** Learning rates, regularisation strengths, discount
  factors, step sizes, policies and seeds stay chosen or random. Tensors a problem takes as
  input (weights, gradients, hidden states) are drawn from real values as a stand-in, and the
  helper says that it is one.
- **Keep generated values next to real ones where no measurement reaches.** Subnormals,
  infinities, NaN payloads and magnitudes near overflow never occur in a measurement, so the
  floating-point tests keep their generated words after the real ones. Say which is which.
- **A problem that takes only sizes, counts or rates reads them off real data.** The row count
  of a dataset, the features and classes of a network built for it, the updates in one epoch
  over it, the word count of a passage, a rate counted from a table. The values of the dataset
  never enter such a function, and the audit lists these problems as "sizes and rates".
- **What may still be generated:** the function's own seeded protocol when the oracle replays it
  (dropout masks, initialisers, inverse-transform draws, bootstrap indices), a simulation against
  a known truth (power, coverage, calibration, a closed-form Monte Carlo check), positions of real
  items, scalars, and a timing case. The audit lists every such draw and its reason.
- **Never read a dataset at import.** A parametrized case names the dataset or the rate
  (`"iris"`, `"invoices-per-day"`) and the test resolves the name in its body.
- **Gaps and dirt the source lacks may be placed by the draw,** on real values, when a contract
  rule needs them (missing measurements, a customer with no region, a status typed with a
  capital, a label in another letter case). The docstring says which part is real and which is
  placed. Use real dirt first: the `label` column of `retail_events()` has stray and doubled
  spaces as typed.
- **Set the property, keep the real values.** Where a property must be exact, set only that:
  real principal axes with a chosen spectrum, real columns mixed to a chosen rank, real clocks
  with gaps placed at exactly thirty minutes, real dates moved by a fixed number of days onto
  the calendar window a statement names. A 200,000-row timing case and a simulation with a
  known truth stay generated.
- **Real rows, the problem's own table.** A SQL problem keeps its schema. Map real retail rows
  onto it (an invoice line as a session, a posting, a view or a shipment) and say in the
  fixture which real column stands behind each column of the table.
- After any such edit: `exercise_gate.py gate <id> --mode refactor`. It replays every recorded
  wrong solution and every mutant on the old and the new tests.

`rng` mixes the `DOGMA_DATA_SALT` environment variable into the key. The grader never sets it.
`exercise_audit seeds` sets it to a new value for each run, so a test that passes for one lucky
draw shows up.

## Datasets

Ids are the values accepted by `load`, `load_series`, `load_frame` and `fingerprint`. The
licence column is the one recorded in `src/dogma_data/catalog.yaml`. Provenance for the vendored
CSVs is in `src/dogma_data/data/DATA_SOURCES.md`.

| Id | Source | Licence | Shape | Quirks that make it useful | Used by (topics) |
|---|---|---|---|---|---|
| `breast_cancer` | scikit-learn bundle | BSD-3 (UCI data) | 569 × 31 | Column means from about 0.004 to about 880. 21 feature pairs with \|corr\| > 0.9. Nearly separable. | logistic-regression, covariance-and-correlation, random-forests-and-feature-importance, roc-pr-curves-and-auc, bootstrap-and-permutation-resampling |
| `diabetes` | scikit-learn bundle | BSD-3 | 442 × 10 | Loaded raw (`scaled=False`). Binary `sex` coded 1/2. Continuous target. | a-complete-training-loop, adaboost-and-gradient-boosting, arithmetic-intensity-and-the-roofline, bias-variance-decomposition, counting-network-parameters, k-fold-cross-validation, lasso-and-elastic-net, learning-rate-decay-schedules, learning-rate-warmup, lora-parameter-and-memory-arithmetic, normal-equation, one-training-step, principal-component-analysis, ridge-regression-and-weight-decay, stochastic-and-mini-batch-gradient-descent, training-an-mlp-from-scratch, xavier-and-he-initialization |
| `digits` | scikit-learn bundle | BSD-3 (UCI data) | 1797 × 64 | 3 zero-variance pixels. 20 pixels with IQR = 0. Integer features 0 to 16. | arithmetic-intensity-and-the-roofline, backprop-through-a-dense-layer, bfs-and-dfs-on-graphs-and-grids, binary-search, building-tensors-by-broadcasting, convolution-forward-pass, cosine-similarity, counting-network-parameters, cross-entropy, dense-layer-forward-pass, epsilon-greedy-exploration, feature-scaling, infonce-contrastive-loss, learning-rate-decay-schedules, learning-rate-warmup, lora-parameter-and-memory-arithmetic, matrix-multiplication, matrix-shapes-and-transpose, model-parameter-and-memory-footprint, precision-recall-and-f1, prefix-sums, random-forests-and-feature-importance, sarsa-and-expected-sarsa, shuffling-splitting-and-batching-data, split-conformal-prediction, t-tests-and-anova, tensors-in-pytorch-and-tinygrad, vector-operations, xavier-and-he-initialization |
| `iris` | scikit-learn bundle | BSD-3 | 150 × 4 | One exact duplicate row. Values to one decimal, so many ties. Three balanced classes. | arithmetic-intensity-and-the-roofline, bayes-theorem, binary-search, bootstrap-and-permutation-resampling, choosing-a-tree-split, counting-network-parameters, expectation-and-variance, gaussian-mixtures-and-em, k-fold-cross-validation, kl-divergence, learning-rate-decay-schedules, learning-rate-warmup, llm-as-judge-and-agreement-with-humans, lora-parameter-and-memory-arithmetic, monte-carlo-simulation, random-forests-and-feature-importance, shuffling-splitting-and-batching-data, t-tests-and-anova, xavier-and-he-initialization |
| `linnerud` | scikit-learn bundle | BSD-3 | 20 × 3 (+ 3 targets) | Multi-output target (Weight, Waist, Pulse). Very small n. | no problem uses it at present |
| `wine` | scikit-learn bundle | BSD-3 (UCI data) | 178 × 13 | Proline (std about 314) holds about 99.8% of variance unscaled. | arithmetic-intensity-and-the-roofline, backprop-through-a-chain-of-scalars, bayes-theorem, binary-tree-recursion-and-traversal, counting-network-parameters, covariance-and-correlation, derivatives, determinants-and-trace, dropout, eigenvalues-and-eigenvectors, epsilon-greedy-exploration, feature-scaling, gaussian-mixtures-and-em, gradient-descent, k-fold-cross-validation, lagrange-multipliers, learning-rate-decay-schedules, learning-rate-warmup, lora-parameter-and-memory-arithmetic, matrix-inverse-and-rank, matrix-multiplication, matrix-shapes-and-transpose, maximum-likelihood-and-map, monte-carlo-simulation, normal-and-multivariate-gaussian, partial-derivatives-and-gradients, pass-k-and-majority-voting, principal-component-analysis, q-learning, random-forests-and-feature-importance, reverse-mode-autograd-engine, shuffling-splitting-and-batching-data, singular-value-decomposition, vector-and-matrix-norms, vector-operations, xavier-and-he-initialization |
| `longley` | statsmodels bundle | Public domain | 16 × 6 | Condition number about 4.9e9 with intercept. Regressors on very different scales. | normal-equation |
| `stackloss` | statsmodels bundle | Public domain | 21 × 3 | Observation 21 has the largest studentized residual. Fits are sensitive to it. | normal-equation |
| `engel` | statsmodels bundle | Public domain | 235 × 1 | Right-skewed income and food expenditure (skew about 2.8 and 1.7). Uneven spread. | descriptive-statistics |
| `macrodata` | statsmodels bundle | Public domain | 203 × 14 | Quarterly, trending macro series. No target: every column is a feature. | no problem uses it at present |
| `spector` | statsmodels bundle | Used with the author's permission (see `NOTICE`) | 32 × 3 | Binary `GRADE`. Very small logit set. | a-b-test-design-and-analysis, logistic-regression |
| `co2` (series) | statsmodels bundle | Public domain | 2284 × 1 | Weekly from 1958 with a trend and yearly season. 59 NaN kept. | floating-point-formats, rolling-statistics-stationarity-and-autocorrelation, two-pointers-and-sliding-windows |
| `elnino` (series) | statsmodels bundle | Public domain | 732 × 1 | 61 years × 12 months flattened. Lag-12 autocorrelation about 0.76. No NaN. | heaps-and-priority-queues, mdps-and-discounted-returns, prefix-sums, rolling-statistics-stationarity-and-autocorrelation, stacks-and-queues, two-pointers-and-sliding-windows |
| `nile` (series) | statsmodels bundle | Public domain | 100 × 1 | Level change near 1898: mean about 1098 before, about 850 after. | descriptive-statistics, leakage-and-walk-forward-validation, mdps-and-discounted-returns, rolling-statistics-stationarity-and-autocorrelation |
| `sunspots` (series) | statsmodels bundle | Public domain | 309 × 1 | About an 11-year cycle. ACF about 0.66 at lag 10. No NaN. | descriptive-statistics, heaps-and-priority-queues, rolling-statistics-stationarity-and-autocorrelation, stacks-and-queues |
| `penguins` | Palmer Penguins, vendored | CC0 1.0 | 344 × 7 | Two NaN in each measurement, 11 NaN in `sex`. String categoricals. | missing-values-and-outliers |
| `online_retail` | Online Retail II (UCI), HF snapshot | CC BY 4.0 | 8768 × 8 | Every invoice line of the 60 customers whose id ends in 77, December 2009 to December 2011. 568 invoices, 112 of them cancellations with negative quantities. 226 exact duplicate lines. The lines of an invoice share its minute. 83% of lines from one country. No sales on Saturdays. Descriptions as typed: stray and doubled spaces, a few not in capitals, 7 products under spellings that differ only in spaces. | a-b-test-design-and-analysis, action-value-estimates, bernoulli-binomial-and-poisson, binary-search, chat-templates-and-tool-call-formats, continuous-batching, cost-per-token-and-break-even, dynamic-programming, heaps-and-priority-queues, inverse-transform-sampling, lru-cache, mixed-precision, pandas-groupby-aggregate-and-transform, pandas-merge-and-reshape, sort-then-sweep, sql-cohort-retention, sql-conditional-aggregation, sql-date-handling, sql-first-and-last-activity-per-user, sql-group-by-and-having, sql-joins-and-unions, sql-lag-and-lead, sql-ranking-window-functions, sql-running-totals-and-frames, sql-select-filter-and-sort, sql-streaks-gaps-and-islands, sql-subqueries-and-ctes, sql-top-n-and-latest-row-per-group, topological-sort, ttft-itl-and-throughput |
| `mt_bench_judgments` | MT-Bench human judgments (LMSYS), HF snapshot | CC BY 4.0 | 3355 × 5 | One expert verdict per row on a question, a turn and an ordered pair of six chat models: `model_a`, `model_b` or `tie` (23% ties). 1,814 distinct items, 582 of them judged in both orders, by different experts who do not always agree. | llm-as-judge-and-agreement-with-humans |
| `customers` | interview-prep, vendored, synthetic | Part of this repository | 5000 × 11 | 50 duplicate `customer_id`. Impossible ages and NaN ages. Casing noise. Malformed dates. Lognormal income with NaN. | no problem uses it at present |
| `orders` | interview-prep, vendored, synthetic | Part of this repository | 15000 × 5 | Lognormal amounts with 300 NaN (returns). `customer_id` drawn from `customers`. String dates. | no problem uses it at present |
| `surveys` | interview-prep, vendored, synthetic | Part of this repository | 3000 × 4 | NaN in `satisfaction_score` and `nps_score`. Customers sampled without replacement. | no problem uses it at present |

Statsmodels sets load through `statsmodels.datasets.<name>`. `statsmodels.api` is never imported
because it is slow.

## Synthetic generators (`dogma_data.synth`)

Every generator takes the `rng` from `dogma_data.rng` first. The knobs after `*` are
keyword-only. Code is pure numpy with no BLAS calls, so output is bit-identical across BLAS
builds. Fingerprint tests pin the output for fixed keys.

| Generator | Knobs (defaults) | Realism it adds |
|---|---|---|
| `correlated_mixed_scale(rng, n, d)` | `n_factors=2`, `strength=3.0`, `scale_range=(0.01, 1000)`, `offset_range=(-10, 10)` | Correlated features (collinearity, PCA structure) on scales spanning orders of magnitude. Breaks unscaled gradient descent and distance metrics. |
| `heavy_tailed_noise(rng, n)` | `df=3.0`, `scale=1.0`, `spread=None` | Student-t draws with heavy tails and optional per-row spread. Breaks mean-based estimates and the normal bootstrap. |
| `imbalanced_labels(rng, n)` | `positive_rate=0.01`, `flip_rate=0.0`, `min_positives=1` | Rare positive class with optional label noise. Breaks accuracy as a score. |
| `informative_scores(rng, y)` | `auc=0.8`, `decimals=None` | Scores with a known ROC AUC (binormal model, 0.7 to 0.95 is realistic). `decimals` rounds them and creates ties. |
| `seasonal_ar_series(rng, n)` | `period=12`, `amplitude=1.0`, `trend=0.0`, `phi=(0.6,)`, `sigma=1.0`, `level_shifts=0`, `shift_size=3.0`, `missing_rate=0.0`, `start=0.0` | Trend, season, AR(p) memory, level shifts and NaN gaps. Breaks white-noise shortcuts in ACF and MASE. |
| `event_log(rng, n_events)` | `n_users=50`, `zipf_a=1.3`, `start`, `days=90`, `amount_median=30`, `amount_sigma=1.0`, `categories`, `dirty_rate=0.0`, `duplicate_rate=0.0` | Zipf-skewed users, lognormal amounts, naive timestamps, dirty categorical keys and exact duplicates. Returns a dict of columns. |
| `latencies(rng, n)` | `median_ms=50`, `sigma=0.5`, `tail_fraction=0.02`, `tail_multiplier=20` | Lognormal body with a slow tail. Makes p99 differ from the mean. |
| `outlier_channel_activations(rng, n_tokens, n_channels)` | `n_outlier_channels=2`, `outlier_scale=50.0` | Normal activations with a few channels scaled up. Breaks per-tensor quantization. |
| `peaked_probs(rng, y, n_classes)` | `accuracy=0.85`, `temperature=0.5`, `overconfidence=1.5`, `margin_scale=2.0` | Probabilities that are accurate but over-confident. Confidence is above accuracy, the case where calibration and conformal sets matter. |

## HuggingFace snapshots

Some problems need text or labelled data that is not bundled. Those sources are HuggingFace
datasets, and they enter the repo as snapshots:

- Each snapshot is a text file (CSV, JSONL or TXT) under `src/dogma_data/data/hf/`. Tests read it
  with the stdlib or pandas.
- `scripts/sync_hf_datasets.py` produces the snapshots. It reads the manifest (pinned revision
  sha, licence, selected rows and columns), writes the text files, and runs the privacy audit
  over them. Run it with `uv run --frozen --isolated --group hf python -I scripts/sync_hf_datasets.py`.
  The `hf` dependency group holds `datasets`, which is opt-in. `--isolated` keeps its pyarrow
  out of the shared `.venv`.
- `--check` re-verifies every file's hash against the manifest, offline.
- Licence policy: CC0, public domain, MIT, Apache-2.0 or BSD are accepted. CC-BY is accepted with
  an entry in `src/dogma_data/data/NOTICE`. NC, ND, SA, unknown and missing licences are refused.
- `trust_remote_code` is always off.
- Tests read the vendored files through `load_frame` and `load_text`. Nothing in `tests/` or
  `content/` downloads or calls the sync script.

| Id | Source (pinned revision in `hf_manifest.yaml`) | Licence | Selection | Use |
|---|---|---|---|---|
| `gsm8k` | `openai/gsm8k`, test split | MIT | first 300 rows, question and answer | pass@k, majority vote, answer extraction (`#### <number>`) |
| `adult` | `scikit-learn/adult-census-income` | CC0 | rows 8000-14000 (18.8% >50K) | categoricals, imbalance, `?` as missing, drift |
| `california_housing` | `gvlassis/california_housing` | MIT | first 6000 rows | target censored at 5.00001, heavy-tailed features |
| `online_retail` | `apurba-pal/online_retail` (Online Retail II, UCI, doi 10.24432/C5CG6D) | CC BY 4.0, with a notice entry | every line of the customers whose id ends in 77; every column except the description is unchanged, and the description is kept as typed | orders, refunds, activity logs, daily revenue with closed days |
| `mt_bench_judgments` | `lmsys/mt_bench_human_judgments`, human split | CC BY 4.0, with a notice entry | every row; the two conversation columns and the annotator id are left out | pairwise judge verdicts in both presentation orders |
| `gutenberg_frankenstein`, `gutenberg_pride_prejudice`, `gutenberg_walden` | `sedthh/gutenberg_english` (public-domain books) | MIT card | about 40 KB from each book's first sentence | BPE, tokenizers, TF-IDF/BM25, chunking, retrieval |

Attributions are in `src/dogma_data/data/hf/NOTICE`.

## When the lock changes

Generators and bundled datasets depend on numpy, scikit-learn and statsmodels. A lock bump can
change their output. The fingerprints then fail in `tests/unit/dogma_data/`.

1. Read the failing test. It names the dataset or generator.
2. Find the problems that use it: `grep -rl dogma_data content/`.
3. Re-check each affected problem. Confirm the contract still holds on the new data.
4. Bump the `version` in each affected `meta.yaml`. A `reviewed` topic goes back to `verified`
   until it is reviewed again.
5. Update the fingerprint only after step 3. Never update it to hide a drift.

Fingerprints protect lesson numbers, not correctness. A problem's correctness must not depend on
them.
