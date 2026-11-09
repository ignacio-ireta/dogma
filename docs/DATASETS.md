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

data = load("breast_cancer")  # RealData: X, y, feature_names, target_name, frame()
values = load_series("co2")  # 1-D float64 in time order; NaNs kept
table = load_frame("penguins")  # pandas DataFrame, as-is
book = load_text("gutenberg_walden")  # about 40 KB of real English prose
ids = available()  # every dataset id, sorted
digest = fingerprint("iris")  # sha256 of the CSV text, used by the unit tests
gen = rng("py-realism:cms:0")  # seeded numpy Generator from a string key
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

Load in module-scoped pytest fixtures, not at import. A collection error would otherwise look
like a hidden-test failure.

`rng` mixes the `DOGMA_DATA_SALT` environment variable into the key. The grader never sets it.
`exercise_audit seeds` sets it to a new value for each run, so a test that passes for one lucky
draw shows up.

## Datasets

Ids are the values accepted by `load`, `load_series`, `load_frame` and `fingerprint`. The
licence column is the one recorded in `src/dogma_data/catalog.yaml`. Provenance for the vendored
CSVs is in `src/dogma_data/data/DATA_SOURCES.md`.

| Id | Source | Licence | Shape | Quirks that make it useful | Fits |
|---|---|---|---|---|---|
| `breast_cancer` | scikit-learn bundle | BSD-3 (UCI data) | 569 × 31 | Column means from about 0.004 to about 880. 21 feature pairs with \|corr\| > 0.9. Nearly separable. | logistic-regression, covariance-and-correlation, random-forests-and-feature-importance, roc-pr-curves-and-auc, bootstrap-and-permutation-resampling |
| `diabetes` | scikit-learn bundle | BSD-3 | 442 × 10 | Loaded raw (`scaled=False`). Binary `sex` coded 1/2. Continuous target. | ridge-regression-and-weight-decay, lasso-and-elastic-net, random-forests-and-feature-importance |
| `digits` | scikit-learn bundle | BSD-3 (UCI data) | 1797 × 64 | 3 zero-variance pixels. 20 pixels with IQR = 0. Integer features 0 to 16. | principal-component-analysis, singular-value-decomposition, k-means |
| `iris` | scikit-learn bundle | BSD-3 | 150 × 4 | One exact duplicate row. Values to one decimal, so many ties. Three balanced classes. | k-means, nearest-neighbour-search-and-knn, t-tests-and-anova, feature-scaling |
| `linnerud` | scikit-learn bundle | BSD-3 | 20 × 3 (+ 3 targets) | Multi-output target (Weight, Waist, Pulse). Very small n. | ridge-regression-and-weight-decay, covariance-and-correlation |
| `wine` | scikit-learn bundle | BSD-3 (UCI data) | 178 × 13 | Proline (std about 314) holds about 99.8% of variance unscaled. | principal-component-analysis, feature-scaling, k-means |
| `longley` | statsmodels bundle | Public domain | 16 × 6 | Condition number about 4.9e9 with intercept. Regressors on very different scales. | normal-equation, ridge-regression-and-weight-decay, regression-metrics-and-robust-losses |
| `stackloss` | statsmodels bundle | Public domain | 21 × 3 | Observation 21 has the largest studentized residual. Fits are sensitive to it. | regression-metrics-and-robust-losses, missing-values-and-outliers |
| `engel` | statsmodels bundle | Public domain | 235 × 1 | Right-skewed income and food expenditure (skew about 2.8 and 1.7). Uneven spread. | bootstrap-and-permutation-resampling, regression-metrics-and-robust-losses, normal-equation |
| `macrodata` | statsmodels bundle | Public domain | 203 × 14 | Quarterly, trending macro series. No target: every column is a feature. | rolling-statistics-stationarity-and-autocorrelation, leakage-and-walk-forward-validation |
| `spector` | statsmodels bundle | Used with the author's permission (see `NOTICE`) | 32 × 3 | Binary `GRADE`. Very small logit set. | logistic-regression, handling-class-imbalance |
| `co2` (series) | statsmodels bundle | Public domain | 2284 × 1 | Weekly from 1958 with a trend and yearly season. 59 NaN kept. | rolling-statistics-stationarity-and-autocorrelation, missing-values-and-outliers, leakage-and-walk-forward-validation |
| `elnino` (series) | statsmodels bundle | Public domain | 732 × 1 | 61 years × 12 months flattened. Lag-12 autocorrelation about 0.76. No NaN. | rolling-statistics-stationarity-and-autocorrelation, leakage-and-walk-forward-validation |
| `nile` (series) | statsmodels bundle | Public domain | 100 × 1 | Level change near 1898: mean about 1098 before, about 850 after. | rolling-statistics-stationarity-and-autocorrelation, leakage-and-walk-forward-validation |
| `sunspots` (series) | statsmodels bundle | Public domain | 309 × 1 | About an 11-year cycle. ACF about 0.66 at lag 10. No NaN. | rolling-statistics-stationarity-and-autocorrelation, leakage-and-walk-forward-validation |
| `penguins` | Palmer Penguins, vendored | CC0 1.0 | 344 × 7 | Two NaN in each measurement, 11 NaN in `sex`. String categoricals. | preprocessing-pipelines, missing-values-and-outliers, gaussian-mixtures-and-em, choosing-a-tree-split |
| `customers` | interview-prep, vendored, synthetic | Part of this repository | 5000 × 11 | 50 duplicate `customer_id`. Impossible ages and NaN ages. Casing noise. Malformed dates. Lognormal income with NaN. | sql-select-filter-and-sort, sql-conditional-aggregation, handling-class-imbalance |
| `orders` | interview-prep, vendored, synthetic | Part of this repository | 15000 × 5 | Lognormal amounts with 300 NaN (returns). `customer_id` drawn from `customers`. String dates. | sql-group-by-and-having, pandas-merge-and-reshape, sql-lag-and-lead |
| `surveys` | interview-prep, vendored, synthetic | Part of this repository | 3000 × 4 | NaN in `satisfaction_score` and `nps_score`. Customers sampled without replacement. | sql-joins-and-unions, missing-values-and-outliers |

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
