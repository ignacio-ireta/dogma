# Plan: I-31 Data realism: real, semi-synthetic and realistic synthetic data for problems

## Context

Problems test on two kinds of data:

- **Hand-built data.** This part is strong: ties, constant columns, collinear designs and overflow cases.
- **Bulk random data.** Features are iid and on the same scale (`rng.normal`). Scores don't depend on the labels. Classes are nearly balanced. "Time series" are white noise. Text comes from 7–14 word vocabularies drawn uniformly. SQL hidden generators are uniform.

No problem uses a real dataset, although sklearn and statsmodels already ship about 36 offline datasets in `uv.lock`. The inventory found two kinds of risk:

- **(P) A solution passes here but fails on real data.**
  - `logistic-regression/p01` never sees a large linear score, so a naive `log(1+exp(η))` never overflows. It never sees mixed feature scales, so plain gradient descent converges only because the data is easy.
  - ACF and MASE mistakes can't be seen on white noise.
  - GMM never sees elongated clusters.
- **(M) The data teaches the wrong intuition.**
  - PCA and SVD run on flat spectra.
  - In kNN, PCA and GMM, feature scaling never matters.
  - ROC, AP and NDCG are only tested at AUC ≈ 0.5.
  - Permutation importance never meets correlated features.
  - The bootstrap never meets skew.
  - BPE, the tokenizers and BM25 only see toy text.
  - Conformal sets use Dirichlet(1) probabilities.
  - p99 latency comes from uniform gaps.
  - Quantization has no outlier channels.
  - SQL has no skewed users and no dirty categories.

The goal is a shared test-data layer, `dogma_data`, followed by one sweep over every affected problem so that its tests, `problem.md` and lesson use the layer. The layer has three parts:

- real offline datasets with documented quirks;
- small HuggingFace snapshots, pinned and vendored;
- realistic synthetic generators that don't change under the current `uv.lock`.

**Decisions (2026-11-09):**

- Add the HF `datasets` dependency.
- Roll out to all affected problems at once, W1–W2 included. A version bump turns an already-solved problem into re-solve material.
- Real data shows in tests, in `problem.md` and in lessons.

**Change from the design review: where `datasets` goes.** With pandas 3.1, installing pyarrow (a dependency of `datasets`) switches the default string dtype to pyarrow storage. That would change behaviour for 21 pandas problems and the 22 SQL problems that call `.fetchdf()`, and it would slow every import.

- `datasets` is added and locked, but in a non-default `hf` dependency group used only by the snapshot sync script.
- Tests read the vendored text files with pandas or the stdlib.
- The C0 spike promotes `datasets` to a main dependency only if all four of its checks pass:
  1. the lock diff downgrades none of numpy, pandas, scipy, sklearn, statsmodels or duckdb;
  2. `make verify-content` gives the same pass set and is at most ~10% slower;
  3. `import datasets` takes under 1.5 s under Seatbelt with no bytecode;
  4. `Dataset.from_pandas` writes nothing.

## Data strategy

| Kind | When | Oracle |
|---|---|---|
| Hand-built edge cases | Exact contract rules | Literal values. **Keep every existing one.** |
| Real dataset | Behaviour depends on real structure: scales, spectra, autocorrelation, correlated features, rounding ties | A library oracle on the same data, or a metamorphic invariant |
| Semi-synthetic (*plasmode*) | You need a known truth on realistic inputs: real `X`, then `y = X @ beta + noise` written inline | The known β plus a library oracle |
| Realistic synthetic | No real set fits, or a property must be controlled: AUC, imbalance, tail weight | A library or reference oracle |

**Rules for robust randomness.** These go in a new CONTENT_SPEC §6.6.

- Oracles are computed at runtime on the same data. Never hard-code values read off random draws.
- Statistical assertions use an explicit binomial or CLT bound.
- Seeds come from `dogma_data.rng("<problem>:<purpose>:<i>")`, a sha256 of the key. This ends the shared `range(8)` streams.
  - The key mixes in a `DOGMA_DATA_SALT` environment variable. The grader never sets it.
  - `exercise_audit seeds` varies the salt, so a test that only passes for one lucky data draw shows up.
- Generators are pure numpy, use no BLAS calls, and keep their output while `uv.lock` is unchanged.
  - Golden fingerprints (hash of `np.round(a, 9)`) catch drift after a lock bump. `grep -rl dogma_data content/` then lists which problems need a version bump.
  - Fingerprints only protect lesson numbers. Correctness never depends on them.
- Every realism test must map to a sentence in the Contract or Constraints. A requirement that is only implied gets reported, not tested (the I-28 rule).
  - Example: on longley (cond ≈ 4.9e9), compare fitted values or the objective, not coefficients at `rtol=1e-6`.
- Public tests stay literal (they are the `problem.md` examples). The one exception is a self-describing `load("iris")`.
- Real data is loaded in module-scoped fixtures, never at import. A collection error would otherwise show up as a hidden failure that names no mistake.
- The time budget is checked with a naive blind solution, not the reference: the whole run must stay under 10 s. Text slices stay at 20–50 KB so that pure-Python BPE fits.

## Dataset catalog (initial)

- **sklearn, bundled with the library:**

  | Dataset | Quirks |
  |---|---|
  | iris | 1 duplicate row, ties at 1 decimal |
  | wine | proline dominates unless scaled |
  | breast_cancer | scales from 0.004 to 880, correlated features, nearly separable |
  | diabetes | `scaled=False` gives the raw data; binary `sex` |
  | digits | 3 zero-variance columns and many columns with IQR = 0 |
  | linnerud | — |

- **statsmodels, bundled, public domain:**

  | Dataset | Quirks |
  |---|---|
  | co2 | 59 NaN, seasonal |
  | sunspots | about an 11-year cycle |
  | nile | changepoint |
  | longley | cond ≈ 4.9e9 |
  | stackloss | outliers |
  | engel | skew, uneven spread |
  | macrodata | trending |
  | elnino | — |
  | spector | small logit set |

  Load from `statsmodels.datasets.<name>`, never `statsmodels.api`, which takes 3.1 s to import.
- **Vendored from `sources/`**, copies pinned by sha against the source:
  - penguins: CC0, NaN, categoricals.
  - interview-prep `customers`, `orders` and `surveys`: deliberately messy synthetic data (duplicate ids, casing noise, impossible ages, lognormal amounts).
- **HuggingFace snapshots**, pinned to a revision sha.
  - Licence policy: CC0, public domain, MIT, Apache or BSD. CC-BY needs a `NOTICE` entry. NC, ND, `unknown` and a missing licence are refused.
  - Card licences checked on 2026-11-09:

  | Repo | Licence | Use |
  |---|---|---|
  | `openai/gsm8k` | MIT | pass@k, majority vote, answer extraction |
  | `scikit-learn/adult-census-income` | CC0 | categoricals, imbalance, `?` as missing |
  | `gvlassis/california_housing` | MIT | target censored at 500001, heavy tails, robust metrics |
  | `sedthh/gutenberg_english` | MIT (public-domain books) | Zipf-distributed real text for BPE, tokenizers, TF-IDF/BM25 and chunking; a few chapters of 20–50 KB |

  - Rejected:
    - no licence on the card: `karpathy/tiny_shakespeare` (also 1.1 MB, over the commit limit), `SetFit/20_newsgroups`;
    - non-free or unknown licence: `stanfordnlp/imdb` (`other`), `fancyzhx/ag_news` (`unknown`), `roneneldan/TinyStories` (CDLA-Sharing);
    - unsuitable content: `Anthropic/hh-rlhf`;
    - IR benchmarks: SciFact (non-commercial), MS MARCO (non-commercial), and the BEIR web text, because email and phone patterns in it would trip the privacy audit.
  - For retrieval, a short set of original queries and a relevance file are written by hand over the public-domain corpus.

## I-31a: platform

Done by worktree agents on disjoint paths; the coordinator creates each worktree.

**C0, spike (no commit).** In a scratch worktree, run `uv add --group hf datasets==5.1.0` (released 2026-10-05; requires `fsspec<=2026.7.0`, while torch's fsspec is locked at 2026.9.0). Then check:

- the lock diff;
- the four promotion checks above;
- that `check_content_types.py` follows the editable `.pth` to `dogma_data`.

The results go into ADR-0020.

**C1, dependencies.** `pyproject.toml` gets `[dependency-groups] hf = ["datasets==5.1.0"]`, and `uv.lock` is updated, in a commit of its own (§11). The sync script then runs with `uv run --frozen --isolated --group hf python -I …`. `--isolated` keeps pyarrow out of the shared `.venv`.

**C2, the `src/dogma_data/` package** (new, test-only):

- **`__init__.py`, `py.typed`.**
  - `import dogma_data` loads numpy only. Other libraries are imported inside functions.
  - `test_lazy_imports` enforces this in a subprocess.
- **`seeds.py`.** `rng(key) -> np.random.Generator`: a sha256 of the key plus `DOGMA_DATA_SALT`.
- **`real.py`.** `load(name) -> RealData`, a frozen dataclass with `X`, `y`, `feature_names`, `target_name` and `frame()`.
  - Data is raw by default: diabetes uses `scaled=False`.
  - Sources: sklearn `load_*`, `statsmodels.datasets.<name>.load_pandas()`, and the vendored files read with `importlib.resources`.
- **`synth.py`.** Pure numpy. A generator lands together with its first consumer. Initial set:
  - `correlated_mixed_scale` (controlled condition number, scales spread over several orders of magnitude)
  - `heavy_tailed_noise` (Student-t, uneven spread)
  - `imbalanced_labels` (ratio and label-flip rate)
  - `informative_scores(y, auc, decimals=None)` (binormal, AUC = Φ(d/√2); rounding creates ties)
  - `seasonal_ar_series` (trend, season, AR(p), level shifts, NaN gaps)
  - `event_log` (Zipf users, lognormal amounts, casing noise and duplicates, naive timestamps to match the SQL contracts)
  - `latencies` (lognormal or mixture)
  - `outlier_channel_activations`
  - `peaked_probs` (temperature, miscalibration)
- **`catalog.yaml`.** One entry per dataset: id, loader, licence or copyright, shape, sha256 and quirks.
- **`data/`.** The vendored CSVs, `DATA_SOURCES.md` and `NOTICE`.
  - Files are text only (CSV, JSONL or TXT), 1 MB or less each, about 8 MB in total, so the privacy audit (which skips files with NUL bytes) and `check_commit_files` both apply.
  - Columns with trailing spaces are quoted.
- **`pyproject.toml`.**
  - Add the package to hatch `packages`, mypy `packages`, coverage `source` and import-linter `root_packages`.
  - Add two contracts: `dogma_data` never imports `dogma` or `dogma_harness`, and neither of them imports `dogma_data`.
- **`tests/unit/dogma_data/`.**
  - `test_catalog` checks offline that each dataset's shape, sha256 and quirks hold, which catches lock drift.
  - Also `test_seeds`, `test_synth_fingerprints`, `test_synth_properties` (each property lands within binomial bounds) and `test_lazy_imports`.
- **Docs.** ADR-0020 and the ARCHITECTURE §8 layout and dependency rules.

**C3, submissions can't import the package.** Ban `dogma_data`, `datasets` and `huggingface_hub` platform-wide in `src/dogma/grading/static.py` `check_submission`, not per problem in `meta.yaml`: there, `validate.py:484` would require every `problem.md` to name the ban. Add tests in `tests/unit/grading/test_grading_static.py`.

**C4, an allowlist for test imports.** `src/dogma/content/validate.py` `_check_test_module` (around line 558) currently enforces no allowlist.

- Enforce the CONTENT_SPEC §6.4 list plus `dogma_data`, and forbid `statsmodels.api` and `datasets`.
- Add tests in `tests/unit/content/test_validate.py`.
- `verify-content --all` must stay clean.

**C5, a Seatbelt integration test.** A new fixture `tests/fixtures/problems/py-realism/` and `tests/integration/runner/test_runner_dogma_data.py` check that:

- the problem passes in under 10 s;
- nothing is written outside the run dir;
- network access is denied;
- a submission that imports `dogma_data` is rejected.

**C6, scripts.**

- **`scripts/sync_hf_datasets.py`.**
  - Reads `src/dogma_data/hf_manifest.yaml` and writes a lock file.
  - Settings: a temporary `cache_dir`, `trust_remote_code=False`, a pinned sha, a licence check against the allowlist, and selection of rows and columns.
  - Writes text files and runs `audit_privacy` over them.
  - `--check` re-verifies the hashes offline.
- **`scripts/exercise_audit.py`.**
  - `cmd_seeds` (about line 810) sets `DOGMA_DATA_SALT` for each seed.
  - A new `realism ID WRONG...` command. The reference and the blind solution must pass the new tests. Each hand-written wrong file must pass the HEAD tests (fetched with `git show`) and fail the new ones.
  - Tests in `tests/unit/scripts/`.

**C7, HF snapshots** (the coordinator runs this, with network). Add `src/dogma_data/data/hf/*` and update `NOTICE`.

**C8, docs.**

- `docs/CONTENT_SPEC.md`: a new §6.6 "Test data" (decision table, realism checklist, robustness rules) and the §6.4 import list.
- `docs/ENGINEERING_STANDARDS.md`: a §5.2 checklist item and a `data` commit scope.
- `docs/ARCHITECTURE.md` §4.3.
- A new `docs/DATASETS.md`: a readable catalog with each dataset's quirks and the topics it fits.
- `templates/topic/problems/p01-todo/test_hidden.py.tmpl`: a fixture example.
- `docs/ROADMAP.md`: status rows and cards for I-31a/b/c.

## I-31b: content sweep

Six Haiku authors run at once. Each works in the shared main tree on separate topic directories and never commits; S7 starts when the first slice finishes. The coordinator reviews and commits each slice.

**Recipe for each problem:**

1. Keep every hand-built edge case. Never weaken a test or widen a tolerance.
2. Replace the flagged bulk generators with `dogma_data.synth`. This also removes the duplicated helpers: the 4 copies of the scores pattern, the PCA/SVD and ridge/normal-equation `_data` functions, and the 8 SQL `_random_rows`.
3. Add at least one hidden test on real or semi-synthetic data, checked against a library oracle or an invariant, and tied to a Contract sentence. Add a Contract sentence when the behaviour is a real requirement.
4. In `problem.md`, add one example that names the dataset, and list the realism families under "Done looks like" (families only, no values).
5. In the lesson, add a "Where it breaks" paragraph (or extend the worked example) with a real-data failure. Its numbers are reproduced by a snippet recorded in the handoff and rounded, because BLAS libraries differ in the last digits.
6. Bump `meta.yaml` `version`. A `reviewed` topic goes back to `verified` until I-31c.
7. Gates:
   - `dogma content verify <topic>` passes.
   - `exercise_audit realism` passes with at least one hand-written realistic wrong solution per new test, for example a naive `log(1+exp)`, a population ACF divisor, or gradient descent on raw scales.
   - `exercise_audit seeds` passes with the salt.
   - The blind solution passes in under 10 s.

| Slice | Owns | Data |
|---|---|---|
| S1 Linear models and conditioning | `cml/{normal-equation, ridge-regression-and-weight-decay, lasso-and-elastic-net, logistic-regression, regression-metrics-and-robust-losses}` | longley, raw diabetes, raw breast_cancer, spector, stackloss, engel, california housing |
| S2 Scores and calibration | `cml/{roc-pr-curves-and-auc, precision-recall-and-f1, ranking-metrics, handling-class-imbalance, confusion-matrix-and-count-based-scores}`, `safe/split-conformal-prediction` | `informative_scores` (AUC 0.7–0.95, ties, 1:100), a breast_cancer feature as the score, `peaked_probs` |
| S3 Geometry and unsupervised | `cml/{principal-component-analysis, singular-value-decomposition, covariance-and-correlation, gaussian-mixtures-and-em, k-means, nearest-neighbour-search-and-knn}` | unscaled wine, digits, iris (duplicates, ties: use a brute-force reference), longley |
| S4 Preprocessing, trees, statistics, time series | `cml/{feature-scaling, leakage-and-walk-forward-validation, random-forests-and-feature-importance, rolling-statistics-stationarity-and-autocorrelation, bootstrap-and-permutation-resampling, t-tests-and-anova, missing-values-and-outliers, preprocessing-pipelines, choosing-a-tree-split}` | digits (IQR = 0), breast_cancer (correlated), co2, sunspots, nile, engel, iris ANOVA, penguins, adult |
| S5 Text | `cml/tf-idf-and-bm25`, `llm/{bpe-merge-training, bpe-encode-and-decode, word-and-character-tokenizers, retrieval-augmented-generation-pipeline, pass-k-and-majority-voting}` (the RAG topic includes chunking) | Gutenberg slices, original queries and relevance file, gsm8k |
| S6 Serving and DL | `inf/{data-quality-and-drift-detection, ttft-itl-and-throughput, absmax-and-zero-point-int8-quantization}`, `dl/{a-complete-training-loop, training-an-mlp-from-scratch}` | adult (categorical drift), `latencies`, `outlier_channel_activations`, digits (an accuracy floor; the d ≤ 32 cap is only raised if the contract allows) |
| S7 SQL and pandas | `content/sql/*` | `event_log` (Zipf, lognormal, dirty categories, naive timestamps kept), interview-prep CSVs for the pandas problems. Expected results are computed with pandas over the same data. |

The exact topic directory names are confirmed against `content/` when the slices are assigned. Problems the inventory found realism doesn't matter for are left alone: primitives, layers and optimizers checked against torch, algo, math, tabular RL, and inf arithmetic.

## I-31c: review

A second agent reviews each slice against the CONTENT_SPEC §6.6 checklist. S1 and S3 need conditioning judgement, so a stronger model than Haiku reviews them. Then:

- the coordinator runs `exercise_audit` `baseline`, `mutate` and `seeds` again with a fresh `--out var/audit-i31/`;
- topics go back to `reviewed`;
- the handoff log entry is written.

## Verification

```bash
# I-31a
uv run --frozen pytest tests/unit/dogma_data tests/unit/grading tests/unit/content tests/unit/scripts tests/unit/runner -q
uv run --frozen pytest -m integration tests/integration/runner -q
uv run --frozen mypy && uv run --frozen lint-imports
uv run --frozen --group torch python scripts/check_content_types.py
uv run --frozen --isolated --group hf python -I scripts/sync_hf_datasets.py --check
uv run --frozen python scripts/audit_privacy.py
# per slice
uv run --frozen --group torch dogma content verify <topic ids>
uv run --frozen --group torch python scripts/exercise_audit.py realism --out var/audit-i31/<slice> <problem id> wrong_*.py
uv run --frozen --group torch python scripts/exercise_audit.py seeds --out var/audit-i31/<slice> --only <topic>
# coordinator gate (macOS, clean worktree on the exact commit)
time make verify-content    # compare with the C0 baseline
make check && make test-all && make audit
```

Report the commands that were actually run in the ROADMAP handoff entry.
