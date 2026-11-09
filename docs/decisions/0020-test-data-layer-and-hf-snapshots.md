# 0020. A test-data layer (`dogma_data`) and pinned HuggingFace snapshots

Date: 2026-11-09 · Status: accepted

## Context

Hidden tests mostly draw bulk data from iid same-scale normals, labels independent of scores,
white-noise series and uniform toy text. That hides failures that real data exposes (overflow on
large linear scores, gradient descent on mixed scales, autocorrelation, ties from rounding) and
teaches the wrong intuition (scaling never matters, AUC is always 0.5). sklearn and statsmodels
already ship about 36 offline datasets in `uv.lock`, but tests can't share helpers: the runner
stages only the problem's own files, and copied generators had drifted apart across problems. The
learner asked for real datasets, HuggingFace data and realistic generators.

## Decision

Add `src/dogma_data`, a test-only package that content tests import (`src/` is already on the
interpreter path through the editable install). It loads real offline datasets with documented
quirks, vendored text snapshots, and pure-numpy realistic generators seeded by
`dogma_data.rng(key)`. Submissions may not import it (a platform-wide static ban). Add `datasets`
in an opt-in `hf` dependency group, used only by `scripts/sync_hf_datasets.py` to vendor small
snapshots pinned by revision; tests read the vendored text files, never `datasets`.

## Consequences

Tests can use real structure and controlled realism without per-problem copies, and the data a
problem sees is reviewable in the repo. A generator's output is frozen only under the current
`uv.lock` (NumPy streams carry no cross-version guarantee), so golden fingerprints in
`tests/unit/dogma_data` flag drift after a lock bump and `grep -rl dogma_data content/` lists the
problems to re-check. Re-grading an old run uses the current package, which is safe only while the
outputs stay frozen. The `hf` group moves the locked `fsspec` from 2026.9.0 to 2026.7.0 (shared
with the torch group); nothing else in the default environment changes.

Spike (2026-11-09, macOS, warm file cache): the lock diff changes no version of numpy, pandas,
scipy, scikit-learn, statsmodels or duckdb. With the `hf` group installed, `import datasets` takes
about 1.15 s without bytecode (7.3 s on a cold first run) and `Dataset.from_pandas` writes no
files. pandas 3.1 switches its default string storage from `python` to `pyarrow` when pyarrow is
present. `dogma content verify --all` with pyarrow installed passes the same 251 problems as
without it, but took 1348 s against 774 s (CPU 1584 s against 899 s). That run shared the machine
with other jobs and used a fresh venv without compiled bytecode, so the slowdown is an upper
bound, not a measurement. The promotion bar (at most 10% slower) is not met, so `datasets` stays
in the opt-in `hf` group.

## Alternatives

- `datasets` as a default dependency, loaded by tests at runtime: changes pandas string storage
  for every pandas and SQL problem and adds over a second per run, while no problem teaches the
  `datasets` API. Revisit only if all four spike checks pass and a problem needs the API.
- Per-problem `fixtures/` copies: no shared generators, data duplicated per attempt, and the
  exercise audit's witness can't add fixture files.
- `fetch_*` / `get_rdataset` / `seaborn.load_dataset` at test time: the sandbox denies network.
- Generators built on sklearn `make_*`: legacy `RandomState` streams and a 0.6 s import.
