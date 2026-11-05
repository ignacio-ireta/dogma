# Multivariate Statistics — exam preparation notebooks

A practice-first series for a university **extraordinary exam in Estadística Multivariada**
It covers the five official syllabus units — multiple regression, principal component
analysis, factor analysis, cluster analysis, and discriminant analysis (incl. logistic) — plus the linear-algebra
foundations, an integrated case study, and a timed mock exam.

The goal is not a digital textbook. It is that you can look at an exam problem or some software output,
**identify the statistical structure, choose a defensible method, run it, and explain what it means.**

## How each notebook works — the five-stage loop

Every notebook has the same shape, so structure never surprises you:

> **🔁 Retrieve** (warm-up from earlier units) → **🔮 Predict** (guess before being taught) →
> **🧩 Learn the system** (method as a system) → **🛠️ Practice** (mixed concept / hand-calc / code / interpret /
> choose / diagnose) → **✅ Close** (write an exam-style answer + fill a method card) → **🔑 Answer key** (separated at the bottom).

Conventions:
- **Type answers in the `YOUR ANSWER` cells** — plain words, numbers, or Python. You never have to write LaTeX.
- `# TODO` code cells have **escalating hints** (concept → formula → code). They self-check: a blank prints a
  reminder, a *wrong* answer raises a clear error, a right answer prints `OK`.
- Colours mean the same thing everywhere: **blue = original variables, orange = components/factors,
  green = fitted/model, red = residual/error**, qualitative palette = groups/clusters.
- Each notebook starts with an **estimated time** and a **Minimum-Viable-Pass path** for low-energy days.

## Setup (with `uv`)

This project uses [`uv`](https://docs.astral.sh/uv/). From the repo root:

```bash
uv sync --group dev          # creates .venv with Python 3.12 and all libraries
uv run python -m ipykernel install --user --name python3 --display-name "Python (mvs)"   # optional
uv run jupyter lab           # open the notebooks
```

No `uv`? Use any Python 3.12 environment and `pip install -r requirements.txt` (pinned).

> **Note on scikit-learn:** pinned to `1.7.x` because `factor_analyzer` (notebook 04) still calls a
> scikit-learn argument removed in 1.8. Everything else uses stable APIs.

Datasets download once and cache to `data/` (see [`DATA_SOURCES.md`](DATA_SOURCES.md)); after that they work
offline.

## Run order, objectives, and time

| # | Notebook | Unit | You will be able to… | Time | Min-viable-pass |
|---|---|---|---|---|---|
| 00 | `00_orientation_and_method_map` | — | choose a method from a research question | 45–60 m | Retrieve + method-map table + scenario drill |
| 01 | `01_linear_algebra_and_covariance_geometry` | foundations | read covariance/eigen/SVD geometry | 90–120 m | covariance-from-scratch + ellipse/eigenvector cells |
| 02 | `02_multiple_regression` | 1 | fit OLS, read a coefficient table, diagnose | 100–130 m | normal-equations + coefficient-table cells |
| 03 | `03_principal_component_analysis` | 2 | derive & run PCA, avoid the scaling trap | 90–120 m | scratch-vs-sklearn + scaling-trap cells |
| 04 | `04_factor_analysis` | 3 | run FA, read loadings/communalities, vs PCA | 100–130 m | recover-factors + PCA-vs-FA cells |
| 05 | `05_cluster_analysis` | 4 | cluster, choose k, read a dendrogram | 100–130 m | scaling-trap + choose-k cells |
| 06 | `06_discriminant_analysis_and_logistic_bridge` | 5 | LDA/QDA/logistic, confusion-matrix metrics | 110–140 m | scratch-LDA + confusion-metrics cells |
| 07 | `07_integrated_case_study` | all | run a full analysis end-to-end (method not pre-told) | 2–3 h | one question, 8 steps, 5-sentence report |
| 08 | `08_exam_practical_simulation` | all | sit a timed practical (no inline solutions) | 2.5 h | attempt every section |
| — | `08_exam_practical_simulation_solutions` | all | self-grade against the rubric | — | after the attempt only |

Work them in order — each notebook's **🔁 Retrieve** deliberately recalls earlier units.

## Check that everything still runs

```bash
./run_all.sh                 # executes every notebook top-to-bottom in a clean kernel
./run_all.sh 03_*.ipynb      # or a subset
```

A freshly delivered (un-filled) notebook passes: blank `# TODO` cells print reminders, not errors. If you fill
an answer **incorrectly**, that cell will raise — that is the feedback working.

## Repository layout

```
README.md                 this file
pyproject.toml / uv.lock  environment (uv-managed, Python 3.12)
requirements.txt          pinned export for non-uv users
DATA_SOURCES.md           dataset provenance & licensing
METHOD_COMPARISON.md      one-page revision sheet (the five method cards)
run_all.sh                clean-kernel execution check
mvslib/                   shared helpers: palette, plots, cached data loaders, self-check
data/                     cached datasets (created on first run)
00_…08_….ipynb            the notebooks
tools/                    one-time generator scripts (see below)
```

## Reference sheets

- [`METHOD_COMPARISON.md`](METHOD_COMPARISON.md) — the five method cards side by side; your last-night review.
- [`DATA_SOURCES.md`](DATA_SOURCES.md) — where the data comes from and its licence.

## Regenerating notebooks (maintainers)

The notebooks were authored from generator scripts in `tools/` (`build_NN.py`) using `nbformat`, with shared
section templates in `tools/nbbuild.py`. **Re-running a generator overwrites its notebook**, so do *not*
regenerate after you have started filling in answers — your notebooks are the working copy. To tweak content,
edit the matching `tools/build_NN.py` and run `uv run python tools/build_NN.py`.
