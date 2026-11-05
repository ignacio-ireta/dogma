# Library refresh: evidence and curriculum design

Date: 2026-10-28. Status: proposed content work; the audit and plan were requested, content
authoring is a subsequent increment. Roadmap: I-25a–f. Existing core content remains complete.

## Intent and evidence

Use the expanded Library to strengthen explanations, assumptions, numerical reasoning and
citations in the existing curriculum. Preserve the 164-topic, 13-week core path and interview
week. Extend optional material where the new sources close a real interview or implementation
gap. Imported notes are seeds, not lessons or correctness oracles.

The two acquisition batches contain **71 PDFs and 35 notes**, with **10,164 PDF pages** reported
by their acquisition reports. This audit independently checked those files and the earlier Goldberg primary PDF and
companion note against their catalog SHA-256 values: **72 PDFs and 36 notes selected**. Their exact paths, titles, hashes, local source relationships and
available source URLs are frozen in [the inventory](../../source-refresh-2026-10-28.json).
That inventory contains Library-relative paths, not runtime dependencies.

The complete reorganized catalog contains 206 PDFs; all 206 current files were independently
hashed and match that catalog. All **124** PDFs in the original dogma import still occur by
checksum; **82** catalog PDFs have no checksum match in the import lock.
Thus the 71 acquired PDFs are not the entire historical import gap: 11 older or alternate
copies also need triage. Goldberg is explicitly selected from that gap; the other 10 remain
outside this import. Do not copy all 82 blindly. The Library's working tree includes
uncommitted reorganization and acquisition work, so its HEAD alone does not identify this
snapshot; file hashes are authoritative.

The current manifest selects `AI-ML/**/*.md`, `Mathematics/**/*.md` and a flat PDF root. The
Library now uses `01-Mathematics/`, `02-AI-ML/` and subject-local PDFs. Re-running that manifest
would miss the expansion and risk misleading provenance. Keep existing imported paths stable:
published citations already use them.

## Approach

**Recommended: additive import and focused revisions.** Freeze an explicit allowlist, add a
refresh source id, preserve the old snapshot and old citation ids, then revise existing topics
in small reviewed batches. This gives immediate value without changing the study budget.

Alternatives considered: replacing the old snapshot would break citations and scrub history;
importing the whole vault would duplicate papers and include archives, personal material and
out-of-scope learning-science notes; rebuilding the syllabus around the Library would displace
completed core content. None serves the requested source refresh as well.

## Core improvement map

Each row owns the complete listed topic packages: metadata/citations, lesson, cards, interview
notes and, only where needed, problems and tests. Preserve existing problem ids. New practice
is optional p02/p03 material; the core time budget must remain unchanged.

| Batch | Existing topic ids | New sources and concrete improvement |
|---|---|---|
| I-25b: numerical foundations | `cml/matrix-inverse-and-rank`, `cml/normal-equation`, `cml/singular-value-decomposition`, `cml/principal-component-analysis` | Penrose; Golub–Reinsch; Boyd–Vandenberghe applied linear algebra; Axler; MML. Distinguish existence, uniqueness and minimum-norm solutions; normal equations square conditioning; compare QR/SVD with inverse formulas; distinguish approximation error from roundoff. Include rank-deficient and nearly-collinear examples. |
| I-25b: derivatives | `cml/partial-derivatives-and-gradients`, `cml/chain-rule`, `dl/gradients-with-framework-autograd` | Baydin; Matrix Cookbook; calculus/Jacobian note. Teach derivative shapes, JVP versus VJP, and accumulation at shared graph nodes; finite differences are a check with truncation/roundoff limits. |
| I-25b: optimization | `cml/gradient-descent`, `cml/stochastic-and-mini-batch-gradient-descent`, `cml/lasso-and-elastic-net`, `cml/ridge-regression-and-weight-decay` | Boyd–Vandenberghe convex optimization; Bottou–Curtis–Nocedal; Parikh–Boyd; Bubeck. State smoothness/convexity and stochastic-gradient assumptions; explain proximal soft thresholding versus ridge shrinkage; qualify convergence claims. |
| I-25c: probability | `cml/probability-basics`, `cml/bayes-theorem`, `cml/expectation-and-variance`, `cml/maximum-likelihood-and-map`, `math/monte-carlo-simulation` | Kolmogorov; Durrett; MacKay; Bishop; dominated-convergence note. Clarify conditioning, sampling assumptions and likelihood versus posterior; explain when gradients and expectations can be interchanged without requiring measure theory to solve the core exercises. |
| I-25c: inference | `cml/sampling-distributions-standard-errors-and-cis`, `cml/bootstrap-and-permutation-resampling`, `cml/t-tests-and-anova`, `cml/a-b-test-design-and-analysis`, `cml/leakage-and-walk-forward-validation`, `safe/split-conformal-prediction` | Neyman–Pearson; Efron; Hernán–Robins; Angelopoulos–Bates; Barber et al. Separate confidence from probability about a fixed parameter, power from significance, exchangeability from independence, causal identification from prediction, and marginal from conditional coverage. Preserve the finite-sample conformal rank convention. |
| I-25d: precision and low rank | `dl/mixed-precision`, `llm/lora-adapters`, `llm/lora-parameter-and-memory-arithmetic` | Goldberg; Hu et al.; LSA/embedding/LoRA bridge. Distinguish range, precision, cancellation and accumulation; explain that a low-rank update is a parameterization, not proof that base weights are low rank; verify LoRA scaling, initialization and memory assumptions against the paper. |
| I-25d: models and alignment | `cml/adaboost-and-gradient-boosting`, `cml/xgboost-second-order-objective`, `llm/direct-preference-optimization-dpo`, plus existing RAG/MoE/RLHF lessons selected during source review | Friedman; Chen–Guestrin; complete Lewis, Jacobs, Jordan–Jacobs, Switch/GShard/GLaM, Christiano/Stiennon/Ouyang/Bai/Rafailov papers. Replace note-only support with exact paper sections; distinguish older conference versions and summaries from complete articles. Re-check DPO beta and reference-policy assumptions. |

These are improvement opportunities, not a claim that every existing lesson is wrong. For each
topic, first record the existing paragraph/card/question that needs improvement; if it already
teaches the point accurately, add provenance only and avoid duplicate prose or exercises.

## Optional extensions and deferred scope

**I-25e, after the core revisions:** use existing stable ids for `cml/lagrange-multipliers` and
`cml/gaussian-mixtures-and-em` (tier B), `llm/document-packing-and-cross-document-masks` (B),
and `rl/sarsa-and-expected-sarsa` (B). The first two gain direct support from Boyd/Bishop and
the Dempster–Laird–Rubin paper; the latter two close links in existing lessons but need their
own source checks. Randomized PCA/JL fits the existing tier-C
`cml/incremental-randomized-pca-and-random-projection`; keep it C. Graphical/variational
inference, sparse recovery and neural ODEs are reading extensions, not new core requirements.

**I-25f:** complete `sys/predictive-maintenance-case`, `sys/recommendation-system-case`,
`sys/logistics-optimization-case`, and `sys/rag-system-design-case`. LP/duality and
LP-versus-gradients-versus-evolution help the logistics case; Pareto comparisons and fair
evaluation budgets help model/serving choices. Write original synthetic cases with reveals,
twists, debriefs and rubric scoring. No personal project narratives.

The new GA, differential evolution, CMA-ES, NSGA-II and NEAT notes justify optional references
and interview follow-ups, not five new core topics or a new track. A later role-specific
extension may propose stable tier-B/C ids and independent implementation exercises. LP should
teach feasibility, boundedness, degeneracy and dual certificates before solver mechanics;
evolutionary claims need multiple seeds, equal evaluation budgets and no universal winner.

## Citation limits

LoRA and DPO now have local primary PDFs in the Library and can close corresponding unsupported
references after import and direct section review. This catalog has no Sutton–Barto,
PagedAttention or Bradley–Terry primary source. Keep those gaps explicit; do not replace a
missing original with an unrelated bridge note or claim they have been verified.

`Understanding Machine Learning` is explicitly recorded as an unavailable local download.
Keep its author-page reference remote; never create a corpus citation to a nonexistent PDF.
Catalog metadata and study notes do not establish technical correctness. Inspect primary text
before changing claims. Do not import learning-science/health notes or model-family marketing
claims into the curriculum automatically.

## Constraints and completion

No runtime/build dependency outside this repository; no content id renumbering; no promotion
to tier A or core-path scheduling changes. Preserve scrubbed source files, lock provenance and learner history. Keep problem ids stable;
bump problem versions when tests, tolerances or contracts change, as CONTENT_SPEC requires.
Never weaken hidden tests to fit an explanation.
Copyrighted PDFs and conversions stay gitignored. Imports must pass privacy review before
being committed; stale or unsafe internal links in copied notes must be rewritten or removed.

Every batch requires the content gate, appropriate oracle/gradient/mutation checks, a second
content reviewer and the full `make check`. A status is `reviewed` only after that review.
The [implementation plan](../plans/2026-10-28-library-refresh.md) defines the file-level work.
