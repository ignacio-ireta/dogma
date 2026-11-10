# 0021. The runtime sets what a submission may import

Date: 2026-11-10 · Status: accepted

## Context

A problem's `meta.yaml` names a runtime (`py-stdlib`, `py-numpy`, `py-data`, `py-torch`), and its
`problem.md` says what a solution may use ("Allowed: NumPy and the Python standard library").
Nothing enforced either: every runtime shares one environment, and `banned_imports` lists only the
library that would do the problem's whole job. In 178 of 231 Python problems a submission could
import libraries its statement rules out, for example scikit-learn's `KBinsDiscretizer` in the
PSI problem or `MaxAbsScaler` in absmax quantization (found during I-31b). No reference solution
or starter imports outside its runtime, and no `py-numpy` statement allows SciPy.

## Decision

The static check rejects any import outside the problem's runtime before the code runs (rule
`runtime-import`). `py-stdlib` allows the standard library; `py-numpy` adds NumPy; `py-data` adds
SciPy, pandas, scikit-learn and statsmodels; `py-torch` adds PyTorch. `banned_imports` still
narrows a runtime, and the platform-wide test-data ban (ADR-0020) still applies.

## Consequences

Statements and grading agree, and the app's own packages (`dogma`, `dogma_harness`) are now outside
every runtime too. A learner who reaches for SciPy in a `py-numpy` problem is told the runtime,
before any test runs. No content version changes, since the references already comply. A problem
that genuinely needs SciPy moves to `py-data`. Dynamic imports (`importlib.import_module`) remain
uncaught, as for every static rule. ARCHITECTURE's runtime table previously listed SciPy under
`py-numpy`; it now matches the statements.

## Alternatives

- Add scikit-learn and statsmodels to the PSI and absmax bans only: closes two problems, leaves 176.
- Document that "Allowed" is guidance: keeps statements and grading out of step.
- Install a separate environment per runtime: heavier, and the grader already isolates attempts.
