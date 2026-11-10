# 0022. Problem variants and a gate for test edits

Date: 2026-11-09 · Status: accepted

## Context

The evidence that a problem's tests work was produced by one-off audits and kept in git-ignored
`var/`: blind solutions, wrong solutions, triage verdicts. The I-28 set no longer exists, so a
later test edit could only be re-checked by a new audit. Mutants are one-token edits of the
reference and miss conceptual mistakes: on 2026-11-10 a hand-written solution that rescaled `b`
in place passed every test of the example problem, which the mutation audit had scored as having
no gap. The rules for hidden tests in CONTENT_SPEC §6.4 had no check, and 214 sites in 73
problems broke them.

## Decision

Each problem package carries `variants/`: solutions its tests must accept (one written blind,
plus correct solutions that differ from the reference where the Contract allows) and wrong
solutions they must reject, each tied to a Contract clause and to the test that kills it. The
manifest is written only by `scripts/exercise_gate.py`, from what the grader showed. The same
tool lints hidden tests and checks every test edit in one of three modes (add, text, refactor).
Static manifest rules run in `make check`. The replay through the grader runs in `make test-all`.

## Consequences

A test edit is checked in about a minute instead of by a full audit, and a correct solution that
starts failing is noticed. The repository gains roughly 2,000 small files. A version bump or a
statement edit must be followed by a replay, which `VARIANT_VERSION_STALE` and the clause check
force. Variants show known mistakes. A mistake nobody wrote down is still unmeasured, so the
sampled mutation audit stays.

## Alternatives

- Keep the evidence in `var/`: free today, lost again later.
- A mirror tree under `tests/`: `content/` stays unchanged, but two trees drift apart and a
  content commit no longer carries its proof.
- A clause-by-test matrix for all hidden tests: complete, but too costly to write and to keep.
