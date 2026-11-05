# Optional PyTorch runtime (I-19)

The approved learning-sequence design calls for optional torch support before week 8.
Keep existing runtimes and the default environment light. The current runner already
uses the application's interpreter; no second virtual environment is needed.

## Decisions

- Add the optional `torch` dependency group, never a default group. Pin PyTorch 2.14.1
  (released 30 September 2026, beyond the two-week package-age window). Use the normal
  PyPI wheel on this Apple Silicon host; no torchvision, torchaudio or CUDA indexes.
- `py-torch` problems require torch to be discoverable in the current interpreter.
  Missing support raises a structured infrastructure error before creating a run
  directory, with the exact `uv run --frozen --group torch dogma ...` remedy. Existing
  static rejection behavior remains first and never executes rejected code.
- Default problem execution remains CPU for deterministic grading. Learner content
  can request MPS explicitly where its contract permits; the platform does not silently
  select a device or promise MPS availability.
- Do not import torch into the app merely to detect it. A broken native installation
  can still fail during actual harness import; the normal runner report contains that
  failure. This guard diagnoses an absent optional package.

## Tasks

- [ ] Add failing missing/available/non-torch guard tests and an absent-runtime
  integration test proving no sandbox or run directory is created.
- [ ] Add `runner/runtime.py`, call it after static checks, and map a stable
  `RUNNER_RUNTIME_UNAVAILABLE` error to API 503.
- [ ] Add frozen optional group and a tiny actual tensor/autograd runner fixture.
  Verify public/hidden grading and repeatability using CPU; skip this optional
  integration test in a default environment without torch.
- [ ] Independently review; run targeted/full checks; update runtime setup docs,
  roadmap and handoff; commit separately from runner hardening.

## Ownership and verification

Coordinator owns dependency/lock changes, runner guard insertion, error mapping,
new runtime tests/fixture and shared docs. I-21 starts editing the runner only after
this change is handed over. All runner tests use scratch Settings and never record
in the learner repository. No runtime download occurs during grading.

Sources checked 25 October 2026:
[PyPI release and macOS wheels](https://pypi.org/project/torch/2.14.1/),
[official local installation](https://docs.pytorch.org/get-started/locally/),
[official MPS backend](https://docs.pytorch.org/docs/2.14/notes/mps.html).
