# CLAUDE.md

Local ML-interview practice platform (deep-ml-like), built in increments across sessions.
Start every session here:

1. `docs/ROADMAP.md`: pick your increment, respect its **Owns** paths, and update its status and
   the handoff log before finishing.
2. `docs/ENGINEERING_STANDARDS.md`: binding rules. §16 is written for agents.
3. `docs/ARCHITECTURE.md` (design), `docs/CONTENT_SPEC.md` (content format, example at
   `content/cml/vector-operations/`), `docs/LEARNER_PROFILE.md` (general learning-design principles).

Hard rules (details in the standards):

- **Self-contained:** never read or depend on paths outside this repo. The only exception is
  the AI runtime (local model servers) through the gateway.
- **No PII and no employer/client data**, anywhere. `make audit` must stay clean. Never commit
  or quote `private/`.
- Never weaken tests, widen tolerances, or edit `test_hidden.py` to make a solution pass.
- Content ids in `curriculum/syllabus.yaml` and `content/` are stable. Never renumber. Retire
  instead of deleting.
- No deep-ml problem text, solutions or tests. Only the topic outline is used.
- Use `uv run --frozen`. Don't touch `pyproject.toml` / `uv.lock` unless your increment owns
  dependency changes.
- Run `make check` before claiming anything works, and report what you actually ran.
