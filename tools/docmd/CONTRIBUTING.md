# Contributing

## Setup

```bash
uv sync
uv run pre-commit install
```

## Workflow

- `main` is protected in spirit: work on feature branches, keep commits small
  with conventional messages (`feat(pdf): ...`, `fix(docx): ...`,
  `test(xlsx): ...`, `docs(cli): ...`).
- Before pushing, the full local gate must pass:

```bash
uv run ruff check . && uv run ruff format --check .
uv run mypy src
uv run pytest && uv run pytest -m slow
```

## Changing extractor output

1. Bump that extractor's `version` class attribute (goldens and sidecars are
   keyed to it).
2. Regenerate goldens: `uv run pytest -m golden --update-golden`.
3. Review `git diff tests/expected` deliberately — golden churn is the
   regression alarm, never commit it unread.
4. Update `docs/extraction_contract.md` if supported behavior changed.

## Adding a fixture

Extend `tests/fixtures/generate_fixtures.py` (fixtures are generated, never
committed) and reference it from `tests/integration/test_fixture_generation.py`'s
expected list.

## Security expectations

- Never execute document macros or active content.
- Never log extracted document content by default.
- No network calls in the conversion path.
- Parse untrusted files defensively: new failure modes get a domain exception
  and an error code, not a traceback.
