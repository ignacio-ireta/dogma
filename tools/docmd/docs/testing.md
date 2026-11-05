# Testing

## Running tests

```bash
uv run pytest                      # fast suite (slow tests excluded by default)
uv run pytest tests/unit           # unit tests only
uv run pytest tests/integration    # integration tests only
uv run pytest tests/e2e            # CLI end-to-end tests
uv run pytest -m slow              # huge-sheet + SIGINT subprocess tests
uv run pytest --cov=docmd          # with coverage (CI gate: 85%)
```

Markers (declared in `pyproject.toml`, `--strict-markers` enforced): `slow`,
`golden`, `integration`, `e2e`, `ocr`. The default `addopts` excludes `slow`.
`ocr`-marked tests drive the real Apple Vision engine and skip themselves
anywhere it cannot run (non-macOS, or the ocr extra not installed); the
extractor's OCR wiring stays covered everywhere through a fake backend in
`tests/integration/test_ocr_extraction.py`.

## Fixtures

Binary fixtures are **generated, not committed**. OOXML containers embed save
timestamps, so the binaries are not byte-stable; committing them would invite
mystery diffs. Instead `tests/fixtures/generate_fixtures.py` builds them
reproducibly into the gitignored `tests/fixtures/_build/`:

- a session-scoped pytest fixture (`fixtures_dir` in `tests/conftest.py`)
  builds them once per session and rebuilds when the generator changes;
- standalone: `uv run python tests/fixtures/generate_fixtures.py
  [--out DIR] [--include-huge]`.

Document core properties are pinned to a fixed epoch so extracted *content* is
deterministic. The set covers good files (docx/pptx/xlsx/pdf, empty and
special-character variants, cached and uncached formulas, ruled PDF tables,
scanned and encrypted PDFs), a corrupt set (truncated zip, fake extensions,
zero bytes, OLE-encrypted, truncated PDF), scanner-skip cases, and a nested
`tree/` for end-to-end runs.

Known limitation: fixtures are library-authored, not Word/Excel-authored.
Real-world samples can be added under `tests/fixtures/external/` with a
provenance README.

## Golden files

`tests/expected/` holds committed Markdown outputs for hand-built IR documents
and for each format's fixture. Golden tests force deterministic mode (no
timestamps). On mismatch the test prints a unified diff.

Update workflow:

```bash
uv run pytest -m golden --update-golden
git diff tests/expected        # review every change deliberately
git commit -m "test: update golden files for <reason>"
```

Extractors carry their own versions (`docx/0.1.0` etc.) so goldens only churn
when the responsible extractor actually changes behavior.

## What runs in CI

- lint job: ruff check, ruff format --check, mypy (strict)
- test matrix: pytest with coverage on Python 3.11/3.12/3.13 (gate: 85%)
- slow job: `pytest -m slow` on 3.12 (50k-row sheet, subprocess SIGINT/resume)
- macOS job: full fast suite with `--extra ocr`, exercising real Apple Vision
  OCR (no coverage gate; the ubuntu matrix keeps it)

## External dependencies

None beyond the Python packages in `uv.lock`. No poppler/tesseract/libreoffice
needed. OCR uses Apple's Vision framework through the `ocr` extra (`ocrmac` +
pyobjc), which only installs on macOS. OCR'd text is asserted by content
(`"Quarterly revenue" in text`), never byte-compared — recognition output may
shift across macOS versions.
