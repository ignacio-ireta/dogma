# Vendored: docmd

- **What:** docmd 0.3.0, a local CLI that extracts DOCX, PPTX, XLSX and PDF content into normalized Markdown.
- **License:** MIT. The full license text is in [`LICENSE`](LICENSE), copyright notice unchanged. Author and copyright holder: as named in `LICENSE`.
- **Origin commit:** `127aa5d`
- **Vendored:** 2026-10-23
- **Local changes:** package author metadata omitted; illustrative document authors in two IR tests
  and one golden Markdown file replaced with a generic example. No extractor or rendering behavior
  changed. The original MIT license and its required attribution remain unchanged. Import-time
  and current scrubbed hashes are in `sources/manifest.lock.json` (`source_id: "docmd"`).
- **Contents:** `src/`, `tests/` (without generated fixtures), `docs/`, `pyproject.toml`, `uv.lock`, `LICENSE`, `README.md`, `CHANGELOG.md`, `CONTRIBUTING.md`, `.gitignore`. Generated test fixtures, caches and the origin's usage notes were not copied.

## How to run

```bash
uv run --project tools/docmd docmd --help
```

Tests run from inside this folder, because pytest collects from the current directory:

```bash
cd tools/docmd && uv sync && uv run pytest -q
```

Running `uv run --project tools/docmd pytest -q` from the repository root collects the repository's own tests instead. The docmd tests write generated fixtures to `tests/fixtures/_build/`, which this folder's `.gitignore` excludes.
