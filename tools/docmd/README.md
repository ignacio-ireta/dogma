# docmd

Local-first CLI that scans a file or folder tree, extracts readable content
from **DOCX, PPTX, XLSX, and PDF** files, and writes normalized **Markdown**
suitable for downstream LLM ingestion — with structured error reporting,
resumable runs, and full source traceability.

```
$ docmd scan ./documents --out ./markdown --recursive

SUCCESS: 91 files converted
WARNING: 7 files converted with warnings
FAILED: 2 files failed

Failed files:
- contract_scan.pdf: PDF contains no extractable text (scanned document?); re-run with --ocr auto to recognize it
- report_old.docx: file appears corrupted or is not a valid document of its type

Run report: markdown/runs/20260610T153012Z
```

## Highlights

- **One output contract for four formats** — every file goes through a shared
  intermediate document model, so DOCX headings, PPTX slides, XLSX sheets, and
  PDF pages all land in the same predictable Markdown shape (frontmatter,
  sections, GFM tables, warnings). See `docs/extraction_contract.md`.
- **Failures are data, not crashes** — corrupt, encrypted, scanned, or
  misnamed files are skipped with structured error codes; one bad file never
  kills a folder run (unless you ask with `--fail-fast`).
- **Resumable and idempotent** — every run writes a manifest; re-runs skip
  unchanged files by content hash; `--resume` finishes an interrupted run;
  outputs are written atomically; Ctrl+C stops gracefully (twice for force).
- **Traceable** — each output carries a `.meta.json` sidecar (source sha256,
  extractor version, pipeline steps, warnings), and each run leaves
  `manifest.json`, `errors.json`, and `run.log` under `<out>/runs/<run_id>/`.
- **Tracked repo sync** — `docmd sync` diffs a source tree against its last
  synced state: adds and updates convert, moves relocate the existing Markdown
  without re-extraction, deleted and superseded outputs are archived, every
  action is recorded in `changes.json`, and a `registry.json` indexes each
  repo synced into the output root.
- **Local-first and non-networked** — no document content leaves your machine,
  macros never execute, content is never logged.
- **Opt-in OCR for scanned PDFs** (macOS) — `--ocr auto` recognizes pages that
  have no text layer using Apple's on-device Vision framework; `--ocr force`
  re-reads garbled text layers. Install with `uv sync --extra ocr`.

## Install & quickstart

Requires Python ≥ 3.11 and [uv](https://docs.astral.sh/uv/).

```bash
git clone <repo-url> && cd document_conversion
uv sync                  # add --extra ocr on macOS for scanned-PDF OCR
uv run docmd scan ./documents --out ./markdown --recursive
uv run docmd sync ./documents --out ./markdown          # tracked repo sync
uv run docmd process ./report.docx --out ./markdown
uv run docmd scan ./scans --out ./markdown --ocr auto   # OCR scanned PDFs
```

Outputs mirror the source tree with collision-proof names
(`report.docx` → `report.docx.md`). Full CLI reference: `docs/usage.md`.

## Documentation

| Doc | Contents |
|---|---|
| [docs/usage.md](docs/usage.md) | CLI commands, flags, exit codes |
| [docs/extraction_contract.md](docs/extraction_contract.md) | What each format supports, error/warning codes |
| [docs/architecture.md](docs/architecture.md) | Components, data flow, concurrency, idempotency |
| [docs/testing.md](docs/testing.md) | Test suite, fixture generation, golden-file workflow |
| [docs/troubleshooting.md](docs/troubleshooting.md) | Common failures and what to do |

## Development

```bash
uv run pytest                 # fast suite
uv run pytest -m slow         # huge-sheet + interruption tests
uv run ruff check . && uv run ruff format --check . && uv run mypy src
uv run pre-commit install     # optional hooks
```

## License

This project's own code is **MIT** (see `LICENSE`). Note: the PDF backend
([PyMuPDF](https://pymupdf.readthedocs.io/)) is **AGPL-3.0**, which has
copyleft implications if you distribute docmd as a combined work. For
local/internal use this does not affect you; the extractor abstraction keeps a
swap to an MIT-licensed backend (e.g. pdfplumber) cheap if plans change.
