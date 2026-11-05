# Corpus

Third-party material used as retrieval sources. Nothing in this folder is committed to git except this README.

## What is here

| Path | Files | Size | Contents |
|---|---|---|---|
| `raw/library-papers/` | 124 PDFs | about 938 MiB | Research papers, historical and modern (arXiv preprints, journal articles, books) |
| `raw/library-refresh/` | 72 PDFs | about 337 MiB | Selected primary sources and textbooks from the Library refresh, by Library-relative path |
| `md/<source-id>/` | Generated | Varies | Incremental PDF-to-Markdown conversions |
| `index.db` | Generated | Varies | SQLite FTS5 search index, separate from learner progress |

The imported raw collections contain 196 PDFs, about 1.3 GiB, copied byte-for-byte from the
configured local library source. These local files are not included in a fresh clone.

## Why it is gitignored

The PDFs are copyrighted third-party works and large. The root `.gitignore` excludes `/corpus/*` except this README. Keep a local backup of `raw/` with the rest of the machine. The `sources/manifest.lock.json` file records every copy with its sha256 so that the set can be verified after a restore.

## Index and search

From the repository root, index committed notes, authored lessons and any existing Markdown
conversions without running the PDF converter:

```bash
uv run --frozen dogma corpus sync --no-convert
uv run --frozen dogma corpus search "Adam bias correction" --limit 5
```

To convert available PDFs with the vendored `tools/docmd` project before refreshing the index:

```bash
uv run --frozen dogma corpus sync
```

Both operations are incremental. Search uses literal terms with optional repeated
`--topic <topic-id>` filters and returns passages with source/page citations. Retrieval and the
tutor use this same index; neither requires an embedding model.

OCR is off by default. `corpus sync --ocr auto` or `--ocr force` opts in to OCR; on macOS these
options also select the converter's locked OCR extra, which may require a first-time download.
Search and indexing existing Markdown work offline after dependency setup.

## Restore raw imports

```bash
uv run --frozen python scripts/import_sources.py
```

This import step needs the original folders listed in `private/import_roots.yaml` (gitignored)
to exist on this machine. With unchanged origins it writes nothing. It is only needed to
restore or refresh the original imports; ordinary practice and indexing the committed notes
do not need those folders. Run `dogma corpus sync` after restoring raw documents to rebuild
their conversions and search index.
