# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/) and the project adheres to
semantic versioning.

## [0.3.0] - 2026-07-02

### Added

- `docmd sync` — tracked repo conversion. A synced folder becomes a known
  repo (`registry.json`), each run diffs the source tree against the last
  synced snapshot (added/updated/unchanged/moved/deleted), moved files
  relocate their existing `.md` + sidecar instead of re-extracting, every
  superseded conversion is archived under `.docmd/archive/<repo>/<run_id>/`,
  and each run writes a `changes.json` change report. Re-running the same
  sync resumes an interrupted one — there is no `--resume` flag.
- `docmd repos` — read-only views over tracked state: `list` (registry),
  `show --files` (per-repo snapshot), and `history` (per-run actions with
  archive locations, filterable by `--path` and `--run`).
- Sync safety guards: deletions above `max(10, tracked/4)` files are skipped
  until confirmed with `--confirm-deletes` (`DOCMD_CONFIRM_DELETES`); a scan
  depth change skips deletions and re-converts moves as adds for that run;
  an empty source tree on a known repo never archives anything (exit 4).
- Per-repo single-writer lockfiles under `.docmd/locks/`: a concurrent sync
  of the same repo exits 2 with `REPO_LOCKED`; stale locks from dead local
  processes are replaced automatically. The registry itself is updated under
  a short-lived shared lock so different repos can sync concurrently.
- Archive retention: `--archive-keep-runs N` / `DOCMD_ARCHIVE_KEEP_RUNS`
  keeps only the newest N archived runs per repo (default 0 = keep forever).
- Sync hashes every file once up front and the pipeline reuses that hash for
  skip-unchanged checks and sidecars instead of re-hashing.
- New error codes: `REPO_NAME_RESERVED` (exit 2; source folder name collides
  with docmd's own output structure), `REPO_LOCKED` (exit 2),
  `REGISTRY_CORRUPT` (exit 3; fails loudly rather than orphaning every
  tracked repo), and `SNAPSHOT_CORRUPT` (recovered by treating the repo as
  new), plus per-file `HASH_FAILED`, `ARCHIVE_FAILED`, and `RELOCATE_FAILED`
  codes in run reports.

### Fixed

- `scan --resume` no longer resumes against the latest manifest when it was
  produced by a different command or a different source tree (the runs
  directory is shared with `sync`); such manifests are now ignored.

## [0.2.0] - 2026-06-11

### Added

- Opt-in OCR for scanned PDFs via Apple's on-device Vision framework
  (macOS-only, `uv sync --extra ocr`): `--ocr auto` recognizes pages without
  a text layer, `--ocr force` OCRs every page; `--ocr-lang` and `--ocr-dpi`
  tune recognition (also `DOCMD_OCR*` env vars). OCR'd pages emit plain
  paragraphs and are flagged `OCR_APPLIED` (plus `OCR_LOW_CONFIDENCE` when
  mean confidence is below 0.5).
- New error codes: `OCR_NO_TEXT` (image-only PDF where OCR recognized
  nothing) and `OCR_FAILED` (engine failure on a page). Requesting OCR where
  no backend can run exits 2 with an install hint.
- CI: macOS job running the suite against the real Vision engine.

### Changed

- PDF extractor version bumped to 0.2.0; the `SCANNED_PDF_OCR_DISABLED`
  message now points at `--ocr auto`.

### Fixed

- PDF tables could silently disappear when several PDFs were processed
  concurrently: pymupdf's table module keeps its working state in
  module-level globals, so parallel `find_tables()` calls clobbered each
  other. Table detection is now serialized across worker threads (text
  extraction and OCR stay parallel). Latent since 0.1.0; surfaced by the
  v0.2 multi-worker determinism check.

### Removed

- The unused `pytesseract`/`pillow`/`pdf2image` optional extra (replaced by
  `ocrmac`).

## [0.1.0] - 2026-06-10

Initial release.

### Added

- `docmd scan` — convert every supported file under a folder (optionally
  recursive) with a bounded parallel worker pool, progress bar, and
  spec-formatted run summary.
- `docmd process` — convert a single file, with `--steps` to run a subset of
  the pipeline (extract/normalize/write).
- Extractors for DOCX (interleaved block order, lists, tables, hyperlinks,
  image metadata), PPTX (slides, group-shape recursion, speaker notes,
  tables), XLSX (cached values + formula text, merged cells, sheet caps), and
  PDF (per-page text, ruled-table detection, font-size heading inference,
  scanned-page detection, empty-password retry on encrypted files).
- Shared intermediate document model (pydantic discriminated block union) and
  one Markdown writer: YAML frontmatter, section labels, GFM tables, escaping.
- Structured error hierarchy with stable error codes; corrupt/encrypted/
  scanned/misnamed files fail per-file without killing the run.
- Run reports: `manifest.json` (write-through, atomic), `errors.json`,
  `run.log` under `<out>/runs/<run_id>/`; sidecar `.md.meta.json` per output.
- Idempotency: sha256-based skip-unchanged, `--force`, `--resume`, atomic
  output writes, deterministic mode (`--deterministic` / `DOCMD_DETERMINISTIC`).
- Graceful Ctrl+C with resume hint (exit 130); second Ctrl+C force-quits.
- Pre-flight casefold collision check for output paths.
- Configuration via `DOCMD_*` env vars / `.env`, overridden by CLI flags.
- Test suite: unit/integration/e2e/golden with a reproducible binary fixture
  generator (including corrupt and adversarial fixtures); CI on Python
  3.11/3.12/3.13 with lint, format, strict mypy, and an 85% coverage gate.
