# Architecture

## Data flow

```
File discovery        discovery/scanner.py, discovery/file_types.py
      |
Format-specific       extractors/{docx,pptx,xlsx,pdf}_extractor.py
extraction            -> DocumentIR + ExtractionWarnings
      |
Intermediate model    types.py (pydantic, discriminated block union)
      |
Normalization         markdown/normalizer.py (pure IR -> IR text hygiene)
      |
Markdown rendering    markdown/writer.py + markdown/metadata.py
      |
Output writing        atomic write + sidecar .meta.json (fsio.py)
      |
Run state/reporting   pipeline/state.py (manifest, errors.json), run.log
```

No extractor writes Markdown directly. Every format normalizes to the same
`DocumentIR` (headings, paragraphs, list items, tables, image refs, code
blocks, organized into body/slide/sheet/page/notes sections), and one writer
renders it. This keeps extraction bugs separable from rendering bugs, makes
block-level testing cheap, and leaves room for new output writers (JSON, etc.)
without touching extractors.

## Key components

| Component | Responsibility |
|---|---|
| `discovery/file_types.py` | Extension map + magic-byte sniffing (zip/PDF/OLE); macro files skipped |
| `discovery/scanner.py` | Deterministic (sorted) walk; hidden/lock-file/output-dir exclusion |
| `extractors/base.py` | `Extractor` ABC; per-extractor version; `ExtractionResult` with warnings |
| `errors.py` | Domain exception hierarchy; `wrap_library_error` maps parser exceptions |
| `markdown/normalizer.py` | NFC, control chars, whitespace, empty-block dropping |
| `markdown/writer.py` | Output contract (frontmatter, section labels, GFM tables, escaping) |
| `pipeline/steps.py` | extract/normalize/write steps over a `FileContext`; output naming |
| `pipeline/state.py` | Manifest schema, thread-safe atomic write-through store |
| `pipeline/runner.py` | Bounded thread pool, failure isolation, SIGINT protocol, skip-unchanged |
| `tracking/registry.py` | `registry.json` repo index; reserved-name validation; corrupt registry fails loudly |
| `tracking/snapshot.py` | Per-repo diff baseline (source relpath -> sha256/output); folds in only completed work |
| `tracking/diff.py` | Pure tree-vs-snapshot classification: added/updated/unchanged/moved/deleted |
| `tracking/archive.py` + `relocate.py` | Archive displaced outputs (move/copy); relocate moved conversions with `source_file` rewrite |
| `tracking/lock.py` | Single-writer lockfiles: per-repo for a whole run, `_registry` for the upsert |
| `tracking/sync.py` | Sync orchestrator: hash pass, guards, collision preflight, ordered execution, changes.json |
| `cli.py` + `cli_repos.py` | typer commands: `scan`, `process`, `sync`, `repos list/show/history`, `version` |

## Concurrency model

A bounded `ThreadPoolExecutor` (default `min(4, cpu)`, `--workers 1..32`).
Threads were chosen over processes for v0.1: results stay in-process (no
pickling), cancellation and progress stay simple, and pymupdf/lxml release the
GIL for meaningful stretches. Pure-Python parsing limits speedup — this is a
known ceiling, and worker records are picklable pydantic models, so a
`ProcessPoolExecutor` is a drop-in future change.

## Interruption protocol

First Ctrl+C sets a cancellation token: no new files start, in-flight workers
hit a checkpoint between steps and finish as `interrupted`, the manifest is
finalized, and a resume hint is printed (exit 130). A second Ctrl+C restores
default signal behavior (hard stop). The manifest is written through
atomically on every completion, so even a hard kill leaves a resumable state.

## Idempotency

Two independent layers:

1. **Sidecar skip-unchanged** — before processing, if `<out>.md` and its
   `.meta.json` exist and the sidecar's `source_sha256` matches the current
   file hash, the file is skipped (`skipped/unchanged`). Works without any
   manifest. `--force` defeats it.
2. **Manifest resume** (`--resume`) — loads the latest run manifest under
   `<out>/runs/` and re-attempts only files that did not complete
   (failed/interrupted/pending).

Output writes are atomic (temp file in the destination directory +
`os.replace`), so readers never observe half-written files.

## Repo tracking (sync)

`docmd sync` treats a source folder as a repo (identity = root folder name)
and converts only what changed since the last sync:

```
Registry lookup       tracking/registry.py: is this a known repo?
      |
Snapshot load         tracking/snapshot.py: last synced relpath -> sha256/output
      |
Hash pass             sha256 every discovered source file up front
      |
Classify              tracking/diff.py: added/updated/unchanged/moved/deleted
      |
Execute               tracking/sync.py: relocate, archive, convert, persist
```

**Move matching.** Paths present in both trees resolve first (unchanged or
updated), so content landing on a still-existing path can never be mistaken
for a move. The remaining deleted/added candidates are matched strictly
one-to-one by sha256: a hash pairs an old path with a new path only when it
identifies exactly one candidate on each side. Duplicate content makes
intent undecidable, so ambiguous sets fall back to delete+add — the archive
preserves the deleted outputs and re-extraction costs the same as a fresh
add. A matched move relocates the existing `.md` + sidecar and rewrites
their `source_file` references; no re-extraction.

**Crash-safety ordering.** Relocations and archiving run before any
conversion. Deleted outputs are MOVED into the archive (the source is gone,
nothing stays live); prior versions of updated outputs are COPIED, so a
failed conversion never destroys the only good output. Added and updated
files then convert through the normal pipeline, `changes.json` is written,
and the snapshot is written last — reflecting only completed work — before
the registry upsert. There is no `--resume` flag: re-running sync IS resume.
The diff is recomputed against the (possibly partially advanced) snapshot,
and every disk mutation is atomic or idempotent under re-classification. An
empty source tree on a known repo exits 4 without archiving anything — an
unmounted or mistyped directory must not read as whole-repo deletion.

**On-disk layout** under the output root:

```
<out>/registry.json                        repo index (corrupt = fatal)
<out>/<repo>/<mirrored tree>.md            live conversions (+ .md.meta.json)
<out>/.docmd/repos/<repo>/snapshot.json    per-repo diff baseline
<out>/.docmd/archive/<repo>/<run_id>/      every displaced conversion
<out>/runs/<run_id>/changes.json           per-run change report
```

`.docmd` is hidden, so archived Markdown can never be re-ingested by a scan
of the output root. Directories emptied by relocations and deletions are
pruned.

**Guards and concurrency.** Deletion is the only destructive inference, so
it is doubly guarded: a mass-deletion threshold (`max(10, tracked/4)` files,
overridable with `--confirm-deletes`) and a scan-depth check (a snapshot
built at a different `--recursive` depth cannot prove absence — deletions
are skipped and moves re-convert as adds for that run; the snapshot records
the depth that wrote it). Each repo takes a single-writer lockfile
(`tracking/lock.py`, `.docmd/locks/<repo>.lock`) for the whole run; stale
locks from dead local pids are replaced. The registry upsert re-reads
`registry.json` under a separate short-lived `_registry` lock, so syncs of
different repos run concurrently without dropping each other's entries.
Archive retention is opt-in: `--archive-keep-runs N` prunes a repo's oldest
archived runs beyond N at the end of every non-interrupted sync (individual
file failures do not block pruning).

## Licensing note

The PDF backend (PyMuPDF) is AGPL-3.0. docmd's own code is MIT. For
local/internal use this has no effect; distributing docmd as a combined work
has copyleft implications. The `Extractor` interface keeps a swap to an
MIT-licensed backend (e.g. pdfplumber) cheap.
