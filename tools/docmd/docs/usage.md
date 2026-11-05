# Usage

## Install

```bash
uv sync            # dev setup from a checkout
uv run docmd version
```

For OCR of scanned PDFs (macOS only, uses Apple's on-device Vision framework):

```bash
uv sync --extra ocr
```

## Convert a folder

```bash
docmd scan ./documents --out ./markdown
docmd scan ./documents --out ./markdown --recursive
```

Outputs mirror the source tree with collision-proof names:

```
documents/report.docx      ->  markdown/report.docx.md
documents/report.pdf       ->  markdown/report.pdf.md
documents/q3/budget.xlsx   ->  markdown/q3/budget.xlsx.md
```

Each output gets a sidecar `*.md.meta.json` (source hash, extractor version,
pipeline steps, warnings) and each run writes a report under
`markdown/runs/<run_id>/` (`manifest.json`, `errors.json`, `run.log`).

## Convert a single file

```bash
docmd process ./file.docx --out ./markdown
docmd process ./file.pdf --out ./markdown --steps extract --steps normalize
```

## Sync a document repo

```bash
docmd sync ./onboarding-docs --out ./markdown
```

`sync` converts a folder the way `scan` does, but tracks it as a repo named
after its root folder (here `onboarding-docs`). Each run diffs the source
tree against the last synced state: new and changed files are converted,
moved files get their existing Markdown relocated (no re-extraction), deleted
and superseded outputs are archived instead of lost, and every action is
recorded in the run's `changes.json`. A `registry.json` at the output root
indexes the synced repos; one output root can hold any number of them:

```
<out>/
  registry.json                   # index of synced repos
  <repo>/                         # mirrored source tree: *.md + *.md.meta.json
  .docmd/
    repos/<repo>/snapshot.json    # per-repo diff baseline
    archive/<repo>/<run_id>/      # displaced outputs, grouped by run
  runs/<run_id>/
    changes.json                  # every sync action, with archived_to paths
    manifest.json / errors.json / run.log
```

Sync-specific flags:

| Flag | Effect |
|---|---|
| `--dry-run` | Classify and write `changes.json`; change nothing on disk |
| `--no-recursive` | Sync the top level only (unlike `scan`, sync is recursive by default; the `DOCMD_RECURSIVE` env var applies to `scan` only) |
| `--confirm-deletes` | Archive deletions past the mass-deletion safety threshold |
| `--archive-keep-runs N` | Keep only the newest N archived runs per repo (0 = keep forever) |

`--fail-fast`, `--workers`, `--deterministic`, the `--ocr*` flags,
`--log-level`, and `--no-progress` work exactly as in "Common flags" below.

Sync protects the archive from accidents: when more than `max(10, tracked/4)`
files disappear at once (an unmounted share, a wrong folder), the deletions
are skipped with a note until you re-run with `--confirm-deletes`; when the
scan depth changes (`--recursive` vs `--no-recursive`), deletions are skipped
and moves re-convert as adds for that run, because absence against a
different-depth snapshot proves nothing. Each repo is also locked for the
duration of a run — a second concurrent sync of the same repo exits 2 with
`REPO_LOCKED` (locks from dead processes are replaced automatically).

### Inspect tracked repos

```bash
docmd repos list --out ./markdown                 # every repo in the knowledge base
docmd repos show onboarding-docs --out ./markdown --files
docmd repos history onboarding-docs --out ./markdown --path guides/setup.docx
```

`history` answers "what went where": every add/update/move/delete per run,
newest first, with the archive location of any displaced conversion.

There is no `--resume` flag: re-running the same `sync` command resumes an
interrupted one — the diff is recomputed against what actually completed and
only unfinished work is redone. There is no `--force` either; to re-extract a
whole repo, run

```bash
docmd scan <source> --out <out>/<repo> --force
```

which rebuilds every output in place and stays consistent with the repo's
snapshot (the next sync still sees those files as unchanged). Note that this
`scan` writes its own run report under `<out>/<repo>/runs/` — inside the
repo's output tree; it is safe to delete that folder afterwards.

## Common flags

| Flag | Effect |
|---|---|
| `--recursive` | Descend into subdirectories |
| `--resume` | Re-attempt only files that failed or were interrupted in the last run |
| `--force` | Reprocess even when the output is current |
| `--fail-fast` | Stop the whole run on the first failure |
| `--workers N` | Bounded worker pool (1-32, default `min(4, cpus)`) |
| `--deterministic` | Omit timestamps so identical inputs give identical bytes |
| `--ocr auto` | OCR scanned PDF pages that have no text layer (macOS + ocr extra) |
| `--ocr force` | OCR every PDF page, ignoring embedded text (for garbled text layers) |
| `--ocr-lang en-US,de-DE` | Preferred OCR languages (default: auto-detect) |
| `--ocr-dpi N` | Page render resolution for OCR (72-600, default 300) |
| `--log-level debug` | Verbose logging (console + run.log) |
| `--no-progress` | Disable the progress bar |

All flags can also be set via environment variables (prefix `DOCMD_`, see
`.env.example`); CLI flags win over env vars.

## OCR for scanned PDFs

OCR is opt-in. Without it, image-only PDFs fail with
`SCANNED_PDF_OCR_DISABLED`; with `--ocr auto` only the pages lacking a text
layer are rendered and recognized:

```bash
docmd scan ./documents --out ./markdown --recursive --ocr auto
```

Requirements and caveats:

- macOS with the `ocr` extra installed (`uv sync --extra ocr`); on other
  platforms `--ocr` exits with code 2 and an explanatory message. Recognition
  runs entirely on-device — nothing leaves your machine.
- OCR'd pages produce plain paragraphs: no heading inference and no table
  detection (the engine reports no reliable font sizes or table structure).
- Each OCR'd page is flagged with an `OCR_APPLIED` warning in the output and
  sidecar; low-confidence pages also get `OCR_LOW_CONFIDENCE`.
- Results are stable on one machine but may differ across macOS versions, so
  OCR'd outputs are not byte-reproducible across machines.

## Re-runs and resume

Plain re-runs are incremental: files whose content hash matches an existing
output are skipped. Changed files are reprocessed. After Ctrl+C or a crash:

```bash
docmd scan ./documents --out ./markdown --recursive --resume
```

## Exit codes

| Code | Meaning |
|---|---|
| 0 | All files succeeded (possibly with warnings) or were skipped as unchanged |
| 1 | Run completed but at least one file failed |
| 2 | Usage or configuration error, including a reserved repo name (`REPO_NAME_RESERVED`) or an already-running sync of the same repo (`REPO_LOCKED`) |
| 3 | Fatal environment error (input missing, output unwritable, corrupt registry for `sync`) |
| 4 | No supported files found; a `sync` of a known repo stops here and does NOT process deletions |
| 130 | Interrupted by Ctrl+C after graceful shutdown |
