# Troubleshooting

## A file failed — where do I look?

Every run writes a report: `<out>/runs/<run_id>/`:

- `errors.json` — machine-readable list of failed/interrupted files with error codes
- `manifest.json` — status, hash, timing, and output path for every file
- `run.log` — full log (DEBUG level lands here even when the console is INFO)

Re-run a single file with debug logging:

```bash
docmd process ./bad_file.docx --out /tmp/probe --log-level debug
```

## Common failures

| Symptom / code | Cause | What to do |
|---|---|---|
| `PASSWORD_PROTECTED` | Encrypted Office file (OLE container) or PDF | Decrypt with the owning application, re-save, re-run |
| `SCANNED_PDF_OCR_DISABLED` | PDF has no text layer (scan) and OCR is off | Re-run with `--ocr auto` (macOS + ocr extra) |
| `OCR_NO_TEXT` | OCR ran but recognized nothing (blank scan, photo, very low quality) | Try `--ocr-dpi 600`; if it stays empty the source has no readable text |
| `OCR_FAILED` | The OCR engine raised on a page | Check `detail` in errors.json; an invalid `--ocr-lang` code is the usual cause |
| "OCR was requested but no backend is available" (exit 2) | Not on macOS, or the ocr extra is missing | `uv sync --extra ocr` on macOS; other platforms have no OCR backend in v0.2 |
| `CORRUPT_FILE` | Truncated/invalid file, or a misnamed file (e.g. DOCX bytes named `.pdf`) | Check the `detail` field in errors.json; fix the extension or re-export |
| `OUTPUT_COLLISION` | Two sources map to the same output path under casefolding | Rename one source |
| `SHEET_TRUNCATED` warning | Sheet exceeds row/col caps | Raise `DOCMD_MAX_ROWS_PER_SHEET` / `DOCMD_MAX_COLS_PER_SHEET` |
| `FORMULA_NO_CACHED_VALUE` warning | Workbook was never saved by a calculating app | Open and save in Excel/LibreOffice to cache values |
| Files silently missing from output | Hidden files, `~$` lock files, macro-enabled (`.docm/.xlsm/.pptm`) are skipped by design | See `manifest.json` skip reasons |
| Exit 4 | No supported files under the source path | Check `--recursive` and the extension list |

## Re-runs do nothing

That is skip-unchanged working: the source hash matches an existing output's
sidecar. Use `--force` to rebuild everything.

## Run was interrupted

```bash
docmd scan <source> --out <out> [--recursive] --resume
```

Only failed/interrupted/unfinished files are re-attempted.

An interrupted `sync` has no `--resume` flag: re-run the same `docmd sync`
command and it picks up where it stopped.

## Sync treats my repo as new

Repo identity is the source root folder name. Renaming the folder creates a
new identity by design — the next sync starts a fresh repo under the new name,
and the old repo's outputs stay under `<out>/<old-name>` until you remove
them. Syncing the same folder from a new location is fine; the path change is
just noted in `changes.json` and the registry.

The other cause is a corrupt snapshot: sync recovers by treating the repo as
new and rebuilds the snapshot in one run. Check the `notes` array in
`<out>/runs/<run_id>/changes.json` — a corrupt snapshot is reported there.

## REGISTRY_CORRUPT (exit 3)

`<out>/registry.json` is unreadable. It is never rebuilt automatically:
silently recreating it would orphan every repo's snapshot, archive, and
history. Restore the file from a backup, or delete it deliberately if you
accept starting a fresh knowledge base.

## REPO_LOCKED (exit 2)

Another sync of the same repo is running — each repo takes a single-writer
lock under `<out>/.docmd/locks/` for the duration of its run. Wait for it to
finish. Locks left by dead processes on the same machine are detected and
replaced automatically; a lock from another machine cannot be probed, so if
you are certain that run is gone, delete the `.lock` file the message names.

## Sync skipped my deletions

Two guards refuse to archive outputs when absence is not trustworthy, and
both say so in the run's `changes.json` notes: more than `max(10, tracked/4)`
files vanished at once (re-run with `--confirm-deletes` if that was
intentional), or the scan depth changed between runs (`--recursive` vs
`--no-recursive`) — re-run at the same depth and the next diff reconciles
deletions normally.

## Where did an old output go?

Superseded conversions are archived under
`<out>/.docmd/archive/<repo>/<run_id>/<original output relpath>`, grouped by
the run that displaced them. The `archived_to` field of the matching entry in
that run's `changes.json` points at the exact file. Deletions move the
`.md` + sidecar pair there; updates archive a copy before re-converting.
Clean moves archive nothing — the existing output is relocated to the new
path.

## A move showed up as delete + add

Moves are matched by content hash, strictly one-to-one. When several files
share identical content the original path is undecidable, so sync falls back
to delete + add rather than guess. Nothing is lost: the deleted outputs are
in the archive and the add is re-extracted like any new file.

## Output looks wrong (tables, headings)

PDF headings are inferred from font sizes and PDF tables are detected
heuristically — see docs/extraction_contract.md for the documented limits.
PPTX shape order is XML order, which can differ from visual order.

## Performance

- Increase `--workers` (bounded 1-32); pure-Python parsing limits scaling.
- Huge XLSX files stream in read-only mode and are row/column-capped.
- Re-runs are incremental by design; only changed files cost time.
