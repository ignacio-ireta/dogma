# Extraction contract (v0.2)

What each format supports, what is best-effort, and what is out of scope.
"Extract document to Markdown" sounds simple; this page is the authoritative
answer to the ambiguity.

## Output shape (all formats)

```markdown
---
source_file: "<relative path>"
source_type: "docx|pptx|xlsx|pdf"
extraction_timestamp: "<ISO 8601>"        # omitted in --deterministic mode
extractor_version: "<extractor>/<semver>"
status: "success" | "warning"
---

# <source filename>

## Document metadata          # only when properties exist (Title/Author/Created/...)

## Content

### <section label>           # Slide N[: title] / Sheet: name / Page N /
                              # Speaker notes (slide N); body content renders bare

## Extraction warnings        # only when warnings exist; one bullet per warning
```

- Content headings are shifted +2 (an H1 in a DOCX renders as `###`) so they
  nest under `## Content`; levels cap at `######`.
- Tables render as GFM pipe tables; `|` is escaped, embedded newlines become
  `<br>`; ragged rows are padded; headerless tables get synthesized
  `Column N` headers.
- Paragraph text starting with a Markdown structural marker (`#`, `-`, `*`,
  `+`, `>`, `1.`) is backslash-escaped; mid-text characters are left readable.
- Output is deterministic: same input + same extractor version -> same bytes,
  except `extraction_timestamp`/`created_at` (suppressed by `--deterministic`).
- Identical runs are idempotent; outputs are written atomically.

## DOCX (python-docx)

| Element | Support |
|---|---|
| Paragraphs, headings (Heading 1-9, Title) | Supported (block order preserved, incl. tables interleaved) |
| Tables | Supported; first row treated as header; merged cells repeat text |
| Lists (List Bullet/Number styles) | Supported, incl. nesting via style suffix or numbering XML |
| Hyperlinks | Supported, rendered inline as `[text](url)` |
| Inline images | Metadata only (`[image: name (content-type)]`) |
| Footnotes | **Not extracted**; presence flagged (`FOOTNOTES_NOT_EXTRACTED`) |
| Comments | **Not extracted**; presence flagged |
| Tracked changes | **Not extracted**; markup ignored, presence flagged |
| Bold/italic inline styling | Dropped (plain text IR in v0.1) |
| Non-English style names | Heading/list detection may miss custom-localized styles |
| Password-protected (OLE) | Fails safely: `PASSWORD_PROTECTED` |

## PPTX (python-pptx)

| Element | Support |
|---|---|
| Slide titles | Section label (`### Slide N: title`) |
| Body/content placeholder text | List items with indent levels |
| Other text boxes | Paragraphs |
| Grouped shapes | Recursed into (text preserved) |
| Tables | Supported |
| Speaker notes | Separate `Speaker notes (slide N)` section |
| Pictures | Metadata only (filename, content type, alt text when present) |
| Reading order | XML (z-)order — usually authoring order, **not guaranteed visual order** |
| Charts/SmartArt/diagrams | Not extracted semantically |

## XLSX (openpyxl)

| Element | Support |
|---|---|
| Sheets | One section per sheet (`### Sheet: name`), in workbook order |
| Cell values | Cached computed values; dates ISO-formatted, booleans TRUE/FALSE |
| Formulas | Formula text shown: `42 (=SUM(B2:B3))`; if the file has no cached value (never saved by a calculating app), the formula alone is shown + `FORMULA_NO_CACHED_VALUE` warning |
| Merged cells | Anchor cell carries the value; other cells empty |
| Large sheets | Capped at `DOCMD_MAX_ROWS_PER_SHEET` (10000) x `DOCMD_MAX_COLS_PER_SHEET` (256); truncation flagged (`SHEET_TRUNCATED`) |
| Trailing empty rows/columns | Trimmed (interior gaps kept) |
| `Modified` property | Omitted (writer libraries reset it at save time) |
| Pivot tables, charts | Not extracted |
| `.xlsm` (macro-enabled) | Skipped entirely (security posture; macros never execute) |

## PDF (PyMuPDF — AGPL, see README)

| Element | Support |
|---|---|
| Embedded text | Supported, one section per page |
| Headings | **Heuristic** from font-size ratio vs the page's dominant body size |
| Tables | Detected via `find_tables()`; reliable for ruled tables, borderless tables are best-effort; detected table regions are excluded from the text pass |
| Reading order | Blocks sorted by vertical position |
| Scanned pages (OCR off, the default) | `SCANNED_PAGE` warning; an entirely image-only document fails with `SCANNED_PDF_OCR_DISABLED` |
| Scanned pages (`--ocr auto`) | Pages without a text layer are rendered and recognized via Apple Vision (macOS + ocr extra); recognized text lands as plain paragraphs with an `OCR_APPLIED` warning. If nothing is recognized anywhere in an image-only document, fails `OCR_NO_TEXT` |
| `--ocr force` | Every page is OCR'd and any embedded text layer is ignored (for garbled text); no table detection or heading inference happens on OCR'd pages |
| OCR'd page structure | **Paragraphs only** — Vision reports no reliable font sizes or table structure; paragraph splits come from vertical line gaps |
| OCR determinism | Stable for one machine/macOS version; may vary across macOS versions, so OCR'd outputs are excluded from byte-level golden guarantees |
| Encrypted PDFs | Empty user password tried; otherwise fails `PASSWORD_PROTECTED` |
| Non-PDF content under .pdf name | Rejected (`CORRUPT_FILE`) even though MuPDF could render some formats |
| Images | Not extracted (text only in v0.1) |

## Error codes

| Code | Meaning |
|---|---|
| `UNSUPPORTED_TYPE` | No extractor for the file type |
| `CORRUPT_FILE` | Parser rejected the file (bad zip, invalid package, broken PDF) |
| `PASSWORD_PROTECTED` | Encrypted Office (OLE container) or PDF |
| `SCANNED_PDF_OCR_DISABLED` | Image-only PDF and OCR is disabled (re-run with `--ocr auto`) |
| `OCR_NO_TEXT` | OCR ran but found no recognizable text in an image-only PDF |
| `OCR_FAILED` | The OCR engine raised while recognizing a page |
| `OUTPUT_COLLISION` | Two outputs collide under casefolding |
| `OUTPUT_WRITE_FAILED` | Filesystem error writing output |
| `EXTRACTION_FAILED` | Unexpected parser failure (wrapped) |

## Warning codes

`FOOTNOTES_NOT_EXTRACTED`, `COMMENTS_NOT_EXTRACTED`,
`TRACK_CHANGES_NOT_EXTRACTED`, `FORMULA_NO_CACHED_VALUE`, `SHEET_TRUNCATED`,
`SCANNED_PAGE`, `TABLE_DETECTION_FAILED`, `DETECTION_MISMATCH`,
`OCR_APPLIED` (page was recognized via OCR), `OCR_LOW_CONFIDENCE` (mean
recognition confidence below 0.5).
