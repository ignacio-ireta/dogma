# Import Plan: what comes into this repo, and how

*Status: approved. §3.A is done. The rest is increment **I-01** in [ROADMAP](ROADMAP.md).
Exact paths are in [`sources/manifest.yaml`](../sources/manifest.yaml); this document holds the
reasoning.*

The 2026-10-28 Library audit found an expansion and upstream directory reorganization after
this original import. Its additive refresh is planned in
[I-25a–f](superpowers/plans/2026-10-28-library-refresh.md), with an explicit checksum inventory
and topic map. That plan preserves existing imported paths and citations. The refresh's 36 notes
and 72 PDFs are imported (section 6, increments I-25a and I-25g), and its authoring is done.

---

## 1. Principles

1. **This project stands on its own.** Anything it needs is copied in. Nothing depends at
   runtime or build time on other projects on this machine, which may be temporary. **The only
   exception is the AI runtime:** model weights, the llama.cpp / Ollama servers and similar
   inference tools stay where they are and are reached over loopback HTTP.
2. **Imported material is committed, after scrubbing.** `sources/` is the canonical copy of
   every seed note, drill and dataset we use. Once imported, the originals can disappear.
   `sources/manifest.lock.json` records provenance: origin path, origin commit, the sha256
   before and after scrubbing, and whether the file was edited.
3. **No PII, and nothing from the learner's employer or clients, anywhere in the repo.** That
   means no names, contact details, locations, health information, employer, client or project
   names, colleagues, or interview histories. Files that are mostly personal are not imported.
   Files that are mostly about the subject get scrubbed. A denylist audit (`make audit`) enforces
   this on every check.
4. **Licensed third-party works come in too, but stay out of git.** The paper PDFs are copied into
   `corpus/raw/` and converted into `corpus/md/` inside the repo directory, but gitignored:
   they're copyrighted and large (about 950 MB). Back up `corpus/raw/` with the rest of the
   machine; the manifest says how to rebuild it.
5. **Tools we depend on are vendored.** docmd (MIT) is copied into `tools/docmd/` with its
   license and provenance, and runs as its own uv project.
6. **Sources are seeds, not content.** Nothing in `sources/` is shown to the learner directly.
   Items are rewritten into topic packages in `content/` ([CONTENT_SPEC](CONTENT_SPEC.md)),
   where they get tests, citations and review.
7. **Original authorship.** From deep-ml we take only the **topic outline**. Problems,
   solutions, tests and explanations are written here and checked against independent
   library implementations.

## 2. Classes

| Class | Meaning | Where it goes | In git? |
|---|---|---|---|
| `own` | Self-authored material about the subject | `sources/<id>/`, scrubbed | **yes** |
| `excluded-personal` | Mostly about the learner's life, work, employer or interviews | not imported | — |
| `third-party` | Copyrighted works (papers, books) | `corpus/raw/` (+ `corpus/md/`) | no (gitignored) |
| `vendored` | Third-party code we run | `tools/<name>/`, license kept | yes |
| `client` | Employer / client engagement material | **never imported or indexed** | — |

**Scrub rules** (applied to every `own` file during import):

1. Delete passages that narrate the learner's own projects, employers, clients, colleagues,
   interviews, schedule, health or personal life.
2. In otherwise generic text, rewrite incidental mentions neutrally ("<name>'s churn project at <employer>" →
   "a churn project"), or delete them.
3. Rename files whose names carry personal context. The manifest's `rename` regex strips the
   ` - <context>` suffix from `* Interview Bridge - <context>.md` files.
4. Done means `make audit` reports **zero** denylist hits across `sources/`, `content/`, `docs/`,
   `src/`, `scripts/`, `templates/` and `tests/`. The denylist lives in `private/denylist.txt`,
   gitignored, because the terms themselves are personal. The audit requires a nonempty local
   rules file and fails closed if it is missing or unreadable. It checks filenames, file contents,
   public assets, the browser build and publication metadata. Diagnostics identify private rules
   by number and redact matching filenames instead of printing the terms.
   Pre-commit also checks staged index bytes with `--staged`; no audit result covers old Git
   history. Required upstream attribution is retained, while illustrative metadata is generic.

## 3. Inventory and decisions

### A. deep-ml curriculum outline: ✅ done

`curriculum/syllabus.yaml`: 14 tracks, 538 topics, stable ids, all track counts checked, tiers
assigned (A 164 / B 221 / C 153). `curriculum/paths/core.yaml`: tier A in study order over 13
themed weeks plus an interview week.

### B. Library (Obsidian lineage vault)

| Item | Decision | Why |
|---|---|---|
| `AI-ML/**/*.md` (125 notes, including 11 "Interview Bridge" notes) | **import + scrub** → `sources/library/` | Lineage and "why it works" seeds for DL/LLM/optimizer topics. The bridges' generic interview reasoning is valuable once the personal stories are stripped |
| `Mathematics/**/*.md` (53), `LINEAGE.md`, `_MOC.md` files | **import + scrub** | Linear algebra, probability, information theory, optimization lineage. The prerequisite graph seeds `requires:` |
| `papers/*.pdf` (124 files, 938 MB) | **copy** → `corpus/raw/library-papers/` | third-party. Retrieval and page-level citations |
| Non-ML source documents outside the reviewed import set | **exclude** | Import only approved technical material; source paths and titles must also be suitable for publication |
| `scripts/arxiv-fetch.py` | **import** → `sources/library/scripts/` | May be reused to fetch papers for later tracks |

### C. Interview-prep repository

| Item | Decision | Why |
|---|---|---|
| `frameworks/` (case framework, model selection, Fermi estimates, consulting basics) | **import + scrub** | → `sys` track (system design & cases), model-selection "when to use" sections |
| `lessons/ref-failure-modes.md`, `ref-model-selection.md`, `ref-rass.md`, `ref-three-phrases.md` | **import + scrub** | Failure-mode table → "Where it breaks" sections |
| Narrative and dated-schedule lessons | **exclude** | Use an explicit allowlist of the four reviewed reference lessons, without recording excluded personal filenames |
| `drills/` (pandas, probability, SQL lessons, 5 case scenarios, pandas notebook) | **import + scrub** | → sql/pandas problems, probability drills, case bank |
| `mock-interviews/SCORING_RUBRIC.md`, `INTERVIEWER_README.md`, `mock_assessment_v2*`, `pack-A…F/` | **import + scrub** | Rubric → mock scorer. Packs → mock bank |
| other `mock-interviews/` files, `*.zip` | **exclude** | Superseded or duplicate |
| `reference/` flashcards ×2, interview refresher, sklearn & syntax cheatsheets, business vocabulary | **import + scrub** | → `cards.yaml` seeds |
| Private teaching guides and research summaries | **exclude** | Keep private source material separate from the general [learning-design principles](LEARNER_PROFILE.md) |
| `reference/` webinar transcripts | **exclude** | Third-party transcript, not needed |
| `data/*.csv` + `generate_mock_data.py` | **import** | Synthetic SQL/pandas fixtures |
| Routine notes and archived engagement guides | **exclude** | Withdrawn after source review; `sources/withdrawn.yaml` prevents re-import |
| rest of `archive/`, `PLAN.md`, `*.apkg` | **exclude** | Out of scope or binary |

### D. Multivariate statistics notebooks

| Item | Decision | Why |
|---|---|---|
| `tools/build_*.py`, `tools/nbbuild.py`, `*.ipynb`, `mvslib/*.py` | **import + scrub** | Regression, PCA, FA, clustering, LDA/logistic. The `selfcheck` pattern and semantic palette get ported into the harness and design tokens |
| `data/*.csv` (penguins, wine, diabetes) + `DATA_SOURCES.md` | **import** | Public datasets (licenses listed in `DATA_SOURCES.md`) |
| `README.md`, `METHOD_COMPARISON.md` | **import + scrub** | The 5-stage loop and method map |

### E. Study-material sources

| Item | Decision | Why |
|---|---|---|
| Multivariate-statistics daily lessons (15, with answers) | **import + scrub** | Matrix algebra → MLE problem seeds |
| Calculus study notes (4) | **import + scrub** | Derivatives, chain rule |
| RAG, prompt-flow, fine-tuning lessons + answers | **import + scrub** | Light LLM seeds |
| Differential-equations course material, other exam-specific files | **exclude** | Not needed |

### F. Earlier visualization site

| Item | Decision | Why |
|---|---|---|
| 6 explainer pages (`ml-metrics`, `ml-concepts`, `model-selection`, `probability-stats`, `sql-concepts`, `data-wrangling`) | **import + scrub** → `sources/legacy-viz/` | Widgets get ported into `web/` per topic |
| `case-frameworks.html`, `index.html`, nested duplicates | **exclude** | Case content is covered by C. Duplicates |

### G. Learning-design guidelines

[`docs/LEARNER_PROFILE.md`](LEARNER_PROFILE.md) specifies general product behavior, neutral
examples and accessible learning flows. It contains no individual biography or profile.
Private teaching notes are not imported.

### H. docmd tooling: vendored

| Item | Decision |
|---|---|
| `tools/document_conversion` (docmd 0.3.0, MIT, origin commit `127aa5d`): `src/`, `tests/` (minus generated `fixtures/_build/`), `docs/`, `pyproject.toml`, `uv.lock`, `LICENSE`, `README.md`, `CHANGELOG.md`, `CONTRIBUTING.md`, `.gitignore` | **vendor** → `tools/docmd/` + `VENDORED.md` (origin, commit, date, local changes). Runs as its own uv project: `uv run --project tools/docmd docmd …` |
| the workbench-specific `*_USAGE.md` file and **everything else in that workbench** | **never touch** (`client`) |

### I. AI runtime: referenced, not copied (the allowed exception)

Model weights on the external SSD, the llama-server router at `127.0.0.1:8080/v1` (default model
`qwen3.8-27b-q6`), and any Ollama or llama.cpp installs. The app reaches them only through the
loopback gateway (ADR-0008) and works without them, minus the tutor.

## 4. Execution (increment I-01)

1. Write `private/denylist.txt` and `private/import_roots.yaml` (origin paths). ✅ Done in planning.
2. `scripts/import_sources.py`: copy per manifest (into `sources/<id>/`, `corpus/raw/<id>/`,
   `tools/docmd/`), rename per scrub rule 3, write `sources/manifest.lock.json`.
3. Scrub every `own` file per §2, review each denylist hit by hand, and record `scrubbed: true`
   plus the post-scrub sha256 in the lock.
4. `scripts/audit_privacy.py` + `make audit`: zero hits required.
5. Spot-check: lock hashes match the files on disk, `uv run --project tools/docmd docmd --help`
   works, and the docmd test suite passes.

Footprint: `sources/` about 8 MB (committed), `tools/docmd/` about 1 MB (committed),
`corpus/raw/` about 950 MB (gitignored).

## 5. Explicitly not doing

- Scraping or copying deep-ml problems, solutions, tests or editorial content.
- Committing PDFs, model weights, Anki binaries or course videos.
- Reading anything from employer or client workspaces beyond the vendored docmd code.
- Importing personal stories, schedules, health, contact or location information.

## 6. Library refresh source (I-25a)

Source id `library-refresh`, described by the frozen inventory
[`source-refresh-2026-10-28.json`](source-refresh-2026-10-28.json): 108 items with
Library-relative paths, 36 notes and 72 PDFs.

- **Notes (snapshot):** one exact `import` rule per note, copied to
  `sources/library-refresh/<Library-relative path>`. Committed after the scrub below. The
  index reports them as `sources:library-refresh/<path>`, which is the citation id for later
  content (`source: library-refresh/<path>`).
- **PDFs (corpus):** exact `include:` paths, copied to `corpus/raw/library-refresh/<path>`.
  Gitignored, third-party. Their conversions are the corpus source ids
  `library-refresh/<path>.pdf.md`, cited as `corpus: library-refresh/<path>.pdf.md p.<page>`.
- The existing `library` and `library-papers` rules and their lock entries are unchanged. The
  refresh only adds lock entries.

Importer options (`scripts/import_sources.py`):

- `--source-id ID` (repeatable): process only that manifest entry. Unselected origin roots are
  not resolved or read, so a notes-only run does not need the PDF root.
- `--kind snapshot|corpus|vendor` (repeatable): restrict to entry kinds. An excluded entry named
  by `--source-id` is reported as `not requested`.
- `--inventory JSON`: before anything is written, every inventory item that a selected entry
  would import must exist under that entry's root as a regular file and match its `sha256`.
  Failures list relative paths only, exit nonzero, and write nothing.
- Corpus `include:`: exact relative paths instead of a `glob`. A missing listed file stops the
  run before any copy.
- `--relock` with a selector rewrites only the selected sources' lock entries.

Commands:

```
uv run --frozen python scripts/import_sources.py --source-id library-refresh --kind snapshot \
    --inventory docs/source-refresh-2026-10-28.json     # notes only
uv run --frozen python scripts/import_sources.py --source-id library-refresh --kind corpus \
    --inventory docs/source-refresh-2026-10-28.json     # PDFs, needs them on this machine
```

**Outcome where the PDFs are absent** (a Library git clone, which has none): the 36 notes import
and validate. The corpus part is not requested, so its 72 files are reported as not imported and
are not read. A run that includes the corpus part exits nonzero before any write, listing the 72
missing files by relative path.

**Outcome where the PDFs are present** (I-25g, 2026-10-31): all 72 matched the inventory and were
copied. The lock gained 72 entries and changed no existing entry, and a second run copied nothing.
Docmd treats a PDF with any text as converted, so a scan with a text cover page yields a
near-empty conversion and no failure. Check the size of each conversion after a sync.

**Scrub of the 36 notes** (`scrubbed: true` in the lock for all 36): vault navigation links
removed (8 lines); 105 relative links to material outside this import reduced to their link
text; 67 links to other imported refresh notes kept; "Local source/copy/text/paper" labels
reworded to "Full text (not imported)" (6); acquisition narration removed: two sentences that
named where the copies were obtained, and one "downloaded draft" phrase; "local" used
for provenance (for example, "local copy", "local PDF") reworded, about 20 places, while
mathematical uses of "local" are unchanged. No personal, employer or client content was found.

**Disposition of the other PDFs:** the 82 catalog PDFs without a lock checksum include 11 older
or alternate copies. Goldberg is selected; the other 10 stay outside this allowlist and are
not copied or indexed. They need separate review before any import.
