# dogma

A local-first practice platform for ML / DS / DL / RL / LLM engineering. Practise with
verified Python and SQL problems, lessons, daily study sessions, spaced review and a tutor that
runs on local models. The CLI and browser share workspace files and progress.

**Status:** the 13-week core path is written and reviewed. Timed re-solves, interview readiness
and mock interviews are available; see the [roadmap](docs/ROADMAP.md) for follow-ups and the
planned Library refresh.

## Requirements

- macOS. Run and Submit execute your code under the system sandbox (`sandbox-exec`), with no
  network and writes confined to the attempt directory; on other systems the grader refuses to
  run rather than run unsandboxed. To try dogma without installing anything, use the
  interactive demo, which grades in the browser. Its link is the website in the About box of
  this repository's page.
- Python 3.12, managed by `uv` (see `.python-version`).
- For the browser: Node.js 22 or newer and pnpm 12.8.1, as specified in
  [web/package.json](web/package.json).
- For development checks: `make` and the Python and browser dependencies below.

Run commands from the repository root. Initial dependency installation needs network access;
practice works offline once the dependencies and browser assets are installed. A local model
server is optional.

## Practise now

```bash
uv sync
uv run --frozen dogma today  # today's next session and first action
uv run --frozen dogma path   # this week's topics
```

Read the topic's `lesson.md`, then open its core problem:

```bash
uv run --frozen dogma open   cml/vector-operations/p01-vector-projection
# edit workspace/cml/vector-operations/p01-vector-projection.py, then:
uv run --frozen dogma run    cml/vector-operations/p01-vector-projection
uv run --frozen dogma submit cml/vector-operations/p01-vector-projection
```

`run` checks public tests; `submit` also checks hidden tests. A passing submission of the core
problem marks the topic solved. `open` preserves an existing workspace file; `open --fresh`
keeps a copy under `var/workspace-history/` before restoring the starter.

```bash
uv run --frozen dogma close              # save a transition note; enrol a Learn topic's cards
uv run --frozen dogma review             # reveal due cards and rate recall from 1 to 4
uv run --frozen dogma progress           # solved topics and recorded attempts
uv run --frozen dogma backup             # export progress to var/backups/
uv run --frozen dogma today --energy low # a shorter session with familiar material
```

Study order and dates live in [curriculum/paths/core.yaml](curriculum/paths/core.yaml).
After a break, `uv run --frozen dogma warp` moves the calendar so this week is the week of your
furthest exercise. It shows the new dates first and asks. `--to-week N`, `--ahead N`, `--back N`
and `--original` pick another week, and `dogma course reset` starts over after a backup. The
Progress page has the same controls.
Use `uv run --frozen dogma --help` or append `--help` to any command for its options.

## Use the browser

```bash
pnpm --dir web install --frozen-lockfile
pnpm --dir web build
uv run --frozen dogma serve
```

Open <http://127.0.0.1:8765>. A first visit opens a short introduction, which stays in the menu as
Start here. The browser provides Today, Problems, Review, Map and Progress pages.
The Problem page includes the lesson, statement, hints, editor, Run/Submit and results. Fonts,
math rendering and editor assets are bundled for offline use. The API also exposes its contract
at <http://127.0.0.1:8765/api/v1/openapi.json>. It listens only on loopback.

The interface uses the Memphis Study Club theme: neutral off-white surfaces, navy text,
vibrant coral and violet accents, and hand-drawn SVG figures on Today and Start here. Today and
exercise headers use subject-specific motifs; lessons and code use quiet white panels.
The interface uses locally bundled OpenDyslexic Bold. Rendered Markdown
uses Inter for reading alongside KaTeX formulas, with tables and separately displayed equations.
The code editor uses fixed-width JetBrains Mono. The light theme stays on regardless of the
operating system's color preference.

Run `pnpm --dir web build` after frontend changes, then restart `dogma serve` if the build did
not exist when the server started. For live reload, see the [developer guide](docs/DEVELOPMENT.md).

## Hints and local search

```bash
uv run --frozen dogma hint cml/vector-operations/p01-vector-projection --level 1
uv run --frozen dogma debug cml/vector-operations/p01-vector-projection
uv run --frozen dogma corpus sync --no-convert
uv run --frozen dogma corpus search "Adam bias correction"
```

Hint levels 1–3 show the authored hint even when the model is unavailable. Personalised nudges
and `debug` use a local OpenAI-compatible server, defaulting to
`http://127.0.0.1:8080/v1` with model `qwen3.8-27b-q6`. `debug` needs a recorded failing attempt.
The level-4 walkthrough unlocks after two failed submits or explicit confirmation.

`corpus sync --no-convert` indexes local notes, lessons and existing converted documents. Use
`corpus sync` to convert locally available PDFs as well; see [corpus/README.md](corpus/README.md).

## Configuration and development

Settings use `DOGMA_` environment variables or an optional gitignored `.env` file. Start with
[.env.example](.env.example); defaults work for ordinary local practice. The
[developer guide](docs/DEVELOPMENT.md) covers configuration, optional PyTorch support, frontend
development, checks and troubleshooting.

Your solutions live in `workspace/`; progress lives in `var/dogma.db`. Both directories are
gitignored. `dogma backup` saves an existing SQLite database before any schema upgrade, then
exports its current tables as JSON under `var/backups/`; keep a separate copy of your workspace
files too. Imported notes in `sources/` are committed after scrubbing, while personal material
in `private/` and third-party documents and indexes in `corpus/` stay local.

## Project documentation

| Read | For |
|---|---|
| [docs/DEVELOPMENT.md](docs/DEVELOPMENT.md) | Setup, configuration, development commands and troubleshooting |
| [docs/ROADMAP.md](docs/ROADMAP.md) | What's built, what's next, the session handoff log |
| [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) | Components, learning engine, budget, decisions |
| [docs/ENGINEERING_STANDARDS.md](docs/ENGINEERING_STANDARDS.md) | How code and content are written, tested and reviewed |
| [docs/CONTENT_SPEC.md](docs/CONTENT_SPEC.md) | Exact topic-package format for authors |
| [docs/LEARNER_PROFILE.md](docs/LEARNER_PROFILE.md) | General learning and interface design principles |
| [docs/IMPORT_PLAN.md](docs/IMPORT_PLAN.md) | What material was imported, and the privacy rules |

## License

The code is released under the [MIT License](LICENSE). The curriculum (lessons, problems, tests,
hints and cards) is not: it is all rights reserved, free to study with but not to republish. See
[LICENSE-CONTENT.md](LICENSE-CONTENT.md).
