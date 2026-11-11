# Development

Read [ENGINEERING_STANDARDS.md](ENGINEERING_STANDARDS.md) before changing code or content.
[ROADMAP.md](ROADMAP.md) tracks increment ownership, acceptance status and session handoffs.

## Set up the checkout

Use Python 3.12 (`.python-version`), `uv`, Node.js 22 or newer, pnpm 12.8.1
(`web/package.json`) and `make`. From the repository root:

```bash
uv sync
pnpm --dir web install --frozen-lockfile
```

`uv sync` installs the default `dev` group as well as runtime dependencies. Keep `uv.lock` and
`web/pnpm-lock.yaml` unchanged unless intentionally updating dependencies. Subsequent Python
commands use `uv run --frozen`.

The optional `torch` dependency group supplies `py-torch` problems. Include it on commands
that need PyTorch, including the server process when grading from the browser:

```bash
uv run --frozen --group torch dogma run <problem-id>
uv run --frozen --group torch dogma serve
```

Plain `uv run --frozen` does not select this optional group. `make check` does: content
verification and content typing run with `--group torch`, so the first gate run installs it. JAX, tinygrad, RL and GPU runtimes
in the architecture are planned stacks; they do not currently have dependency groups.

## Develop the browser

Start the API in one terminal:

```bash
uv run --frozen dogma serve
```

Start Vite in another:

```bash
pnpm --dir web dev
```

Open <http://127.0.0.1:5173>. Vite proxies `/api` to `http://127.0.0.1:8765`, so keep the API
on that port or update `web/vite.config.ts` when changing it. The API allows the Vite origin
by default. `dogma serve --allow-origin <origin>` accepts a repeatable explicit origin list.

For the bundled app, run `pnpm --dir web build`, then `uv run --frozen dogma serve` and open
<http://127.0.0.1:8765>. The server mounts the UI only if `web/dist/index.html` exists at startup.
CLI and browser operations use the same learner data.

After changing API schemas, regenerate the browser contract:

```bash
pnpm --dir web gen:api
pnpm --dir web gen:api --check
```

Generation exports the Python app's OpenAPI schema in-process; no running server is needed.
Commit both `web/src/api/openapi.json` and `web/src/api/schema.d.ts`. Interactive `/docs` and
`/redoc` pages are disabled to avoid runtime CDN dependencies.

## Configuration

[Settings](../src/dogma/core/config.py) reads process environment variables and an optional
`.env` in the current working directory. Environment variables take precedence. Copy
[.env.example](../.env.example) to `.env` if needed; run commands from the repository root.

| Variable | Default | Purpose |
|---|---|---|
| `DOGMA_REPO_ROOT` | This checkout | Root for curriculum, content, workspace and runtime state |
| `DOGMA_LLM_BASE_URL` | `http://127.0.0.1:8080/v1` | Loopback OpenAI-compatible model endpoint |
| `DOGMA_LLM_MODEL` | `qwen3.8-27b-q6` | Model id sent to the local server |
| `DOGMA_ALLOW_EXTERNAL` | `false` | Reserved for future external providers; does not enable one |
| `DOGMA_RUN_TIMEOUT_S` | `30` | Harness timeout when the problem does not override it |
| `DOGMA_RUN_TIMEOUT_SCALE` | `1.0` | Multiplies every run's time limit, including a problem's own `timeout_s`. For machines slower than the one the limits were calibrated on; CI sets `3`. Cannot be below 1 |
| `DOGMA_RUN_MEMORY_MB` | `2048` | Harness process-tree RSS limit |
| `DOGMA_RECORD_ATTEMPTS` | `true` | Whether graded attempts are recorded in learner progress |

`DOGMA_RECORD_ATTEMPTS=false` does not disable session or card writes. For manual verification,
use test fixtures or a separate data root. To grade an alternative file without recording a
solve, use `dogma submit <problem-id> --submission <file>`; recording requires `--record`.
The `--no-record` option suppresses recording for an individual `run` or `submit`.

The API bind address and port are command options (`serve --host`, `--port`), not settings.
Only loopback addresses are accepted. Local model weights and the inference server are managed
separately from the Python environment.

## Checks

```bash
make check
```

The gate runs Python lint/format checks, strict application typing, import contracts, default
pytest suites, all content verification, the privacy audit, browser lint/types/unit tests,
generated API freshness and a production browser build. Privacy runs after the build so newly
generated assets are included. Install both dependency sets first.

Before running privacy checks, create a gitignored `private/denylist.txt` containing one private
regex per line. Choose names and context terms that must never enter your publication; do not
put the real values in shared tests or documentation. Empty/missing rules block certification.
The working-tree audit includes code, metadata, filenames and built assets. It checks UTF-8,
BOM-marked UTF-16 and common escaped text, but does not claim to inspect arbitrary binary data.
Known upstream license attribution and opaque parser data are checksum-pinned; changed copies
and surrounding code are checked normally.

The pre-commit hook adds `--staged`, checking actual Git index blobs rather than assuming they
match disk. Stage the cleaned versions of all affected files before committing. A current-tree
check does not certify prior Git history; a history-free source export avoids publishing old
commits and author metadata. Private state, configuration values and local agent artifacts must
remain outside that export.

| Command | Scope |
|---|---|
| `make lint` / `make fmt` | Check Python style / apply Python formatting and lint fixes |
| `make types` | Strict application types, import contracts and per-package content typing |
| `make test` | Python tests excluding `slow`, `e2e` and live `llm` tests |
| `make verify-content` | Verify every authored topic, including solution/starter grading (uses the torch group) |
| `make audit` | Privacy audit over publishable project files |
| `make web-check` | Browser checks and production build |
| `make test-all` | Full gate, then slow and Python/browser end-to-end tests; no live model calls |

The browser smoke suite needs Chromium installed once with
`pnpm --dir web exec playwright install chromium`. It starts its own fixture API and keeps
learner state separate. Live model tests are opt-in: `uv run --frozen pytest -m llm`.
The live scorer checks usable, verbatim evidence and records accepted/dropped counts; partial
advisory results are allowed and never replace self-scores. To retain those diagnostics, add
`--junitxml=/tmp/dogma-live-model.xml -o junit_family=legacy`. The interviewer smoke checks both
a matching and an unrelated question. The browser suite scores all 30 mock items at 320px and
390px, checks horizontal fit and tap targets, then saves the debrief in a scratch database.

Before upgrading an existing learner database, run `uv run --frozen dogma backup`. It saves
the original schema and committed WAL rows to a `.sqlite3` snapshot before session initialization
applies migrations, and then writes the current-schema JSON export. If the snapshot cannot be
saved, the command stops before migration. Keep both files and a separate workspace backup.

Optional local hooks use the locked tools:

```bash
uv run --frozen pre-commit install
uv run --frozen pre-commit run --all-files
```

They check Python style, privacy, whitespace, file sizes and frontend lint. Hooks complement
the full gate. For content authoring, follow [CONTENT_SPEC.md](CONTENT_SPEC.md); use
`uv run --frozen dogma content verify <topic-id>` while working on one topic.

## Run artifacts

Every graded run writes `var/runs/<attempt_id>/`. After grading and child cleanup, the runner
adds a small `.dogma-completed.json` marker. Nothing is deleted automatically. Cleanup is an
explicit command that is a dry run unless you add `--apply`:

```bash
uv run --frozen dogma runs prune --older-than-days 30          # dry run: lists candidates
uv run --frozen dogma runs prune --older-than-days 30 --apply  # removes them
```

A directory is removed only when it has a valid marker, finished at least N days ago (minimum 1,
default 30), is not an attempt the database references, and is an ordinary directory, not a link.
Recorded attempts and their directories are never removed. Runs without a marker (active,
interrupted or created before this change) are always kept. The command reads the database
read-only and never creates or migrates it; a damaged database stops the command before any
deletion.

## Troubleshooting

| Symptom | Action |
|---|---|
| API works but `/` returns 404 | Build `web/`, then restart `dogma serve` so it mounts the UI |
| Browser cannot reach the API in development | Start `dogma serve` on port 8765 and use Vite at `127.0.0.1:5173` |
| Generated API check fails | Run `pnpm --dir web gen:api`, review and commit both generated files |
| Model nudge unavailable | Check the local server, `DOGMA_LLM_BASE_URL` and model id; authored hints still work |
| Corpus search has no index or matches | Run `uv run --frozen dogma corpus sync --no-convert`; see the [corpus guide](../corpus/README.md) |
| `RUNNER_RUNTIME_UNAVAILABLE` for `py-torch` | Include `--group torch` in the `uv run --frozen` invocation |
| Review rating conflicts with another view | Reload the queue and reveal the current card before rating again |

The runner provides time, memory and disk-growth limits with a scrubbed environment, and on
macOS runs learner code under `sandbox-exec` with no network and writes only inside the attempt
directory; tests run from a sealed copy the learner code cannot change. Other platforms are
refused rather than run without enforcement. Run-directory retention is
explicit (see [Run artifacts](#run-artifacts)); see [ARCHITECTURE §4.2](ARCHITECTURE.md#42-runner-sandboxed-execution).
