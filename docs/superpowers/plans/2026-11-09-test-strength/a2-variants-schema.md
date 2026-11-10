# I-32a2: the `variants/` manifest: schema, loader, static validation

**Owns:** `src/dogma/content/schemas.py`, `src/dogma/content/loader.py`,
`src/dogma/content/validate.py`, `tests/unit/content/test_variants.py` (new),
`tests/fixtures/content/packages/tst/alpha/problems/p01-double-all/variants/` (new).
**Commit message (coordinator):** `feat(content): load and validate problem variants (I-32a2)`

Nothing here runs a variant. This runbook only reads files and checks them against each other.

## Read first

- `src/dogma/content/schemas.py` lines 30 to 45 (`ContentModel`) and 220 to 252 (`ProblemMeta`).
- `src/dogma/content/loader.py` lines 49 to 60 (`ProblemPackage`) and 136 to 157 (`load_problem`).
- `src/dogma/content/validate.py` lines 93 to 100 (`Issue`), 439 to 455 (`_check_problem`) and
  583 to 631 (`_check_test_module`, the style to copy).
- `tests/unit/content/conftest.py` (fixtures `packages`, `alpha`, `alpha_problem`, `edit`) and the
  first 60 lines of `tests/unit/content/test_validate.py` (helpers `_codes`, `_topic_codes`).

## Build

### 1. Schema (`schemas.py`, after `ProblemMeta`)

```python
VARIANT_ACCEPT_PATTERN = r"^accept_[a-z0-9_]+\.(py|sql)$"
VARIANT_REJECT_PATTERN = r"^reject_[a-z0-9_]+\.(py|sql)$"
TEST_FUNCTION_PATTERN = r"^test_[A-Za-z0-9_]+$"

class AcceptVariant(ContentModel):
    file: str            # matches VARIANT_ACCEPT_PATTERN
    origin: Literal["blind", "freedom"]
    freedom: str | None = None   # what this solution does differently from the reference
    clause: str | None = None

class RejectVariant(ContentModel):
    file: str            # matches VARIANT_REJECT_PATTERN
    mistake: str         # min_length 1
    clause: str          # min_length 1
    killed_by: str       # matches TEST_FUNCTION_PATTERN
    visible: Literal["public", "hidden"]
    message_has: str | None = None

class UncoveredClause(ContentModel):
    clause: str          # min_length 1
    reason: str          # min_length 1

class VariantsManifest(ContentModel):
    problem_version: int           # ge 1
    accept: tuple[AcceptVariant, ...] = ()
    reject: tuple[RejectVariant, ...] = ()
    uncovered: tuple[UncoveredClause, ...] = ()
    sweeps: tuple[Literal["reject", "mutants"], ...] = ()
```

Model validators (raise `ValueError` with a plain sentence):

- `AcceptVariant`: `origin == "blind"` requires the file stem to be exactly `accept_blind` and
  `freedom is None`. `origin == "freedom"` requires a non-empty `freedom`.
- `VariantsManifest`: no `file` appears twice across `accept` and `reject`; at most one accept
  has `origin == "blind"`; no value appears twice in `sweeps`.

### 2. Loader (`loader.py`)

Add two fields to `ProblemPackage`, **with defaults**, after the existing fields:

```python
variants: VariantsManifest | None = None
variant_files: frozenset[str] = frozenset()
```

In `load_problem`: if `problem_dir / "variants"` is a directory, `variant_files` is the name of
every plain file directly inside it, and `variants` is the manifest parsed with `_validate_yaml`
when `variants/manifest.yaml` exists. A malformed manifest raises `ContentError` like any other
YAML file.

### 3. Validation (`validate.py`)

Add a module constant `VARIANTS_REQUIRED = False` and a **public** function:

```python
def validate_variants(problem: ProblemPackage) -> list[Issue]: ...
```

Call it from `_check_problem`. Let `squash(text) = " ".join(text.split())`.

| Code | Reported when | `path` |
|---|---|---|
| `VARIANT_MANIFEST_MISSING` | `variant_files` is not empty and has no `manifest.yaml` | `variants/` |
| `VARIANT_FILE_UNLISTED` | A file in `variant_files` other than `manifest.yaml` is not named in the manifest | the file |
| `VARIANT_FILE_MISSING` | The manifest names a file that is not in `variant_files` | `manifest.yaml` |
| `VARIANT_SUFFIX` | A listed file ends in `.sql` for a Python problem, or `.py` for a SQL problem | `manifest.yaml` |
| `VARIANT_CLAUSE_NOT_IN_STATEMENT` | A `clause` (reject, uncovered, or accept when set) is not in `problem.md`: `squash(clause) not in squash(problem.md text)` | `manifest.yaml` |
| `VARIANT_KILLED_BY_UNKNOWN` | A reject's `killed_by` is not a test function of `test_public.py` (`visible: public`) or `test_hidden.py` (`visible: hidden`). Use `problem.python[...].test_functions` | `manifest.yaml` |
| `VARIANT_VERSION_STALE` | `manifest.problem_version != meta.version` | `manifest.yaml` |
| `VARIANTS_REQUIRED` | Only when the constant is `True` and `meta.status` is `verified` or `reviewed`: there is no manifest, or no blind accept, or neither a reject nor an uncovered entry | the problem directory |

With no `variants/` directory and the constant `False`, the function returns `[]`. When
`problem.meta` is `None`, skip the checks that need it.

### 4. Fixture

Create under `tests/fixtures/content/packages/tst/alpha/problems/p01-double-all/variants/`:

`accept_blind.py`

```python
from __future__ import annotations


def double_all(values: list[float]) -> list[float]:
    """Return a new list with every value doubled."""
    doubled = []
    for value in values:
        doubled.append(float(value) + float(value))
    return doubled
```

`reject_mutates_input.py`

```python
from __future__ import annotations


def double_all(values: list[float]) -> list[float]:
    """Wrong on purpose: doubles the caller's list in place."""
    for index, value in enumerate(values):
        values[index] = 2.0 * value
    return values
```

`manifest.yaml` (read `problem_version` from that fixture's `meta.yaml` and use its value):

```yaml
problem_version: 1
accept:
  - file: accept_blind.py
    origin: blind
reject:
  - file: reject_mutates_input.py
    mistake: doubles the caller's list in place
    clause: Must not mutate its input.
    killed_by: test_input_list_is_not_mutated
    visible: hidden
```

## Tests to write (`tests/unit/content/test_variants.py`)

Each test mutates the `alpha_problem` copy and checks the issue code with `_topic_codes`-style
helpers (copy the two small helpers; do not import from another test module).

1. The fixture as written gives no `VARIANT_*` issue.
2. One test per code in the table, each a single small mutation: delete `manifest.yaml`; add an
   unlisted file; delete a listed file; rename a listed file to `.sql` in the manifest; change
   the clause by one word; change `killed_by` to `test_nope`; change `problem_version`.
3. `visible: public` with a hidden test name gives `VARIANT_KILLED_BY_UNKNOWN`.
4. A clause that differs from the statement only in line breaks and spacing passes.
5. `VARIANTS_REQUIRED`: with `monkeypatch.setattr(validate, "VARIANTS_REQUIRED", True)`, a
   `verified` problem without `variants/` is reported; the fixture with variants is not; a
   `draft` problem is not.
6. Schema: a bad file name, `origin: freedom` without `freedom`, two blind accepts, a file listed
   twice, and an unknown key each raise `ContentError` from `load_topic`.
7. Loader: `variant_files` and `variants` are filled for the fixture and empty for the beta
   fixture.

## Accept

```bash
uv run --frozen pytest tests/unit/content -q
uv run --frozen ruff check src/dogma/content tests/unit/content
uv run --frozen ruff format --check src/dogma/content tests/unit/content tests/fixtures/content
uv run --frozen mypy
uv run --frozen lint-imports
uv run --frozen --group torch dogma content verify cml/vector-operations
grep -rn "ProblemPackage(" src tests
```

All pass. The `grep` must show the one constructor in `loader.py` and nothing that now lacks
arguments. `content verify` must end with `All 1 topic(s) verified.`

## Traps

- Give the two new `ProblemPackage` fields defaults, or every existing constructor breaks.
- Do not make `validate_variants` run or import a variant. Static only.
- Do not set `VARIANTS_REQUIRED = True`. I-32e does that after the sweeps.
- The fixture's `test_hidden.py` has asserts without messages. Leave it. I-32e fixes fixtures
  when it turns the lint on.
- `tests/unit/content/test_validate.py` has tests that expect the fixtures to validate cleanly.
  They must still pass with the new files in place.
