# Quality gate (docs/ENGINEERING_STANDARDS.md §13). Targets grow as increments land.
.PHONY: check lint types test test-all verify-content verify-variants audit web-check fmt

check: lint types test verify-content web-check
	$(MAKE) --no-print-directory audit

lint:
	uv run --frozen ruff check .
	uv run --frozen ruff format --check .

fmt:
	uv run --frozen ruff format .
	uv run --frozen ruff check --fix .

types:
	uv run --frozen mypy
	uv run --frozen lint-imports
	uv run --frozen --group torch python scripts/check_content_types.py

test:
	uv run --frozen pytest

test-all: check verify-variants
	uv run --frozen pytest -m "(e2e or slow) and not llm"
	cd web && pnpm test:e2e

verify-content:
	uv run --frozen --group torch dogma content verify --all

verify-variants:
	uv run --frozen --group torch python scripts/exercise_gate.py variants --out var/audit-variants

audit:
	uv run --frozen python scripts/audit_privacy.py

web-check:
	cd web && pnpm check
	cd web && pnpm build
