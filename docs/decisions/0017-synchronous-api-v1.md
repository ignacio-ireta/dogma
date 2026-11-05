# 0017. Synchronous API calls in v1, streaming before the browser tutor
Date: 2026-10-25 · Status: accepted (amends ENGINEERING_STANDARDS §8 for API v1)

## Context
ENGINEERING_STANDARDS §8 says long operations return an id or stream, and no request blocks longer
than 30 s. In API v1 (I-14), `POST /attempts` grades synchronously, bounded by the problem's
`timeout_s` (30 s for every current problem) plus staging, and `POST /tutor/hint|debug` waits for the
local model (33–42 s measured, gateway total timeout 180 s). A job/SSE layer would add complexity
before any browser client exists.

## Decision
API v1 may block synchronously, bounded by the runner's `timeout_s` (attempts) and the tutor gateway's
total timeout (tutor). Before the browser relies on the tutor (I-15/I-16), the tutor endpoints gain a
streaming (SSE) variant built on `dogma.tutor.service.prepare_hint()`. The browser must show progress
for any call that can exceed 1 s.

## Consequences
Simpler v1 and identical semantics to the CLI. A slow grade or tutor call ties up one request, which is
fine for a single local user. §8's 30 s rule applies again once SSE exists.

## Alternatives
- A job queue with polling now: rejected, because there's no client yet and it duplicates the runner's lifecycle.
- SSE for everything now: rejected for attempts. Grading is bounded and short, so streaming adds nothing.
