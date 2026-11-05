# Interview notes — Reverse-mode autograd engine

## Q1. What are the essential parts of a reverse-mode engine?

**60-second answer.** Each forward operation records parent edges and local derivatives. To differentiate one scalar root, I discover a topological order, clear reachable gradients, seed the root with one, and walk the order backward. Every edge adds its contribution. Reverse mode fits training because a scalar loss depends on many parameters.

**Follow-ups.** Why addition rather than assignment? Shared paths contribute separately. Why a visited set? It prevents repeated node scheduling while preserving duplicate operand edges.

## Q2. What should happen if backward is called twice?

**60-second answer.** The answer is an API contract. This engine resets gradients on the reachable graph before each call, so repeated calls are idempotent. A different root recomputes only its reachable nodes and leaves unrelated gradients intact. Tensor frameworks often accumulate leaf gradients instead, so users must clear them intentionally.

**Follow-ups.** Can you mutate forward values? Cached local partials would become stale; this engine exposes read-only data. What is missing for production? Tensor broadcasting, graph lifetime controls, and scalable traversal.

## Common mistakes

- Deduplicating operand edges along with the topological node schedule.
- Overwriting accumulated gradients from shared paths.
- Calling backward again without applying the stated reset policy.

Source: [book: Goodfellow, Bengio and Courville, Deep Learning, chapter 6.5]. See lesson.md for the derivation and explicit conventions.
