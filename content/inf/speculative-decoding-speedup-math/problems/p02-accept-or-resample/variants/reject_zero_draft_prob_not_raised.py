from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray


def _distribution(values: Any, name: str) -> NDArray[np.float64]:
    """Validate a probability vector before any arithmetic touches it."""
    raw = np.asarray(values)
    if raw.dtype.kind not in "iuf":
        raise ValueError(f"{name} must be a real numeric array")
    array = raw.astype(np.float64)
    if array.ndim != 1 or array.shape[0] < 1:
        raise ValueError(f"{name} must be a nonempty 1-D array")
    if not np.all(np.isfinite(array)) or np.any(array < 0.0):
        raise ValueError(f"{name} must be finite and nonnegative")
    if abs(float(array.sum()) - 1.0) > 1e-9:
        raise ValueError(f"{name} must sum to 1")
    return array


def accept_or_resample(p: Any, q: Any, draft_token: Any, rng: Any) -> int:
    """Wrong on purpose: return the token one speculative step emits at a single position."""
    target = _distribution(p, "p")
    draft = _distribution(q, "q")
    if target.shape != draft.shape:
        raise ValueError("p and q must have the same length")
    vocab = target.shape[0]
    if isinstance(draft_token, bool) or not isinstance(draft_token, (int, np.integer)):
        raise ValueError("draft_token must be an integer index")
    if not 0 <= draft_token < vocab:
        raise ValueError("draft_token must index the vocabulary")
    x = int(draft_token)
    if draft[x] < 0.0:
        raise ValueError(
            "draft_token has zero draft probability, so the draft cannot have proposed it"
        )

    # Step 1, acceptance: accept x with probability min(1, p(x)/q(x)). The uniform is drawn
    # unconditionally so every call uses the same draw pattern, and the test is strict.
    threshold = min(1.0, target[x] / draft[x])
    u_accept = rng.random()
    if u_accept < threshold:
        return x

    # Step 2, rejection: sample from the residual norm(max(0, p - q)). This is the leftover
    # target mass the draft did not cover, which is what makes the output exactly p.
    residual = np.maximum(target - draft, 0.0)
    total = float(residual.sum())
    if total <= 0.0:
        # p is elementwise at most q (only up to rounding). The rejection branch has no
        # residual mass, so sample from p to keep a valid distribution.
        residual = target
        total = float(residual.sum())
    cdf = np.cumsum(residual) / total
    cdf[-1] = 1.0  # rounding can leave the last cumulative value just below 1

    # Inverse-CDF draw: the first index whose cumulative mass exceeds the second uniform.
    # side="right" makes a draw exactly on a boundary move past it.
    u_resample = rng.random()
    return int(np.searchsorted(cdf, u_resample, side="right"))
