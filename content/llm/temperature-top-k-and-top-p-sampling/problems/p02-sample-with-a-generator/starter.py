from __future__ import annotations

import torch


def sample_token(probs: torch.Tensor, generator: torch.Generator) -> int:
    """Draw one token index from probs using only the passed-in generator.

    probs is a float64 (V,) tensor, V >= 1, with finite non-negative entries summing to 1 within
    1e-9. The draw is the first index whose cumulative sum exceeds one uniform value in [0, 1)
    taken from generator. A zero-probability index is never returned. Returns a Python int.

    Raises ValueError for a non-1-D or empty probs, a dtype other than float64, a non-finite or
    negative entry, a sum that is not 1 within 1e-9, or a generator that is not a torch.Generator.
    Never mutates probs and never uses the global random state.
    """
    raise NotImplementedError
