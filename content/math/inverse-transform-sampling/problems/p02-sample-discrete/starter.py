from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray


def sample_discrete(probs: Sequence[float], n: int, rng: np.random.Generator) -> NDArray[np.intp]:
    """Draw ``n`` category indices from ``probs`` by inverting the cumulative distribution.

    Makes exactly one call, ``u = rng.random(n)``. With ``c = cumsum(probs)``, category ``i`` is
    returned when ``c[i-1] <= u < c[i]`` (``c[-1] = 0``). A draw at or above ``c[k-1]`` is returned
    as category ``k - 1``, which covers rounding at the top of the cumulative sums.

    Returns
    -------
    (n,) integer array of category indices in ``0 .. k - 1``.

    Raises
    ------
    ValueError
        If ``probs`` is not a non-empty 1-D sequence, has a negative or non-finite entry, does not
        sum to 1 within 1e-9, or if ``n`` is negative.
    """
    raise NotImplementedError
