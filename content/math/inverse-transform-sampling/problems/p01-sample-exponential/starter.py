from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def sample_exponential(rate: float, n: int, rng: np.random.Generator) -> NDArray[np.float64]:
    """Draw ``n`` exponential(rate) values by the inverse-CDF method.

    Makes exactly one call, ``u = rng.random(n)``, then maps each u to
    ``-log(1 - u) / rate`` (computed as ``-log1p(-u) / rate``).

    Returns
    -------
    (n,) float64 array of draws.

    Raises
    ------
    ValueError
        If ``rate`` is not finite and positive, or if ``n`` is negative.
    """
    raise NotImplementedError
