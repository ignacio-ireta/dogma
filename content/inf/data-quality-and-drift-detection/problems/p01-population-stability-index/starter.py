from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray


def population_stability_index(
    expected: Sequence[float] | NDArray[np.float64],
    actual: Sequence[float] | NDArray[np.float64],
    *,
    bins: int = 10,
    eps: float = 1e-6,
) -> float:
    """Population stability index of ``actual`` against the reference sample ``expected``.

    Bins: the interior edges are the quantiles of ``expected`` at k / bins (linear method), with
    outer edges at -inf and +inf. A value goes to the bin whose index is the number of interior
    edges strictly below it. Shares are counts divided by sample sizes; a zero share becomes eps.

    Returns
    -------
    psi : float, sum over bins of (q - p) * ln(q / p), with p the expected and q the actual shares.

    Raises
    ------
    ValueError
        If an input is not 1-D, is empty, or has non-finite values; if bins is not an integer of at
        least 2; or if eps is not finite and positive.
    """
    raise NotImplementedError
