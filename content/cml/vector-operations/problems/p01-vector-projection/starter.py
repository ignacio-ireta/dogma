from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray


def vector_projection(a: ArrayLike, b: ArrayLike) -> NDArray[np.float64]:
    """Return the orthogonal projection of ``a`` onto ``b``.

    Parameters
    ----------
    a, b : 1-D array-likes of equal length n >= 1 (ints allowed).

    Returns
    -------
    (n,) float64 array equal to (a·b / b·b) · b.

    Raises
    ------
    ValueError
        If ``a`` or ``b`` is not 1-D, their lengths differ, or ``b`` is the zero vector.
    """
    raise NotImplementedError
