from __future__ import annotations

from numpy.typing import ArrayLike


def permutation_test_mean_diff(
    a: ArrayLike,
    b: ArrayLike,
    *,
    n_permutations: int = 5000,
    seed: int,
) -> float:
    """Return the two-sided permutation p-value for the difference in means of ``a`` and ``b``.

    Parameters
    ----------
    a, b : 1-D array-likes of finite real numbers, each non-empty (ints allowed).
    n_permutations : int >= 1, number of random relabellings of the pooled sample.
    seed : int, seeds ``np.random.default_rng``.

    Returns
    -------
    float in (0, 1]: ``(count + 1) / (n_permutations + 1)``, where ``count`` is the number of
    relabellings with ``|mean(A*) - mean(B*)| >= |mean(a) - mean(b)| * (1 - 1e-9)``.

    Raises
    ------
    ValueError
        If ``a`` or ``b`` is not a non-empty 1-D finite array, or if ``n_permutations < 1``.
    """
    raise NotImplementedError
