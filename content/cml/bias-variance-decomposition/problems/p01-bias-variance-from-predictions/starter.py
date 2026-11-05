from __future__ import annotations

from numpy.typing import ArrayLike


def bias_variance(
    predictions: ArrayLike,
    f_true: ArrayLike,
    noise_var: float,
) -> tuple[float, float, float]:
    """Return ``(bias2, variance, expected_mse)`` averaged over the test points.

    Parameters
    ----------
    predictions : 2-D array-like, shape ``(B, m)``: row ``b`` holds model ``b``'s predictions at the
        ``m`` test points. Each model was trained on its own training set from the same process.
    f_true : 1-D array-like of length ``m``, the noise-free target at each test point.
    noise_var : the variance of the label noise, so that a test label is ``f_true + noise``.

    Returns
    -------
    bias2 : mean over test points of ``(mean_b predictions - f_true) ** 2``.
    variance : mean over test points of the variance of the predictions over models (divisor ``B``).
    expected_mse : mean over test points of ``bias2_j + variance_j + noise_var``, which equals
        ``bias2 + variance + noise_var``. Preserve representable errors at large common offsets;
        squared quantities outside float64 range may overflow to infinity or underflow to zero.

    Raises
    ------
    ValueError
        If ``predictions`` is not 2-D or has no model or no test point, if ``f_true`` is not 1-D
        or does not match the number of test points, if any value is non-finite, or if
        ``noise_var`` is negative or non-finite.
    """
    raise NotImplementedError
