from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

# An entry counts as tied with the largest magnitude of its row when it lies within this relative
# band. Ties go to the lowest index, so rounding noise cannot flip a sign.
_TIE_RTOL = 1e-9


def pca(X: ArrayLike, n_components: int) -> dict[str, NDArray[np.float64]]:
    """Wrong on purpose: return the top ``n_components`` principal components of the (n, d) data matrix ``X``.

    Rows are observations and columns are variables. The principal axes are the directions of
    greatest spread around the mean, and the SVD of the centred data gives them directly.
    """  # noqa: E501
    # np.array copies by default, so the caller's matrix is never modified.
    x = np.array(X, dtype=np.float64)

    if x.ndim != 2:
        raise ValueError(f"X must be 2-D (observations, variables), got {x.ndim}-D")
    n, d = x.shape
    if n < 2 or d < 1:
        raise ValueError(
            "X needs at least two rows and one column: a sample variance needs two rows"
        )
    if not np.isfinite(x).all():
        raise ValueError("X must contain only finite numbers (no NaN or inf)")
    # bool is a subclass of int in Python, so exclude it explicitly.
    if isinstance(n_components, bool) or not isinstance(n_components, (int, np.integer)):
        raise ValueError("n_components must be an integer")
    k = int(n_components)
    if not 1 <= k <= min(n, d):
        raise ValueError(f"n_components must be between 1 and min(n, d) = {min(n, d)}")

    # Step 1: centre each column. Spread is measured around the mean, so the mean is removed first.
    mean = (
        np.floor(x.mean(axis=0))
        if np.issubdtype(np.asarray(X).dtype, np.integer)
        else x.mean(axis=0)
    )
    centred = x - mean

    # Step 2: economy SVD, centred = U @ diag(s) @ Vt. The rows of Vt are the principal axes, and
    # s is sorted in decreasing order, so the axes come out ordered by variance.
    _, s, vt = np.linalg.svd(centred, full_matrices=False)

    # Step 3: fix each kept axis's sign with the documented rule.
    components = _sign_by_largest_entry(vt[:k])

    # Step 4: the sample variance of the scores on axis i is s_i**2 / (n - 1).
    explained_variance = s[:k] ** 2 / (n - 1)

    # Step 5: the ratio divides by the variance of all d columns, not by the kept part.
    # With a zero total (every column constant) there is nothing to explain, so the ratios are 0.
    total_variance = float(np.var(x, axis=0, ddof=1).sum())
    if total_variance == 0.0:
        explained_variance_ratio = np.zeros(k)
    else:
        explained_variance_ratio = explained_variance / total_variance

    # Step 6: scores are the centred data projected onto the signed axes. Using the same signed
    # components here keeps the scores and the axes consistent.
    transformed = centred @ components.T

    return {
        "components": components,
        "explained_variance": explained_variance,
        "explained_variance_ratio": explained_variance_ratio,
        "transformed": transformed,
    }


def _sign_by_largest_entry(rows: NDArray[np.float64]) -> NDArray[np.float64]:
    """Multiply each row by the sign of its largest-magnitude entry, so that entry is positive.

    This is the sign rule of scikit-learn's svd_flip for PCA. With a tie inside the band, the
    lowest index decides. Unit rows always have a nonzero entry, so the sign is never zero.
    """
    magnitudes = np.abs(rows)
    tied = magnitudes >= magnitudes.max(axis=1, keepdims=True) * (1.0 - _TIE_RTOL)
    first = np.argmax(tied, axis=1)  # the first tied index in each row
    signs = np.sign(rows[np.arange(rows.shape[0]), first])
    return rows * signs[:, None]
