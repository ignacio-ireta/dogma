import numpy as np


def pseudo_inverse(A, *, rcond=1e-15):
    if isinstance(rcond, bool) or not isinstance(rcond, (int, float, np.integer, np.floating)):
        raise ValueError("rcond must be a real number")
    if not (np.isfinite(float(rcond)) and float(rcond) >= 0):
        raise ValueError("rcond must be finite and >= 0")
    M = np.array(A, dtype=np.float64)
    if M.ndim != 2 or M.shape[0] < 1 or M.shape[1] < 1:
        raise ValueError("A must be a non-empty 2-D array")
    if not np.isfinite(M).all():
        raise ValueError("A must be finite")
    # SVD of the transpose, then transpose back: a different route to the same pseudo-inverse.
    u, s, vt = np.linalg.svd(M.T, full_matrices=False)
    keep = s > float(rcond) * s.max()
    inv = np.zeros_like(s)
    inv[keep] = 1.0 / s[keep]
    # pinv(M^T) = V diag(1/s) U^T, so pinv(M) = U diag(1/s) V^T
    return (u * inv) @ vt
