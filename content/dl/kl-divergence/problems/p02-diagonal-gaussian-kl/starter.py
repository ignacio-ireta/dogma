from __future__ import annotations

from numpy.typing import ArrayLike


def diagonal_gaussian_kl(
    mu_p: ArrayLike, sigma_p: ArrayLike, mu_q: ArrayLike, sigma_q: ArrayLike
) -> float:
    """Return KL(N(mu_p, diag(sigma_p^2)) || N(mu_q, diag(sigma_q^2))) in nats.

    Inputs are 1-D real arrays of one common length d >= 1 with finite entries. Every sigma
    entry must be strictly positive. Closed form, summed over coordinates:
    sum_i [ log(sq_i / sp_i) + (sp_i^2 + (mp_i - mq_i)^2) / (2 sq_i^2) - 1/2 ].
    Returns a Python float. Raise ValueError for invalid input. Do not mutate inputs.
    """
    raise NotImplementedError
