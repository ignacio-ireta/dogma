from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray


def kl_divergence_from_logits(p_logits: ArrayLike, q_logits: ArrayLike) -> NDArray[np.float64]:
    """Return KL(P_i || Q_i) in nats for each row i, with P_i = softmax(p_logits[i]).

    KL = sum_j p_j (log p_j - log q_j). A term with p_j = 0 contributes 0 (0 log 0 = 0), even if
    q_j = 0. A term with p_j > 0 and q_j = 0 makes the row's KL +inf. Both inputs are (n, k) with
    n >= 1, k >= 1; -inf logits are allowed (zero probability); NaN and +inf are not. Every row
    needs a finite entry. Returns a float64 array of shape (n,). Raise ValueError for invalid
    input. Compute the log-probabilities with log-sum-exp from the logits. Do not mutate inputs.
    """
    raise NotImplementedError
