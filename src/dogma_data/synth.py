"""Realistic synthetic generators with controlled properties.

Every generator takes ``rng`` first (from ``dogma_data.rng``), keyword-only knobs after ``*``,
validates its arguments with ``ValueError``, and returns float64 unless the docstring says
otherwise. The code is pure numpy with no BLAS calls: products are built from explicit loops
or broadcasting, so the output is bit-identical across BLAS builds. Fingerprints in the tests
pin the output for a fixed key.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from statistics import NormalDist
from typing import Any

import numpy as np
from numpy.typing import NDArray

_BURN_IN = 100  # discarded AR steps so the series starts near stationarity


def _check_rng(rng: object) -> np.random.Generator:
    if not isinstance(rng, np.random.Generator):
        raise ValueError("rng must be a numpy.random.Generator (use dogma_data.rng(key))")
    return rng


def _check_int(name: str, value: object, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int | np.integer):
        raise ValueError(f"{name} must be an int, got {value!r}")
    if value < minimum:
        raise ValueError(f"{name} must be >= {minimum}, got {value}")
    return int(value)


def _check_float(
    name: str,
    value: object,
    *,
    low: float | None = None,
    high: float | None = None,
    low_open: bool = False,
    high_open: bool = False,
) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float | np.floating):
        raise ValueError(f"{name} must be a number, got {value!r}")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite, got {number}")
    if low is not None and (number < low or (low_open and number == low)):
        bound = "> " if low_open else ">= "
        raise ValueError(f"{name} must be {bound}{low}, got {number}")
    if high is not None and (number > high or (high_open and number == high)):
        bound = "< " if high_open else "<= "
        raise ValueError(f"{name} must be {bound}{high}, got {number}")
    return number


def _check_fraction(name: str, value: object) -> float:
    return _check_float(name, value, low=0.0, high=1.0)


def correlated_mixed_scale(
    rng: np.random.Generator,
    n: int,
    d: int,
    *,
    n_factors: int = 2,
    strength: float = 3.0,
    scale_range: tuple[float, float] = (1e-2, 1e3),
    offset_range: tuple[float, float] = (-10.0, 10.0),
) -> NDArray[np.float64]:
    """Return an (n, d) matrix of correlated features on very different scales.

    A latent factor model: ``F`` (n x k) standard normal, loadings ``W`` (k x d) with variance
    ``strength**2 / k``, unique N(0, 1) noise. Column j is then multiplied by a log-uniform
    scale from ``scale_range`` and shifted by a uniform offset from ``offset_range``. Realism:
    correlated features (collinearity, PCA structure) on scales spanning orders of magnitude
    (unscaled gradient descent, distance metrics).
    """
    gen = _check_rng(rng)
    n = _check_int("n", n, 2)
    d = _check_int("d", d, 1)
    k = _check_int("n_factors", n_factors, 1)
    strength = _check_float("strength", strength, low=0.0, low_open=True)
    lo, hi = (
        _check_float("scale_range[0]", scale_range[0], low=0.0, low_open=True),
        _check_float("scale_range[1]", scale_range[1], low=0.0, low_open=True),
    )
    if lo > hi:
        raise ValueError(f"scale_range must be (low, high) with low <= high, got {scale_range}")
    off_lo = _check_float("offset_range[0]", offset_range[0])
    off_hi = _check_float("offset_range[1]", offset_range[1])
    if off_lo > off_hi:
        raise ValueError(f"offset_range must be (low, high) with low <= high, got {offset_range}")

    factors = gen.standard_normal((n, k))
    loadings = gen.normal(0.0, strength / math.sqrt(k), size=(k, d))
    out = gen.standard_normal((n, d))
    for j in range(k):
        out = out + factors[:, j : j + 1] * loadings[j : j + 1, :]
    scales = np.exp(gen.uniform(math.log(lo), math.log(hi), size=d)) if lo < hi else np.full(d, lo)
    offsets = gen.uniform(off_lo, off_hi, size=d)
    return np.asarray(out * scales + offsets, dtype=np.float64)


def heavy_tailed_noise(
    rng: np.random.Generator,
    n: int,
    *,
    df: float = 3.0,
    scale: float = 1.0,
    spread: NDArray[Any] | None = None,
) -> NDArray[np.float64]:
    """Return n Student-t draws (shape (n,)), times ``scale``, times ``spread`` if given.

    ``spread`` is an optional (n,) array of positive per-row multipliers, for uneven spread
    (heteroscedastic noise). Realism: heavy tails (outliers, unstable means) and fan-shaped
    residuals.
    """
    gen = _check_rng(rng)
    n = _check_int("n", n, 1)
    df = _check_float("df", df, low=0.0, low_open=True)
    scale = _check_float("scale", scale, low=0.0, low_open=True)
    draws = gen.standard_t(df, size=n).astype(np.float64) * scale
    if spread is not None:
        mult = np.asarray(spread, dtype=np.float64)
        if mult.shape != (n,):
            raise ValueError(f"spread must have shape ({n},), got {mult.shape}")
        if not np.all(np.isfinite(mult)) or np.any(mult <= 0):
            raise ValueError("spread must hold finite, strictly positive values")
        draws = draws * mult
    return draws


def imbalanced_labels(
    rng: np.random.Generator,
    n: int,
    *,
    positive_rate: float = 0.01,
    flip_rate: float = 0.0,
    min_positives: int = 1,
) -> NDArray[np.int64]:
    """Return 0/1 labels (int64, shape (n,)) with a rare positive class.

    Exactly ``max(min_positives, round(n * positive_rate))`` positives go at random positions,
    then ``round(n * flip_rate)`` random labels are flipped (label noise). Realism: rare-event
    classes (1:100 and worse) where accuracy and ROC behave differently.
    """
    gen = _check_rng(rng)
    n = _check_int("n", n, 1)
    positive_rate = _check_fraction("positive_rate", positive_rate)
    flip_rate = _check_fraction("flip_rate", flip_rate)
    min_pos = _check_int("min_positives", min_positives, 0)
    n_pos = max(min_pos, round(n * positive_rate))
    if n_pos > n:
        raise ValueError(f"cannot place {n_pos} positives among {n} rows")
    labels = np.zeros(n, dtype=np.int64)
    labels[gen.choice(n, size=n_pos, replace=False)] = 1
    n_flip = round(n * flip_rate)
    flip_idx = gen.choice(n, size=n_flip, replace=False)
    labels[flip_idx] = 1 - labels[flip_idx]
    return labels


def informative_scores(
    rng: np.random.Generator,
    y: NDArray[Any],
    *,
    auc: float = 0.8,
    decimals: int | None = None,
) -> NDArray[np.float64]:
    """Return scores in (0, 1) whose population ROC AUC equals ``auc``.

    Binormal model: negatives N(0, 1), positives N(delta, 1) with
    ``delta = sqrt(2) * Phi^-1(auc)``, mapped through the logistic function like model
    probabilities. ``decimals`` rounds the scores and creates ties. Realism: a model with a
    known, imperfect ranking quality (AUC 0.7 to 0.95), with probabilities that look real.
    """
    gen = _check_rng(rng)
    labels = np.asarray(y)
    if labels.ndim != 1 or labels.size == 0:
        raise ValueError("y must be a non-empty 1-D array")
    if not np.all((labels == 0) | (labels == 1)):
        raise ValueError("y must hold only 0 and 1")
    if labels.sum() in (0, labels.size):
        raise ValueError("y must contain both classes")
    auc = _check_float("auc", auc, low=0.5, high=1.0, low_open=True, high_open=True)
    if decimals is not None:
        decimals = _check_int("decimals", decimals, 0)
    delta = math.sqrt(2.0) * NormalDist().inv_cdf(auc)
    raw = gen.standard_normal(labels.size) + delta * labels.astype(np.float64)
    scores = 1.0 / (1.0 + np.exp(-raw))
    if decimals is not None:
        scores = np.round(scores, decimals)
    return np.asarray(scores, dtype=np.float64)


def seasonal_ar_series(
    rng: np.random.Generator,
    n: int,
    *,
    period: int = 12,
    amplitude: float = 1.0,
    trend: float = 0.0,
    phi: Sequence[float] = (0.6,),
    sigma: float = 1.0,
    level_shifts: int = 0,
    shift_size: float = 3.0,
    missing_rate: float = 0.0,
    start: float = 0.0,
) -> NDArray[np.float64]:
    """Return a seasonal, autocorrelated series of length n with NaN gaps.

    AR(p) noise by explicit recursion (after a burn-in), plus ``amplitude * sin(2 pi t / period)``,
    plus ``trend * t``, plus ``start``. ``level_shifts`` random steps of +-``shift_size`` happen
    at random times. ``round(n * missing_rate)`` random positions become NaN. Requires
    ``sum(|phi|) < 1`` (stationary AR). Realism: autocorrelated, seasonal, trending series with
    level changes and gaps, where white-noise assumptions break ACF, MASE and naive forecasts.
    """
    gen = _check_rng(rng)
    n = _check_int("n", n, 1)
    period = _check_int("period", period, 2)
    amplitude = _check_float("amplitude", amplitude)
    trend = _check_float("trend", trend)
    coefs = [_check_float("phi", p) for p in phi]
    if not coefs:
        raise ValueError("phi must hold at least one coefficient")
    if sum(abs(p) for p in coefs) >= 1.0:
        raise ValueError(f"phi is not stationary: sum(|phi|) must be < 1, got {coefs}")
    sigma = _check_float("sigma", sigma, low=0.0, low_open=True)
    n_shifts = _check_int("level_shifts", level_shifts, 0)
    shift = _check_float("shift_size", shift_size, low=0.0)
    if n_shifts > max(n - 1, 0):
        raise ValueError(f"level_shifts={n_shifts} needs at least that many rows after the first")
    missing = _check_fraction("missing_rate", missing_rate)
    start_level = _check_float("start", start)

    total = n + _BURN_IN
    noise = gen.normal(0.0, sigma, size=total)
    ar = np.zeros(total, dtype=np.float64)
    order = len(coefs)
    for t in range(total):
        value = noise[t]
        for lag in range(1, order + 1):
            if t - lag >= 0:
                value += coefs[lag - 1] * ar[t - lag]
        ar[t] = value
    series = ar[_BURN_IN:].copy()

    steps = np.zeros(n, dtype=np.float64)
    if n_shifts > 0:
        times = gen.choice(np.arange(1, n), size=n_shifts, replace=False)
        signs = gen.choice(np.array([-1.0, 1.0]), size=n_shifts)
        for when, sign in zip(times, signs, strict=True):
            steps[int(when)] += sign * shift
    series = series + np.cumsum(steps)

    t_index = np.arange(n, dtype=np.float64)
    series = series + amplitude * np.sin(2.0 * np.pi * t_index / period)
    series = series + trend * t_index + start_level

    n_missing = round(n * missing)
    if n_missing > 0:
        series[gen.choice(n, size=n_missing, replace=False)] = np.nan
    return np.asarray(series, dtype=np.float64)


def event_log(
    rng: np.random.Generator,
    n_events: int,
    *,
    n_users: int = 50,
    zipf_a: float = 1.3,
    start: str = "2024-01-01",
    days: int = 90,
    amount_median: float = 30.0,
    amount_sigma: float = 1.0,
    categories: Sequence[str] = ("web", "ios", "android"),
    dirty_rate: float = 0.0,
    duplicate_rate: float = 0.0,
) -> dict[str, NDArray[Any]]:
    """Return an event log as a dict of column arrays, shuffled, with duplicates appended.

    Columns: ``event_id`` (int64, unique before duplication), ``user_id`` (int64, user k drawn
    with probability proportional to ``1 / k**zipf_a``, ids 1..n_users), ``ts`` (datetime64[s],
    naive, uniform over ``days`` from ``start``), ``amount`` (lognormal around ``amount_median``,
    rounded to cents), ``category`` (str). A ``dirty_rate`` fraction of categories gets casing
    noise or a leading/trailing space. ``round(n * duplicate_rate)`` exact duplicate rows are
    appended and the whole log is shuffled. Realism: heavy-user skew, skewed money amounts,
    naive timestamps, and messy categorical keys, the things SQL aggregations get wrong.
    """
    gen = _check_rng(rng)
    n = _check_int("n_events", n_events, 1)
    users = _check_int("n_users", n_users, 1)
    zipf = _check_float("zipf_a", zipf_a, low=0.0, low_open=True)
    days_n = _check_int("days", days, 1)
    median = _check_float("amount_median", amount_median, low=0.0, low_open=True)
    sigma = _check_float("amount_sigma", amount_sigma, low=0.0)
    dirty = _check_fraction("dirty_rate", dirty_rate)
    dup = _check_fraction("duplicate_rate", duplicate_rate)
    if isinstance(categories, str) or not categories:
        raise ValueError("categories must be a non-empty sequence of strings")
    cats = [str(c) for c in categories]
    try:
        window_start = np.datetime64(start, "s")
    except ValueError as exc:
        raise ValueError(f"start must be an ISO date or datetime, got {start!r}") from exc

    weights = 1.0 / np.arange(1, users + 1, dtype=np.float64) ** zipf
    user_idx = gen.choice(users, size=n, p=weights / weights.sum())
    event_id = np.arange(1, n + 1, dtype=np.int64)
    user_id = (user_idx + 1).astype(np.int64)
    offsets = gen.integers(0, days_n * 86400, size=n, dtype=np.int64)
    ts = window_start + offsets.astype("timedelta64[s]")
    amount = np.round(gen.lognormal(math.log(median), sigma, size=n), 2)

    picks = gen.integers(0, len(cats), size=n)
    labels = [cats[i] for i in picks]
    dirty_mask = gen.random(n) < dirty
    kinds = gen.integers(0, 3, size=n)
    side = gen.random(n) < 0.5
    for i in np.flatnonzero(dirty_mask):
        value = labels[i]
        kind = int(kinds[i])
        if kind == 0:
            labels[i] = value.upper()
        elif kind == 1:
            labels[i] = value.capitalize()
        else:
            labels[i] = f" {value}" if side[i] else f"{value} "

    n_dup = round(n * dup)
    if n_dup > n:
        raise ValueError("duplicate_rate must be <= 1")
    dup_rows = gen.choice(n, size=n_dup, replace=False)
    columns: dict[str, NDArray[Any]] = {
        "event_id": np.concatenate([event_id, event_id[dup_rows]]),
        "user_id": np.concatenate([user_id, user_id[dup_rows]]),
        "ts": np.concatenate([ts, ts[dup_rows]]),
        "amount": np.concatenate([amount, amount[dup_rows]]),
        "category": np.array(labels + [labels[i] for i in dup_rows], dtype=str),
    }
    order = gen.permutation(n + n_dup)
    return {key: values[order] for key, values in columns.items()}


def latencies(
    rng: np.random.Generator,
    n: int,
    *,
    median_ms: float = 50.0,
    sigma: float = 0.5,
    tail_fraction: float = 0.02,
    tail_multiplier: float = 20.0,
) -> NDArray[np.float64]:
    """Return n request latencies in milliseconds: a lognormal body plus a slow tail.

    A ``tail_fraction`` of requests comes from a slow component, lognormal around
    ``median_ms * tail_multiplier``. Realism: a long right tail, so p99/p50 is large and mean and
    median disagree (serving metrics, SLO tables).
    """
    gen = _check_rng(rng)
    n = _check_int("n", n, 1)
    median = _check_float("median_ms", median_ms, low=0.0, low_open=True)
    sig = _check_float("sigma", sigma, low=0.0)
    tail = _check_fraction("tail_fraction", tail_fraction)
    mult = _check_float("tail_multiplier", tail_multiplier, low=1.0)
    body = gen.lognormal(math.log(median), sig, size=n)
    slow = gen.lognormal(math.log(median * mult), sig, size=n)
    is_slow = gen.random(n) < tail
    return np.asarray(np.where(is_slow, slow, body), dtype=np.float64)


def outlier_channel_activations(
    rng: np.random.Generator,
    n_tokens: int,
    n_channels: int,
    *,
    n_outlier_channels: int = 2,
    outlier_scale: float = 50.0,
) -> NDArray[np.float64]:
    """Return (n_tokens, n_channels) normal activations with a few channels scaled up.

    ``n_outlier_channels`` randomly chosen columns are multiplied by ``outlier_scale``, the
    outlier-feature pattern of large transformers (LLM.int8). Realism: one absmax scale per
    tensor is dominated by a few channels, and the rest of the values lose precision.
    """
    gen = _check_rng(rng)
    tokens = _check_int("n_tokens", n_tokens, 1)
    channels = _check_int("n_channels", n_channels, 1)
    n_out = _check_int("n_outlier_channels", n_outlier_channels, 0)
    if n_out > channels:
        raise ValueError(f"n_outlier_channels={n_out} exceeds n_channels={channels}")
    scale = _check_float("outlier_scale", outlier_scale, low=0.0, low_open=True)
    acts = gen.standard_normal((tokens, channels))
    cols = gen.choice(channels, size=n_out, replace=False)
    acts[:, cols] = acts[:, cols] * scale
    return np.asarray(acts, dtype=np.float64)


def peaked_probs(
    rng: np.random.Generator,
    y: NDArray[Any],
    n_classes: int,
    *,
    accuracy: float = 0.85,
    temperature: float = 0.5,
    overconfidence: float = 1.5,
    margin_scale: float = 2.0,
) -> NDArray[np.float64]:
    """Return (n, n_classes) class probabilities that are accurate but over-confident.

    For each row the predicted class is the true class with probability ``accuracy``, otherwise
    a uniformly random wrong class. Logits are N(0, 1) noise; the predicted class's logit is set
    to the largest other logit plus a half-normal margin with scale ``margin_scale``, so the
    argmax is always the predicted class and confidence varies from row to row. Probabilities
    are ``softmax(overconfidence * logits / temperature)`` (max-subtracted for stability), so
    rows sum to 1. With the defaults and 10 classes, mean confidence is about 0.875 against an
    accuracy of 0.85, the median confidence is about 0.97, and the 5% quantile about 0.48.
    Realism: a classifier whose confidence is higher than its accuracy, the situation where
    calibration and conformal sets matter.
    """
    gen = _check_rng(rng)
    labels = np.asarray(y)
    if labels.ndim != 1 or labels.size == 0:
        raise ValueError("y must be a non-empty 1-D array")
    if not np.issubdtype(labels.dtype, np.integer):
        raise ValueError("y must hold integer class labels")
    k = _check_int("n_classes", n_classes, 2)
    if labels.min() < 0 or labels.max() >= k:
        raise ValueError(f"y labels must lie in [0, {k})")
    acc = _check_fraction("accuracy", accuracy)
    temp = _check_float("temperature", temperature, low=0.0, low_open=True)
    over = _check_float("overconfidence", overconfidence, low=0.0, low_open=True)
    margin = _check_float("margin_scale", margin_scale, low=0.0, low_open=True)
    labels = labels.astype(np.int64)
    n = labels.size
    rows = np.arange(n)
    correct = gen.random(n) < acc
    wrong = (labels + gen.integers(1, k, size=n)) % k
    predicted = np.where(correct, labels, wrong)
    logits = gen.standard_normal((n, k))
    others = logits.copy()
    others[rows, predicted] = -np.inf
    gaps = np.abs(gen.normal(0.0, margin, size=n))
    logits[rows, predicted] = others.max(axis=1) + gaps
    scaled = over * logits / temp
    scaled = scaled - scaled.max(axis=1, keepdims=True)
    expv = np.exp(scaled)
    probs = expv / expv.sum(axis=1, keepdims=True)
    return np.asarray(probs, dtype=np.float64)
