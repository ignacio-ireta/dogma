"""Each synth knob does what it says, checked against explicit statistical bounds.

Bounds use binomial or CLT standard errors (stated per test) with fixed rng keys. A bound
of k standard errors fails by chance with probability about 6e-5 for k = 4 (two-sided).
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from numpy.typing import NDArray

from dogma_data import rng, synth


def _mann_whitney_auc(scores: NDArray[np.float64], y: NDArray[np.int64]) -> float:
    neg = np.sort(scores[y == 0])
    pos = scores[y == 1]
    less = np.searchsorted(neg, pos, side="left")
    leq = np.searchsorted(neg, pos, side="right")
    return float(np.sum(less + 0.5 * (leq - less)) / (pos.size * neg.size))


def _hanley_mcneil_se(auc: float, n_pos: int, n_neg: int) -> float:
    q1 = auc / (2 - auc)
    q2 = 2 * auc * auc / (1 + auc)
    var = (auc * (1 - auc) + (n_pos - 1) * (q1 - auc * auc) + (n_neg - 1) * (q2 - auc * auc)) / (
        n_pos * n_neg
    )
    return math.sqrt(var)


# --- correlated_mixed_scale ---


def test_correlated_mixed_scale_spans_scales_and_correlates() -> None:
    design = synth.correlated_mixed_scale(rng("props:mixed-scale"), 2000, 20)
    assert design.shape == (2000, 20)
    assert design.dtype == np.float64
    stds = design.std(axis=0)
    assert stds.max() / stds.min() >= 1e2
    corr = np.corrcoef(design, rowvar=False)
    upper = np.abs(corr[np.triu_indices(20, k=1)])
    assert float(upper.max()) > 0.5


def test_correlated_mixed_scale_no_factors_is_uncorrelated() -> None:
    design = synth.correlated_mixed_scale(
        rng("props:mixed-scale-noise"),
        5000,
        4,
        strength=1e-9,
        scale_range=(1.0, 1.0),
        offset_range=(0.0, 0.0),
    )
    corr = np.corrcoef(design, rowvar=False)
    off = corr[np.triu_indices(4, k=1)]
    assert float(np.max(np.abs(off))) < 0.06  # 4 SE of corr for n = 5000 is about 0.057


# --- heavy_tailed_noise ---


def test_heavy_tailed_noise_has_heavy_tails_and_scale() -> None:
    x = synth.heavy_tailed_noise(rng("props:heavy"), 50000, df=3.0, scale=2.0)
    assert x.shape == (50000,)
    # Student-t with df=3 has P(|T| > 6) near 0.01; a normal with the same scale has about 2e-9.
    assert float(np.mean(np.abs(x) > 12.0)) > 0.002
    assert float(np.mean(np.abs(x) > 12.0)) < 0.02


def test_heavy_tailed_noise_spread_scales_rows() -> None:
    n = 50000
    spread = np.where(np.arange(n) < n // 2, 1.0, 10.0)
    x = synth.heavy_tailed_noise(rng("props:spread"), n, df=5.0, spread=spread)
    ratio = float(np.mean(np.abs(x[n // 2 :])) / np.mean(np.abs(x[: n // 2])))
    assert 9.0 < ratio < 11.0


# --- imbalanced_labels ---


def test_imbalanced_labels_exact_positive_count() -> None:
    y = synth.imbalanced_labels(rng("props:imbalanced"), 10000, positive_rate=0.01)
    assert y.dtype == np.int64
    assert set(np.unique(y).tolist()) <= {0, 1}
    assert int(y.sum()) == 100


def test_imbalanced_labels_min_positives_floor() -> None:
    y = synth.imbalanced_labels(rng("props:min-pos"), 100, positive_rate=0.00001, min_positives=3)
    assert int(y.sum()) == 3


def test_imbalanced_labels_flip_rate_changes_exactly_that_many_labels() -> None:
    clean = synth.imbalanced_labels(rng("props:flip"), 5000, positive_rate=0.02, flip_rate=0.0)
    flipped = synth.imbalanced_labels(rng("props:flip"), 5000, positive_rate=0.02, flip_rate=0.03)
    assert int(np.sum(clean != flipped)) == round(5000 * 0.03)


# --- informative_scores ---


@pytest.mark.parametrize("target", [0.7, 0.8, 0.95])
def test_informative_scores_hit_target_auc(target: float) -> None:
    n = 20000
    y = np.arange(n) % 2
    scores = synth.informative_scores(rng(f"props:auc:{target}"), y, auc=target)
    assert float(scores.min()) > 0.0 and float(scores.max()) < 1.0
    empirical = _mann_whitney_auc(scores, y)
    se = _hanley_mcneil_se(target, n // 2, n // 2)
    assert abs(empirical - target) < 4 * se


def test_informative_scores_imbalanced_ratio_hits_target_auc() -> None:
    n = 40000
    y = synth.imbalanced_labels(rng("props:auc-imb-y"), n, positive_rate=0.01)
    scores = synth.informative_scores(rng("props:auc-imb"), y, auc=0.85)
    n_pos, n_neg = int(y.sum()), int(n - y.sum())
    se = _hanley_mcneil_se(0.85, n_pos, n_neg)
    assert abs(_mann_whitney_auc(scores, y) - 0.85) < 4 * se


def test_informative_scores_decimals_create_ties() -> None:
    y = np.arange(4000) % 2
    scores = synth.informative_scores(rng("props:ties"), y, auc=0.8, decimals=2)
    assert len(np.unique(scores)) <= 101
    np.testing.assert_allclose(scores, np.round(scores, 2))


# --- seasonal_ar_series ---


def test_seasonal_series_has_lag_period_autocorrelation() -> None:
    x = synth.seasonal_ar_series(rng("props:season"), 2400, period=12, amplitude=3.0)
    c = x - x.mean()
    lag12 = float(np.sum(c[:-12] * c[12:]) / np.sum(c * c))
    assert lag12 > 0.5


def test_seasonal_series_without_amplitude_has_weak_lag_period_autocorrelation() -> None:
    x = synth.seasonal_ar_series(rng("props:no-season"), 2400, period=12, amplitude=0.0)
    c = x - x.mean()
    lag12 = float(np.sum(c[:-12] * c[12:]) / np.sum(c * c))
    assert abs(lag12) < 0.1  # AR(1) phi=0.6 gives 0.6**12 = 0.002 at lag 12


def test_seasonal_series_nan_count_is_exact() -> None:
    x = synth.seasonal_ar_series(rng("props:missing"), 2400, missing_rate=0.01)
    assert int(np.isnan(x).sum()) == 24


def test_seasonal_series_level_shifts_are_exact_steps() -> None:
    x = synth.seasonal_ar_series(
        rng("props:shifts"),
        600,
        amplitude=0.0,
        phi=(0.0,),
        sigma=0.05,
        level_shifts=3,
        shift_size=10.0,
    )
    jumps = np.abs(np.diff(x))
    assert int(np.sum(jumps > 5.0)) == 3


def test_seasonal_series_trend_slope() -> None:
    x = synth.seasonal_ar_series(
        rng("props:trend"), 500, amplitude=0.0, phi=(0.0,), sigma=0.01, trend=0.5
    )
    slope = float(np.polyfit(np.arange(500), x, 1)[0])
    assert abs(slope - 0.5) < 0.01


# --- event_log ---


def test_event_log_user_skew_matches_zipf_share() -> None:
    n_users, zipf_a, n = 50, 1.3, 20000
    cols = synth.event_log(rng("props:zipf"), n, n_users=n_users, zipf_a=zipf_a)
    weights = 1.0 / np.arange(1, n_users + 1) ** zipf_a
    p_top = float(weights[0] / weights.sum())
    share = float(np.mean(cols["user_id"] == 1))
    assert share > 10 / n_users
    assert abs(share - p_top) < 4 * math.sqrt(p_top * (1 - p_top) / n)


def test_event_log_duplicates_are_exact() -> None:
    cols = synth.event_log(rng("props:dups"), 2000, duplicate_rate=0.05)
    assert cols["event_id"].size == 2000 + 100
    assert cols["event_id"].size - np.unique(cols["event_id"]).size == 100


def test_event_log_dirty_categories_at_roughly_dirty_rate() -> None:
    cats = ("web", "ios", "android")
    cols = synth.event_log(rng("props:dirty"), 20000, dirty_rate=0.3, categories=cats)
    clean = np.isin(cols["category"], np.array(cats))
    dirty_share = float(1.0 - clean.mean())
    assert abs(dirty_share - 0.3) < 4 * math.sqrt(0.3 * 0.7 / 20000)


def test_event_log_timestamps_naive_and_inside_window() -> None:
    cols = synth.event_log(rng("props:ts"), 3000, start="2024-01-01", days=90)
    ts = cols["ts"]
    assert ts.dtype == np.dtype("datetime64[s]")
    lo = np.datetime64("2024-01-01T00:00:00", "s")
    hi = lo + np.timedelta64(90, "D")
    assert bool(np.all(ts >= lo)) and bool(np.all(ts < hi))


def test_event_log_amounts_are_cents_and_lognormal() -> None:
    cols = synth.event_log(rng("props:amount"), 20000, amount_median=30.0, amount_sigma=1.0)
    amounts = cols["amount"]
    np.testing.assert_allclose(amounts * 100, np.round(amounts * 100), atol=1e-6)
    assert abs(float(np.median(amounts)) / 30.0 - 1.0) < 0.05
    assert float(np.mean(amounts)) > float(np.median(amounts))


def test_event_log_rows_are_shuffled() -> None:
    cols = synth.event_log(rng("props:shuffle"), 500)
    assert not bool(np.all(np.diff(cols["event_id"]) > 0))


# --- latencies ---


def test_latencies_have_a_heavy_right_tail() -> None:
    x = synth.latencies(rng("props:latency"), 20000, median_ms=50.0)
    p50, p99 = np.percentile(x, [50, 99])
    assert 47.5 < float(p50) < 53.0  # median shifts by about 1% from the body alone
    assert float(p99 / p50) > 10.0


def test_latencies_without_tail_are_lognormal_body() -> None:
    x = synth.latencies(rng("props:latency-body"), 20000, tail_fraction=0.0)
    p50, p99 = np.percentile(x, [50, 99])
    assert float(p99 / p50) < 5.0  # exp(0.5 * 2.326) = 3.2


# --- outlier_channel_activations ---


def test_outlier_channels_are_exactly_the_scaled_ones() -> None:
    acts = synth.outlier_channel_activations(
        rng("props:outliers"), 2000, 32, n_outlier_channels=3, outlier_scale=50.0
    )
    assert acts.shape == (2000, 32)
    stds = acts.std(axis=0)
    median_std = float(np.median(stds))
    assert int(np.sum(stds > 10 * median_std)) == 3


# --- peaked_probs ---


def test_peaked_probs_are_accurate_and_overconfident() -> None:
    n, k, acc = 20000, 10, 0.85
    y = rng("props:peaked-y").integers(0, k, size=n)
    probs = synth.peaked_probs(rng("props:peaked"), y, k, accuracy=acc)
    assert probs.shape == (n, k)
    np.testing.assert_allclose(probs.sum(axis=1), 1.0, atol=1e-12)
    assert bool(np.all(probs >= 0.0)) and bool(np.all(np.isfinite(probs)))
    argmax_acc = float(np.mean(np.argmax(probs, axis=1) == y))
    assert abs(argmax_acc - acc) < 4 * math.sqrt(acc * (1 - acc) / n)
    mean_confidence = float(probs.max(axis=1).mean())
    assert mean_confidence > acc + 0.05


def test_peaked_probs_stable_for_large_logits() -> None:
    y = np.zeros(50, dtype=np.int64)
    probs = synth.peaked_probs(rng("props:peaked-stable"), y, 3, overconfidence=500.0)
    assert bool(np.all(np.isfinite(probs)))
    np.testing.assert_allclose(probs.sum(axis=1), 1.0, atol=1e-12)


# --- argument validation: every public generator rejects bad input with ValueError ---

BAD_CALLS = [
    lambda: synth.correlated_mixed_scale(rng("bad"), 1, 3),
    lambda: synth.correlated_mixed_scale(rng("bad"), 5, 3, scale_range=(1.0, 0.5)),
    lambda: synth.correlated_mixed_scale(rng("bad"), 5, 3, strength=0.0),
    lambda: synth.correlated_mixed_scale(None, 5, 3),  # type: ignore[arg-type]
    lambda: synth.heavy_tailed_noise(rng("bad"), 5, df=0.0),
    lambda: synth.heavy_tailed_noise(rng("bad"), 5, spread=np.array([1.0, -1.0, 1, 1, 1])),
    lambda: synth.heavy_tailed_noise(rng("bad"), 5, spread=np.ones(4)),
    lambda: synth.imbalanced_labels(rng("bad"), 10, positive_rate=1.5),
    lambda: synth.imbalanced_labels(rng("bad"), 10, flip_rate=-0.1),
    lambda: synth.imbalanced_labels(rng("bad"), 10, min_positives=20),
    lambda: synth.informative_scores(rng("bad"), np.zeros(10), auc=0.8),
    lambda: synth.informative_scores(rng("bad"), np.array([0, 1, 2]), auc=0.8),
    lambda: synth.informative_scores(rng("bad"), np.array([0, 1, 1]), auc=0.5),
    lambda: synth.informative_scores(rng("bad"), np.array([0, 1, 1]), auc=1.0),
    lambda: synth.seasonal_ar_series(rng("bad"), 50, phi=(0.6, 0.5)),
    lambda: synth.seasonal_ar_series(rng("bad"), 50, phi=()),
    lambda: synth.seasonal_ar_series(rng("bad"), 5, level_shifts=10),
    lambda: synth.seasonal_ar_series(rng("bad"), 50, missing_rate=2.0),
    lambda: synth.seasonal_ar_series(rng("bad"), 50, period=1),
    lambda: synth.event_log(rng("bad"), 0),
    lambda: synth.event_log(rng("bad"), 10, dirty_rate=1.5),
    lambda: synth.event_log(rng("bad"), 10, categories=()),
    lambda: synth.event_log(rng("bad"), 10, start="not-a-date"),
    lambda: synth.latencies(rng("bad"), 10, tail_multiplier=0.5),
    lambda: synth.latencies(rng("bad"), 10, median_ms=-1.0),
    lambda: synth.outlier_channel_activations(rng("bad"), 4, 2, n_outlier_channels=3),
    lambda: synth.peaked_probs(rng("bad"), np.array([0, 3]), 3),
    lambda: synth.peaked_probs(rng("bad"), np.array([0.0, 1.0]), 3),
    lambda: synth.peaked_probs(rng("bad"), np.array([0, 1]), 3, accuracy=1.2),
]


@pytest.mark.parametrize("call", BAD_CALLS)
def test_invalid_arguments_raise_value_error(call: object) -> None:
    with pytest.raises(ValueError):
        call()  # type: ignore[operator]


def test_outputs_are_fresh_not_shared() -> None:
    a = synth.latencies(rng("props:fresh"), 10)
    a[:] = 0.0
    b = synth.latencies(rng("props:fresh"), 10)
    assert float(b.sum()) > 0.0
