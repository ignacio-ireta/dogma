import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from scipy.special import ndtr
from submission import early_stopping

from dogma_data import rng as data_rng
from dogma_data import sample_rows


def _real_normal(gen, size=None, loc=0.0, scale=1.0):
    """Real numbers where this test used to draw from a normal distribution.

    Standardised breast_cancer measurements (mean 0 and standard deviation 1 over the dataset,
    skewed, with a few values far out), drawn by ``gen``. A vector is one real measurement
    over that many tumours, a matrix is real tumours by real measurements, and any other
    shape is filled from whole rows. ``loc`` and ``scale`` shift and stretch them as before.
    """
    shape = () if size is None else tuple(int(side) for side in np.atleast_1d(size))
    total = int(np.prod(shape)) if shape else 1
    if len(shape) == 1 and 1 <= shape[0] <= 569:
        values = sample_rows("breast_cancer", gen, shape[0], 1, standardize=True)[0][:, 0]
    elif len(shape) == 2 and 1 <= shape[0] <= 569 and 1 <= shape[1] <= 30:
        values = sample_rows("breast_cancer", gen, shape[0], shape[1], standardize=True)[0]
    else:
        rows = max(1, min(569, -(-total // 30)))
        drawn = sample_rows("breast_cancer", gen, rows, standardize=True)[0]
        values = np.resize(drawn.ravel(), total).reshape(shape)
    values = values * scale + loc
    return values if shape else float(values)


def _real_unit(gen, size=None):
    """Real numbers between 0 and 1 where this test used to draw uniform ones.

    Each is the share of a standard normal that lies below a standardised breast_cancer
    measurement, so the values keep the order, the ties and the skew of real measurements.
    """
    return ndtr(_real_normal(gen, size))


def _real_ints(gen, low, high, size=None):
    """Real whole numbers from ``low`` up to ``high - 1``: the band a real measurement falls in.

    The range is cut into ``high - low`` bands of equal normal probability, and each value is
    the band of one standardised measurement. Real measurements are skewed, so the bands are
    not equally full, and neighbouring values tie as real categories and grades do.
    """
    bands = np.floor(_real_unit(gen, size) * (high - low))
    return (low + np.minimum(bands, high - low - 1)).astype(np.int64)


def _real_shares(gen, alpha, size=None):
    """Real probability vectors: positive real numbers (see `_real_unit`) divided by their sum."""
    k = int(np.size(alpha))
    shape = (k,) if size is None else (int(size), k)
    weights = _real_unit(gen, shape) + 1e-3
    return weights / weights.sum(axis=-1, keepdims=True)


def _validation_losses(gen, length):
    """Validation losses of a real training run, one per epoch.

    A logistic model is trained on 60 breast_cancer tumours (five standardised features) and
    scored on 60 others. Each epoch is one large step on four tumours, so the curve is noisy:
    it falls, jumps back up and levels off, as a small-batch run does. The recorded loss is
    four times the mean validation log-loss, rounded to a multiple of 0.25, so equal losses
    are common and every comparison at the improvement boundary is exact.
    """
    X, y = sample_rows("breast_cancer", gen, 120, 5, standardize=True)
    w = np.zeros(5)
    losses = []
    for _ in range(length):
        rows = _real_ints(gen, 0, 60, 4)
        p = 1.0 / (1.0 + np.exp(-(X[rows] @ w)))
        w = w - 3.0 * X[rows].T @ (p - y[rows]) / 4.0
        z = X[60:] @ w
        loss = float(np.mean(np.logaddexp(0.0, z) - y[60:] * z))
        losses.append(float(np.round(4.0 * loss * 4.0) / 4.0))
    return losses


def _call(losses, *, patience, min_delta=0.0):
    """Call the learner's function and check the result type, naming the mistake if it is off."""
    try:
        result = early_stopping(losses, patience=patience, min_delta=min_delta)
    except ValueError:
        pytest.fail("a valid sequence raised ValueError: check the patience and improvement rules")
    except Exception as exc:
        pytest.fail(f"a valid sequence raised {type(exc).__name__}: check the bookkeeping")
    if not (isinstance(result, tuple) and len(result) == 2):
        pytest.fail("the result must be a (best_epoch, stop_epoch) pair")
    best, stop = result
    if not (isinstance(best, int) and isinstance(stop, int)):
        pytest.fail("best_epoch and stop_epoch must be Python ints, not arrays or floats")
    return best, stop


def _require(condition, mistake):
    if not condition:
        pytest.fail(mistake)


def _reference(losses, patience, min_delta):
    """Brute force, written differently: list the improving epochs, then find the first stall."""
    improving = []
    best = float("inf")
    for epoch, loss in enumerate(losses):
        if loss < best - min_delta:
            improving.append(epoch)
            best = loss
    for epoch in range(len(losses)):
        last = (
            max(e for e in improving if e <= epoch) if any(e <= epoch for e in improving) else None
        )
        if last is not None and epoch - last >= patience:
            return last, epoch
    return improving[-1], len(losses) - 1


# Values are multiples of 0.25 and min_delta is a multiple of 0.25 too, so every comparison at the
# improvement boundary is exact in floating point.
@pytest.mark.parametrize(
    ("patience", "min_delta", "seed"),
    [(1, 0.0, 0), (2, 0.0, 1), (3, 0.25, 2), (4, 0.5, 3), (2, 0.25, 4), (5, 0.0, 5)],
    ids=[
        "patience-one",
        "patience-two",
        "small-delta",
        "large-delta",
        "mid-delta",
        "patience-five",
    ],
)
def test_matches_the_brute_force_reference_on_seeded_sequences(patience, min_delta, seed):
    rng = data_rng(f"early-stopping-p01:brute-force:{seed}")
    for _ in range(40):
        length = int(rng.integers(1, 25))
        losses = _validation_losses(rng, length)
        exp_best, exp_stop = _reference(losses, patience, min_delta)
        got_best, got_stop = _call(losses, patience=patience, min_delta=min_delta)
        _require(
            0 <= got_stop <= length - 1,
            "the stop epoch must lie inside the sequence: without a stop it is the last epoch",
        )
        _require(
            got_stop == exp_stop,
            "the stop must be the first epoch whose wait, counted from the last improvement,"
            " reaches patience: a tie with the best is not an improvement and does not reset it",
        )
        _require(
            got_best == exp_best,
            "the best checkpoint must be the last epoch that beat the best loss by more than"
            " min_delta: ties and exact min_delta decreases do not count",
        )


def test_a_decrease_of_exactly_min_delta_is_not_an_improvement():
    _require(
        _call([1.0, 0.75, 0.6], patience=5, min_delta=0.25) == (2, 2),
        "a decrease of exactly min_delta is not an improvement: use a strict comparison",
    )
    _require(
        _call([1.0, 0.5, 0.25], patience=1, min_delta=0.5) == (0, 1),
        "a decrease of exactly min_delta must not reset the wait counter",
    )


def test_ties_with_the_best_loss_are_not_improvements():
    _require(
        _call([0.5, 0.5, 0.5, 0.5], patience=2) == (0, 2),
        "ties with the best loss are not improvements: the best checkpoint is the first one",
    )


def test_wait_counter_resets_after_each_improvement():
    losses = [1.0, 1.25, 1.25, 0.75, 1.0, 1.0, 1.0]
    _require(
        _call(losses, patience=3) == (3, 6),
        "the wait counter must restart from zero after every improvement",
    )


def test_patience_one_stops_at_the_first_epoch_without_improvement():
    _require(
        _call([1.0, 2.0, 0.5, 0.75], patience=1) == (0, 1),
        "with patience 1 the first non-improving epoch must stop training",
    )


def test_single_epoch_is_its_own_best_checkpoint_and_stop():
    _require(_call([4.25], patience=1) == (0, 0), "a single epoch must give (0, 0)")


def test_stop_comes_exactly_patience_epochs_after_the_best_checkpoint():
    rng = data_rng("early-stopping-p01:patience-after-best")
    for _ in range(60):
        losses = _validation_losses(rng, 30)
        patience = int(rng.integers(1, 6))
        best, stop = _call(losses, patience=patience)
        if stop < len(losses) - 1:
            _require(
                stop - best == patience,
                "a stop must come exactly patience epochs after the best checkpoint",
            )


@settings(max_examples=100, deadline=None, report_multiple_bugs=False)
@given(
    st.lists(st.integers(min_value=-8, max_value=8), min_size=1, max_size=30),
    st.integers(min_value=1, max_value=6),
)
def test_with_zero_min_delta_the_best_checkpoint_is_the_first_running_minimum(raw, patience):
    losses = [v / 4.0 for v in raw]
    best, stop = _call(losses, patience=patience, min_delta=0.0)
    _require(0 <= best <= stop <= len(losses) - 1, "epochs must satisfy 0 <= best <= stop <= last")
    prefix = losses[: stop + 1]
    _require(
        losses[best] == min(prefix) and best == prefix.index(min(prefix)),
        "the best checkpoint must be the first epoch holding the lowest loss seen before the stop",
    )


INVALID = [
    ("empty", [], 2, 0.0, "an empty loss sequence has no epochs: raise ValueError"),
    ("two_dimensional", [[1.0, 0.5], [0.4, 0.3]], 2, 0.0, "losses must be 1-D: raise ValueError"),
    ("nan_loss", [1.0, float("nan"), 0.5], 2, 0.0, "a NaN loss must raise ValueError"),
    ("zero_patience", [1.0, 0.5], 0, 0.0, "patience below 1 must raise ValueError"),
    ("negative_min_delta", [1.0, 0.5], 2, -0.1, "a negative min_delta must raise ValueError"),
]


@pytest.mark.parametrize(
    ("case", "losses", "patience", "min_delta", "mistake"),
    INVALID,
    ids=[case[0] for case in INVALID],
)
def test_invalid_arguments_raise_value_error(case, losses, patience, min_delta, mistake):
    try:
        early_stopping(losses, patience=patience, min_delta=min_delta)
    except ValueError:
        return
    except Exception as exc:
        pytest.fail(f"{mistake} (raised {type(exc).__name__} instead)")
    pytest.fail(mistake)


def test_input_array_is_not_mutated():
    losses = np.array([0.9, 0.4, 0.6, 0.7, 0.8])
    before = losses.copy()
    early_stopping(losses, patience=2)
    _require(bool(np.array_equal(losses, before)), "the loss array was modified in place")


def test_tiny_improvement_below_single_precision_resolution_is_an_improvement():
    _require(
        early_stopping([1.0, 1.0 - 1e-9], patience=1) == (1, 1),
        "a loss lower than the best by 1e-9 is an improvement: compare the losses at full "
        "float64 precision, not after rounding them to a coarser format",
    )


def test_default_min_delta_is_zero_so_a_tiny_decrease_counts():
    # The default min_delta is 0.0, so a decrease of 1e-13 from 1.0 is an improvement:
    # epoch 1 is the best checkpoint, and with patience 1 the stop is the last epoch.
    got = early_stopping([1.0, 1.0 - 1e-13], patience=1)
    _require(
        tuple(got) == (1, 1),
        "the default min_delta is 0.0, so any decrease in loss must count as an improvement",
    )


def test_non_finite_min_delta_raises_value_error():
    # min_delta must be finite: infinity and NaN are both invalid and must raise ValueError.
    for value in (float("inf"), float("nan")):
        try:
            early_stopping([1.0, 0.5], patience=2, min_delta=value)
        except ValueError:
            continue
        pytest.fail("a min_delta that is not finite must raise ValueError")
