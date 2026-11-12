import math

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from submission import population_stability_index

from dogma_data import rng as data_rng
from dogma_data import sample_rows


def _incomes(gen, n):
    """Median income of n real california_housing block groups, standardised.

    The feature is right-skewed, with a long upper tail and a cap. A drifted sample is a
    second draw of real values, shifted and rescaled by the amount the case states.
    """
    return sample_rows("california_housing", gen, n, standardize=True)[0][:, 0]


def _require(ok, mistake):
    """Fail with the mistake's name only. Hypothesis and the grader show this first line."""
    if not ok:
        raise AssertionError(mistake)


def _run(call, mistake):
    try:
        result = call()
    except Exception as exc:
        raise AssertionError(f"{mistake} (raised {type(exc).__name__} instead)") from None
    _require(isinstance(result, float), "PSI must be returned as a Python float")
    return result


def _expect_value_error(call, mistake):
    try:
        call()
    except ValueError:
        return
    except Exception as exc:
        raise AssertionError(f"{mistake} (raised {type(exc).__name__} instead)") from None
    raise AssertionError(mistake)


def _linear_quantile(sorted_values, q):
    """Linear interpolation written out: position q * (n - 1) from the smallest value."""
    n = len(sorted_values)
    position = q * (n - 1)
    lower = math.floor(position)
    upper = min(lower + 1, n - 1)
    fraction = position - lower
    return sorted_values[lower] + fraction * (sorted_values[upper] - sorted_values[lower])


def _reference(expected, actual, bins, eps, *, edge_sample=None, equal_width=False):
    """Independent loop-based PSI: explicit quantiles, explicit counting, explicit sum.

    The keyword options build the common wrong conventions, so a failing learner can be named.
    """
    reference = sorted(float(v) for v in expected)
    if equal_width:
        low, high = reference[0], reference[-1]
        edges = [low + (high - low) * k / bins for k in range(1, bins)]
    else:
        source = reference if edge_sample is None else sorted(float(v) for v in edge_sample)
        edges = [_linear_quantile(source, k / bins) for k in range(1, bins)]

    def shares(values, size):
        counts = [0] * bins
        for value in values:
            counts[sum(1 for edge in edges if edge < value)] += 1
        return [count / size if count else eps for count in counts]

    p = shares(reference, len(reference))
    actual_values = [float(v) for v in actual]
    q = shares(actual_values, len(actual_values))
    return sum((qi - pi) * math.log(qi / pi) for pi, qi in zip(p, q, strict=True))


def _check_psi(psi, correct, expected, actual, bins, eps, mistake):  # noqa: PLR0917
    """Compare with the reference computed before the learner ran.

    The wrong-convention values are computed only after a mismatch, from the same inputs.
    """
    if math.isclose(psi, correct, rel_tol=1e-9, abs_tol=1e-12):
        return
    wrong_conventions = [
        (
            "bins were taken from the actual sample: the expected sample must define the edges",
            _reference(expected, actual, bins, eps, edge_sample=actual),
        ),
        (
            "bins must be the expected quantiles, not equal-width bins over the expected range",
            _reference(expected, actual, bins, eps, equal_width=True),
        ),
        ("the logarithm must be natural (ln), not base 10", correct / math.log(10.0)),
    ]
    for named_mistake, wrong_value in wrong_conventions:
        if math.isclose(psi, wrong_value, rel_tol=1e-9, abs_tol=1e-12):
            raise AssertionError(named_mistake)
    raise AssertionError(mistake)


@pytest.mark.parametrize(
    ("seed", "n_expected", "n_actual", "bins", "shift", "spread"),
    [
        (0, 500, 400, 10, 0.0, 1.0),
        (1, 300, 900, 5, 0.3, 1.0),
        (2, 1000, 250, 20, -0.5, 1.5),
        (3, 80, 60, 4, 1.0, 0.5),
        (4, 2000, 2000, 10, 0.0, 1.0),
    ],
    ids=["ten-bins", "five-bins-shifted", "twenty-bins", "small-samples", "equal-sizes"],
)
def test_matches_loop_reference_on_seeded_samples(  # noqa: PLR0917
    seed, n_expected, n_actual, bins, shift, spread
):
    rng = data_rng(f"psi-p01:reference:{seed}")
    expected = _incomes(rng, n_expected)
    actual = _incomes(rng, n_actual) * spread + shift
    correct = _reference(expected, actual, bins, 1e-6)
    psi = _run(
        lambda: population_stability_index(expected, actual, bins=bins),
        "population_stability_index raised on valid samples",
    )
    _check_psi(
        psi, correct, expected, actual, bins, 1e-6, "PSI differs from the bin-rule reference"
    )


def test_value_exactly_on_an_edge_belongs_to_the_lower_bin():
    expected = np.arange(101.0)  # quartile edges are exactly 25, 50 and 75
    actual = np.array([25.0, 50.0, 50.0, 75.0, 10.0, 60.0, 0.0, 100.0])
    correct = _reference(expected, actual, 4, 1e-6)
    psi = _run(
        lambda: population_stability_index(expected, actual, bins=4),
        "population_stability_index raised on integer samples",
    )
    _check_psi(
        psi,
        correct,
        expected,
        actual,
        4,
        1e-6,
        "values on an edge must go to the lower bin: an edge value belongs to the bin below it",
    )


def test_actual_values_outside_the_expected_range_are_kept_in_the_outer_bins():
    rng = data_rng("psi-p01:beyond-range")
    expected = _incomes(rng, 400)
    actual = np.concatenate([_incomes(rng, 300), _incomes(rng, 100) + 10.0])
    correct = _reference(expected, actual, 10, 1e-6)
    psi = _run(
        lambda: population_stability_index(expected, actual, bins=10),
        "population_stability_index raised on samples with values beyond the reference range",
    )
    _check_psi(
        psi,
        correct,
        expected,
        actual,
        10,
        1e-6,
        "values outside the expected range must stay in the outer bins, not be dropped",
    )


def test_bins_come_from_the_expected_sample_not_the_actual_one():
    rng = data_rng("psi-p01:shifted")
    expected = _incomes(rng, 600)
    actual = _incomes(rng, 600) + 2.0
    correct = _reference(expected, actual, 10, 1e-6)
    psi = _run(
        lambda: population_stability_index(expected, actual, bins=10),
        "population_stability_index raised on a shifted sample",
    )
    _require(
        psi > 0.25,
        "a sample shifted by two standard deviations looks stable: bins must come from expected",
    )
    _check_psi(
        psi,
        correct,
        expected,
        actual,
        10,
        1e-6,
        "PSI of a shifted sample differs from the reference",
    )


@pytest.mark.parametrize("eps", [1e-6, 1e-3], ids=["default-eps", "larger-eps"])
def test_empty_bins_take_eps_as_their_floor(eps):
    expected = np.linspace(1.0, 40.0, 40)
    actual = np.linspace(1.0, 10.0, 25)  # every actual value falls in the lowest bins
    correct = _reference(expected, actual, 4, eps)
    psi = _run(
        lambda: population_stability_index(expected, actual, bins=4, eps=eps),
        "population_stability_index raised when a bin is empty",
    )
    _require(math.isfinite(psi), "an empty bin made PSI non-finite: floor zero shares at eps")
    _check_psi(
        psi, correct, expected, actual, 4, eps, "empty-bin handling differs from the eps floor"
    )


def test_constant_reference_and_a_moved_actual_sample_are_handled():
    expected = np.full(20, 5.0)
    correct_moved = _reference(expected, np.full(20, 6.0), 10, 1e-6)
    moved = population_stability_index(expected, np.full(20, 6.0), bins=10)
    unchanged = population_stability_index(expected, np.full(20, 5.0), bins=10)
    _require(
        math.isfinite(moved) and moved > 0.0,
        "a constant reference with a moved sample must give a positive, finite PSI",
    )
    _check_psi(
        moved,
        correct_moved,
        expected,
        np.full(20, 6.0),
        10,
        1e-6,
        "constant reference PSI differs from reference",
    )
    _require(
        unchanged == 0.0,
        "a constant reference with an identical constant sample must give PSI exactly 0",
    )


def test_shifting_and_rescaling_both_samples_leaves_psi_unchanged():
    rng = data_rng("psi-p01:shift-and-scale")
    expected = _incomes(rng, 300)
    actual = _incomes(rng, 200) + 0.4
    base = _run(lambda: population_stability_index(expected, actual, bins=8), "call raised")
    shifted = _run(
        lambda: population_stability_index(expected + 7.0, actual + 7.0, bins=8),
        "call raised on shifted samples",
    )
    scaled = _run(
        lambda: population_stability_index(expected * 3.0, actual * 3.0, bins=8),
        "call raised on scaled samples",
    )
    _require(
        math.isclose(shifted, base, rel_tol=1e-9, abs_tol=1e-12),
        "adding one constant to both samples changed PSI: bins must be location-free",
    )
    _require(
        math.isclose(scaled, base, rel_tol=1e-9, abs_tol=1e-12),
        "multiplying both samples by one factor changed PSI: bins must be scale-free",
    )


def test_identical_samples_have_exactly_zero_psi():
    values = _incomes(data_rng("psi-p01:identical"), 150)
    psi = _run(lambda: population_stability_index(values, values.copy(), bins=7), "call raised")
    _require(psi == 0.0, "identical samples must give exactly zero PSI")


@pytest.mark.parametrize("bins", [1, 0, 2.5, True], ids=["one", "zero", "fraction", "bool"])
def test_invalid_bins_raise_value_error(bins):
    _expect_value_error(
        lambda: population_stability_index([1.0, 2.0, 3.0], [1.0, 2.0], bins=bins),
        "an invalid bins was accepted: bins must be an integer of at least 2",
    )


@pytest.mark.parametrize("eps", [0.0, -1e-6], ids=["zero", "negative"])
def test_non_positive_eps_raises_value_error(eps):
    _expect_value_error(
        lambda: population_stability_index([1.0, 2.0, 3.0], [1.0, 2.0], bins=2, eps=eps),
        "a non-positive eps was accepted: eps must be a positive number",
    )


def test_empty_sample_raises_value_error():
    _expect_value_error(
        lambda: population_stability_index([], [1.0, 2.0], bins=2),
        "an empty expected sample was accepted: it must contain at least one value",
    )


@pytest.mark.parametrize(
    ("expected", "actual"),
    [([1.0, float("nan"), 3.0], [1.0, 2.0]), ([1.0, 2.0, 3.0], [float("inf"), 2.0])],
    ids=["nan-in-expected", "infinity-in-actual"],
)
def test_non_finite_values_raise_value_error(expected, actual):
    _expect_value_error(
        lambda: population_stability_index(expected, actual, bins=2),
        "a NaN or infinite value was accepted: samples must be finite",
    )


def test_two_dimensional_input_raises_value_error():
    _expect_value_error(
        lambda: population_stability_index(np.ones((4, 2)), [1.0, 2.0], bins=2),
        "a two-dimensional sample was accepted: samples must be one-dimensional",
    )


def test_integer_input_gives_a_float_result():
    psi = _run(
        lambda: population_stability_index([1, 2, 3, 4], [1, 2, 3, 4, 4, 4], bins=2),
        "population_stability_index raised on integer input",
    )
    _require(isinstance(psi, float), "PSI must be returned as a float for integer input")


def test_inputs_are_not_mutated():
    expected = np.array([3.0, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0, 6.0])
    actual = np.array([2.0, 7.0, 1.0, 8.0])
    e_before, a_before = expected.copy(), actual.copy()
    _run(lambda: population_stability_index(expected, actual, bins=3), "call raised")
    _require(np.array_equal(expected, e_before), "expected was modified in place")
    _require(np.array_equal(actual, a_before), "actual was modified in place")


@settings(max_examples=100, deadline=None, report_multiple_bugs=False)
@given(
    st.lists(
        st.floats(-100.0, 100.0, allow_nan=False, allow_infinity=False), min_size=2, max_size=60
    ),
    st.lists(
        st.floats(-100.0, 100.0, allow_nan=False, allow_infinity=False), min_size=2, max_size=60
    ),
    st.integers(min_value=2, max_value=12),
)
def test_property_matches_reference_and_is_never_negative(expected, actual, bins):
    correct = _reference(expected, actual, bins, 1e-6)
    psi = _run(
        lambda: population_stability_index(expected, actual, bins=bins),
        "population_stability_index raised on valid lists",
    )
    _require(math.isfinite(psi), "PSI must be finite for finite samples")
    _require(
        psi >= -1e-12, "PSI must never be negative: each term (q - p) ln(q / p) is non-negative"
    )
    _check_psi(
        psi,
        correct,
        expected,
        actual,
        bins,
        1e-6,
        "PSI differs from the bin-rule reference on a list",
    )


def test_default_number_of_bins_is_ten():
    rng = data_rng("psi-p01:default-bins")
    expected = _incomes(rng, 400)
    actual = _incomes(rng, 300) + 0.5
    default = population_stability_index(expected, actual)
    explicit = population_stability_index(expected, actual, bins=10)
    _require(default == explicit, "the default for bins must be 10, as the signature states")
