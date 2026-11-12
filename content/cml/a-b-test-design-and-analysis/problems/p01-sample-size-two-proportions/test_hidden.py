import numpy as np
import pytest
from scipy import stats
from submission import sample_size_two_proportions

from dogma_data import load, retail_invoices

# (p_baseline, mde, alpha, power). Chosen so the continuous n sits well away from an integer,
# which makes ceil differ from round (for example at 0.30 + 0.05) and from floor everywhere.
GRID: list[tuple[float | str, float, float, float]] = [
    (0.10, 0.02, 0.05, 0.8),
    (0.30, 0.05, 0.05, 0.8),
    (0.05, 0.01, 0.05, 0.8),
    (0.50, 0.05, 0.05, 0.8),
    (0.02, 0.005, 0.05, 0.8),
    (0.20, -0.04, 0.05, 0.8),
    (0.10, 0.02, 0.01, 0.9),
    (0.40, 0.03, 0.01, 0.9),
    (0.25, 0.025, 0.05, 0.9),
    (0.50, 0.001, 0.05, 0.8),
]


# Baseline rates counted from real tables, each with an effect a team might look for. The name
# is replaced by the rate inside the test (`_baseline`), so no dataset is read at import.
GRID += [
    ("cancelled-invoices", -0.03, 0.05, 0.8),
    ("malignant-tumours", 0.05, 0.05, 0.9),
    ("improved-grades", 0.04, 0.01, 0.8),
]


def _baseline(value):
    """A number as it is; a name as the real rate it stands for.

    The share of invoices that were cancelled (Online Retail II sample), the share of tumours
    that were malignant (breast_cancer) and the share of students whose grade improved
    (spector).
    """
    if value == "cancelled-invoices":
        return float(retail_invoices()["cancelled"].mean())
    if value == "malignant-tumours":
        return float(1.0 - load("breast_cancer").y.mean())
    if value == "improved-grades":
        return float(load("spector").y.mean())
    return value


def _reference(p_baseline, mde, alpha, power):
    """Independent implementation of the two-proportion sample size formula."""
    p1 = p_baseline
    p2 = p_baseline + mde
    pbar = (p1 + p2) / 2.0
    z_alpha = stats.norm.ppf(1.0 - alpha / 2.0)
    z_beta = stats.norm.ppf(power)
    n = (
        z_alpha * np.sqrt(2.0 * pbar * (1.0 - pbar))
        + z_beta * np.sqrt(p1 * (1.0 - p1) + p2 * (1.0 - p2))
    ) ** 2 / mde**2
    return int(np.ceil(n))


def _simulated_power(p1, p2, n, alpha, *, reps=20000, seed=3):
    """Power of the pooled two-proportion z-test, by simulation at n users per arm."""
    rng = np.random.default_rng(seed)
    x_a = rng.binomial(n, p1, size=reps)
    x_b = rng.binomial(n, p2, size=reps)
    p_a = x_a / n
    p_b = x_b / n
    pooled = (x_a + x_b) / (2.0 * n)
    se = np.sqrt(pooled * (1.0 - pooled) * (2.0 / n))
    usable = se > 0
    z = np.zeros(reps)
    z[usable] = (p_b - p_a)[usable] / se[usable]
    critical = stats.norm.ppf(1.0 - alpha / 2.0)
    return float(np.mean(np.abs(z) > critical))


def _must_raise_value_error(call, mistake):
    """Fail with the kind of mistake, instead of pytest's generic 'DID NOT RAISE'."""
    try:
        call()
    except ValueError:
        return
    except Exception as exc:
        pytest.fail(f"{mistake} (raised {type(exc).__name__} instead)")
    pytest.fail(mistake)


@pytest.mark.parametrize(
    ("p_baseline", "mde", "alpha", "power"),
    GRID,
    ids=[f"grid{index}" for index in range(len(GRID))],
)
def test_matches_independent_formula_on_grid(p_baseline, mde, alpha, power):
    p_baseline = _baseline(p_baseline)
    assert sample_size_two_proportions(p_baseline, mde, alpha=alpha, power=power) == _reference(
        p_baseline, mde, alpha, power
    ), (
        "users per arm differ from the two-proportion formula: pooled variance under H0, unpooled "
        "under H1, two-sided z_alpha, and ceil"
    )


@pytest.mark.parametrize(
    ("p_baseline", "mde"),
    [(0.10, 0.02), (0.20, -0.04)],
    ids=["power-case-a", "power-case-b"],
)
def test_simulated_power_at_the_returned_size_is_near_the_target(p_baseline, mde):
    n = sample_size_two_proportions(p_baseline, mde)
    power = _simulated_power(p_baseline, p_baseline + mde, n, 0.05)
    assert abs(power - 0.8) <= 0.02, (
        "the returned size does not reach the target power in simulation: check the variance terms "
        "and the factor of two between total and per-arm users"
    )


def test_more_power_smaller_effect_or_smaller_alpha_need_more_users():
    base = sample_size_two_proportions(0.10, 0.02)
    assert sample_size_two_proportions(0.10, 0.03) < base, (
        "a larger effect needs fewer users: check that mde enters squared in the denominator"
    )
    assert sample_size_two_proportions(0.10, 0.02, power=0.9) > base, (
        "higher power needs more users: check which quantile uses the power argument"
    )
    assert sample_size_two_proportions(0.10, 0.02, alpha=0.01) > base, (
        "a stricter alpha needs more users: check the two-sided critical value uses alpha / 2"
    )


def test_return_value_is_an_integer_count_of_users():
    n = sample_size_two_proportions(0.30, 0.05)
    assert isinstance(n, (int, np.integer)) and not isinstance(n, bool), (
        "the sample size must be an integer count of users: return math.ceil of the continuous n"
    )


def test_very_small_effect_returns_a_large_integer_without_overflow():
    n = sample_size_two_proportions(0.50, 0.001)
    assert n == _reference(0.50, 0.001, 0.05, 0.8), (
        "a tiny effect needs millions of users: the value must still be the ceiling of the formula"
    )


INVALID_CASES = [
    (
        "baseline_zero",
        {"p_baseline": 0.0, "mde": 0.02},
        "a zero baseline is not a rate: raise ValueError",
    ),
    (
        "baseline_one",
        {"p_baseline": 1.0, "mde": 0.02},
        "a baseline of one is not a rate: raise ValueError",
    ),
    (
        "baseline_negative",
        {"p_baseline": -0.1, "mde": 0.02},
        "a negative baseline must raise ValueError",
    ),
    (
        "baseline_nan",
        {"p_baseline": float("nan"), "mde": 0.02},
        "a NaN baseline must raise ValueError",
    ),
    ("mde_zero", {"p_baseline": 0.10, "mde": 0.0}, "mde = 0 must raise ValueError"),
    (
        "treatment_above_one",
        {"p_baseline": 0.98, "mde": 0.05},
        "p_baseline + mde above 1 must raise ValueError",
    ),
    (
        "treatment_below_zero",
        {"p_baseline": 0.05, "mde": -0.10},
        "p_baseline + mde below 0 must raise ValueError",
    ),
    (
        "alpha_zero",
        {"p_baseline": 0.10, "mde": 0.02, "alpha": 0.0},
        "alpha = 0 must raise ValueError",
    ),
    (
        "alpha_one",
        {"p_baseline": 0.10, "mde": 0.02, "alpha": 1.0},
        "alpha = 1 must raise ValueError",
    ),
    (
        "power_zero",
        {"p_baseline": 0.10, "mde": 0.02, "power": 0.0},
        "power = 0 must raise ValueError",
    ),
    (
        "power_one",
        {"p_baseline": 0.10, "mde": 0.02, "power": 1.0},
        "power = 1 must raise ValueError",
    ),
]


@pytest.mark.parametrize(
    ("case", "kwargs", "mistake"),
    INVALID_CASES,
    ids=[case[0] for case in INVALID_CASES],
)
def test_invalid_input_raises_value_error(case, kwargs, mistake):
    _must_raise_value_error(lambda: sample_size_two_proportions(**kwargs), mistake)


def test_treatment_rate_exactly_one_raises_value_error():
    _must_raise_value_error(
        lambda: sample_size_two_proportions(0.5, 0.5),
        "a treatment rate of exactly 1 was accepted: p_baseline + mde must be strictly below 1",
    )


def test_treatment_rate_exactly_zero_raises_value_error():
    _must_raise_value_error(
        lambda: sample_size_two_proportions(0.5, -0.5),
        "a treatment rate of exactly 0 was accepted: p_baseline + mde must be strictly above 0",
    )


def test_treatment_rate_above_one_with_no_domain_error_raises_value_error():
    _must_raise_value_error(
        lambda: sample_size_two_proportions(0.5, 0.6),
        "a treatment rate of 1.1 was accepted: p_baseline + mde must be strictly below 1",
    )


def test_baseline_exactly_one_with_a_negative_effect_raises_value_error():
    _must_raise_value_error(
        lambda: sample_size_two_proportions(1.0, -0.02),
        "a baseline of exactly 1 was accepted: p_baseline must be strictly below 1",
    )


def test_baseline_above_one_with_a_valid_treatment_rate_raises_value_error():
    _must_raise_value_error(
        lambda: sample_size_two_proportions(1.2, -0.7),
        "a baseline above 1 was accepted: p_baseline must be strictly between 0 and 1",
    )


def test_control_rate_enters_the_formula_with_all_its_digits():
    # The control rate is used as given, p1 = p_baseline. Rounding it to six decimals moves the
    # count by several users at this effect size, so the ceiling differs.
    p_baseline, mde = 0.2345678, 0.001
    assert sample_size_two_proportions(p_baseline, mde) == _reference(p_baseline, mde, 0.05, 0.8), (
        "the control rate must enter the formula unchanged: p1 = p_baseline, with every digit"
    )
