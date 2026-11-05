# Public tests: the examples from problem.md plus an edge case (CONTENT_SPEC §6.4).
# The learner's code is imported from submission.py; tests never import solution.py.
import numpy as np
import pytest
from submission import fit_power_law


def test_exact_square_root_law_recovers_coefficients():
    x = np.array([1.0, 2.0, 4.0, 8.0])
    a, b = fit_power_law(x, 2.0 * np.sqrt(x))
    assert abs(a - 2.0) < 1e-9, "the intercept must give a = 2 (check exp of the intercept)"
    assert abs(b - 0.5) < 1e-9, "the slope must give b = 0.5 (check the log-space slope)"


def test_constant_y_gives_zero_exponent():
    a, b = fit_power_law(np.array([1.0, 10.0, 100.0]), np.array([5.0, 5.0, 5.0]))
    assert abs(a - 5.0) < 1e-9, "a flat law y = 5 must give a = 5"
    assert abs(b) < 1e-12, "a flat law must give exponent b = 0"


def test_all_equal_x_raises_value_error():
    with pytest.raises(ValueError):
        fit_power_law(np.array([3.0, 3.0, 3.0]), np.array([1.0, 2.0, 4.0]))
