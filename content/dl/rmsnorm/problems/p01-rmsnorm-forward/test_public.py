# Public tests: the examples from problem.md plus the zero-input edge case (CONTENT_SPEC §6.4).
# The learner's code is imported from submission.py; tests never import solution.py.
import torch
from submission import rms_norm


def _t(values):
    return torch.tensor(values, dtype=torch.float64)


def test_unit_gain_example_divides_by_root_mean_square():
    """Mean square of [1, 3] is 5, so the output is x / sqrt(5)."""
    y = rms_norm(_t([[1.0, 3.0]]), _t([1.0, 1.0]), eps=1e-12)
    assert y.dtype == torch.float64, "output must keep the dtype of x; do not downcast"
    assert torch.allclose(y, _t([[0.4472135955, 1.3416407865]]), atol=1e-8), (
        "RMSNorm output is wrong: divide by sqrt(mean(x**2) + eps), the root mean square of x, "
        "not by its standard deviation"
    )


def test_gain_applies_per_feature_and_mean_is_not_removed():
    """Mean square of [3, 4] is 12.5. The gain acts per feature; no mean is subtracted."""
    y = rms_norm(_t([[3.0, 4.0]]), _t([2.0, 0.5]), eps=1e-12)
    assert y.dtype == torch.float64, "output must keep the dtype of x; do not downcast"
    assert torch.allclose(y, _t([[1.6970562748, 0.5656854249]]), atol=1e-8), (
        "output must be gain * x / RMS(x): the gain multiplies each feature after the division, "
        "and no mean is subtracted (RMSNorm keeps the offset)"
    )


def test_zero_input_returns_zeros_not_nan():
    """An all-zero example has zero mean square; eps keeps the output finite and zero."""
    y = rms_norm(_t([[0.0, 0.0]]), _t([1.0, 1.0]), eps=1e-6)
    assert torch.equal(y, _t([[0.0, 0.0]])), "zero input must give zero output, not NaN or inf"
