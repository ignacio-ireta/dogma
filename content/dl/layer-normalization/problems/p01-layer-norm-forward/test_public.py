# Public tests: the examples from problem.md plus the single-feature edge case (CONTENT_SPEC §6.4).
# The learner's code is imported from submission.py; tests never import solution.py.
import torch
from submission import layer_norm


def _t(values):
    return torch.tensor(values, dtype=torch.float64)


def test_three_feature_example_normalizes_the_last_dimension():
    """One example of three features: mean 2, biased variance 2/3."""
    y = layer_norm(_t([[1.0, 2.0, 3.0]]), _t([1.0, 1.0, 1.0]), _t([0.0, 0.0, 0.0]), eps=1e-12)
    assert y.dtype == torch.float64, "output must keep the dtype of x; do not downcast"
    assert torch.allclose(y, _t([[-1.2247448714, 0.0, 1.2247448714]]), atol=1e-8), (
        "layer norm output is wrong: check the mean and the biased variance over the last "
        "dimension, the centring before squaring, and that eps is inside the square root"
    )


def test_gain_bias_per_feature_and_constant_example_returns_bias():
    """Gain and bias apply per feature; a constant example returns the bias unchanged."""
    x = _t([[1.0, 2.0, 3.0], [10.0, 10.0, 10.0]])
    y = layer_norm(x, _t([2.0, 1.0, 0.5]), _t([0.0, 0.0, 1.0]), eps=1e-12)
    assert y.dtype == torch.float64, "output must keep the dtype of x; do not downcast"
    expected = _t([[-2.4494897428, 0.0, 1.6123724357], [0.0, 0.0, 1.0]])
    assert torch.allclose(y, expected, atol=1e-8), (
        "output must be weight * (x - mean) / sqrt(var + eps) + bias: gain and bias are per "
        "feature (shape (D,)) and act after normalizing; a constant row must return the bias"
    )


def test_single_feature_returns_bias():
    """With D = 1 every example is constant, so the output equals the bias."""
    y = layer_norm(_t([[5.0], [-3.0]]), _t([7.0]), _t([0.25]), eps=1e-5)
    assert torch.allclose(y, _t([[0.25], [0.25]]), atol=1e-12), (
        "a single feature must return the bias: the centred value is zero"
    )
