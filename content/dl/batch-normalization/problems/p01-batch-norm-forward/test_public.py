# Public tests: the examples from problem.md plus the single-row edge case (CONTENT_SPEC §6.4).
# The learner's code is imported from submission.py; tests never import solution.py.
import torch
from submission import batch_norm_forward


def _x():
    return torch.tensor([[1.0, 2.0], [3.0, 6.0]], dtype=torch.float64)


def _t(values):
    return torch.tensor(values, dtype=torch.float64)


def test_training_example_normalizes_per_feature_and_updates_running_stats():
    """Training normalizes each column with its biased variance and moves the running stats."""
    y, rm, rv = batch_norm_forward(
        _x(),
        _t([1.0, 1.0]),
        _t([0.0, 0.0]),
        _t([0.0, 0.0]),
        _t([1.0, 1.0]),
        training=True,
        momentum=0.1,
        eps=1e-12,
    )
    assert y.dtype == torch.float64, "output y must keep the dtype of x; do not downcast"
    assert torch.allclose(y, _t([[-1.0, -1.0], [1.0, 1.0]]), atol=1e-9), (
        "training output is wrong: normalize each feature with the biased variance (divide by N), "
        "put eps inside the square root, then apply gamma and beta"
    )
    assert torch.allclose(rm, _t([0.2, 0.4]), atol=1e-9), (
        "running mean update is wrong: new = (1 - momentum) * old + momentum * batch mean"
    )
    assert torch.allclose(rv, _t([1.1, 1.7]), atol=1e-9), (
        "running variance update is wrong: running_var takes the unbiased batch variance "
        "(divide by N - 1), blended with the same momentum rule"
    )


def test_inference_example_uses_running_stats_and_leaves_them_unchanged():
    """Inference normalizes with the stored statistics and returns them as they were."""
    rm_in, rv_in = _t([0.5, 1.0]), _t([4.0, 9.0])
    y, rm, rv = batch_norm_forward(
        _x(),
        _t([2.0, 1.0]),
        _t([0.0, 1.0]),
        rm_in,
        rv_in,
        training=False,
        eps=1e-12,
    )
    expected = _t([[0.5, 4.0 / 3.0], [2.5, 8.0 / 3.0]])
    assert torch.allclose(y, expected, atol=1e-9), "inference output must use running statistics"
    assert torch.equal(rm, _t([0.5, 1.0])), "eval mode changed the running mean"
    assert torch.equal(rv, _t([4.0, 9.0])), "eval mode changed the running variance"


def test_single_row_in_training_is_rejected():
    """One row cannot give a batch variance, so training mode must raise ValueError."""
    try:
        batch_norm_forward(
            _t([[1.0, 2.0]]),
            _t([1.0, 1.0]),
            _t([0.0, 0.0]),
            _t([0.0, 0.0]),
            _t([1.0, 1.0]),
            training=True,
        )
    except ValueError:
        return
    raise AssertionError("training with one row must raise ValueError: batch variance is undefined")
