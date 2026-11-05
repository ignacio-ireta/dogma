from __future__ import annotations

import pytest
from submission import forward_backward


def test_worked_example_matches_stated_loss_and_gradients():
    loss, grads = forward_backward(1.0, 0.5, 0.0, 0.0)
    assert loss == pytest.approx(3.3069053347706587, rel=1e-9), "loss of the worked example"
    assert grads["x"] == pytest.approx(2.9853340297787616, rel=1e-9), "dL/dx of the worked example"
    assert grads["w"] == pytest.approx(7.6513787795891925, rel=1e-9), (
        "dL/dw of the worked example: w has two paths, through p and through q"
    )
    assert grads["b"] == pytest.approx(5.970668059557523, rel=1e-9), "dL/db of the worked example"
    assert grads["t"] == pytest.approx(-3.6369797001196797, rel=1e-9), "dL/dt of the worked example"


def test_zero_inputs_give_gradients_only_through_bias_and_target():
    loss, grads = forward_backward(0.0, 0.0, 0.0, 0.0)
    assert loss == 1.0, "with every input zero, y = exp(0) = 1 and L = 1"
    assert grads == {"x": 0.0, "w": 0.0, "b": 2.0, "t": -2.0}, (
        "with x = w = 0 the input paths vanish, so only b and t carry gradient"
    )


def test_nan_input_raises_value_error():
    with pytest.raises(ValueError):
        forward_backward(float("nan"), 0.5, 0.0, 0.0)
