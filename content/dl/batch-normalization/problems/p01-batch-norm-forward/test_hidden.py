# Hidden tests: oracle comparisons against PyTorch's functional batch norm, plus edge cases.
# The oracle is used only here; the learner's submission may not use torch.nn or torch.batch_norm.
# Messages name the kind of mistake and never show hidden inputs or expected values.
from __future__ import annotations

import pytest
import torch
import torch.nn.functional as ref_f  # oracle only
from submission import batch_norm_forward

ATOL = 1e-9  # float64 agreement; far below any real mistake in this problem
RTOL = 1e-9


def _gen(seed: int) -> torch.Generator:
    return torch.Generator().manual_seed(seed)


def _randn(shape, seed, scale=1.0, offset=0.0, dtype=torch.float64):
    g = _gen(seed)
    return (torch.randn(shape, generator=g, dtype=dtype) * scale + offset).clone()


def _call(*args, **kwargs):
    snapshots = [a.clone() for a in args if torch.is_tensor(a)]
    try:
        out = batch_norm_forward(*args, **kwargs)
    except Exception as exc:  # a valid call must not fail with a generic error
        pytest.fail(
            f"valid input raised {type(exc).__name__}: check shapes, dtypes and the contract",
            pytrace=False,
        )
    for before, after in zip(snapshots, [a for a in args if torch.is_tensor(a)], strict=True):
        if not torch.equal(before, after):
            pytest.fail("an input tensor was modified in place; return new tensors", pytrace=False)
    if isinstance(out, tuple) and len(out) == 3 and all(torch.is_tensor(t) for t in out):
        for label, got, want in (
            ("output y", out[0], args[0].dtype),
            ("new running mean", out[1], args[3].dtype),
            ("new running variance", out[2], args[4].dtype),
        ):
            if got.dtype != want:
                pytest.fail(
                    f"{label} must keep the dtype of its input ({want}); do not downcast",
                    pytrace=False,
                )
    return out


def _expect_value_error(thunk, message):
    try:
        thunk()
    except ValueError:
        return
    except Exception:
        pytest.fail(
            "invalid input must raise ValueError, not another exception type", pytrace=False
        )
    pytest.fail(message, pytrace=False)


def _oracle(x, gamma, beta, rm, rv, *, training, momentum, eps):
    """PyTorch functional batch norm on clones: returns y and the updated running statistics."""
    rm_o, rv_o = rm.clone(), rv.clone()
    y = ref_f.batch_norm(x, rm_o, rv_o, gamma, beta, training=training, momentum=momentum, eps=eps)
    return y, rm_o, rv_o


def _result(out, n, c):
    if not isinstance(out, tuple | list) or len(out) != 3:
        pytest.fail(
            "return three values: y, new running mean and new running variance", pytrace=False
        )
    if not all(torch.is_tensor(t) for t in out):
        pytest.fail("return torch tensors, not NumPy arrays or Python lists", pytrace=False)
    y, rm, rv = out
    if tuple(y.shape) != (n, c):
        pytest.fail("output y must have the input shape (N, C)", pytrace=False)
    if tuple(rm.shape) != (c,) or tuple(rv.shape) != (c,):
        pytest.fail(
            "running statistics must have shape (C,): reduce over axis 0 (the batch), "
            "leaving one value per feature",
            pytrace=False,
        )
    return y, rm, rv


def _close(actual, expected, message, atol=ATOL, rtol=RTOL):
    if not torch.allclose(actual, expected, atol=atol, rtol=rtol):
        pytest.fail(message, pytrace=False)


SHAPES = [(2, 2), (3, 5), (5, 3), (4, 1), (8, 4), (16, 6)]


def test_training_output_matches_reference_on_random_batches():
    for seed, (n, c) in enumerate(SHAPES):
        x = _randn((n, c), seed, scale=2.0, offset=0.5)
        gamma, beta = _randn((c,), seed + 100), _randn((c,), seed + 200)
        rm, rv = _randn((c,), seed + 300), _randn((c,), seed + 400).abs() + 0.5
        y, new_rm, new_rv = _result(
            _call(x, gamma, beta, rm, rv, training=True, momentum=0.1, eps=1e-5), n, c
        )
        want_y, want_rm, want_rv = _oracle(
            x, gamma, beta, rm, rv, training=True, momentum=0.1, eps=1e-5
        )
        _close(
            y,
            want_y,
            "training output differs from the reference batch norm: check the biased variance "
            "(divide by N), eps inside the square root, gamma and beta, and that statistics run "
            "per feature over the rows",
        )
        _close(
            new_rm,
            want_rm,
            "running mean after a training step differs from the reference: new = (1 - momentum) "
            "* old + momentum * batch mean, with the batch mean taken over the rows",
        )
        _close(
            new_rv,
            want_rv,
            "running variance after a training step differs from the reference: the running update "
            "uses the unbiased batch variance (divide by N - 1) with the same momentum rule",
        )


def test_eval_output_matches_reference_on_random_inputs():
    for seed, (n, c) in enumerate(SHAPES):
        x = _randn((n, c), seed + 7)
        gamma, beta = _randn((c,), seed + 11), _randn((c,), seed + 13)
        rm, rv = _randn((c,), seed + 17), _randn((c,), seed + 19).abs() + 0.2
        y, new_rm, new_rv = _result(
            _call(x, gamma, beta, rm, rv, training=False, momentum=0.3, eps=1e-3), n, c
        )
        want_y, _, _ = _oracle(x, gamma, beta, rm, rv, training=False, momentum=0.3, eps=1e-3)
        _close(
            y,
            want_y,
            "eval mode must normalize with the stored running statistics, not batch statistics",
        )
        _close(new_rm, rm, "eval mode changed the running mean; statistics update only in training")
        _close(
            new_rv, rv, "eval mode changed the running variance; statistics update only in training"
        )


def test_running_variance_uses_unbiased_estimate():
    x = _randn((4, 3), 31)  # N=4: divide-by-N and divide-by-(N-1) differ by 33%
    gamma, beta = torch.ones(3, dtype=torch.float64), torch.zeros(3, dtype=torch.float64)
    rm, rv = torch.zeros(3, dtype=torch.float64), torch.ones(3, dtype=torch.float64)
    _, _, new_rv = _result(
        _call(x, gamma, beta, rm, rv, training=True, momentum=0.5, eps=1e-5), 4, 3
    )
    batch_var_biased = x.var(dim=0, unbiased=False)
    biased_update = 0.5 * rv + 0.5 * batch_var_biased
    _close(
        new_rv,
        0.5 * rv + 0.5 * batch_var_biased * 4 / 3,
        "running variance used biased (divide-by-N) variance; the running estimate needs N - 1",
    )
    assert not torch.allclose(new_rv, biased_update, atol=1e-6), (
        "running variance used biased (divide-by-N) variance; the running estimate needs N - 1"
    )


def test_normalization_uses_biased_variance():
    x = _randn((3, 2), 41)
    gamma, beta = torch.ones(2, dtype=torch.float64), torch.zeros(2, dtype=torch.float64)
    rm, rv = torch.zeros(2, dtype=torch.float64), torch.ones(2, dtype=torch.float64)
    y, _, _ = _result(_call(x, gamma, beta, rm, rv, training=True, eps=1e-12), 3, 2)
    centred = x - x.mean(dim=0)
    want = centred / torch.sqrt((centred * centred).mean(dim=0) + 1e-12)
    _close(
        y,
        want,
        "normalization used unbiased variance; the normalizing variance divides by N",
    )


def test_momentum_weights_the_new_batch_statistic():
    x = _randn((5, 3), 51)
    gamma, beta = torch.ones(3, dtype=torch.float64), torch.zeros(3, dtype=torch.float64)
    rm = _randn((3,), 52)
    rv = _randn((3,), 53).abs() + 0.5
    m = 0.3  # asymmetric on purpose: a flipped convention uses 0.7 instead
    _, new_rm, new_rv = _result(
        _call(x, gamma, beta, rm, rv, training=True, momentum=m, eps=1e-5), 5, 3
    )
    batch_mean = x.mean(dim=0)
    batch_var_unbiased = x.var(dim=0, unbiased=True)
    _close(
        new_rm,
        (1 - m) * rm + m * batch_mean,
        "momentum weights the old running statistic instead of the new batch statistic",
    )
    _close(
        new_rv,
        (1 - m) * rv + m * batch_var_unbiased,
        "momentum weights the old running statistic instead of the new batch statistic",
    )


def test_normalizes_per_feature_over_the_batch_axis():
    # N != C so a reduction over the wrong axis cannot match the reference by accident.
    x = _randn((3, 5), 61, scale=3.0, offset=-1.0)
    gamma, beta = torch.ones(5, dtype=torch.float64), torch.zeros(5, dtype=torch.float64)
    rm, rv = torch.zeros(5, dtype=torch.float64), torch.ones(5, dtype=torch.float64)
    y, _, _ = _result(_call(x, gamma, beta, rm, rv, training=True, eps=1e-12), 3, 5)
    per_feature_mean = y.mean(dim=0)
    if not torch.allclose(per_feature_mean, torch.zeros(5, dtype=torch.float64), atol=1e-9):
        pytest.fail("normalized over the wrong axis: each feature must be centred over the rows")
    want_y, _, _ = _oracle(x, gamma, beta, rm, rv, training=True, momentum=0.1, eps=1e-12)
    _close(y, want_y, "normalized over the wrong axis: statistics must be per feature, across rows")


def test_eps_is_added_inside_the_square_root():
    x = _randn((4, 2), 71)
    gamma, beta = torch.ones(2, dtype=torch.float64), torch.zeros(2, dtype=torch.float64)
    rm = torch.zeros(2, dtype=torch.float64)
    rv = torch.zeros(2, dtype=torch.float64)  # zero variance makes eps the only denominator term
    eps = 0.25
    y, _, _ = _result(_call(x, gamma, beta, rm, rv, training=False, eps=eps), 4, 2)
    _close(
        y,
        x / torch.sqrt(torch.tensor(eps, dtype=torch.float64)),
        "eps is added outside the square root, or inference did not use the running statistics; "
        "inference divides by sqrt(running_var + eps)",
    )
    centred = x - x.mean(dim=0)
    want_train = centred / torch.sqrt((centred * centred).mean(dim=0) + eps)
    y_train, _, _ = _result(_call(x, gamma, beta, rm, rv, training=True, eps=eps), 4, 2)
    _close(
        y_train,
        want_train,
        "eps is added outside the square root in training mode; add it to the batch variance "
        "before taking sqrt",
    )


def test_eval_mode_does_not_update_running_statistics_in_place_or_returned():
    x = _randn((4, 3), 81)
    rm, rv = _randn((3,), 82), _randn((3,), 83).abs() + 0.5
    rm0, rv0 = rm.clone(), rv.clone()
    _, new_rm, new_rv = _result(
        _call(
            x,
            torch.ones(3, dtype=torch.float64),
            torch.zeros(3, dtype=torch.float64),
            rm,
            rv,
            training=False,
            momentum=0.9,
            eps=1e-5,
        ),
        4,
        3,
    )
    assert torch.equal(rm, rm0) and torch.equal(rv, rv0), (
        "eval mode modified an input running buffer in place"
    )
    _close(new_rm, rm0, "eval mode changed the returned running mean")
    _close(new_rv, rv0, "eval mode changed the returned running variance")


def test_inputs_and_running_buffers_are_not_mutated():
    x = _randn((4, 3), 91)
    gamma, beta = _randn((3,), 92), _randn((3,), 93)
    rm, rv = _randn((3,), 94), _randn((3,), 95).abs() + 0.5
    originals = [t.clone() for t in (x, gamma, beta, rm, rv)]
    _call(x, gamma, beta, rm, rv, training=True, momentum=0.2, eps=1e-5)
    for name, now, before in zip(
        ("x", "gamma", "beta", "running_mean", "running_var"),
        (x, gamma, beta, rm, rv),
        originals,
        strict=True,
    ):
        assert torch.equal(now, before), f"input {name} was modified in place; return new tensors"


def test_large_common_offset_keeps_variance_accurate():
    # The naive E[x^2] - E[x]^2 formula loses most of its digits when the mean is large.
    x = _randn((6, 3), 101, scale=1.0, offset=1e6)
    gamma, beta = torch.ones(3, dtype=torch.float64), torch.zeros(3, dtype=torch.float64)
    rm, rv = torch.zeros(3, dtype=torch.float64), torch.ones(3, dtype=torch.float64)
    y, _, new_rv = _result(
        _call(x, gamma, beta, rm, rv, training=True, momentum=1.0, eps=1e-5), 6, 3
    )
    want_y, _, want_rv = _oracle(x, gamma, beta, rm, rv, training=True, momentum=1.0, eps=1e-5)
    _close(
        y, want_y, "statistics lost precision under a large common offset: centre before squaring"
    )
    _close(
        new_rv,
        want_rv,
        "statistics lost precision under a large common offset: centre before squaring",
        atol=1e-7,
        rtol=1e-7,
    )


def test_small_magnitude_inputs_match_reference():
    x = _randn((5, 2), 111, scale=1e-6)  # variance near 1e-12, comparable to eps
    gamma, beta = torch.ones(2, dtype=torch.float64), torch.zeros(2, dtype=torch.float64)
    rm, rv = torch.zeros(2, dtype=torch.float64), torch.ones(2, dtype=torch.float64)
    y, new_rm, new_rv = _result(_call(x, gamma, beta, rm, rv, training=True, eps=1e-12), 5, 2)
    want_y, want_rm, want_rv = _oracle(
        x, gamma, beta, rm, rv, training=True, momentum=0.1, eps=1e-12
    )
    _close(y, want_y, "small-magnitude training output differs from the reference")
    _close(new_rm, want_rm, "small-magnitude running mean differs from the reference")
    _close(new_rv, want_rv, "small-magnitude running variance differs from the reference")


def test_float32_inputs_return_float32():
    x = _randn((4, 3), 121, dtype=torch.float32)
    gamma = torch.ones(3, dtype=torch.float32)
    beta = torch.zeros(3, dtype=torch.float32)
    rm, rv = torch.zeros(3, dtype=torch.float32), torch.ones(3, dtype=torch.float32)
    y, new_rm, new_rv = _result(_call(x, gamma, beta, rm, rv, training=True, eps=1e-5), 4, 3)
    assert y.dtype == torch.float32, "output dtype must follow the input dtype"
    assert new_rm.dtype == torch.float32 and new_rv.dtype == torch.float32, (
        "running statistics must keep their input dtype"
    )
    want_y, _, _ = _oracle(x, gamma, beta, rm, rv, training=True, momentum=0.1, eps=1e-5)
    _close(y, want_y, "float32 training output differs from the reference", atol=1e-5, rtol=1e-5)


def test_single_row_training_is_rejected_with_value_error():
    x = _randn((1, 3), 129)
    ones, zeros = torch.ones(3, dtype=torch.float64), torch.zeros(3, dtype=torch.float64)
    _expect_value_error(
        lambda: batch_norm_forward(x, ones, zeros, zeros, ones, training=True),
        "training with one row must raise ValueError: batch variance is undefined",
    )


def test_single_row_inference_is_allowed():
    x = _randn((1, 3), 131)
    gamma, beta = _randn((3,), 132), _randn((3,), 133)
    rm, rv = _randn((3,), 134), _randn((3,), 135).abs() + 0.5
    y, _, _ = _result(_call(x, gamma, beta, rm, rv, training=False, eps=1e-5), 1, 3)
    want_y, _, _ = _oracle(x, gamma, beta, rm, rv, training=False, momentum=0.1, eps=1e-5)
    _close(y, want_y, "inference with one row must use the running statistics")


def test_invalid_arguments_raise_value_error():
    good = {
        "gamma": torch.ones(3, dtype=torch.float64),
        "beta": torch.zeros(3, dtype=torch.float64),
        "running_mean": torch.zeros(3, dtype=torch.float64),
        "running_var": torch.ones(3, dtype=torch.float64),
    }
    x = _randn((4, 3), 141)
    cases = {
        "a one-dimensional input must raise ValueError": lambda: batch_norm_forward(
            x[0], training=False, **good
        ),
        "an integer input must raise ValueError": lambda: batch_norm_forward(
            torch.ones(4, 3, dtype=torch.int64), training=False, **good
        ),
        "an empty batch must raise ValueError": lambda: batch_norm_forward(
            torch.empty(0, 3, dtype=torch.float64), training=False, **good
        ),
        "a parameter with the wrong feature count must raise ValueError": lambda: (
            batch_norm_forward(
                x, training=False, **{**good, "gamma": torch.ones(2, dtype=torch.float64)}
            )
        ),
        "momentum above 1 must raise ValueError": lambda: batch_norm_forward(
            x, training=True, momentum=1.5, **good
        ),
        "a negative eps must raise ValueError": lambda: batch_norm_forward(
            x, training=True, eps=-1.0, **good
        ),
        "a zero eps must raise ValueError": lambda: batch_norm_forward(
            x, training=True, eps=0.0, **good
        ),
    }
    for message, thunk in cases.items():
        _expect_value_error(thunk, message)


@pytest.mark.parametrize("seed", [0, 1, 2], ids=["low-momentum", "half-momentum", "high-momentum"])
def test_random_momentum_and_eps_match_reference(seed):
    n, c = 7, 4
    x = _randn((n, c), 150 + seed, scale=4.0, offset=2.0)
    gamma, beta = _randn((c,), 160 + seed), _randn((c,), 170 + seed)
    rm, rv = _randn((c,), 180 + seed), _randn((c,), 190 + seed).abs() + 0.1
    momentum = [0.05, 0.5, 0.95][seed]
    eps = [1e-8, 1e-3, 0.1][seed]
    y, new_rm, new_rv = _result(
        _call(x, gamma, beta, rm, rv, training=True, momentum=momentum, eps=eps), n, c
    )
    want_y, want_rm, want_rv = _oracle(
        x, gamma, beta, rm, rv, training=True, momentum=momentum, eps=eps
    )
    _close(
        y,
        want_y,
        "training output differs from the reference for this momentum and eps: check eps inside "
        "the square root and the biased variance",
    )
    _close(
        new_rm,
        want_rm,
        "running mean differs from the reference for this momentum: the new batch statistic is "
        "weighted by momentum",
    )
    _close(
        new_rv,
        want_rv,
        "running variance differs from the reference for this momentum: unbiased batch variance "
        "(divide by N - 1), weighted by momentum",
    )


def test_zero_momentum_is_accepted_and_keeps_running_statistics():
    x = _randn((4, 3), 301)
    gamma, beta = torch.ones(3, dtype=torch.float64), torch.zeros(3, dtype=torch.float64)
    rm, rv = _randn((3,), 302), _randn((3,), 303).abs() + 0.5
    out = _call(x, gamma, beta, rm, rv, training=True, momentum=0.0, eps=1e-5)
    _, new_rm, new_rv = _result(out, 4, 3)
    _close(new_rm, rm, "momentum of exactly 0 is valid: the running mean must stay unchanged")
    _close(new_rv, rv, "momentum of exactly 0 is valid: the running variance must stay unchanged")


def test_default_momentum_and_eps_match_the_signature():
    # Small variance (about 1e-5) so that a different default eps changes the output clearly.
    x = _randn((6, 3), 201, scale=3e-3)
    gamma, beta = torch.ones(3, dtype=torch.float64), torch.zeros(3, dtype=torch.float64)
    rm, rv = torch.zeros(3, dtype=torch.float64), torch.ones(3, dtype=torch.float64)
    y, new_rm, new_rv = _result(_call(x, gamma, beta, rm, rv, training=True), 6, 3)
    want_y, want_rm, want_rv = _oracle(
        x, gamma, beta, rm, rv, training=True, momentum=0.1, eps=1e-5
    )
    _close(
        y,
        want_y,
        "default eps or momentum differs from the documented signature (momentum 0.1, eps 1e-5)",
    )
    _close(new_rm, want_rm, "default momentum differs from the documented signature (0.1)")
    _close(new_rv, want_rv, "default momentum differs from the documented signature (0.1)")


def test_eval_returned_running_statistics_are_new_tensors():
    x = _randn((4, 3), 311)
    gamma, beta = torch.ones(3, dtype=torch.float64), torch.zeros(3, dtype=torch.float64)
    rm, rv = _randn((3,), 312), _randn((3,), 313).abs() + 0.5
    rm0, rv0 = rm.clone(), rv.clone()
    _, new_rm, new_rv = batch_norm_forward(
        x, gamma, beta, rm, rv, training=False, momentum=0.1, eps=1e-5
    )
    # The rule is to return new tensors. Writing to a returned statistic must not reach the inputs.
    new_rm.add_(1.0)
    new_rv.mul_(0.0)
    if not torch.equal(rm, rm0) or not torch.equal(rv, rv0):
        pytest.fail(
            "eval mode returned the input running statistics themselves; return new tensors",
            pytrace=False,
        )
