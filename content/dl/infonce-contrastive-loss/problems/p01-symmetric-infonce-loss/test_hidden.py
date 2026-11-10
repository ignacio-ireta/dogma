# Hidden tests. The oracle is torch.nn.functional.cross_entropy on the similarity matrix and its
# transpose (tests may use torch.nn; the learner may not). Messages name the kind of mistake only.
from __future__ import annotations

import math

import torch
import torch.nn.functional as ref_f  # oracle only
from hypothesis import given, settings
from hypothesis import strategies as st
from submission import symmetric_infonce_loss

ATOL = 1e-9
RTOL = 1e-9
# float32 oracle comparison: float32 carries about 7 significant digits, and the similarity matrix
# is divided by a small temperature. 1e-5 does not fail correct float32 code, and it is far below a
# real mistake (missing normalisation, one-directional loss, a naive exponential).
ATOL_F32 = 1e-5
RTOL_F32 = 1e-5


def _call(a, b, temperature):
    """Run the learner on a valid call. Failures are AssertionErrors that name the mistake."""
    snap_a, snap_b = a.clone(), b.clone()
    try:
        out = symmetric_infonce_loss(a, b, temperature=temperature)
    except Exception as exc:
        raise AssertionError(
            f"valid input raised {type(exc).__name__}: check the shapes, the dtype, and that "
            "temperature is the keyword argument"
        ) from None
    if not torch.equal(a, snap_a) or not torch.equal(b, snap_b):
        raise AssertionError("an input was modified; normalise into new tensors, never in place")
    if not torch.is_tensor(out) or out.ndim != 0:
        raise AssertionError("return a 0-dimensional tensor: the loss is one number")
    if out.dtype != a.dtype:
        raise AssertionError("the loss must keep the dtype of the embeddings; do not downcast")
    return out


def _oracle(a, b, temperature):
    """Cross-entropy on the cosine matrix over rows and over columns, then averaged."""
    a_hat = ref_f.normalize(a, dim=1)
    b_hat = ref_f.normalize(b, dim=1)
    logits = a_hat @ b_hat.T / temperature
    labels = torch.arange(a.shape[0])
    loss_ab = ref_f.cross_entropy(logits, labels, reduction="mean")
    loss_ba = ref_f.cross_entropy(logits.T, labels, reduction="mean")
    return (loss_ab + loss_ba) / 2


def _expect_value_error(thunk, message):
    raised = None
    try:
        thunk()
    except Exception as exc:
        raised = exc
    if raised is None:
        raise AssertionError(message)
    if not isinstance(raised, ValueError):
        raise AssertionError(f"invalid argument must raise ValueError, not {type(raised).__name__}")


def _gen(seed: int) -> torch.Generator:
    return torch.Generator().manual_seed(seed)


def test_random_pairs_match_cross_entropy_oracle_at_several_temperatures():
    for seed, (n, d) in enumerate([(2, 3), (5, 4), (8, 16), (32, 8)]):
        g = _gen(seed)
        a = torch.randn((n, d), generator=g, dtype=torch.float64)
        b = torch.randn((n, d), generator=g, dtype=torch.float64)
        for tau in (0.07, 0.5, 2.0):
            got = _call(a, b, tau)
            want = _oracle(a, b, tau)
            if not torch.allclose(got, want, atol=ATOL, rtol=RTOL):
                raise AssertionError(
                    "loss differs from the cross-entropy oracle: average the row and column "
                    "directions, use the diagonal as the label, and divide by the temperature"
                )


def test_loss_is_symmetric_in_its_two_arguments():
    g = _gen(9)
    a = torch.randn((6, 5), generator=g, dtype=torch.float64)
    b = torch.randn((6, 5), generator=g, dtype=torch.float64)
    forward = _call(a, b, 0.2)
    backward = _call(b, a, 0.2)
    if not torch.allclose(forward, backward, atol=ATOL, rtol=RTOL):
        raise AssertionError("swapping a and b must not change the symmetric loss")


def test_identical_embeddings_give_chance_level_log_n():
    n = 7
    v = torch.ones((n, 4), dtype=torch.float64)
    got = _call(v, v.clone(), 0.3)
    if abs(float(got) - math.log(n)) > ATOL:
        raise AssertionError(
            "identical embeddings make every similarity equal, so the loss must be log N (chance)"
        )


def test_row_scaling_leaves_the_loss_unchanged():
    g = _gen(13)
    a = torch.randn((5, 6), generator=g, dtype=torch.float64)
    b = torch.randn((5, 6), generator=g, dtype=torch.float64)
    base = _call(a, b, 0.4)
    scale = torch.tensor([0.01, 3.0, 100.0, 1.0, 7.5], dtype=torch.float64).reshape(5, 1)
    moved = _call(a * scale, b, 0.4)
    if not torch.allclose(base, moved, atol=ATOL, rtol=RTOL):
        raise AssertionError(
            "the loss must depend on directions only: normalise each row of a and b"
        )


def test_small_temperature_stays_finite_and_matches_oracle():
    g = _gen(17)
    a = torch.randn((6, 8), generator=g, dtype=torch.float64)
    b = torch.randn((6, 8), generator=g, dtype=torch.float64)
    got = _call(a, b, 1e-3)  # logits reach about 1000 in magnitude; a naive exp overflows float64
    if not bool(torch.isfinite(got)):
        raise AssertionError("a small temperature gave inf or NaN: use the row-maximum log-sum-exp")
    if not torch.allclose(got, _oracle(a, b, 1e-3), atol=ATOL, rtol=RTOL):
        raise AssertionError("small-temperature loss differs from the reference")


def test_single_pair_is_exactly_zero():
    got = _call(
        torch.tensor([[3.0, 4.0]], dtype=torch.float64),
        torch.tensor([[1.0, 0.0]], dtype=torch.float64),
        0.5,
    )
    if float(got) != 0.0:
        raise AssertionError("with one pair there is one candidate, so the loss must be exactly 0")


def test_float32_inputs_return_float32_within_declared_bound():
    g = _gen(23)
    a = torch.randn((8, 12), generator=g, dtype=torch.float32)
    b = torch.randn((8, 12), generator=g, dtype=torch.float32)
    got = _call(a, b, 0.07)
    if got.dtype != torch.float32:
        raise AssertionError("float32 embeddings must give a float32 loss")
    if not torch.allclose(got, _oracle(a, b, 0.07), atol=ATOL_F32, rtol=RTOL_F32):
        raise AssertionError("float32 loss differs from the reference beyond the declared bound")


def test_mismatched_shapes_raise_value_error():
    _expect_value_error(
        lambda: symmetric_infonce_loss(torch.ones((3, 2)), torch.ones((4, 2)), temperature=1.0),
        "a and b must have the same shape (N, D)",
    )


def test_one_dimensional_embeddings_raise_value_error():
    _expect_value_error(
        lambda: symmetric_infonce_loss(torch.ones(3), torch.ones(3), temperature=1.0),
        "embeddings must be two-dimensional (N, D)",
    )


def test_zero_row_raises_value_error():
    a = torch.tensor([[1.0, 0.0], [0.0, 0.0]], dtype=torch.float64)
    _expect_value_error(
        lambda: symmetric_infonce_loss(a, a.clone(), temperature=1.0),
        "a zero embedding has no direction and must raise ValueError",
    )


def test_non_positive_temperature_raises_value_error():
    a = torch.eye(2, dtype=torch.float64)
    for bad in (0.0, -0.1):
        _expect_value_error(
            lambda bad=bad: symmetric_infonce_loss(a, a, temperature=bad),
            "temperature must be strictly positive",
        )


def test_non_finite_temperature_raises_value_error():
    a = torch.eye(2, dtype=torch.float64)
    _expect_value_error(
        lambda: symmetric_infonce_loss(a, a, temperature=float("nan")),
        "a NaN temperature must raise ValueError",
    )


def test_bool_temperature_raises_value_error():
    a = torch.eye(2, dtype=torch.float64)
    _expect_value_error(
        lambda: symmetric_infonce_loss(a, a, temperature=True),
        "a bool temperature must raise ValueError, not be read as 1",
    )


def test_non_finite_embedding_raises_value_error():
    a = torch.tensor([[1.0, float("inf")], [0.0, 1.0]], dtype=torch.float64)
    _expect_value_error(
        lambda: symmetric_infonce_loss(a, torch.eye(2, dtype=torch.float64), temperature=1.0),
        "a non-finite embedding must raise ValueError",
    )


def test_empty_batch_raises_value_error():
    _expect_value_error(
        lambda: symmetric_infonce_loss(torch.zeros((0, 3)), torch.zeros((0, 3)), temperature=1.0),
        "an empty batch has no pairs and must raise ValueError",
    )


def test_inputs_are_not_mutated():
    g = _gen(31)
    a = torch.randn((4, 3), generator=g, dtype=torch.float64)
    b = torch.randn((4, 3), generator=g, dtype=torch.float64)
    _call(a, b, 0.1)  # _call compares the inputs with snapshots after the call


@settings(max_examples=100, deadline=None, derandomize=True, report_multiple_bugs=False)
@given(
    n=st.integers(min_value=1, max_value=6),
    d=st.integers(min_value=1, max_value=5),
    seed=st.integers(min_value=0, max_value=10_000),
    tau=st.floats(min_value=0.05, max_value=2.0, allow_nan=False, allow_infinity=False),
)
def test_property_loss_is_non_negative_and_invariant_to_permuting_pairs(n, d, seed, tau):
    g = _gen(seed)
    a = torch.randn((n, d), generator=g, dtype=torch.float64)
    b = torch.randn((n, d), generator=g, dtype=torch.float64)
    base = symmetric_infonce_loss(a, b, temperature=tau)
    if float(base) < -1e-12:
        raise AssertionError("InfoNCE is a cross-entropy, so it cannot be negative")
    perm = torch.randperm(n, generator=g)
    permuted = symmetric_infonce_loss(a[perm], b[perm], temperature=tau)
    if not torch.allclose(base, permuted, atol=ATOL, rtol=RTOL):
        raise AssertionError(
            "permuting the pairs together must not change the loss: pairs, not rows, matter"
        )


def test_list_argument_raises_value_error_not_attribute_error():
    eye = torch.eye(2, dtype=torch.float64)
    _expect_value_error(
        lambda: symmetric_infonce_loss([[1.0, 0.0], [0.0, 1.0]], eye, temperature=1.0),
        "a must be a torch tensor; passing a plain list must raise ValueError",
    )
    _expect_value_error(
        lambda: symmetric_infonce_loss(eye, [[1.0, 0.0], [0.0, 1.0]], temperature=1.0),
        "b must be a torch tensor; passing a plain list must raise ValueError",
    )


def test_non_finite_value_in_second_argument_raises_value_error():
    b = torch.tensor([[1.0, float("nan")], [0.0, 1.0]], dtype=torch.float64)
    _expect_value_error(
        lambda: symmetric_infonce_loss(torch.eye(2, dtype=torch.float64), b, temperature=1.0),
        "a non-finite value in b must raise ValueError, not return a NaN loss",
    )


def test_temperature_is_keyword_only_and_required():
    a = torch.eye(3, dtype=torch.float64)
    raised = None
    try:
        symmetric_infonce_loss(a, a.clone())
    except Exception as exc:
        raised = exc
    if not isinstance(raised, TypeError):
        raise AssertionError(
            "temperature is keyword-only and required: omitting it must raise TypeError"
        )


def test_tiny_nonzero_row_is_normalised_not_rejected():
    # Row (3e-31, 4e-31) has norm 5e-31 and a direction (0.6, 0.8). Only a zero norm may raise.
    tiny = torch.tensor([[3e-31, 4e-31], [1.0, 0.0]], dtype=torch.float64)
    unit = torch.tensor([[0.6, 0.8], [1.0, 0.0]], dtype=torch.float64)
    base = _call(unit, unit.clone(), 0.5)
    moved = _call(tiny, tiny.clone(), 0.5)
    if not torch.allclose(base, moved, atol=ATOL, rtol=RTOL):
        raise AssertionError(
            "a tiny nonzero row has a direction: normalise it, and raise only for a zero norm"
        )
