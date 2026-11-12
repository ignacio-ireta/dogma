from __future__ import annotations

import torch
from submission import batched_sq_distances

from dogma_data import rng, sample_rows

ATOL = 1e-6  # matches meta.yaml tolerance
RTOL = 1e-5  # matches meta.yaml tolerance (justified there)


def _point_sets(key, batch, n, d, dtype=torch.float64):
    """batch sets of n real points with d standardised breast_cancer features each."""
    rows, _ = sample_rows("breast_cancer", rng(key), batch * n, d, standardize=True)
    return torch.tensor(rows.reshape(batch, n, d), dtype=dtype)


def _two_point_sets(key, batch, n, m, d, *, dtype=torch.float64):
    """Two groups of real point sets that share their d features: n and m points per set."""
    both = _point_sets(key, batch, n + m, d, dtype)
    return both[:, :n].clone(), both[:, n:].clone()


def exact_sq_distances(X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
    """Independent float64 reference: explicit difference, no expansion, no broadcasting tricks."""
    X64 = X.double()
    Y64 = Y.double()
    out = torch.zeros(X.shape[0], X.shape[1], Y.shape[1], dtype=torch.float64)
    for b in range(X.shape[0]):
        for i in range(X.shape[1]):
            for j in range(Y.shape[1]):
                out[b, i, j] = ((X64[b, i] - Y64[b, j]) ** 2).sum()
    return out


def checked(call, expected_dtype: torch.dtype | None = None) -> torch.Tensor:
    try:
        out = call()
    except Exception as exc:  # any failure on a valid call is reported as a contract breach
        raise AssertionError(
            f"valid input raised {type(exc).__name__}: honor the documented contract"
        ) from None
    if not isinstance(out, torch.Tensor):
        raise AssertionError("return a torch.Tensor, not a Python number or array")
    if expected_dtype is not None and out.dtype != expected_dtype:
        raise AssertionError(f"wrong result dtype: expected {expected_dtype}, got {out.dtype}")
    return out


def same_shape(out: torch.Tensor, ref: torch.Tensor) -> None:
    if out.shape != ref.shape:
        raise AssertionError("output shape must be (B, N, M): check the query and reference axes")


def rejects(call) -> None:
    try:
        call()
    except ValueError:
        return
    except Exception as exc:
        raise AssertionError(
            f"invalid input must raise ValueError, not {type(exc).__name__}"
        ) from None
    raise AssertionError("invalid input was accepted: check dtype, rank and batch/feature sizes")


def test_random_float64_inputs_match_independent_reference():
    for trial in range(4):
        B, N, M, D = 2 + trial % 2, 1 + trial, 2 + trial, 1 + trial
        X, Y = _two_point_sets(f"batched-sq-distances-p01:reference:{trial}", B, N, M, D)
        out = checked(lambda X=X, Y=Y: batched_sq_distances(X, Y), torch.float64)
        same_shape(out, exact_sq_distances(X, Y))
        assert torch.allclose(out, exact_sq_distances(X, Y), atol=ATOL, rtol=RTOL), (
            "pairwise values differ from the reference: check the reduced axis and the pair axes"
        )


def test_zero_sized_batch_query_reference_and_feature_dimensions():
    cases = [((0, 2, 3), (0, 4, 3)), ((2, 0, 3), (2, 4, 3)), ((2, 3, 4), (2, 0, 4))]
    for x_shape, y_shape in cases:
        out = checked(
            lambda x=x_shape, y=y_shape: batched_sq_distances(torch.ones(x), torch.ones(y))
        )
        assert out.shape == (x_shape[0], x_shape[1], y_shape[1]), (
            "zero-sized dimensions must propagate to the (B, N, M) output shape"
        )
    zero_features = checked(lambda: batched_sq_distances(torch.ones(2, 3, 0), torch.ones(2, 4, 0)))
    assert zero_features.shape == (2, 3, 4) and bool((zero_features == 0).all()), (
        "zero features mean no coordinate differences, so every distance is zero"
    )


def test_identical_points_have_nonnegative_distances_with_zero_diagonal():
    X = _point_sets("batched-sq-distances-p01:self-distance", 3, 16, 5, torch.float32)
    out = checked(lambda: batched_sq_distances(X, X), torch.float32)
    assert bool((out >= 0).all()), "squared distances must never be negative, not even by rounding"
    # Rounding in the self-distance scales with the point's squared norm, so the allowance does too.
    diagonal = torch.diagonal(out, dim1=1, dim2=2)
    norms = (X.double() ** 2).sum(-1)
    assert bool((diagonal.double() <= RTOL * (1.0 + norms)).all()), (
        "a point must have (numerically) zero distance to itself"
    )


def test_large_common_offset_keeps_small_distances_accurate_in_float32():
    # Every coordinate sits near 1000 with spread 0.01. The true squared distances are about 1e-4,
    # far below the float32 rounding of the squared norms (about 3e6), so the naive expansion fails.
    near, far = _two_point_sets("batched-sq-distances-p01:offset", 2, 6, 5, 3)
    X = (1000.0 + 0.01 * near).float()
    Y = (1000.0 + 0.01 * far).float()
    out = checked(lambda: batched_sq_distances(X, Y), torch.float32)
    same_shape(out, exact_sq_distances(X, Y))
    assert torch.allclose(out.double(), exact_sq_distances(X, Y), atol=ATOL, rtol=RTOL), (
        "a shared large offset destroyed the small distances: avoid cancelling squared norms"
    )


def test_small_magnitudes_scale_quadratically():
    X, Y = _two_point_sets("batched-sq-distances-p01:scale", 2, 4, 3, 3)
    scale = 1e-4
    unit = checked(lambda: batched_sq_distances(X, Y), torch.float64)
    tiny = checked(lambda: batched_sq_distances(X * scale, Y * scale), torch.float64)
    assert torch.allclose(tiny / scale**2, unit, atol=ATOL, rtol=RTOL), (
        "distances must scale with the square of the coordinate scale"
    )


def test_float32_with_float64_promotes_to_float64():
    X, Y = _two_point_sets("batched-sq-distances-p01:mixed-dtypes", 2, 3, 4, 2)
    X = X.float()
    out = checked(lambda: batched_sq_distances(X, Y), torch.float64)
    same_shape(out, exact_sq_distances(X, Y))
    assert torch.allclose(out, exact_sq_distances(X, Y), atol=ATOL, rtol=RTOL), (
        "mixed dtypes must be promoted before the arithmetic"
    )


def test_float32_in_float32_out_and_float64_in_float64_out():
    X32, Y32 = _two_point_sets("batched-sq-distances-p01:dtype", 1, 2, 3, 2, dtype=torch.float32)
    checked(lambda: batched_sq_distances(X32, Y32), torch.float32)
    checked(lambda: batched_sq_distances(X32.double(), Y32.double()), torch.float64)


def test_integer_input_is_rejected_as_value_error():
    rejects(
        lambda: batched_sq_distances(torch.ones(1, 2, 2, dtype=torch.int64), torch.ones(1, 2, 2))
    )


def test_complex_input_is_rejected_as_value_error():
    rejects(
        lambda: batched_sq_distances(
            torch.ones(1, 2, 2, dtype=torch.complex64), torch.ones(1, 2, 2)
        )
    )


def test_two_dimensional_input_is_rejected_as_value_error():
    rejects(lambda: batched_sq_distances(torch.ones(2, 2), torch.ones(1, 2, 2)))


def test_mismatched_batch_or_feature_sizes_are_rejected_as_value_error():
    rejects(lambda: batched_sq_distances(torch.ones(2, 3, 2), torch.ones(3, 3, 2)))
    rejects(lambda: batched_sq_distances(torch.ones(2, 3, 2), torch.ones(2, 3, 4)))


def test_inputs_are_not_mutated():
    X, Y = _two_point_sets("batched-sq-distances-p01:not-mutated", 2, 3, 4, 2)
    X_before, Y_before = X.clone(), Y.clone()
    checked(lambda: batched_sq_distances(X, Y))
    assert torch.equal(X, X_before), "the query argument was modified: do not write into inputs"
    assert torch.equal(Y, Y_before), "the reference argument was modified: do not write into inputs"


def test_argument_with_wrong_rank_is_rejected_as_value_error():
    rejects(lambda: batched_sq_distances(torch.ones(1, 2), torch.ones(1, 2, 2)))
    rejects(lambda: batched_sq_distances(torch.ones(1, 2, 2), torch.ones(1, 2)))


def test_batch_of_one_against_many_batches_is_rejected_as_value_error():
    rejects(lambda: batched_sq_distances(torch.ones(1, 2, 2), torch.ones(3, 2, 2)))
    rejects(lambda: batched_sq_distances(torch.ones(3, 2, 2), torch.ones(1, 2, 2)))
