# Public tests: the examples from problem.md plus edge cases (CONTENT_SPEC §6.4).
# The learner's code is imported from submission.py; tests never import solution.py.
import pytest
from submission import estimate_memory


def _call(**kwargs):
    """Call the learner; a crash on valid arguments is reported with the likely cause."""
    try:
        return estimate_memory(**kwargs)
    except ValueError:
        raise
    except Exception as exc:
        pytest.fail(
            f"valid arguments crashed ({type(exc).__name__}): check the byte table and the "
            "optimizer state count for this combination"
        )


def test_fp32_inference_is_four_bytes_per_weight():
    got = _call(num_params=1_000_000, mode="inference", dtype="fp32", optimizer="none",
                master_weights=False, activation_bytes=0)  # fmt: skip
    assert got == {"weights": 4_000_000, "gradients": 0, "master_weights": 0,
                   "optimizer_states": 0, "activations": 0, "total": 4_000_000}, (
        "fp32 inference must count four bytes per weight and nothing else"
    )  # fmt: skip


def test_mixed_precision_adam_is_sixteen_bytes_per_parameter():
    got = _call(num_params=1_000_000, mode="training", dtype="bf16", optimizer="adam",
                master_weights=True, activation_bytes=0)  # fmt: skip
    assert got["total"] == 16_000_000, (
        "bf16 Adam with an fp32 master copy is 2 + 2 + 4 + 8 = 16 bytes per parameter"
    )
    assert got["optimizer_states"] == 8_000_000, (
        "Adam keeps two fp32 states, so 8 bytes per parameter: check the state count and the dtype"
    )


def test_inference_with_an_optimizer_is_rejected():
    with pytest.raises(ValueError):
        estimate_memory(num_params=10, mode="inference", dtype="fp32", optimizer="adam",
                        master_weights=False, activation_bytes=0)  # fmt: skip
