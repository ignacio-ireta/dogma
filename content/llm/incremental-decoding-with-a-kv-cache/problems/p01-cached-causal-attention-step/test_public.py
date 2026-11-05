# Public tests: the examples from problem.md plus edge cases (CONTENT_SPEC §6.4).
# The learner's code is imported from submission.py; tests never import solution.py.
import torch
from submission import cached_attention_step


def _identity_step(x, cache=None):
    eye = torch.eye(2, dtype=torch.float64)
    return cached_attention_step(x, cache, eye, eye, eye, eye, 1)


def test_first_token_attends_only_to_itself():
    out, _ = _identity_step(torch.tensor([[[1.0, 0.0]]], dtype=torch.float64))
    torch.testing.assert_close(
        out,
        torch.tensor([[[1.0, 0.0]]], dtype=torch.float64),
        atol=1e-9,
        rtol=1e-9,
        msg=(
            "a single token with no cache must return its own value: check the softmax over one key"
        ),
    )


def test_second_step_reads_the_cached_key_and_value():
    _, cache = _identity_step(torch.tensor([[[1.0, 0.0]]], dtype=torch.float64))
    out, _ = _identity_step(torch.tensor([[[0.0, 1.0]]], dtype=torch.float64), cache)
    want = torch.tensor([[[0.3302, 0.6698]]], dtype=torch.float64)
    torch.testing.assert_close(
        out,
        want,
        # problem.md quotes 0.3302 to four decimals, so atol 1e-4 covers that rounding only.
        atol=1e-4,
        rtol=0,
        msg=(
            "the new query must attend to the cached key at the right "
            "offset: check the cache append"
        ),
    )


def test_two_token_chunk_is_causal_within_itself_and_cache_grows():
    x = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]], dtype=torch.float64)
    out, cache = _identity_step(x)
    assert out[0, 0].tolist() == [1.0, 0.0], "row 0 must not attend to the later token in its chunk"
    assert cache[0].shape == (1, 1, 2, 2), (
        "the cache must hold one key per position after the chunk"
    )
