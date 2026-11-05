# Hidden tests (CONTENT_SPEC §6.4). Names describe the behaviour; messages name the kind of mistake
# and never print hidden inputs, shapes or expected values.
from __future__ import annotations

import pytest
import torch
from submission import greedy_generate as _learner


def _call(*args, **kwargs):
    """Call the learner; an exception on a valid input is reported as a named contract failure."""
    try:
        return _learner(*args, **kwargs)
    except ValueError:
        raise
    except Exception as exc:
        pytest.fail(f"valid arguments violated the contract: raised {type(exc).__name__}")


def _expect_value_error(*args, **kwargs):
    try:
        _learner(*args, **kwargs)
    except ValueError:
        return
    except Exception as exc:
        pytest.fail(f"invalid argument must raise ValueError, received {type(exc).__name__}")
    pytest.fail("invalid argument was accepted instead of raising ValueError")


def _bigram(table: torch.Tensor):
    """Model whose logits row for the last token is table[last]; table is (V, V) float64."""

    def model(tokens: torch.Tensor) -> torch.Tensor:
        # Finished rows feed pad_id back in, which may be outside the table: wrap it. Their
        # logits are ignored by the learner.
        return table[tokens[:, -1] % table.shape[0]]

    return model


def _reference_one_row(table: torch.Tensor, prompt: list[int], cap: int, eos: int, pad: int):
    """Independent reference: generate one sequence alone, stopping at EOS or the cap."""
    seq = list(prompt)
    generated: list[int] = []
    for _ in range(cap):
        nxt = int(torch.argmax(table[seq[-1]]).item())
        generated.append(nxt)
        seq.append(nxt)
        if nxt == eos:
            break
    return list(prompt) + generated, len(generated)


def _padded_reference(table, prompts, cap, eos, pad):
    rows, lengths = [], []
    for row in prompts.tolist():
        full, gen_len = _reference_one_row(table, row, cap, eos, pad)
        rows.append((full, gen_len))
        lengths.append(gen_len)
    width = prompts.shape[1] + (max(lengths) if lengths else 0)
    out = []
    for full, _ in rows:
        out.append(full + [pad] * (width - len(full)))
    return torch.tensor(out, dtype=torch.int64)


def test_random_bigram_batches_match_one_row_at_a_time_reference():
    gen = torch.Generator().manual_seed(11)
    for trial in range(20):
        vocab = 7
        table = torch.randn(vocab, vocab, generator=gen, dtype=torch.float64)
        prompts = torch.randint(0, vocab, (4, 2), generator=gen, dtype=torch.int64)
        eos, pad, cap = 5, 6, 6
        got = _call(_bigram(table), prompts, cap, eos_id=eos, pad_id=pad)
        want = _padded_reference(table, prompts, cap, eos, pad)
        assert torch.equal(got, want), (
            f"batched output differs from the one-row reference (trial {trial}): "
            "check finished flags, padding and the stop rule"
        )


def test_one_row_hitting_eos_does_not_stop_the_other_rows():
    # Row starting at 0 hits EOS (id 1) at once; row starting at 2 keeps going to id 3, then 0.
    table = torch.tensor(
        [
            [0.0, 9.0, 0.0, 0.0],  # 0 -> 1 (EOS)
            [9.0, 0.0, 0.0, 0.0],  # 1 -> 0
            [0.0, 0.0, 0.0, 9.0],  # 2 -> 3
            [9.0, 0.0, 0.0, 0.0],  # 3 -> 0
        ],
        dtype=torch.float64,
    )
    out = _call(_bigram(table), torch.tensor([[0], [2]]), 3, eos_id=1, pad_id=7)
    expected = torch.tensor([[0, 1, 7, 7], [2, 3, 0, 1]])
    assert torch.equal(out, expected), (
        "one row's EOS stopped the batch: stop only when every row is finished"
    )


def test_finished_rows_are_padded_with_pad_id_not_with_argmax():
    # Row 0 finishes at once. Row 1 keeps going, so row 0 must be padded with pad_id = 8 at every
    # later step, not with the argmax it would otherwise produce.
    table = torch.tensor(
        [
            [0.0, 9.0, 0.0],  # 0 -> 1 (EOS)
            [0.0, 0.0, 9.0],  # 1 -> 2
            [9.0, 0.0, 0.0],  # 2 -> 0
        ],
        dtype=torch.float64,
    )
    out = _call(_bigram(table), torch.tensor([[0], [1]]), 3, eos_id=1, pad_id=8)
    assert out[0].tolist() == [0, 1, 8, 8], "finished row must append pad_id, not keep generating"


def test_eos_token_is_kept_in_the_output():
    table = torch.tensor([[0.0, 9.0], [0.0, 0.0]], dtype=torch.float64)
    out = _call(_bigram(table), torch.tensor([[0]]), 4, eos_id=1, pad_id=0)
    assert out.tolist() == [[0, 1]], "the EOS token must be part of the output, not dropped"


def test_tie_between_logits_chooses_lowest_token_id():
    table = torch.zeros(4, 4, dtype=torch.float64)  # every logit ties
    out = _call(_bigram(table), torch.tensor([[2]]), 1, eos_id=3, pad_id=0)
    assert out.tolist() == [[2, 0]], "ties must go to the lowest id: check argmax tie handling"


def test_step_cap_counts_only_new_tokens():
    table = torch.zeros(3, 3, dtype=torch.float64)
    table[:, 0] = 9.0  # always emit 0, never EOS (EOS = 2)
    out = _call(_bigram(table), torch.tensor([[1, 1, 1]]), 4, eos_id=2, pad_id=2)
    assert out.shape == (1, 3 + 4), "the cap must count new tokens only; the prompt does not count"


def test_model_sees_prefix_that_grows_by_one_column_per_call():
    widths = []
    table = torch.zeros(3, 3, dtype=torch.float64)
    table[:, 1] = 9.0  # every row emits 1, so no row finishes (EOS = 2)

    def model(tokens: torch.Tensor) -> torch.Tensor:
        widths.append(tokens.shape[1])
        return table[tokens[:, -1]]

    _call(model, torch.tensor([[0], [0]]), 3, eos_id=2, pad_id=0)
    assert widths == [1, 2, 3], "each call must receive the prompt plus every token appended so far"


def test_finished_rows_appear_as_pad_in_the_prefix_the_model_sees():
    # Row 0 emits EOS (2) at once; the argmax of its last token (2) is 3, which must not appear.
    table = torch.zeros(4, 4, dtype=torch.float64)
    table[0, 2] = 9.0  # row starting at 0 emits EOS (2) at once
    table[1, 1] = 9.0  # row starting at 1 keeps emitting 1
    table[2, 3] = 9.0  # the token after EOS would be 3 if the row were not finished
    seen = []

    def model(tokens: torch.Tensor) -> torch.Tensor:
        seen.append(tokens.clone())
        return table[tokens[:, -1]]

    _call(model, torch.tensor([[0], [1]]), 3, eos_id=2, pad_id=0)
    assert seen[1][0].tolist() == [0, 2], (
        "the finished row must show its EOS, then the next call pads"
    )
    assert seen[2][0].tolist() == [0, 2, 0], (
        "a finished row must appear as pad_id in the prefix, not as the argmax of its last token"
    )


def test_zero_new_tokens_returns_a_copy_not_an_alias_of_the_prompt():
    prompts = torch.tensor([[1, 2]])
    out = _call(_bigram(torch.zeros(3, 3, dtype=torch.float64)), prompts, 0, eos_id=0, pad_id=0)
    out[0, 0] = 99
    assert prompts[0, 0].item() == 1, "output shares storage with the prompts: return a new tensor"


def test_prompts_are_not_mutated():
    prompts = torch.tensor([[0], [1]])
    before = prompts.clone()
    table = torch.zeros(4, 4, dtype=torch.float64)
    table[:, 3] = 9.0
    _call(_bigram(table), prompts, 3, eos_id=3, pad_id=0)
    assert torch.equal(prompts, before), "the prompts were written to: copy them before appending"


def test_batch_of_one_matches_reference():
    gen = torch.Generator().manual_seed(3)
    table = torch.randn(5, 5, generator=gen, dtype=torch.float64)
    prompts = torch.tensor([[4, 0, 1]])
    got = _call(_bigram(table), prompts, 5, eos_id=2, pad_id=4)
    assert torch.equal(got, _padded_reference(table, prompts, 5, 2, 4)), "single-row batch differs"


def test_large_vocabulary_ids_round_trip_exactly():
    vocab = 1000
    table = torch.zeros(vocab, vocab, dtype=torch.float64)
    for i in range(vocab):
        table[i, (i * 7 + 3) % vocab] = 1.0
    eos = vocab - 1
    out = _call(_bigram(table), torch.tensor([[0]]), 4, eos_id=eos, pad_id=0)
    assert out.tolist() == [[0, 3, 24, 171, 1200 % vocab]], "large token ids were altered"


def test_output_is_int64():
    table = torch.zeros(2, 2, dtype=torch.float64)
    out = _call(_bigram(table), torch.tensor([[0]]), 1, eos_id=1, pad_id=0)
    assert out.dtype == torch.int64, "output must be int64 token ids"


def test_empty_prompt_raises_value_error():
    _expect_value_error(
        _bigram(torch.zeros(2, 2, dtype=torch.float64)),
        torch.zeros(1, 0, dtype=torch.int64),
        2,
        eos_id=0,
        pad_id=0,
    )


def test_zero_batch_raises_value_error():
    _expect_value_error(
        _bigram(torch.zeros(2, 2, dtype=torch.float64)),
        torch.zeros(0, 1, dtype=torch.int64),
        2,
        eos_id=0,
        pad_id=0,
    )


def test_float_prompts_raise_value_error():
    _expect_value_error(
        _bigram(torch.zeros(2, 2, dtype=torch.float64)), torch.zeros(1, 1), 2, eos_id=0, pad_id=0
    )


def test_negative_step_cap_raises_value_error():
    _expect_value_error(
        _bigram(torch.zeros(2, 2, dtype=torch.float64)), torch.tensor([[0]]), -1, eos_id=0, pad_id=0
    )


def test_bool_eos_id_raises_value_error():
    _expect_value_error(
        _bigram(torch.zeros(2, 2, dtype=torch.float64)),
        torch.tensor([[0]]),
        2,
        eos_id=True,
        pad_id=0,
    )


def test_model_returning_wrong_batch_shape_raises_value_error():
    def bad_model(tokens: torch.Tensor) -> torch.Tensor:
        return torch.zeros(1, 2, dtype=torch.float64)

    _expect_value_error(bad_model, torch.tensor([[0], [1]]), 2, eos_id=0, pad_id=0)


def test_model_returning_zero_vocabulary_raises_value_error():
    def model(tokens: torch.Tensor) -> torch.Tensor:
        return torch.zeros(tokens.shape[0], 0, dtype=torch.float64)

    _expect_value_error(model, torch.tensor([[0]]), 2, eos_id=0, pad_id=0)


def test_single_entry_vocabulary_is_a_valid_model_output():
    # V >= 1 is allowed, so a model with one vocabulary entry must work: its only choice is 0.
    def model(tokens: torch.Tensor) -> torch.Tensor:
        return torch.ones(tokens.shape[0], 1, dtype=torch.float64)

    try:
        out = _learner(model, torch.tensor([[3]]), 2, eos_id=5, pad_id=7)
    except Exception as exc:
        pytest.fail(f"a model with one vocabulary entry is valid: raised {type(exc).__name__}")
    assert out.tolist() == [[3, 0, 0]], (
        "with a one-entry vocabulary the only argmax is 0: check the choice and the appended token"
    )
