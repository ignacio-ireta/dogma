# Hidden tests (CONTENT_SPEC §6.4). Names describe the behaviour; messages name the kind of mistake
# and never print hidden inputs, shapes or expected values.
from __future__ import annotations

import collections
import functools
import itertools
import re

import pytest
import torch
from submission import greedy_generate as _learner

from dogma_data import load_text

BOOKS = ("gutenberg_frankenstein", "gutenberg_pride_prejudice", "gutenberg_walden")


@functools.cache
def _token_lines(vocab: int) -> tuple[tuple[int, ...], ...]:
    """The lines of three vendored novels, reduced to their ``vocab`` most frequent words.

    A word's id is its frequency rank over the three books. Every other word is dropped, so a
    line becomes its run of the most common words (the, and, of, to, ...), and a line with
    none of them is left out. No id stands for "any other word": such an id would follow
    almost every word and a greedy model would emit nothing else.
    """
    lines = [
        re.findall(r"[a-z]+", line.lower())
        for name in BOOKS
        for line in load_text(name).splitlines()
    ]
    ranked = collections.Counter(word for line in lines for word in line).most_common()
    rank = {word: index for index, (word, _) in enumerate(ranked)}
    kept = (tuple(rank[word] for word in line if rank[word] < vocab) for line in lines)
    return tuple(line for line in kept if line)


def _bigram_model(
    gen: torch.Generator,
    vocab: int,
    rows: int,
    width: int,
    *,
    eos: int = -1,
    pad: int = -1,
    cap: int = 0,
):
    """A real bigram language model and real prompts, both from a stretch of a novel.

    The table is fitted to forty consecutive lines picked by the generator: entry (p, n) is the
    log of the number of times word id n follows word id p, with a floor of one tenth of a
    count. Forty lines are few, so every stretch gives a different model. The prompts are
    ``width`` consecutive word ids taken from anywhere in ``rows`` lines of the same stretch.
    When ``eos`` is given and the stretch holds that word far enough into a line, the first
    prompt is the one that ends on it: a prompt whose last word is the end-of-sequence id.
    Which word gets which id is drawn afresh each time, because ids are labels: in one model
    the end-of-sequence id is a common word that every row soon reaches, in another a rare one.

    ``pad_id`` may be a real word of the vocabulary. When ``pad`` and ``cap`` are given and the
    rows that generate longest all end on the same word, that word trades ids with ``pad``, so
    the last column of the answer holds a generated token equal to the pad id.
    """
    lines = _token_lines(vocab)
    first = int(torch.randint(0, len(lines) - 40, (1,), generator=gen))
    relabel = torch.randperm(vocab, generator=gen).tolist()
    stretch = [tuple(relabel[word] for word in line) for line in lines[first : first + 40]]
    counts = torch.full((vocab, vocab), 0.1, dtype=torch.float64)
    for line in stretch:
        for previous, current in itertools.pairwise(line):
            counts[previous, current] += 1.0
    long_enough = [line for line in stretch if len(line) >= width]
    prompts = []
    ending = [
        line[stop - width : stop]
        for line in long_enough
        for stop in range(width, len(line) + 1)
        if line[stop - 1] == eos
    ]
    if ending:
        prompts.append(ending[int(torch.randint(0, len(ending), (1,), generator=gen))])
    while len(prompts) < rows:
        line = long_enough[int(torch.randint(0, len(long_enough), (1,), generator=gen))]
        start = int(torch.randint(0, len(line) - width + 1, (1,), generator=gen))
        prompts.append(line[start : start + width])
    batch = torch.tensor(prompts, dtype=torch.int64)
    table = torch.log(counts)
    if pad >= 0:
        runs = [_reference_one_row(table, row, cap, eos, pad) for row in batch.tolist()]
        longest = max(length for _, length in runs)
        last = {full[-1] for full, length in runs if length == longest}
        if longest > 0 and len(last) == 1 and not last & {eos, pad}:
            word = last.pop()
            swap = torch.arange(vocab)
            swap[word], swap[pad] = pad, word
            table, batch = table[swap][:, swap], swap[batch]
    return table, batch


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
        eos, pad, cap = 5, 6, 6
        table, prompts = _bigram_model(gen, vocab, 4, 2, eos=eos, pad=pad, cap=cap)
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
    table, _ = _bigram_model(gen, 5, 1, 3)
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


def test_bool_step_cap_raises_value_error():
    # A bool is not a step count: True would otherwise be read as a cap of one step.
    _expect_value_error(
        _bigram(torch.zeros(2, 2, dtype=torch.float64)),
        torch.tensor([[0]]),
        True,
        eos_id=0,
        pad_id=0,
    )


def test_finished_rows_may_carry_non_finite_logits():
    # Logits of a finished row are ignored, so NaN there must neither raise nor change the output.
    def model(tokens: torch.Tensor) -> torch.Tensor:
        logits = torch.zeros(tokens.shape[0], 3, dtype=torch.float64)
        first = tokens[:, 0] == 0  # row 0: prompt [0], emits EOS (id 1) at its first step
        logits[first, 1] = 1.0
        logits[~first, 2] = 1.0  # row 1: always emits id 2
        if tokens.shape[1] >= 2:
            logits[first] = float("nan")  # row 0 is finished from step 2 on
        return logits

    try:
        out = _learner(model, torch.tensor([[0], [2]]), 3, eos_id=1, pad_id=0)
    except ValueError:
        pytest.fail("logits of finished rows are ignored: a non-finite value there must not raise")
    # Row 0 is [prompt, EOS, pad, pad]; row 1 runs to the cap of 3 new tokens.
    expected = torch.tensor([[0, 1, 0, 0], [2, 2, 2, 2]])
    assert torch.equal(out, expected), (
        "finished rows must be padded and their logits ignored, whatever values they hold"
    )
