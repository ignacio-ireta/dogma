# Hidden tests (CONTENT_SPEC §6.4). Names describe the behaviour; messages name the kind of mistake
# and never print hidden inputs, shapes or expected values.
from __future__ import annotations

import collections
import copy
import functools
import math
import random
import re

import pytest
import torch
import torch.nn.functional as ref_f  # oracle only: the learner may not use torch.nn
from submission import build_sft_batch as _learner

from dogma_data import load_frame, sample_rows
from dogma_data import rng as data_rng

IGNORE = -100


def _real_values(gen, shape, dtype=torch.float64):
    """Real numbers in the shape asked for: standardised breast_cancer measurements.

    The tensors of this problem are weights, gradients, scores or hidden states. No dataset
    holds those, so real measurements stand in for them: skewed, correlated along a row, with
    a few entries far from the rest. The torch generator picks the draw.
    """
    if len(shape) == 1 and isinstance(shape[0], (tuple, list, torch.Size)):
        shape = shape[0]
    shape = tuple(int(side) for side in shape)
    total = math.prod(shape)
    if total == 0:
        return torch.zeros(shape, dtype=dtype)
    draw = int(torch.randint(0, 2**31 - 1, (1,), generator=gen))
    key = f"p01-sft-batch-with-loss-masks:values:{draw}"
    rows = max(1, min(569, -(-total // 30)))
    drawn, _ = sample_rows("breast_cancer", data_rng(key), rows, standardize=True)
    flat = torch.tensor(drawn.ravel(), dtype=torch.float64)
    flat = flat.repeat(-(-total // flat.numel()))[:total]
    return flat.reshape(shape).to(dtype)


@functools.cache
def _gsm8k_ids(vocab):
    """GSM8K questions and worked answers as word ids below ``vocab``: real instruction pairs.

    A word's id is its frequency rank over the 300 pairs (0 for the most frequent). Ranks at
    or above ``vocab - 1`` share the last id, as an unknown-word id does, so that id is common.
    """
    table = load_frame("gsm8k")
    questions = [re.findall(r"[a-z0-9#]+", text.lower()) for text in table["question"]]
    answers = [re.findall(r"[a-z0-9#]+", text.lower()) for text in table["answer"]]
    ranked = collections.Counter(w for words in (*questions, *answers) for w in words).most_common()
    rank = {word: index for index, (word, _) in enumerate(ranked)}

    def ids(words):
        return tuple(min(rank[word], vocab - 1) for word in words)

    return tuple((ids(q), ids(a)) for q, a in zip(questions, answers, strict=True))


def _call(pairs, pad_id):
    snapshot = copy.deepcopy(pairs)
    # Record the outcome, then fail after the except blocks so the message stays clean.
    error = None
    try:
        out = _learner(pairs, pad_id)
    except ValueError as exc:
        error = (
            f"valid pairs were rejected with ValueError ({exc}): check the non-empty and int rules"
        )
    except Exception as exc:
        error = f"valid pairs raised {type(exc).__name__}: check the batch shapes and dtypes"
    if error is not None:
        pytest.fail(error, pytrace=False)
    if pairs != snapshot:
        pytest.fail(
            "the input pairs were modified in place; build new tensors and lists", pytrace=False
        )
    return out


def _reference(pairs, pad_id):
    """Plain-Python rebuild of the three tensors, row by row, from the contract."""
    width = max((len(p) + len(r) for p, r in pairs), default=0)
    ids, lab, msk = [], [], []
    for prompt, resp in pairs:
        n = len(prompt) + len(resp)
        ids.append(list(prompt) + list(resp) + [pad_id] * (width - n))
        lab.append([IGNORE] * len(prompt) + list(resp) + [IGNORE] * (width - n))
        msk.append([1] * n + [0] * (width - n))
    return ids, lab, msk


def _random_pairs(seed, rows, max_prompt, max_resp, vocab):
    """Real prompt and response pairs: the first words of a GSM8K question and of its answer."""
    rng = random.Random(seed)  # noqa: S311 -- seeded choice of rows and lengths, not security
    source = _gsm8k_ids(vocab)
    pairs = []
    for _ in range(rows):
        question, answer = source[rng.randrange(len(source))]
        p = list(question[: rng.randint(1, max_prompt)])
        r = list(answer[: rng.randint(1, max_resp)])
        pairs.append((p, r))
    return pairs


def test_random_batches_match_the_plain_python_reference():
    for seed in range(12):
        pairs = _random_pairs(seed, rows=1 + seed % 5, max_prompt=7, max_resp=6, vocab=50)
        pad_id = 49
        input_ids, labels, mask = _call(pairs, pad_id)
        want_ids, want_lab, want_msk = _reference(pairs, pad_id)
        assert input_ids.tolist() == want_ids, (
            "input_ids differ from the reference: check right padding and the token order"
        )
        assert labels.tolist() == want_lab, (
            "labels differ from the reference: check the prompt-length offset and the -100 fill"
        )
        assert mask.tolist() == want_msk, (
            "attention mask differs from the reference: it must be 1 on exactly the n real columns"
        )


def test_shifted_cross_entropy_scores_exactly_the_response_tokens():
    # The loss is the only place the one-token shift happens. With random logits, scoring
    # logits[:, :-1] against labels[:, 1:] must equal the hand-computed NLL of response tokens,
    # where the response token at input position t is predicted from the logits at position t - 1.
    gen = torch.Generator().manual_seed(7)
    vocab = 13
    pairs = _random_pairs(seed=99, rows=4, max_prompt=5, max_resp=4, vocab=vocab)
    _, labels, _ = _call(pairs, pad_id=0)
    width = labels.shape[1]
    logits = _real_values(gen, [len(pairs), width, vocab], torch.float64)

    loss_sum = ref_f.cross_entropy(
        logits[:, :-1, :].reshape(-1, vocab),
        labels[:, 1:].reshape(-1),
        ignore_index=IGNORE,
        reduction="sum",
    )
    logp = torch.log_softmax(logits, dim=-1)
    want = torch.zeros((), dtype=torch.float64)
    for i, (prompt, resp) in enumerate(pairs):
        for k, tok in enumerate(resp):
            t = len(prompt) + k  # input position of the response token
            want = want - logp[i, t - 1, tok]
    assert abs(float(loss_sum) - float(want)) <= 1e-9, (
        "the shifted loss over labels does not score the response tokens: labels must sit on the"
        " input positions (unshifted) and the response token at t is predicted from t - 1"
    )


def test_response_only_positions_are_targets_and_no_prompt_position_is_one():
    pairs = _random_pairs(seed=5, rows=6, max_prompt=6, max_resp=5, vocab=30)
    _, labels, _ = _call(pairs, pad_id=29)
    for i, (prompt, resp) in enumerate(pairs):
        targets = [t for t in range(labels.shape[1]) if labels[i, t].item() != IGNORE]
        expected = list(range(len(prompt), len(prompt) + len(resp)))
        assert targets == expected, (
            "targets must be exactly the response positions: prompt or padding leaked into the loss"
        )


def test_pad_equal_response_tokens_keep_their_labels_in_every_position():
    pairs = [([3, 0], [0, 0, 4]), ([0], [0])]
    _, labels, mask = _call(pairs, pad_id=0)
    assert labels.tolist() == [[IGNORE, IGNORE, 0, 0, 4], [IGNORE, 0, IGNORE, IGNORE, IGNORE]], (
        "response ids equal to pad_id were dropped: masks must come from lengths, not token values"
    )
    assert mask.tolist() == [[1, 1, 1, 1, 1], [1, 1, 0, 0, 0]], (
        "a real token equal to pad_id got attention mask 0: the mask must follow lengths"
    )


def test_single_pair_without_padding_has_no_pad_columns():
    input_ids, labels, mask = _call([([8, 9], [10, 11])], pad_id=0)
    assert input_ids.tolist() == [[8, 9, 10, 11]], "a single row must not be padded"
    assert labels.tolist() == [[IGNORE, IGNORE, 10, 11]], (
        "single-row labels must mask only the prompt"
    )
    assert mask.tolist() == [[1, 1, 1, 1]], "single-row attention mask must be all ones"


def test_first_response_token_right_after_prompt_is_a_target():
    _, labels, _ = _call([([1, 2, 3], [4])], pad_id=0)
    assert labels.tolist() == [[IGNORE, IGNORE, IGNORE, 4]], (
        "first response token unlabelled: an off-by-one in the prompt length masks it"
    )


def test_large_token_ids_stay_exact_in_int64():
    big = 2**62 + 12345
    input_ids, labels, _ = _call([([big], [2**63 - 1, 2**40 + 1])], pad_id=0)
    assert input_ids.tolist() == [[big, 2**63 - 1, 2**40 + 1]], (
        "token ids lost precision: keep ids in int64, not float32 or int32"
    )
    assert labels.tolist() == [[IGNORE, 2**63 - 1, 2**40 + 1]], (
        "large ids must reach labels exactly"
    )


def test_output_tensors_are_int64_with_batch_shape():
    input_ids, labels, mask = _call(_random_pairs(3, rows=3, max_prompt=4, max_resp=4, vocab=9), 0)
    for name, t in (("input_ids", input_ids), ("labels", labels), ("attention_mask", mask)):
        assert t.dtype == torch.int64, (
            f"{name} must be int64 (the loss and embedding need integer ids)"
        )
        assert t.dim() == 2, f"{name} must be a 2-D (B, L) tensor"
    assert input_ids.shape == labels.shape == mask.shape, (
        "the three tensors must share the (B, L) shape"
    )


def test_empty_batch_returns_zero_by_zero_tensors():
    input_ids, labels, mask = _call([], pad_id=0)
    for name, t in (("input_ids", input_ids), ("labels", labels), ("attention_mask", mask)):
        assert tuple(t.shape) == (0, 0), f"an empty batch must give {name} shape (0, 0)"


def test_width_follows_the_longest_row_not_a_fixed_length():
    input_ids, _, _ = _call([([1], [2]), ([3, 4, 5], [6, 7, 8, 9])], pad_id=0)
    assert input_ids.shape == (2, 7), "width must be the longest prompt + response, here 7"


@pytest.mark.parametrize(
    "bad_pairs, why",
    [
        ([([], [1])], "empty prompt"),
        ([([1], [])], "empty response"),
        ([([True], [1])], "bool token id"),
        ([([1], [-1])], "negative token id"),
        ([([1], [2**63])], "token id above the int64 range"),
        ([([1.0], [2])], "float token id"),
        ("not-a-list", "pairs that is not a list"),
        ([([1], [2], [3])], "item with three elements instead of two"),
    ],
    ids=[
        "empty-prompt",
        "empty-response",
        "bool-id",
        "negative-id",
        "id-too-large",
        "float-id",
        "not-a-list",
        "three-element-item",
    ],
)
def test_invalid_pairs_raise_value_error(bad_pairs, why):
    try:
        _learner(bad_pairs, 0)
    except ValueError:
        return
    except Exception as exc:
        pytest.fail(f"{why} must raise ValueError, received {type(exc).__name__}")
    pytest.fail(f"{why} was accepted instead of raising ValueError")


def test_bool_pad_id_raises_value_error():
    try:
        _learner([([1], [2])], True)
    except ValueError:
        return
    except Exception as exc:
        pytest.fail(f"a bool pad_id must raise ValueError, received {type(exc).__name__}")
    pytest.fail("a bool pad_id was accepted instead of raising ValueError")


@pytest.mark.parametrize(
    "bad_pad",
    [-1, 2**63],
    ids=["negative-pad-id", "pad-id-above-int64"],
)
def test_invalid_pad_id_raises_value_error(bad_pad):
    try:
        _learner([([1], [2])], bad_pad)
    except ValueError:
        return
    except Exception as exc:
        pytest.fail(
            f"pad_id outside [0, 2**63-1] must raise ValueError, received {type(exc).__name__}"
        )
    pytest.fail("pad_id outside [0, 2**63-1] was accepted instead of raising ValueError")


def test_pairs_that_is_none_raises_value_error():
    try:
        _learner(None, 0)
    except ValueError:
        return
    except Exception as exc:
        pytest.fail(
            "pairs that is not a list or tuple must raise ValueError, "
            f"received {type(exc).__name__}"
        )
    pytest.fail("pairs that is not a list or tuple was accepted instead of raising ValueError")


def test_prompt_that_is_none_raises_value_error():
    try:
        _learner([(None, [1])], 0)
    except ValueError:
        return
    except Exception as exc:
        pytest.fail(
            "a prompt that is not a list or tuple must raise ValueError, "
            f"received {type(exc).__name__}"
        )
    pytest.fail("a prompt that is not a list or tuple was accepted instead of raising ValueError")


def test_largest_int64_pad_id_is_accepted():
    big_pad = 2**63 - 1
    input_ids, labels, mask = _call([([1], [2]), ([3], [4, 5])], pad_id=big_pad)
    assert input_ids.tolist() == [[1, 2, big_pad], [3, 4, 5]], (
        "pad_id equal to the largest int64 value was rejected or mis-placed: the range is inclusive"
    )
    assert labels.tolist() == [[IGNORE, 2, IGNORE], [IGNORE, 4, 5]], (
        "labels are wrong for a batch whose pad_id is the largest int64 value"
    )
    assert mask.tolist() == [[1, 1, 0], [1, 1, 1]], (
        "attention mask is wrong for a batch whose pad_id is the largest int64 value"
    )


def test_tuple_prompt_and_response_are_valid_inputs():
    # Contract C1: prompt_ids and response_ids may each be a list or a tuple of ints.
    # Worked by hand with the same rules as the Examples in problem.md:
    # row 0 has n = 5 (prompt 2, response 3), row 1 has n = 2 (prompt 1, response 1),
    # so L = 5 and row 1 is padded with pad_id = 0 in columns 2..4.
    pairs = [((5, 6), (7, 8, 9)), ((4,), (3,))]
    snapshot = copy.deepcopy(pairs)
    input_ids, labels, mask = _call(pairs, 0)
    assert pairs == snapshot, "the input pairs were modified in place"
    assert input_ids.tolist() == [[5, 6, 7, 8, 9], [4, 3, 0, 0, 0]], (
        "tuple prompts and responses must build the same inputs as lists"
    )
    assert labels.tolist() == [
        [IGNORE, IGNORE, 7, 8, 9],
        [IGNORE, 3, IGNORE, IGNORE, IGNORE],
    ], "tuple prompts and responses must give the same response-only labels as lists"
    assert mask.tolist() == [[1, 1, 1, 1, 1], [1, 1, 0, 0, 0]], (
        "tuple prompts and responses must give the same attention mask as lists"
    )
