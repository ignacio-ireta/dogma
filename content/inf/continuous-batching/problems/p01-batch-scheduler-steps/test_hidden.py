from __future__ import annotations

import copy
import random

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from submission import simulate_batching


def run(arrivals, prompts, outputs, max_batch, budget, policy):  # noqa: PLR0917
    """Call the learner; a crash is reported with the contract rule it broke."""
    try:
        result = simulate_batching(arrivals, prompts, outputs, max_batch, budget, policy)
    except ValueError:
        pytest.fail(
            "valid workload raised ValueError: check the policy and length rules", pytrace=False
        )
    except Exception as exc:
        pytest.fail(
            f"valid workload raised {type(exc).__name__}: return a list of (first, finish) pairs",
            pytrace=False,
        )
    if not isinstance(result, list) or any(
        not isinstance(pair, tuple) or len(pair) != 2 for pair in result
    ):
        pytest.fail("return a list of (first_token_step, finish_step) tuples", pytrace=False)
    return result


def step_model(arrivals, prompts, outputs, max_batch, budget, policy):  # noqa: PLR0917
    """Independent model: a plain time loop over active requests with a remaining-token count.

    It does not compute finish = first + output - 1. Each request is decremented once per step.
    """
    n = len(arrivals)
    first, finish = [None] * n, [None] * n
    waiting: list[int] = []
    active: list[list[int]] = []  # [request, tokens still to produce], in admission order
    step = 0
    finished = 0
    while finished < n:
        waiting += [i for i in range(n) if arrivals[i] == step]
        can_admit = policy == "continuous" or not active
        newly: list[int] = []
        if can_admit:
            spent = 0
            while waiting and len(active) + len(newly) < max_batch:
                if spent + prompts[waiting[0]] > budget:
                    break
                req = waiting.pop(0)
                spent += prompts[req]
                newly.append(req)
        for req in newly:
            first[req] = step
            active.append([req, outputs[req]])
        still = []
        for slot in active:
            slot[1] -= 1
            if slot[1] == 0:
                finish[slot[0]] = step
                finished += 1
            else:
                still.append(slot)
        active = still
        step += 1
    return list(zip(first, finish, strict=False))


def random_workload(rng: random.Random):
    n = rng.randint(1, 9)
    budget = rng.randint(6, 12)
    arrivals = [rng.randint(0, 10) for _ in range(n)]
    prompts = [rng.randint(1, 6) for _ in range(n)]
    outputs = [rng.randint(1, 6) for _ in range(n)]
    return arrivals, prompts, outputs, rng.randint(1, 4), budget


@pytest.mark.parametrize("policy", ["static", "continuous"])
@pytest.mark.parametrize("seed", range(40), ids=[f"seed-{s}" for s in range(40)])
def test_random_workloads_match_the_independent_step_model(seed, policy):
    arrivals, prompts, outputs, max_batch, budget = random_workload(random.Random(seed))  # noqa: S311
    want = step_model(arrivals, prompts, outputs, max_batch, budget, policy)
    got = run(arrivals, prompts, outputs, max_batch, budget, policy)
    if len(got) != len(want):
        pytest.fail("one (first, finish) pair per request, in input order", pytrace=False)
    for g, w in zip(got, want, strict=True):
        if g != w:
            pytest.fail(
                f"step mismatch for {policy}: check when freed slots are reused, the one-token-"
                "per-step rule, and finish = first + output - 1",
                pytrace=False,
            )


@settings(max_examples=60, deadline=None, derandomize=True, report_multiple_bugs=False)
@given(st.integers(0, 10_000), st.sampled_from(["static", "continuous"]))
def test_invariants_hold_for_any_workload(seed, policy):
    arrivals, prompts, outputs, max_batch, budget = random_workload(random.Random(seed))  # noqa: S311
    got = run(arrivals, prompts, outputs, max_batch, budget, policy)
    for i, (first, finish) in enumerate(got):
        if finish - first != outputs[i] - 1:
            pytest.fail("finish step must be first step + output length - 1", pytrace=False)
        if first < arrivals[i]:
            pytest.fail("a request cannot produce a token before it arrives", pytrace=False)
    for t in {f for _, f in got} | {s for s, _ in got}:
        running = sum(1 for s, f in got if s <= t <= f)
        if running > max_batch:
            pytest.fail("batch limit broken: more running requests than max_batch", pytrace=False)
    order = sorted(range(len(arrivals)), key=lambda i: (arrivals[i], i))
    firsts = [got[i][0] for i in order]
    if firsts != sorted(firsts):
        pytest.fail(
            "FCFS mistake: an earlier arrival must not start after a later one", pytrace=False
        )


def test_static_holds_freed_slots_idle_until_the_batch_drains():
    args = ([0, 0, 0, 0], [1, 1, 1, 1], [10, 1, 1, 1], 2, 10)
    assert run(*args, "static") == [(0, 9), (0, 0), (10, 10), (10, 10)], (
        "static batching mistake: freed slots must stay idle while the batch still runs"
    )
    assert run(*args, "continuous") == [(0, 9), (0, 0), (1, 1), (2, 2)], (
        "continuous batching mistake: each freed slot must be refilled on the next step"
    )


def test_prefill_budget_is_per_step_not_per_queue():
    # Both requests fit the budget only one at a time, so they start on consecutive steps.
    got = run([0, 0], [5, 5], [2, 2], 4, 6, "continuous")
    assert got == [(0, 1), (1, 2)], "budget mistake: the prefill budget applies to each step"


def test_empty_workload_returns_an_empty_list_for_both_policies():
    assert run([], [], [], 2, 8, "static") == []
    assert run([], [], [], 2, 8, "continuous") == []


def test_invalid_arguments_raise_value_error():
    bad_calls = [
        ([0], [1, 1], [1], 2, 8, "static", "mismatched list lengths"),
        ([-1], [1], [1], 2, 8, "static", "negative arrival"),
        ([0], [0], [1], 2, 8, "static", "zero prompt length"),
        ([0], [1], [0], 2, 8, "static", "zero output length"),
        ([True], [1], [1], 2, 8, "static", "bool arrival"),
        ([0], [9], [1], 2, 8, "static", "prompt longer than the budget"),
        ([0], [1], [1], 0, 8, "static", "max_batch of zero"),
        ([0], [1], [1], 2, 0, "static", "budget of zero"),
        ([0], [1], [1], 2, 8, "dynamic", "unknown policy"),
    ]
    for arrivals, prompts, outputs, max_batch, budget, policy, label in bad_calls:
        try:
            simulate_batching(arrivals, prompts, outputs, max_batch, budget, policy)
        except ValueError:
            continue
        except Exception as exc:
            pytest.fail(f"{label}: expected ValueError, got {type(exc).__name__}", pytrace=False)
        pytest.fail(f"{label}: expected ValueError, nothing was raised", pytrace=False)


def test_inputs_are_not_mutated():
    args = ([0, 1, 1], [2, 3, 1], [2, 1, 3], 2, 5)
    before = copy.deepcopy(args)
    run(*args[:3], *args[3:], "continuous")
    run(*args[:3], *args[3:], "static")
    assert args == before, "input lists must not be mutated"


def test_prompt_longer_than_budget_is_rejected_even_beside_a_short_prompt():
    import signal  # noqa: PLC0415

    # Without the up-front rejection the head of the queue never fits and the simulation never
    # ends, so a timer turns that hang into a failure.
    def on_timeout(signum, frame):
        raise TimeoutError

    previous = signal.signal(signal.SIGALRM, on_timeout)
    signal.alarm(2)
    try:
        simulate_batching([0, 0], [1, 9], [1, 1], 2, 8, "continuous")
    except ValueError:
        return
    except TimeoutError:
        pytest.fail(
            "a prompt longer than prefill_budget must be rejected up front, "
            "not left waiting forever in the queue",
            pytrace=False,
        )
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, previous)
    pytest.fail(
        "a prompt longer than prefill_budget was accepted: expected ValueError", pytrace=False
    )
