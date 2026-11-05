from __future__ import annotations

from submission import simulate_batching

ARRIVALS = [0, 0, 1, 2]
PROMPTS = [4, 4, 4, 4]
OUTPUTS = [3, 1, 2, 2]


def test_static_batch_waits_for_its_longest_request():
    got = simulate_batching(ARRIVALS, PROMPTS, OUTPUTS, 2, 100, "static")
    assert got == [(0, 2), (0, 0), (3, 4), (3, 4)], (
        "static batching mistake: admission must wait until nothing is running"
    )


def test_continuous_batch_refills_a_slot_on_the_next_step():
    got = simulate_batching(ARRIVALS, PROMPTS, OUTPUTS, 2, 100, "continuous")
    assert got == [(0, 2), (0, 0), (1, 2), (3, 4)], (
        "continuous batching mistake: a slot freed at step t must be reused at step t + 1"
    )


def test_prefill_budget_stops_the_queue_without_skipping_ahead():
    got = simulate_batching([0, 0, 0], [4, 4, 2], [1, 1, 1], 3, 6, "continuous")
    assert got == [(0, 0), (1, 1), (1, 1)], (
        "budget mistake: request 2 fits the leftover budget but must not jump the queue"
    )
