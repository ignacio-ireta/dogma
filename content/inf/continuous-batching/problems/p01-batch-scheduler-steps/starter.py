from __future__ import annotations


def simulate_batching(  # noqa: PLR0917 - the six-argument signature is fixed by the contract
    arrivals: list[int],
    prompt_lens: list[int],
    output_lens: list[int],
    max_batch: int,
    prefill_budget: int,
    policy: str,
) -> list[tuple[int, int]]:
    """Return (first_token_step, finish_step) per request under static or continuous batching.

    Steps are integers from 0. In each step: arrivals join the FCFS queue; admission takes
    queue heads while there is room (fewer than max_batch running and the prompt tokens admitted
    this step stay within prefill_budget; no skipping). "static" admits only when nothing is
    running. Every running request emits one token per step, so a request admitted at s with
    output length O finishes at s + O - 1. Raise ValueError for bad lengths, bounds, a prompt
    longer than prefill_budget, or an unknown policy. Never mutate the inputs.
    """
    raise NotImplementedError
