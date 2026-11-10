from __future__ import annotations

from collections import deque


def _checked_ints(values: list[int], name: str, minimum: int) -> None:
    """Reject bool, non-int and out-of-range values before the simulation starts."""
    for value in values:
        if not isinstance(value, int) or value < minimum:
            raise ValueError(f"{name} entries must be ints of at least {minimum}")


def simulate_batching(  # noqa: PLR0917 - the six-argument signature is fixed by the contract
    arrivals: list[int],
    prompt_lens: list[int],
    output_lens: list[int],
    max_batch: int,
    prefill_budget: int,
    policy: str,
) -> list[tuple[int, int]]:
    """Wrong on purpose: bool values are accepted as ints. Return (first, finish) per request."""
    if policy not in ("static", "continuous"):
        raise ValueError("policy must be 'static' or 'continuous'")
    if not len(arrivals) == len(prompt_lens) == len(output_lens):
        raise ValueError("arrivals, prompt_lens and output_lens must have the same length")
    _checked_ints(arrivals, "arrival", 0)
    _checked_ints(prompt_lens, "prompt length", 1)
    _checked_ints(output_lens, "output length", 1)
    for name, value in (("max_batch", max_batch), ("prefill_budget", prefill_budget)):
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"{name} must be an int of at least 1")
    if max(prompt_lens, default=0) > prefill_budget:
        raise ValueError("a prompt is longer than prefill_budget and could never be admitted")

    n = len(arrivals)
    first = [0] * n
    finish = [0] * n
    # FCFS: by arrival step, then by index for ties. Earlier arrivals stay ahead of later ones.
    by_arrival = sorted(range(n), key=lambda i: (arrivals[i], i))
    next_arrival = 0
    queue: deque[int] = deque()
    # Running requests, in admission order, with the tokens each one still has to produce.
    remaining: dict[int, int] = {}
    done = 0
    step = 0
    while done < n:
        # 1. Arrivals at this step join the back of the queue.
        while next_arrival < n and arrivals[by_arrival[next_arrival]] <= step:
            queue.append(by_arrival[next_arrival])
            next_arrival += 1

        # 2. Admission. Static batching only starts a batch when nothing is running. Both
        # policies stop at the first head that does not fit, so there is no skipping.
        may_admit = policy == "continuous" or not remaining
        if may_admit:
            spent = 0
            while queue and len(remaining) < max_batch:
                head = queue[0]
                if spent + prompt_lens[head] > prefill_budget:
                    break
                queue.popleft()
                spent += prompt_lens[head]
                remaining[head] = output_lens[head]
                first[head] = step  # the prefill of this step produces the first token

        # 3. Every running request produces one token; 4. those whose last token it was finish.
        for req in list(remaining):
            remaining[req] -= 1
            if remaining[req] == 0:
                finish[req] = step
                del remaining[req]
                done += 1
        step += 1
    return list(zip(first, finish, strict=False))
