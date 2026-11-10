from __future__ import annotations

from collections import deque


def _is_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def simulate_batching(  # noqa: PLR0912, PLR0915, PLR0917
    arrivals: list[int],
    prompt_lens: list[int],
    output_lens: list[int],
    max_batch: int,
    prefill_budget: int,
    policy: str,
) -> list[tuple[int, int]]:
    """Return (first_token_step, finish_step) per request under static or continuous batching."""
    if policy not in ("static", "continuous"):
        raise ValueError("policy must be 'static' or 'continuous'")
    if not (len(arrivals) == len(prompt_lens) == len(output_lens)):
        raise ValueError("arrivals, prompt_lens and output_lens must have the same length")
    if not _is_int(max_batch) or max_batch < 1:
        raise ValueError("max_batch must be an int of at least 1")
    if not _is_int(prefill_budget) or prefill_budget < 1:
        raise ValueError("prefill_budget must be an int of at least 1")
    for arrival, prompt, output in zip(arrivals, prompt_lens, output_lens, strict=True):
        if not _is_int(arrival) or arrival < 0:
            raise ValueError("arrivals must be ints >= 0")
        if not _is_int(prompt) or prompt < 1:
            raise ValueError("prompt_lens must be ints >= 1")
        if not _is_int(output) or output < 1:
            raise ValueError("output_lens must be ints >= 1")
        if prompt > prefill_budget:
            raise ValueError("a prompt is longer than prefill_budget and can never be admitted")

    n = len(arrivals)
    if n == 0:
        return []

    # FCFS order: by arrival step, then by index.
    pending = sorted(range(n), key=lambda i: (arrivals[i], i))
    first = [0] * n
    finish = [0] * n
    produced = [0] * n
    queue: deque[int] = deque()
    running: list[int] = []
    ptr = 0
    done = 0
    t = 0

    while done < n:
        # Skip idle steps: nothing running and nothing waiting.
        if not running and not queue:
            t = max(t, arrivals[pending[ptr]])

        # 1. Arrivals join the queue.
        while ptr < n and arrivals[pending[ptr]] <= t:
            queue.append(pending[ptr])
            ptr += 1

        # 2. Admission. Static batches are formed only when nothing is running.
        may_admit = policy == "continuous" or not running
        if may_admit:
            admitted_tokens = 0
            while queue and len(running) < max_batch:
                head = queue[0]
                if admitted_tokens + prompt_lens[head] > prefill_budget:
                    break
                queue.popleft()
                running.append(head)
                admitted_tokens += prompt_lens[head]
                first[head] = t

        # 3. Forward pass: every running request produces one token.
        still_running: list[int] = []
        for i in running:
            produced[i] += 1
            # 4. Finish: the last token leaves the batch at the end of the step.
            if produced[i] == output_lens[i]:
                finish[i] = t
                done += 1
            else:
                still_running.append(i)
        running = still_running
        t += 1

    return list(zip(first, finish, strict=True))
