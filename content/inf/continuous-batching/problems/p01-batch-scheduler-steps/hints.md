## Hint 1 — concept

Time advances in steps. Each step, requests arrive, the scheduler admits waiting requests if it has
room, and every running request produces one token. The two policies differ only in when admission is
allowed: static waits until the whole batch has drained, and continuous admits whenever a slot is free.

## Hint 2 — formula / approach

Keep a FCFS queue of arrived requests, the running set with each request's tokens left, and the
current step. A request admitted at step `s` gets `first = s`. It emits a token every step, so with
`O` output tokens it finishes at `s + O - 1`. Admission stops at the first queue head that does not
fit the batch limit or the prefill budget. Under static, admission is allowed only when the running
set is empty at the start of the step.

## Hint 3 — code shape

Write a `while done < n` loop over the step index. First move arrivals into the queue, then admit from
the head while there is room, then decrement every running request's remaining count and record the
finish step when it reaches zero. Validate lengths, types and bounds before the loop, and raise
`ValueError` for an unknown policy.
