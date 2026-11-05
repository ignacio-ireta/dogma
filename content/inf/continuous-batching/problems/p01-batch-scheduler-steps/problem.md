# Simulate static and continuous batching

Implement `simulate_batching(arrivals, prompt_lens, output_lens, max_batch, prefill_budget, policy)`.
It runs a step-by-step model of a generative-model server and returns, for every request, the step
of its **first token** and the step of its **last token**. Requests are admitted first come, first
served (FCFS). `policy` is `"static"` or `"continuous"`.

## Contract

### Step semantics (exact)

Steps are integers `t = 0, 1, 2, ...`. Each step is one forward pass over the running batch.
Within step `t`, in this order:

1. **Arrivals.** Every request with `arrivals[i] == t` joins the FCFS queue. Requests that arrived
   at the same step queue in index order. Earlier arrivals stay ahead.
2. **Admission.** Take requests from the head of the queue while both hold:
   - the number of running requests is below `max_batch`; and
   - the prompt tokens admitted in this step, plus the head's `prompt_lens[i]`, stay within
     `prefill_budget`.

   Stop at the first head that does not fit. A later, smaller request is **not** admitted ahead of
   it (no skipping). Under `"static"`, admission happens only when **no request is running** at the
   start of step `t`, so a batch is formed only when the previous one has fully drained. Under
   `"continuous"`, admission happens whenever there is room, so a slot freed in step `t - 1` can be
   used in step `t`.
3. **Forward pass.** Every running request produces exactly one token this step. A request admitted
   in step `t` produces its first token now (its prefill), so `first_token_step = t`. An older
   running request produces its next token.
4. **Finish.** A request whose last token was produced in this step finishes now. It leaves the
   batch at the end of the step.

Consequence: a request admitted at step `s` with `output_lens[i] = O` produces token `k` at step
`s + k - 1`, so `finish_step = s + O - 1`. A request with `O = 1` has `first_token_step ==
finish_step`.

- `arrivals`, `prompt_lens`, `output_lens`: lists of `int`, the same length `n >= 0`. An arrival is
  `>= 0`; prompt and output lengths are `>= 1`. Bool values are invalid. Otherwise `ValueError`.
- `max_batch >= 1` and `prefill_budget >= 1`, both `int`. Otherwise `ValueError`.
- A prompt longer than `prefill_budget` can never be admitted. Raise `ValueError`.
- `policy` must be `"static"` or `"continuous"`. Otherwise `ValueError`.
- Returns a list of `(first_token_step, finish_step)` tuples, one per request, in input order.
  Return `[]` for empty input.
- Must not mutate the input lists.

## Examples

`arrivals = [0, 0, 1, 2]`, `prompt_lens = [4, 4, 4, 4]`, `output_lens = [3, 1, 2, 2]`,
`max_batch = 2`, `prefill_budget = 100`.

Static: at step 0 requests 0 and 1 start (the batch is empty). Request 1 finishes at step 0.
Request 0 runs until step 2, so no admission happens in steps 1 and 2 even though request 2 is
waiting. At step 3 the batch is empty, so requests 2 and 3 start together.

```python
>>> simulate_batching([0, 0, 1, 2], [4, 4, 4, 4], [3, 1, 2, 2], 2, 100, "static")
[(0, 2), (0, 0), (3, 4), (3, 4)]
```

Continuous: request 1 finishes at step 0, so request 2 starts at step 1. Request 0 and request 2
both finish at step 2, so request 3 starts at step 3.

```python
>>> simulate_batching([0, 0, 1, 2], [4, 4, 4, 4], [3, 1, 2, 2], 2, 100, "continuous")
[(0, 2), (0, 0), (1, 2), (3, 4)]
```

Budget case, FCFS with no skipping: `arrivals = [0, 0, 0]`, `prompt_lens = [4, 4, 2]`,
`output_lens = [1, 1, 1]`, `max_batch = 3`, `prefill_budget = 6`, `"continuous"`. At step 0
request 0 fits (4 of 6 tokens). Request 1 would make 8, so it stops the queue, and request 2 (which
would fit in the 2 tokens left) is not admitted ahead of it. Step 1 admits requests 1 and 2.

```python
>>> simulate_batching([0, 0, 0], [4, 4, 2], [1, 1, 1], 3, 6, "continuous")
[(0, 0), (1, 1), (1, 1)]
```

## Constraints

- Allowed: the Python standard library (for example `collections.deque`).
- Not allowed: `numpy`, `pandas` and `torch`. Use plain loops, so each step is visible.
- Do not implement a closed-form throughput formula. Simulate the steps.

## Done looks like

Public examples pass. The hidden tests compare both policies with an independent step-by-step
model on seeded random workloads (FCFS order, budget and batch limits, arrivals inside the same
step). They also check the invariants `finish - first == output - 1`, the batch limit at every step,
the first-token step never earlier than arrival, and a known static case where freed slots stay idle
until the longest request finishes. They check the `ValueError` rules and that inputs are not mutated.
