## Hint 1 — concept

Each update moves against the gradient, and the learning rate sets how far. The loop has two ways
to end: the gradient is small enough, or the budget is spent. Where you put the stop test decides
whether the count you return is right.

## Hint 2 — formula / approach

The update is `x ← x − lr · grad(x)`. Test ‖grad(x)‖ before you update, so that when the test passes
after k updates, you return k. Check the learning rate before you compute anything, and check
that the iterates stay finite after every update, so a divergent run stops early with `diverged`.
Copy `x0` first, because the update must never write into the caller's array.

## Hint 3 — code shape

`x = np.array(x0, dtype=np.float64)`, then `for k in range(max_iters):`. Inside the loop, compute
`g = np.asarray(grad(x), dtype=np.float64)` and return `x, k` when `np.sqrt(g @ g) <= tol`. Otherwise
set `x = x - lr * g`, check `np.isfinite`, and raise `ValueError("diverged")` if it fails. After the
loop, return `x, max_iters`.
