# Interview notes: Gradients with framework autograd

## Q1. Why do the gradients look wrong on the second training step?

**60-second answer.** `backward()` adds into each leaf's `.grad` rather than replacing it. If nothing clears the field, the second step's gradient is the sum of the first and second. The fix is `zero_grad()` before the forward pass, in the same place every step. A quick diagnostic is to check that `.grad` is `None` or zero before `backward()`.

**Follow-ups.** Why does `zero_grad` set `.grad` to `None` by default? It saves a write and makes an unset gradient visible. Does gradient accumulation (several micro-batches before a step) depend on this? Yes, on purpose: you call `zero_grad` only after the update, and the loss is scaled per micro-batch.

## Q2. How would you get a derivative without changing any tensor state?

**60-second answer.** Use `torch.autograd.grad(output, inputs)`. It returns the gradients as a tuple and writes nothing into `.grad`. Work on a detached copy that requires grad, so the caller's `requires_grad` flag stays as it was. Wrap the computation in `enable_grad` if the caller might be inside `no_grad`. Check the result against central finite differences in float64.

**Follow-ups.** Why not `backward()` on a copy? It works, but it is a second code path to keep consistent, and it accumulates. When does `create_graph=True` matter? When you need a second derivative, for example a gradient penalty. Why does `backward(gradient=v)` need `v` for a non-scalar output? Backward needs a scalar to differentiate, so $\mathbf{v}$ turns $\mathbf{y}$ into the scalar $\mathbf{v}\cdot\mathbf{y}$, whose gradient is $J^\top\mathbf{v}$. It still writes into `.grad`; `autograd.grad(y, x, grad_outputs=v)` returns the same product with no side effect.

## Common mistakes

- Calling `backward()` twice on the same graph without `retain_graph=True`, which raises an error.
- Reading `.grad` on a non-leaf tensor and assuming it is the true gradient. It is `None` unless `retain_grad()` was called.
- Using `round` or `floor` in the forward pass and expecting training to continue. Their derivative is zero almost everywhere.
- Calling `requires_grad_(False)` on a non-leaf tensor, which raises an error. Detach it instead.
