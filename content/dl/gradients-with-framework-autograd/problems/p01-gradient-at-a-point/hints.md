## Hint 1 — concept

The caller's tensor must not become part of the computation. Work on a private copy that starts a graph of its own, and ask autograd for the derivative of that copy. Autograd returns the gradient and writes nothing into `.grad` when you call `torch.autograd.grad`.

## Hint 2 — formula / approach

Let $f:\mathbb{R}^n\to\mathbb{R}$. The gradient is the vector $\nabla f(x)$ of partial derivatives. `torch.autograd.grad(out, leaf)` returns a tuple, and the first entry is $\nabla f$. If $f$ does not use the leaf, the result is `None`, and the derivative is zero. Inside `torch.no_grad()` a graph is not recorded, so you must turn recording back on for the computation.

## Hint 3 — code shape

Check the rank and dtype of `x`. Inside `torch.enable_grad()`, make `leaf = x.detach().clone().requires_grad_(True)`, compute `out = f(leaf)`, and return zeros if `out` does not require grad. Then call `torch.autograd.grad(out, leaf, allow_unused=True)`, replace `None` with zeros, and return the result detached.
