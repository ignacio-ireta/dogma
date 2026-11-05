## Hint 1 — concept

A training step has five stages, and each one depends on the one before it. The gradient describes the parameters that produced the loss. The update must not itself be recorded as part of the function being differentiated.

## Hint 2 — formula / approach

For each parameter $\theta$, the update is $\theta \leftarrow \theta - \eta\,\partial L/\partial\theta$, where $\eta$ is `lr`. The reported loss is $L$ from the forward pass, before the update. Clear the old gradients before the forward pass, because `backward()` adds into `.grad`.

## Hint 3 — code shape

Validate `lr`. Set each parameter's `.grad` to `None`. Compute `prediction` and `loss`, then call `loss.backward()`. Inside `torch.no_grad()`, subtract `lr * p.grad` from each parameter in place with `sub_`. Return `float(loss.detach())`.
