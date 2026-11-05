## Hint 1 — concept

The loop is four nested stages: epochs, a seeded shuffle per epoch, mini-batches, and evaluation. Keep the shuffle on its own generator, so that nothing else draws from it. Validation is a measurement, so it runs with no gradients and in eval mode.

## Hint 2 — formula / approach

Per epoch, the training loss is $\frac{1}{n}\sum_k b_k L_k$, where $b_k$ is each batch's size and $L_k$ its loss before the step. Validation is the mean squared error over all validation rows. Keep the parameters with the lowest validation loss, and replace them only on a strict improvement.

## Hint 3 — code shape

Create `torch.optim.SGD` and one `torch.Generator` seeded with `seed`. Each epoch, draw one `torch.randperm(n, generator=g)`. Loop over `range(0, n, batch_size)` and take `perm[start:start + batch_size]`, which keeps the short last batch. After each epoch, switch to eval mode and measure validation under `torch.no_grad()`. Store a `copy.deepcopy` of the state dict on improvement, and load it at the end.
