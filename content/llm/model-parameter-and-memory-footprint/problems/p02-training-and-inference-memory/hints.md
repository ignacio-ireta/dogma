## Hint 1 — concept

Memory is a sum of separate terms. Inference holds the weights. Training also holds one gradient per
weight and the optimizer's states, and mixed precision adds a fp32 master copy. The optimizer's
states are always fp32, whatever dtype the weights use.

## Hint 2 — formula / approach

Let $N$ be the number of parameters and $b$ the bytes of the working dtype. Then:

- weights $= Nb$
- gradients $= Nb$ in training
- master copy $= 4N$ when training with a lower-precision dtype and `master_weights` true
- optimizer states $= 4\,s\,N$, where $s$ is 0 for sgd, 1 for momentum, and 2 for adam or adamw
- activations: the given number, added unchanged

## Hint 3 — code shape

Check the arguments first, with `type(x) is int` and `type(x) is bool` for the integers and flags.
Look up the bytes per dtype and the number of states per optimizer in two dictionaries. Branch on
`mode`, build a dict with the terms, then add the activation term and a `total` that is their sum.
