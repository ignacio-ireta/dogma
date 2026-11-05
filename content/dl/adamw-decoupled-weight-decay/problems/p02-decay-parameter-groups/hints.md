## Hint 1 — concept

Weight decay pulls a weight toward zero. It helps matrices, which mix many inputs. It does
little for a bias or a gain that scales one feature, so the optimizer should leave those alone.

## Hint 2 — formula / approach

Decide membership per tensor from two facts: its rank, and whether its name ends in `bias`. Put
decayed tensors in one list and the rest in another, keeping the order you saw them. A group
with no tensors is dropped. The decayed group comes first and carries your `weight_decay`.

## Hint 3 — code shape

Validate the inputs before the loop. Loop over `named_params`, test `param.ndim` and the final
component of the name, and append the tensor object itself. Build the list of dicts from the two
non-empty lists. Return it.
