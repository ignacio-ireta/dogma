## Hint 1 — concept

A transformer's parameters are a sum of tables and matrices, and most of them repeat once per layer.
Count each component on its own, then add them. The config only tells you which components exist.
Tying removes the head, and a learned position table adds one more table.

## Hint 2 — formula / approach

With $D$ the width, $F$ the MLP width, $V$ the vocabulary and $N$ the layers:

- attention per block: $4D^2$, plus $4D$ with bias
- gelu MLP per block: $2DF$, plus $F + D$ with bias. Swiglu: $3DF$, plus $2F + D$ with bias
- norms per block: two norms, each $2D$ for layernorm or $D$ for rmsnorm
- token table: $VD$. Position table: $CD$ when learned. Head: $VD$ when untied, $0$ when tied

## Hint 3 — code shape

Validate the config first: exact key set, `type(x) is int` and `type(x) is bool`, the divisibility
rule, and the two string choices. Then compute one value for the norm size, one for the attention block,
one for the MLP block, and one for the norms block. Put the per-block values into the dict multiplied by
`n_layers`, the table values next to them, and add a `total` key as the sum of the others.
