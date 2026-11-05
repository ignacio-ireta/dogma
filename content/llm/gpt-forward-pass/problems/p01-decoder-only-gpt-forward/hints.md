## Hint 1 — concept

The model is a stack of three stages wrapped around the blocks. Ids become vectors by a table
lookup plus a position vector. The blocks refine the stream. A final LayerNorm and one matrix
product turn the stream into one score per vocabulary entry for each position.

## Hint 2 — formula / approach

Compute $x = \mathrm{tok\_emb}[idx] + \mathrm{pos\_emb}[:T]$, then apply each block in order, then
$h = \mathrm{LN}_f(x)$. With tied weights the head is $h\,\mathrm{tok\_emb}^\top$. With an untied
head it is $h\,\mathrm{lm\_head}$. Check the shapes at each step: $(B,T)$, $(B,T,D)$, $(B,T,D)$,
$(B,T,D)$, $(B,T,V)$. Check $T \le C$ and the token id range before you index.

## Hint 3 — code shape

Validate the inputs first and raise `ValueError` with a clear message. Use
`params["tok_emb"][idx]` for the lookup, and add `params["pos_emb"][:T]`, which broadcasts over
the batch. Loop over `params["blocks"]` and call a block helper with the stream and `num_heads`.
Choose the head with `if "lm_head" in params`, then return `h @ head`.
