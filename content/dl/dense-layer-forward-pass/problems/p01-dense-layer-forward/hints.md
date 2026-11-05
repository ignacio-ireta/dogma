## Hint 1 — concept

Every example is one row of `X`, and every output unit is one column of `W`. The whole batch goes
through in one step: multiplying `X` by `W` computes every example's weighted sums at once. Only
after that does the bias come in, and it is one number per output unit, shared by all rows.

## Hint 2 — formula / approach

Write the shapes down first: `(n, d_in) @ (d_in, d_out)` gives `(n, d_out)`, and the bias `(d_out,)`
adds to every row because of broadcasting. The inner axes must match, so check `X.shape[1]` against
`W.shape[0]` before you multiply. A `ValueError` is the contract for every shape rule, and an error
that numpy raises on its own does not count as the contract. Check the ndim of each input, because a
1-D `X` multiplies without complaint and quietly loses its batch axis.

## Hint 3 — code shape

Convert each input with `np.asarray(..., dtype=np.float64)`. Then check in order: the ndim of
`X`, the ndim of `W`, the ndim of `b`, then the two size checks (`X.shape[1]` against `W.shape[0]`, and
`b.shape[0]` against `W.shape[1]`). Each check raises `ValueError` with a message that says which rule
failed. After validation, use one matrix contraction and one shared bias addition. Keep the result
as a two-dimensional array even for a single example.
