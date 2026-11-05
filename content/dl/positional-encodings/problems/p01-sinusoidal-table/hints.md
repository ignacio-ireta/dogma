## Hint 1 — concept

Each row is a position and each pair of columns is one frequency. Within a pair, the sine and the
cosine share one angle. The frequencies get lower as the column pair index grows, and the lowest
frequency is the one with angle equal to the position itself.

## Hint 2 — formula / approach

Write the angle for column pair `i` as `pos / 10000 ** (2*i / d_model)`. Build the matrix of
angles with shape `(max_len, d_model // 2)` by broadcasting a column of positions against a row of
exponents, then fill the even columns with sines and the odd columns with cosines. Check the
ValueError rules before you build anything: negative `max_len`, and `d_model` that is zero, negative
or odd.

## Hint 3 — code shape

Make `pos` with `torch.arange(max_len, dtype=torch.float64).unsqueeze(1)` and the exponent row with
`torch.arange(0, d_model, 2, dtype=torch.float64)`. Compute `angles = pos * rate` with the rate
`10000 ** (-two_i / d_model)`. Allocate `pe` with `torch.empty(max_len, d_model, dtype=torch.float64)`
and assign `pe[:, 0::2]` and `pe[:, 1::2]` from `torch.sin` and `torch.cos` of `angles`.
