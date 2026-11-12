# Sinusoidal positional encoding table

Implement `sinusoidal_encoding(max_len, d_model)`. It returns the table of position vectors that a
Transformer adds to token embeddings. Row `pos` is the encoding of position `pos`, where positions
start at 0. Each pair of columns holds one sine and one cosine at one frequency. Frequencies fall
geometrically from the lowest (`pos` scaled by 1) to the highest (`pos` scaled by about
1/10000), so that the table can tell nearby and distant positions apart.

For column pair `i` (columns `2i` and `2i+1`) and row `pos`, with `i = 0, 1, ..., d_model/2 - 1`:

$$
\mathrm{PE}[pos, 2i] = \sin\!\left(\frac{pos}{10000^{2i/d_{model}}}\right),\qquad
\mathrm{PE}[pos, 2i+1] = \cos\!\left(\frac{pos}{10000^{2i/d_{model}}}\right).
$$

The sine and cosine of one pair use the same angle. Even columns are sines and odd columns are
cosines, interleaved. They are not two blocks (all sines, then all cosines).

## Contract

- `max_len`: Python `int`, `max_len >= 0`. Number of positions (rows).
- `d_model`: Python `int`. It must be a **positive even** integer. An odd value is rejected, not
  truncated or padded.
- Returns: `torch.float64` tensor of shape `(max_len, d_model)`.
- Raises `ValueError` when: `max_len` is negative or not an integer; `d_model` is zero, negative
  or odd.
- `max_len = 0` returns an empty tensor of shape `(0, d_model)`.
- Creates a new tensor; takes no tensor inputs, so there is nothing to mutate.

## Examples

```python
>>> sinusoidal_encoding(3, 4)
tensor([[0.0000, 1.0000, 0.0000, 1.0000],
        [0.8415, 0.5403, 0.0100, 1.0000],
        [0.9093, -0.4161, 0.0200, 0.9998]], dtype=torch.float64)
# Row 1: sin(1), cos(1), sin(1/100), cos(1/100), since 10000**(2/4) = 100.
```

```python
>>> sinusoidal_encoding(2, 3)
ValueError: d_model must be a positive even integer
```

## Constraints

- Allowed: the `torch` package (for example `torch.arange`, `torch.exp`, `torch.log`,
  `torch.sin`, `torch.cos`, `torch.empty`), and the `math` module.
- Not allowed: `torch.nn` in any form, including `torch.nn.functional` and
  `torch.nn.Embedding`. Those are the test oracles, not the tool. Do not import `numpy` for the
  table either; build it with torch.

## Done looks like

All public tests pass. Then **Submit** runs the hidden tests. They cover: agreement with a
position-by-position reference on seeded random shapes, including `max_len` from 0 up to a few dozen; the even/odd interleave, with sines in even columns and cosines in odd columns; the
frequency base and the `2i/d_model` exponent, checked at the smallest and largest `d_model`; each
sine/cosine pair having unit norm; the first row being `[0, 1, 0, 1, ...]`; the empty table; the
`ValueError` rules (negative length, zero width, odd width); float64 output dtype and shape; and
long tables (a few thousand positions) staying finite and bounded in `[-1, 1]`.

Some hidden tests take their inputs from the real dataset `gsm8k` (see `docs/DATASETS.md`).
