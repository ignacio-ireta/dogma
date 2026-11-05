## Hint 1 — concept

Each query or key row is a set of pairs. In the half-split layout the two members of a pair are
half a row apart. A pair is rotated by an angle that grows with the token's absolute position, and
the position starts at `offset`, not at zero. Values are not rotated.

## Hint 2 — formula / approach

Let `a = x[..., :d/2]` and `b = x[..., d/2:]`. For token `t` and pair `i`, use the angle
$\phi = (\mathrm{offset}+t)\,\theta_i$ with $\theta_i = 10000^{-2i/d}$. The rotated pair is
$(a\cos\phi - b\sin\phi,\ a\sin\phi + b\cos\phi)$. Compute `cos` and `sin` once, as tables of shape
$(T, d/2)$, and reuse them for q and k.

## Hint 3 — code shape

Build `positions = torch.arange(offset, offset + T)` and `theta` with `d/2` entries. Form the
`(T, d/2)` angle table as an outer product, and take its `cos` and `sin`. Split each tensor with
slices on the last axis, apply the two output formulas, and join the halves with `torch.cat`. Return
the rotated q and k as a tuple, and validate the inputs before any arithmetic.
