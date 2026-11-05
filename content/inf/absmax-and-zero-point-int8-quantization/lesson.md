---
title: "Absmax and zero-point int8 quantization"
minutes: 12
min_viable: "Read The system and Worked example, then solve p01 with the scale written out on paper first."
---

## The system

Quantization stores a real tensor $x$ as integer codes $q$ plus a scale $s > 0$. Dequantization
rebuilds $\hat{x} = s \cdot q$. Two schemes cover most cases:

| | Symmetric absmax | Asymmetric zero point |
|---|---|---|
| Code range | $[-127, 127]$ | $[-128, 127]$ |
| Scale | $s = \max\lvert x\rvert / 127$ | $s = (hi - lo)/255$, with $lo = \min(\min x, 0)$ and $hi = \max(\max x, 0)$ |
| Zero point | none; $0.0$ has code $0$ | $z = -128 - \operatorname{round}(lo/s)$, the code of $0.0$ |
| Code | $q = \operatorname{round}(x/s)$ | $q = \operatorname{round}(x/s) + z$ |
| Dequantize | $\hat{x} = s\,q$ | $\hat{x} = s\,(q - z)$ |

Codes are clipped into their range after rounding. The symmetric grid leaves $-128$ unused, so the
scale is the same for both signs. For a normal scale the absmax clip never changes a code; the
zero-point clip does, at the top edge (see Why it works).

Four rules set the details:

1. **Granularity.** Per-tensor uses one $s$. Per-channel uses one $s_c$ per index along a chosen
   axis, with the absmax taken over the other axes.
2. **Zero groups.** An all-zero group gets $s = 1.0$. A zero scale would divide by zero.
3. **Rounding.** Round half to even: $0.5 \to 0$, $1.5 \to 2$, $2.5 \to 2$. NumPy's `round` does this.
   Half-away-from-zero sends $2.5 \to 3$. State the rule you use.
4. **Asymmetric range contains zero.** Then $0.0$ is exactly representable, which matters for padding
   and ReLU outputs.

## Worked example

**Symmetric, per tensor.** Take $x = [127,\ 0.5,\ 1.5,\ 2.5,\ -2.5]$. The absmax is $127$, so
$s = 127/127 = 1.0$. The codes are $[127,\ 0,\ 2,\ 2,\ -2]$. Half-away rounding would give
$[127, 1, 2, 3, -3]$.

**Per channel, with an outlier.** Let $X = \begin{bmatrix} 127 & 1 & -2 \\ 0.5 & -0.3 & 0.1 \end{bmatrix}$,
with channels on rows (axis $0$).

- Row 0 has absmax $127$, so $s_0 = 1.0$ and codes $[127,\ 1,\ -2]$.
- Row 1 has absmax $0.5$, so $s_1 = 0.5/127 \approx 0.003937$ and codes $[127,\ -76,\ 25]$.
- One per-tensor scale would be $1.0$, and row 1 would become $[0, 0, 0]$.

Dequantized row 1 is $[0.5,\ -0.2992,\ 0.0984]$, and each error is at most $s_1/2 \approx 0.00197$.

**Asymmetric.** Take $x = [-100,\ 0,\ 155]$. Then $lo = -100$, $hi = 155$, and $s = 255/255 = 1.0$.
The zero point is $z = -128 - \operatorname{round}(-100) = -28$. The codes are $[-128,\ -28,\ 127]$,
and code $-28$ dequantizes to $1.0 \cdot (-28 + 28) = 0.0$ exactly.

## Why it works

Rounding $x/s$ to an integer moves each element by at most half a code, so in real units

$$
\lvert x - s\,q \rvert \le \frac{s}{2}.
$$

This holds because the absmax choice keeps $\lvert x/s \rvert \le 127$, so nothing is clipped. For the
asymmetric scheme, $hi - lo = 255 s$ makes the 256 codes cover every $x$ in $[lo, hi]$, but the rounding
grid is shifted by $z$. When $lo/s$ is a half-integer, $hi/s$ rounds one step past the top: $lo = -253.5$
and $hi = 1.5$ with $s = 1$ give $z = 126$, and $hi$ rounds to code $128$. The clip brings it back to $127$,
and the error stays at $s/2$.

The bound depends only on $s$, so the design question is how small $s$ can be. For a group with absmax
$A$, $s = A/127$ and the error is at most $A/254$. One large value raises $A$ for everything in its
group. Per-channel scales shrink the group to one channel, so channel $c$ gets error at most
$A_c/254$. The formula is unchanged; the group is smaller.

## Signal vs. noise

- **Signal:** the scale formulas, the code ranges, the rounding rule, and the bound $s/2$.
- **Signal:** the granularity choice. It decides what an outlier costs.
- **Noise for now:** scale storage precision, calibration such as percentile clipping, and
  quantization-aware training. They refine $s$; they do not change the rule.

## Where it breaks

- **Outliers.** One value at $100$ among values near $1$ sets $s \approx 0.79$, so most values land on
  a handful of codes. Per-channel scales help when outliers sit in a few channels. An outlier inside
  one channel still costs that channel its resolution.
- **Zero-scale groups.** $s = 0$ divides by zero. Use $s = 1.0$.
- **Subnormal magnitudes.** If $A/127$ underflows (for example $A = 5 \times 10^{-324}$), the scale is 0 or
  has too few bits. The problem raises `ValueError` rather than return a zero scale.
- **Code $-128$ in the symmetric grid.** It has no mirror, so the sign symmetry breaks.
- **Asymmetric range without zero.** An all-positive tensor with $lo > 0$ puts $0.0$ off the grid.
- **Integer overflow.** Adding $z$ in int8 arithmetic wraps $127 + 1$ to $-128$. Add in a wider type,
  then clip.
- **Wrong dequantization.** For an asymmetric code, $\hat{x} = s\,q$ is wrong. Use $s\,(q - z)$.

## Interview angle

**60-second answer.** For int8 weights I use symmetric per-output-channel scales, $s_c = A_c/127$, with
round half to even. Each element is within $s_c/2$ of its value, and zero stays exact. After a ReLU the
values are one-sided, so an asymmetric zero point covers $[0, hi]$ with all 256 codes. The failure mode
is an outlier: it inflates the scale of its whole group, and finer groups are the usual fix.

**Follow-ups.** *Why 127 and not 128?* The grid is symmetric, so $-128$ has no mirror. *Why not clip
the outliers?* Clipping trades error on the outlier for error everywhere else; that is a calibration
choice.

## Connections

This builds on [inf/floating-point-formats](../floating-point-formats/): an int8 code is a point on a
uniform grid, while a float's grid spreads out by exponent. The byte saving is what
[KV-cache size arithmetic](../kv-cache-size-arithmetic/) counts. The affine scheme is from
[paper: Jacob et al. 2018, arXiv:1712.05877].
