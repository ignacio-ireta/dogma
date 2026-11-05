## Hint 1 — concept

The naive `log(p / (1 - p))` is fine when the answer is far from zero, but it fails near $p = \tfrac12$,
where the answer is tiny and the ratio $p/(1-p)$ is close to $1$. Ask which quantity you can compute
without subtracting two nearly equal numbers, and which $\log$ of a value near $1$ should be written with
`log1p`.

## Hint 2 — formula / approach

Work with $m = \min(p, 1-p)$, which lies in $(0, \tfrac12]$, and use $\operatorname{logit}(p) = \pm g(m)$
with $g(m) = \log\dfrac{1-m}{m}$. Write $g$ as $\log(1 - m) - \log(m)$ for very small $m$. Near
$m = \tfrac12$, write $g(m) = \operatorname{log1p}\!\left(\dfrac{1-2m}{m}\right)$. The subtraction $1 - 2m$ is
exact for $m$ in $[\tfrac14, \tfrac12]$.

## Hint 3 — code shape

Validate first: `np.asarray(p, dtype=np.float64)`, then reject any entry that is not finite or is outside
$(0, 1)$ with a `ValueError`. Compute `m = np.where(p < 0.5, p, 1 - p)`. Fill two arrays with boolean masks,
one for `m < 0.25` and one for the rest, and finish with `np.where(p < 0.5, -g, g)`. Avoid evaluating a
form on elements it is not meant for, because a subnormal `m` overflows `1 / m`.
