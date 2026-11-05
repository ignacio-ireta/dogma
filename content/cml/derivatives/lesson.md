---
title: Derivatives
minutes: 12
min_viable: "Read The system and the Worked example, then solve p01."
---

## The system

The derivative of $f:\mathbb{R}\to\mathbb{R}$ at $x$ is the slope of the best linear approximation to $f$ near $x$:

$$f(x+h) = f(x) + f'(x)\,h + o(h), \qquad f'(x) = \lim_{h\to 0}\frac{f(x+h)-f(x)}{h}.$$



Core rules, with $u$ and $v$ differentiable and $c$ a constant:

| Rule | Formula | Example |
|---|---|---|
| Constant multiple | $(cu)' = c\,u'$ | $(3x^2)' = 6x$ |
| Sum | $(u+v)' = u' + v'$ | $(x^2+\sin x)' = 2x + \cos x$ |
| Product | $(uv)' = u'v + uv'$ | $(x\,e^x)' = (1+x)\,e^x$ |
| Quotient | $(u/v)' = (u'v - uv')/v^2$ | $(1/x)' = -1/x^2$ |
| Power | $(x^n)' = n\,x^{n-1}$ | $(x^3)' = 3x^2$ |
| Exp, log, sine | $(e^x)' = e^x$, $(\ln x)' = 1/x$, $(\sin x)' = \cos x$ | |



Without a formula for $f'$, estimate it with a finite step $h > 0$:

- forward: $D_+ f(x) = \dfrac{f(x+h) - f(x)}{h}$
- central: $D_0 f(x) = \dfrac{f(x+h) - f(x-h)}{2h}$

## Worked example

Take $f(x) = x^3$ at $x = 1$, where $f'(1) = 3$.

| $h$ | forward $D_+$ | error | central $D_0$ | error |
|---|---|---|---|---|
| 0.1 | 3.31 | 0.31 | 3.01 | 0.01 |
| 0.05 | 3.1525 | 0.1525 | 3.0025 | 0.0025 |

Halving $h$ halves the forward error and quarters the central error. The arithmetic here is exact, so these are pure truncation errors: $3xh + h^2$ for forward and $h^2$ for central.

Once $f(x)$ is reused, both schemes cost two evaluations.

## Why it works

Taylor's theorem gives $f(x\pm h) = f(x) \pm f'(x)h + \tfrac12 f''(x)h^2 \pm \tfrac16 f'''(x)h^3 + O(h^4)$. Subtracting the two expansions cancels every even power:

$$D_0 f(x) = f'(x) + \frac{h^2}{6}\,f'''(x) + O(h^4).$$

The central error is $O(h^2)$. The forward error is $\tfrac12 f''(x)\,h + O(h^2)$, which is $O(h)$. Both are **truncation errors**. They come from the method and shrink as $h$ shrinks. Burden and Faires' *Numerical Analysis* covers this analysis.

A second error runs the other way: computing $f(x+h)$ carries an absolute error of about $\varepsilon\,|f(x)|$, where $\varepsilon \approx 2.2\times10^{-16}$ is the float64 machine epsilon. Dividing by $h$ turns that into a **round-off error** of about $\varepsilon|f|/h$, which grows as $h$ shrinks. Balancing the two gives the best step:

- forward: $h^\star \sim \varepsilon^{1/2} \approx 1.5\times10^{-8}$, with error about $10^{-8}$;
- central: $h^\star \sim \varepsilon^{1/3} \approx 6.1\times10^{-6}$, with error about $10^{-11}$.

Measured for $f=\sin$ at $x=1$ (exact $f'=\cos 1$):

| $h$ | forward error | central error |
|---|---|---|
| $10^{-2}$ | $4.2\times10^{-3}$ | $9.0\times10^{-6}$ |
| $10^{-4}$ | $4.2\times10^{-5}$ | $9.0\times10^{-10}$ |
| $10^{-6}$ | $4.2\times10^{-7}$ | $2.8\times10^{-11}$ |
| $10^{-8}$ | $1.4\times10^{-8}$ | $3.0\times10^{-9}$ |
| $10^{-10}$ | $5.8\times10^{-8}$ | $5.8\times10^{-8}$ |
| $10^{-12}$ | $4.3\times10^{-5}$ | $1.2\times10^{-5}$ |

Both columns form a U-curve: truncation on the left, round-off on the right. On this grid the central best, $2.8\times10^{-11}$, is about three orders of magnitude below the forward best, $1.4\times10^{-8}$. Individual steps can land luckier, because round-off is not monotone in $h$.

## Signal vs. noise

- **Signal:** $f'(x)$ is a slope, and the difference quotient estimates it. 
- **Signal:** the step size trades truncation against round-off. That trade-off is the whole practical content of numerical differentiation.
- **Noise for now:** the formal $\varepsilon$–$\delta$ definition of a limit, and higher-order Taylor terms.

## Where it breaks

- **Kinks.** $|x|$ and ReLU are not differentiable at 0, yet a finite difference always returns a number. At $x=0$ with $h=10^{-3}$, $|x|$ gives forward $1.0$ and central $0.0$, and ReLU gives forward $1.0$ and central $0.5$. None of these is "the" derivative. Report the one-sided slopes.
- **A fixed step at large $|x|$.** The default step scales with $\max(1,|x|)$. That scaling assumes $f$ varies on the scale of $|x|$. For $f(x)=x^2$ at $x=10^8$, an absolute step of $\varepsilon^{1/3}\approx 6\times10^{-6}$ gives relative error $9\times10^{-4}$, while the scaled step gives $7\times10^{-12}$.
- **A scaled step for an oscillating function.** $\sin$ varies on scale 1 wherever it is evaluated. At $x=10^6$ the default step is $h\approx 6.06$, and the central error is $0.97$, close to the size of the function's whole range. Pass `h` explicitly when you know the function's length scale.
- **Inexact steps.** In floating point $(x+h)-x \neq h$ in general, so the divisor is not exactly the step $f$ was evaluated at. The formulas above ignore this, and its effect is of round-off order.

## Interview angle

*60-second shape:* the derivative is the slope of the best local linear approximation. A difference quotient estimates it. Central differences are $O(h^2)$ accurate and forward differences $O(h)$, at the same cost. The step must balance truncation against round-off, which puts the best central step near $\varepsilon^{1/3}\approx 6\times10^{-6}$ and the error near $10^{-11}$.

*Likely follow-ups:* why not make $h$ tiny (in the table above, central error climbs from $2.8\times10^{-11}$ at $h=10^{-6}$ to $5.8\times10^{-8}$ at $h=10^{-10}$), and why differentiable is not the same as smooth at a kink.

## Connections

- **Prerequisites:** none in this repo. Basic algebra and the idea of a limit are enough.
- **Unlocks:** `cml/partial-derivatives-and-gradients` (the same difference, one coordinate at a time), `cml/chain-rule` (the rule for composed functions), `cml/gradient-descent` (the derivative sets the step direction), and `dl/backprop-through-a-chain-of-scalars`.
- **Lineage:** Newton and Leibniz developed differential calculus independently in the late 17th century.
