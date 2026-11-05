# Interview notes: Derivatives

## Q1. What is a derivative, and how would you estimate one numerically?

**60-second answer.** The derivative $f'(x)$ is the slope of the best linear approximation to $f$ at $x$. It is the limit of the difference quotient $\frac{f(x+h)-f(x)}{h}$ as $h$ goes to zero. Numerically, you evaluate the quotient at a small finite step. Because $f$ is only available as code, that estimate is all you get, and you pick the step to balance truncation error against round-off error.

**Follow-ups.**
- *Why does a finite step help at all?* It turns a limit into one evaluation. Its error comes from curvature (truncation) and from floating-point subtraction (round-off).
- *What if $f$ is not differentiable at $x$?* The estimate still returns a number, and that number depends on the scheme. Check for kinks before trusting it.

## Q2. Forward or central differences, and how do you pick the step?

**60-second answer.** Central differences are the default. They cancel the first-order term by symmetry, so the error is $O(h^2)$, against $O(h)$ for forward. They cost the same number of evaluations once $f(x)$ is reused. For float64, the best central step is about $\varepsilon^{1/3}\approx 6\times10^{-6}$ when $|x|\le 1$, scaled by $\max(1,|x|)$ otherwise. The error there is about $10^{-11}$. Smaller steps make round-off dominate, so the error grows again.

**Follow-ups.**
- *Why not use a very small step to be safe?* Round-off error grows like $\varepsilon|f|/h$. For $\sin$ at $x=1$, the central error climbs from about $3\times10^{-11}$ at $h=10^{-6}$ to about $6\times10^{-8}$ at $h=10^{-10}$.
- *Where does a fixed step fail?* At large $|x|$, a fixed absolute step is too small relative to $x$ and round-off wins. For an oscillating function at large $|x|$, a scaled step can be too large. Pass the step explicitly when you know the function's length scale.

## Common mistakes

- Using the forward difference by default, and then being surprised that gradient checks fail at the $10^{-8}$ level.
- Choosing a step by feel, such as `1e-12`, without checking the U-curve.
- Dividing the central quotient by $h$ instead of $2h$. The result is off by a factor of two.
- Trusting a derivative estimate at a kink, such as $|x|$ or ReLU at zero.
