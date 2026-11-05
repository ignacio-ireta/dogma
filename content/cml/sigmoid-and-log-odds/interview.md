# Interview notes — Sigmoid and log-odds

## Q1. What does logistic regression model, and what does a coefficient mean?

**60-second answer.** Logistic regression models the log-odds of the positive class as a linear function of the features. The probability is the sigmoid of that linear score. A coefficient is the change in log-odds for a one-unit change in its feature, with the other features fixed. Its exponential is the odds ratio, which multiplies the odds, not the probability, so the effect on probability depends on where you start.

**Follow-ups.**
- *Why not a linear probability model?* It can predict values outside $[0, 1]$, and its errors are not well behaved near the boundaries.
- *Is $e^{\beta}$ a risk ratio?* No. It is an odds ratio. The two are close only when the outcome is rare.

## Q2. How would you compute the sigmoid so that it is safe for any logit?

**60-second answer.** Never form $e^{z}$ for large positive $z$, or $e^{-z}$ for large negative $z$. Compute $e = e^{-|z|}$, which lies in $[0, 1]$, then use $1/(1+e)$ for $z \ge 0$ and $e/(1+e)$ for $z < 0$. Both branches use the same $e$, so neither overflows. The small side keeps its relative precision because it is computed directly, not as $1 - \sigma(z)$.

**Follow-ups.**
- *Why does `np.where` still warn?* It evaluates both branches for every element. Each branch has to be safe on its own.
- *What about the derivative?* Use $e/(1+e)^2$ rather than $\sigma(1-\sigma)$. The second loses digits once $\sigma$ rounds to 1.

## Common mistakes

- Writing the sigmoid as `1 / (1 + np.exp(-z))` and accepting the overflow warning as harmless.
- Clipping logits or probabilities to avoid $\log 0$. The clip changes the answer in the tail.
- Reading $e^{\beta}$ as a probability ratio.
- Computing the logit as $\log(p/(1-p))$ and expecting relative precision near $p = 1/2$.
