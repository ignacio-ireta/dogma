# Interview notes — Smooth activations: tanh, GELU, Swish

## Q1. Compare tanh, GELU and Swish.

**60-second answer.** Tanh squashes into a bounded signed range and saturates in both tails. Exact GELU multiplies x by the normal CDF; beta-1 Swish multiplies x by sigmoid(x). The gates are smooth and retain some negative values, unlike a hard sign gate. I state exact versus approximate GELU because the forward and derivative formulas must match the implementation convention.

**Follow-ups.**

- Can a GELU derivative exceed one? Yes: its product-rule derivative is Phi(x) + x*phi(x).
- Is smoothness a training guarantee? No; architecture, initialization and optimization still matter.

## Q2. How do you implement stable Swish and verify its derivative?

**60-second answer.** I compute sigmoid from exp(-abs(x)) using separate expressions by sign so no exponential overflows in the input domain. For s = sigmoid(x), product rule gives s + x*s*(1-s). I test the zero derivative of 0.5 and compare with central differences of an independently written forward function, including negative inputs so missing terms are visible.

**Follow-ups.**

- What about GELU backward? Differentiate x*Phi(x) into Phi(x) + x*phi(x).
- What happens in the tails? Gates can round to their limiting values; tanh derivatives can round to zero.

## Common mistakes

- Using a tanh GELU approximation when the contract requires exact CDF GELU.
- Dropping the gate-derivative term from the product rule.
- Computing exp(-x) directly across the entire domain.
