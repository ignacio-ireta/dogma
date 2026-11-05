## Hint 1 — concept

Count the tokens a round emits, not only the draft tokens it accepts. The target always adds one token of its own, so the count starts at one. Then divide that count by what the round costs in target passes.

## Hint 2 — formula / approach

Expected tokens per round is the sum of alpha to the power k, for k from 0 to gamma. Use the geometric closed form when alpha is below 1. Treat alpha equal to 1 separately, because the closed form divides by 1 - alpha. The round costs gamma * c + 1 target passes.

## Hint 3 — code shape

Validate the three arguments first, one small helper each, and reject bools before the number checks. Branch on alpha being 1, then on alpha being 0, then use the closed form. For the general case, compute the numerator and the denominator on separate lines. Near alpha = 1 the plain difference loses precision, so look at a function in `math` that evaluates exp(x) - 1 accurately for small x. Return the tuple of tokens and tokens divided by the round cost.
