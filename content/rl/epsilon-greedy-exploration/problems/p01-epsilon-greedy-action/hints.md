## Hint 1 — concept

Epsilon-greedy makes two decisions on every step: whether to explore, and which action to take
once that is settled. Both decisions use the same random source, so the order of the draws matters
as much as the probabilities. The contract fixes that order, and the tests check it.

## Hint 2 — formula / approach

Draw one uniform number first, on every call. If it is below epsilon, draw one integer from the
range of actions and return it. Otherwise return the first index of the largest value. A
first-index maximum is the lowest index among ties, and it needs no further draw.

## Hint 3 — code shape

Validate the arguments before any draw, so an invalid call never advances the generator. Check the
generator type, then the values, then epsilon. Then make the uniform draw into a named variable,
branch on it, and in the exploring branch return a Python int from the integer draw. Keep the
exploiting branch to a single expression that picks the first largest index.
