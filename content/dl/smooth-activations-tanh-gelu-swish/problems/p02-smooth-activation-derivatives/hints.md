## Hint 1 — concept

Treat GELU and Swish as products. Both factors depend on x.

## Hint 2 — formula / approach

The derivative of the normal CDF is the normal density; the sigmoid derivative is s*(1-s).

## Hint 3 — code shape

Reuse stored activation factors for each formula. Check the output at zero before testing random inputs.
