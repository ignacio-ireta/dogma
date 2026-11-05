## Hint 1 — concept

A replica is paid for every hour of the day, so its cost per token is a fixed daily bill spread over the tokens it
serves. Utilization sits in the denominator. The break-even volume is the point where the same fixed bill equals the
API bill for that many tokens.

## Hint 2 — formula / approach

Work in exact rational arithmetic and use `Decimal` only for inputs. The daily cost is the hourly price times the GPUs
times 24. Tokens per day are the tokens per second times 86,400 times the utilization over 100. The cost per million is the
daily cost times 10 to the sixth, over the tokens per day. The break-even volume is the daily cost times 10 to the sixth,
over the API price. Round the cost to the cent, half up, and round the volume up to a whole token.

## Hint 3 — code shape

Validate first: the `Decimal` prices must be `Decimal` instances, finite, with the API price above zero, and the
counts must be Python `int`. Then compute, in this order:

- The daily cost as an exact fraction.
- The tokens per day as an exact fraction.
- The cost per million, and round it to cents with an integer floor of the value in cents plus one half.
- The break-even volume, and round it up with integer arithmetic.

Return a `dict` with the two keys. Format the cent value as a `Decimal` with two places.
