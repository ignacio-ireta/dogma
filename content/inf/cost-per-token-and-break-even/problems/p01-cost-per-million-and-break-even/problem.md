# Self-hosted cost per million output tokens and the break-even volume

Implement `cost_and_break_even(gpu_hour_price, gpus, tokens_per_second, utilization_percent, api_price_per_million)`.
It compares serving one model replica on your own GPUs with buying the same output tokens from an API. It returns the
self-hosted cost per million output tokens at the given utilization, and the daily output volume at which self-hosting
costs exactly what the API charges.

**The model (use it exactly).**

- Daily cost of the replica, paid every day whether or not it is busy: `F = gpu_hour_price * gpus * 24`, in USD per day.
- Tokens served per day at utilization `u` percent: `S = tokens_per_second * 86400 * u / 100`. Here `tokens_per_second`
  is the sustained output rate of one replica at full load, and `86400` is seconds per day.
- Self-hosted cost per million output tokens: `F * 10**6 / S`, in USD per million.
- Break-even volume: the daily tokens `V` at which self-hosting costs what the API charges, `F * 10**6 / V = api_price_per_million`,
  so `V = F * 10**6 / api_price_per_million` tokens per day. Above `V` self-hosting is cheaper. The break-even volume does
  not depend on utilization; utilization decides whether the replica can reach `V` at all.

**Rounding rules.** Compute each quantity exactly, with no intermediate rounding.

- `self_host_usd_per_million` is rounded to the cent, half up (a value of exactly half a cent goes up), and is a `Decimal`
  with exactly two decimal places.
- `break_even_tokens_per_day` is rounded up to the next whole token (a ceiling), and is an `int`.

**Why not float.** A binary float cannot hold most decimal cents. `1.005` is stored as `1.00499999...`, so a float
route rounds a half cent down, and the answer is wrong by a cent. Work with `Fraction` (exact rationals built from the
`Decimal` inputs with `Fraction(Decimal)`) and round once, at the end, half up. Use `Decimal` for the output only.

## Contract

- `gpu_hour_price`: `decimal.Decimal`, USD per GPU-hour, finite and at least 0. `int`, `float` and `str` are rejected
  with `TypeError`, because money is exact only as `Decimal`.
- `gpus`: Python `int`, at least 1. GPUs billed for one replica.
- `tokens_per_second`: Python `int`, at least 1. Sustained output tokens per second of one replica at full load.
- `utilization_percent`: Python `int` from 1 to 100 inclusive. Share of the day the replica serves at that rate.
- `api_price_per_million`: `decimal.Decimal`, USD per million output tokens, finite and greater than 0.
- Returns a `dict` with exactly the keys `"self_host_usd_per_million"` (a `Decimal` with two places) and
  `"break_even_tokens_per_day"` (an `int`).
- Raises `TypeError` when a value has the wrong type (`bool` is rejected for the integer arguments, and the `Decimal`
  arguments reject `int`, `float` and `str`). Raises `ValueError` when a value is out of range: a price that is not
  finite or is negative, an API price of 0 or less, `gpus` or `tokens_per_second` below 1, or a utilization outside 1 to 100.

## Examples

```python
>>> from decimal import Decimal as D
>>> cost_and_break_even(D("2.50"), 1, 1000, 100, D("0.60"))
{'self_host_usd_per_million': Decimal('0.69'), 'break_even_tokens_per_day': 100000000}
# F = 60 USD/day; 86.4e6 tokens/day at 100 %; 60e6 / 86.4e6 = 0.694... rounds to 0.69
```

```python
>>> cost_and_break_even(D("2.50"), 1, 1000, 50, D("0.60"))["self_host_usd_per_million"]
Decimal('1.39')      # half the utilization doubles the cost per token; the break-even volume is unchanged
```

```python
>>> cost_and_break_even(D("2.50"), 1, 1000, 100, D("0.70"))["break_even_tokens_per_day"]
85714286             # 60e6 / 0.70 = 85,714,285.71..., rounded up to a whole token
```

```python
>>> cost_and_break_even(D("3.00"), 3, 32, 100, D("80.00"))
{'self_host_usd_per_million': Decimal('78.13'), 'break_even_tokens_per_day': 2700000}
# the exact value is 78.125 USD per million: half up gives 78.13; round-half-even gives 78.12
```

```python
>>> cost_and_break_even(D("2.50"), 1, 1000, 100, D("0"))
ValueError           # the API price must be greater than 0
```

## Constraints

- Allowed: built-in types, `decimal` and `fractions` from the standard library.
- Not allowed: `numpy`, and any helper that prices tokens or rounds money for you. Do not round until the end, and do not
  convert money to `float`.

## Done looks like

The public tests check the examples above. **Submit** adds hidden tests. They compare your result with an independent
computation over seeded random prices, shapes and utilizations, to the cent and to the token. They also cover: a value
of exactly half a cent rounding up; a break-even volume that is not a whole number; the break-even volume staying the
same as utilization changes; a free GPU costing 0.00 with a break-even of 0; the cost falling as utilization rises;
very large throughputs and prices staying exact; the exact key set and the types of the values; and each validation rule.
