# Smallest truck capacity for a delivery window

A depot sends parcels in a fixed order. Each day one truck leaves and carries a run of consecutive
parcels, and the total weight of that run must not exceed the truck's capacity. Implement
`min_ship_capacity(weights, days)`. It returns the smallest capacity that lets every parcel leave
within `days` days. Search over the capacity with a binary search. Do not try every capacity.

## Contract

- `weights`: a non-empty `list` of `int`s, each greater than 0, in shipping order.
- `days`: an `int` of at least 1.
- Returns: an `int` capacity `C` such that the parcels can be cut into at most `days` consecutive
  runs, each with total weight at most `C`, and no smaller capacity does. `C` is at least the
  heaviest parcel and at most the total weight.
- Raises: `ValueError` when `weights` is empty, when any weight is not an `int` greater than 0, or
  when `days` is not an `int` of at least 1.
- Must not mutate `weights`.

## Examples

```python
>>> min_ship_capacity([4, 2, 7, 3, 5, 1, 6], 3)
12
```

Capacity 11 needs four days: `[4, 2]`, `[7, 3]`, `[5, 1]`, `[6]`. Capacity 12 needs three:
`[4, 2]`, `[7, 3]`, `[5, 1, 6]`.

```python
>>> min_ship_capacity([9, 4, 6], 1)    # one day: one truck carries everything
19
>>> min_ship_capacity([9, 4, 6], 3)    # one parcel per day: the heaviest parcel is enough
9
>>> min_ship_capacity([5, 5, 5, 5], 2)
10
```

## Constraints

- Allowed: built-in types, `typing`, `collections.abc`, and the standard library otherwise.
- Not allowed: a scan that tries every capacity from the heaviest parcel up to the total weight.
  The search must run in a logarithmic number of passes over the parcels.

## Done looks like

All public tests pass, then **Submit** runs the hidden tests. They compare your answer with a brute
force over every capacity on small random inputs, and check that the answer is the smallest feasible
capacity on larger inputs. The edge families are: one day, a number of days at least the number of
parcels, the lower end of the range (the heaviest parcel must fit), a day count that is not positive,
an empty list, a weight that is zero or not an integer, a `days` value that is not a positive integer,
inputs that must not change, and a large list where a scan over every capacity reads far too many
parcels.
