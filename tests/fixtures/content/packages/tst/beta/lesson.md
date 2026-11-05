---
title: Beta topic
minutes: 8
min_viable: "Read The system, then solve p01."
---

## The system

`GROUP BY key` partitions rows by `key`. Each aggregate, such as `SUM(amount)`, runs once per
partition and returns one row per key.

## Worked example

Rows `(1, 2.5)`, `(1, 1.5)`, `(2, 4.0)` grouped by user give `(1, 4.0)` and `(2, 4.0)`.

## Why it works

Every row lands in exactly one group, so the totals add up to the grand total.

## Signal vs. noise

- **Signal:** every non-aggregated column must be in the `GROUP BY`.
- **Noise:** the order of output rows, unless the question asks for one.

## Where it breaks

- A `NULL` key forms its own group, which surprises people who expect it to be dropped.

## Interview angle

Explain the difference between `WHERE` (filters rows) and `HAVING` (filters groups).

## Connections

- **Requires:** `tst/alpha`.
