---
title: Alpha topic
minutes: 10
min_viable: "Read The system, then solve p01."
---

## The system

Doubling maps each value $x$ to $2x$. A list is mapped element by element.

## Worked example

Input $[1, 2.5]$ becomes $[2, 5]$. Each entry is multiplied by $2$.

## Why it works

Multiplication by two distributes over addition, so the map is linear.

```python
## this line is inside a fence and is not a heading
```

## Signal vs. noise

- **Signal:** the input list must stay unchanged.
- **Noise:** floating point rounding for small integers.

## Where it breaks

- Mutating the input list in place surprises every caller that holds it.

## Interview angle

Explain pure functions with this example in under a minute.

## Connections

- **Unlocks:** `tst/beta`.
