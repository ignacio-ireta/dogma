## Hint 1 — concept

Each session has a moment it starts and a moment it stops being active. Ask how the number of active sessions
changes over time, and what happens to the count at a moment when one session ends and another starts.

## Hint 2 — formula / approach

Turn each session into two events: a start with +1 and an end with -1. Sort all the events by time. Walk through
them with a running total, and keep the largest total you see. The tie rule is the trap: at one moment, apply the
ends before the starts, because a session that ends at t is not active at t. Sorting tuples of (time, change) does
this, because -1 sorts before +1.

## Hint 3 — code shape

```python
for start, end in sessions:
    if start >= end:
        raise ValueError(...)
events = []
for start, end in sessions:
    events.append((..., +1))
    events.append((..., -1))
events.sort()
active = peak = 0
for _, change in events:
    active += change
    peak = max(peak, active)
return peak
```
