---
title: "SQL first and last activity per user"
minutes: 12
min_viable: "Read The system and Worked example, then solve p01 with the struct tie-break written out first."
---

## The system

**Tie rules first.** When events share a timestamp, the first event is the one with the smallest `event_id`, and the last event is the one with the largest. The grader compares the result rows only, not the query text.

For each user you want two rows of information from one table: the time of the first event and the type of
that event, and the time of the last event and its type. Before you write any SQL, fix three things:

1. **Group key.** One output row per `user_id`.
2. **Order.** "First" means ascending `event_time`. "Last" means descending `event_time`.
3. **Tie rule.** Several events can share a timestamp. Name a deterministic second key, here `event_id`:
   the smallest id is first, the largest id is last.

The timestamp and the type belong to the same row, so every method must keep them together. Three do:

| Method | How it keeps the row together | Watch for |
|---|---|---|
| Rank join | `ROW_NUMBER()` per direction, then join where rank = 1 | the join is two scans; keep both sides filtered |
| `arg_min` / `arg_max` | returns the argument from the row with the extreme key | the tie-break must be inside the key |
| Window with an explicit frame | `FIRST_VALUE` / `LAST_VALUE` on every row, then keep one row | the default frame, see below |

## Worked example

| event_id | user_id | event_time | event_type |
|---|---|---|---|
| 1 | 1 | 2026-01-01 10:00 | login |
| 2 | 1 | 2026-01-01 10:00 | view |
| 3 | 1 | 2026-01-02 09:00 | logout |
| 4 | 1 | 2026-01-02 09:00 | purchase |
| 5 | 2 | 2026-01-03 08:00 | view |

For user 1, the earliest time is 10:00 and two events share it. The smallest id, 1, gives `login`. The latest time
is 09:00 on 2 January, shared by ids 3 and 4. The largest id, 4, gives `purchase`. The result is one row:
`(1, 10:00, login, 09:00 on 2 Jan, purchase)`. User 2 has a single event, which is both first and last.

**The trap in MIN and MAX.** `MIN(event_time)` gives 10:00, but not the type. Joining the table back on that time
returns both `login` and `view`, so the output has two rows for user 1. A join back on timestamp is not one row
per user.

## Why it works

`ROW_NUMBER` with `ORDER BY event_time, event_id` numbers each user's events from the first, so row 1 is the first
event. Ordered descending, row 1 is the last.

`arg_min(x, key)` returns `x` from the row with the smallest `key`. `arg_max` does the same for the largest.
When two rows tie on `key`, DuckDB does not promise which one it returns. Make the key unique by packing the
tie-breaker into it: `struct_pack(t := event_time, id := event_id)`. DuckDB compares structs field by field, so the
key order is time first, then id.  The time of the chosen row is the same `MIN` or `MAX` of `event_time`, because time comes first in the key.

**The LAST_VALUE trap.** When a window has an `ORDER BY`, the default frame is
`RANGE BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW`. The frame ends at the current row and its peers, which share the
same ordering value. Using `LAST_VALUE(event_type) OVER (PARTITION BY user_id ORDER BY event_time)` on the worked
example returns `view` for rows 1 and 2, because they are peers at 10:00. The user's last event is `purchase`.
The fix is an explicit frame, `ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING`, or ordering descending and
taking `FIRST_VALUE`.

## Signal vs. noise

- **Signal:** the tie rule. It decides the answer whenever two events share a timestamp, which happens often in logs.
- **Signal:** `MIN` and `MAX` give values, not records. Use an `arg` function or a rank to get the record.
- **Noise:** the exact spelling of `struct_pack` or the window syntax. The result is what gets graded.

## Where it breaks

- **Join back on timestamp.** Duplicates rows on ties and returns mixed records.
- **Tie-break left out.** A nondeterministic choice among tied events can change between runs.
- **`LAST_VALUE` with default frame.** Returns the last peer at the current time, not the partition's last event.
- **`event_id` as a clock.** A larger id is not necessarily a later time. Order by time first.
- **Forgetting users with one event.** The first and last rows are the same row, and that must still appear once.

## Interview angle

**60-second answer.** I would ask how ties are broken and whether a missing timestamp is possible, because those
choices define the output. Then I would group by user and use `arg_min` and `arg_max` with a struct key that
contains the timestamp and the id, or two rank windows with a join. Each gives one row per user, and the tie rule is
explicit. I would avoid joining back on a `MIN` or `MAX` timestamp, and I would not rely on `LAST_VALUE` without an
explicit frame.


## Connections

This topic needs ranking windows and frames. It leads to top-N per group, where the same tie-break rule applies.
