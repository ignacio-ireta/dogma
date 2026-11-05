## Hint 1 — concept

The time and the type of an event belong to one row, so every method has to keep them together. Also decide what breaks
a tie in time before you write the query: the first event uses the smallest event_id, and the last uses the largest.

## Hint 2 — formula / approach

Group by user. For the first event, take the minimum time and the type of the row with the smallest (time, event_id).
For the last event, take the maximum time and the type of the row with the largest (time, event_id). An arg function with
a key that packs the time and the id gives each type from the same row as its time.

## Hint 3 — code shape

One grouped select with two arg functions on a struct key is enough. Put the tie-breaker in the key, and state the final order:

```sql
SELECT user_id,
    min(event_time) AS first_event_time,
    arg_min(event_type, struct_pack(t := event_time, id := event_id)) AS first_event_type,
    ...
FROM events GROUP BY user_id ORDER BY user_id;
```

Complete the last two columns yourself, and check the direction of the tie-breaker for the last event.
