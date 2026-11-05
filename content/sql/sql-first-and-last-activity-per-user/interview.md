# Interview notes — SQL first and last activity per user

## Q1. For each user, what was the first and last event, and what type was each?
**60-second answer.** I would start with the tie rule. Events can share a timestamp, so the output needs a second key,
such as event id, to be deterministic. Then I would group by user and use two selections of the same row: the
earliest and the latest by time and id. In DuckDB, `arg_min` and `arg_max` with a `struct_pack` key do this in one
pass. Each user gets one row, and the type is from the same event as the time.
**Follow-ups.** Why not `MIN` and `MAX` plus a join? The join matches every tied event, so one user can return several
rows. Why not `LAST_VALUE`? With an `ORDER BY`, its default frame stops at the current row's peers, so it can return the
wrong event. An explicit `ROWS` frame or a reversed order fixes it.

## Q2. How do you get the latest row per group when there can be ties?
**60-second answer.** Assign a row number that orders by time descending and then by a unique key, and keep row 1. The
unique key is the difference between a stable answer and one that changes between runs. If the group has several
rows to keep, the same ordering with a row number limit works for top-N.
**Follow-ups.** What if the timestamp is NULL? Say where NULL sorts, and state it. Does `RANK` work? It keeps ties, which
can return more than one row per group, so it answers a different question.

## Common mistakes
- Joining back on a `MIN` or `MAX` timestamp, which duplicates tied rows.
- Leaving the tie-breaker out, so the chosen event can change between runs.
- Using `LAST_VALUE` with the default frame and expecting the partition's last event.
- Using `event_id` as a clock when ids and times disagree.
