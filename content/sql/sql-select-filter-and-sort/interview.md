# Interview notes: SQL select, filter and sort

## Q1. Walk me through how a SQL engine evaluates a SELECT.

**60-second answer.** The engine reads FROM first, keeps rows with WHERE, computes the SELECT
list, sorts with ORDER BY, and cuts with LIMIT. So WHERE cannot see an alias that SELECT defines,
and LIMIT only sees the rows ORDER BY has already placed. The logical order is also why you
filter early: fewer rows reach the sort. The engine may run the steps differently physically,
but the result must match the logical order.

**Follow-ups.**
- *Can WHERE use a SELECT alias?* Not in standard SQL. Some engines allow it as an extension.
- *Why filter in WHERE rather than after the sort?* Less data to sort, and the answer is the same.

## Q2. What does NULL do to a filter?

**60-second answer.** A comparison with NULL is UNKNOWN, not FALSE. WHERE keeps only TRUE, so a
row with a NULL in the compared column is dropped by `amount > 100` and also by
`NOT (amount > 100)`. To find missing values you test `IS NULL`. The practical trap is
`NOT IN` with a NULL in the list, which returns no rows at all.

**Follow-ups.**
- *Why does `x = NULL` return nothing?* It is UNKNOWN for every x, including NULL.
- *What does `COALESCE` do here?* It replaces NULL with a value you choose. Use it only when
  the substitute is correct for the business question.

## Q3. How do you get the top N rows deterministically?

**60-second answer.** Order by the metric descending and add a unique key ascending, so ties at
the cut are resolved the same way every time. Then apply LIMIT. Filter first, so the sort works
on the rows that qualify. Say out loud which key breaks the ties and why it is a sensible choice.

**Follow-ups.**
- *What if the metric is NULL for some rows?* Decide whether they qualify. In DuckDB they sort last
  by default, but the query should not depend on that.
- *Top N within each group?* That needs a window function or a correlated subquery. It is a later
  topic.

## Common mistakes
- `LIMIT` without `ORDER BY`, which returns an arbitrary set.
- `WHERE status <> 'x'` where a status should have been named exactly.
- `BETWEEN` on a timestamp column with an end date that is midnight.
- Testing NULL with `= NULL`.
- Sorting by a metric without a tie-breaker, then reporting one of the tied rows as "the" answer.
