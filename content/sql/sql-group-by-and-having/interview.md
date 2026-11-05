# Interview notes: SQL GROUP BY and HAVING

## Q1. When does a condition go in WHERE and when in HAVING?

**60-second answer.** Ask what the condition is about. A condition on a single row's own values,
such as status or date, goes in WHERE, because it removes rows before grouping and so changes
what the aggregates see. A condition on an aggregate, such as `COUNT(*) >= 3` or
`SUM(amount) > 500`, goes in HAVING, because aggregates only exist after grouping. Putting a row
condition in HAVING either fails or filters the wrong rows, and it is the mistake I look for first.

**Follow-ups.**
- *Can HAVING use a column that is not grouped?* Only inside an aggregate.
- *Is WHERE faster?* Usually. It shrinks the input before the grouping step.

## Q2. How do COUNT, SUM and AVG treat NULLs?

**60-second answer.** `COUNT(*)` counts rows. `COUNT(col)` counts non-NULL values. `SUM` and
`AVG` skip NULLs. So AVG divides by the non-NULL count, not by the row count. A group with only
NULL values gives a NULL sum, not zero. The right answer depends on what the question means, so I
state the choice explicitly rather than rely on the default.

**Follow-ups.**
- *Why does a NULL sum matter?* A total of NULL that is shown as 0 looks like a real zero.
- *What about COUNT(DISTINCT)?* It counts distinct non-NULL values.

## Q3. Customers with at least N events in the last 30 days: how do you write it?

**60-second answer.** Filter rows with WHERE on the date window and any other row conditions.
Group by the customer key. Keep groups with HAVING on the count. Check the grain first: if the
events join to another table, use `COUNT(DISTINCT event_id)` or aggregate before the join, so the
count is not inflated. Sort the result with a tie-breaker so the output is repeatable.

**Follow-ups.**
- *What if the window changes per customer?* Then the window moves into a join or a window
  function, a later topic.
- *How would you test it?* Build a small table with a customer at exactly the threshold, one just
  below it, and one with NULLs.

## Common mistakes
- A row condition in HAVING, or an aggregate condition in WHERE.
- `COUNT(amount)` where `COUNT(*)` was meant, which drops orders with unknown amounts.
- Expecting `SUM` to return 0 for an empty or all-NULL group.
- Joining before grouping and then reporting inflated counts.
- Returning grouped results without an ORDER BY, so the order is arbitrary.
