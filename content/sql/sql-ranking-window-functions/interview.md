# Interview notes: SQL ranking window functions

## Q1. Return products at the top two distinct revenue values per category, keeping ties.

**60-second answer.** I compute `DENSE_RANK() OVER (PARTITION BY category ORDER BY revenue DESC NULLS LAST)` in a
subquery, because a window cannot be filtered in the same `WHERE`. In the outer query I keep the rows whose rank is at
most 2. I use `DENSE_RANK` because the question asks for the top two values, so two products tied for second both come
back. I state where NULLs sort, and I order the output by category, rank and id, so that the answer is repeatable.

**Follow-ups.**
- *Why not `ROW_NUMBER`?* It gives each row a distinct number, so it would drop one of two tied products at the cut.
- *What if I want exactly two rows per category?* Use `ROW_NUMBER` with an explicit tie-breaker, and say that ties are cut arbitrarily unless the tie-breaker decides them.

## Q2. How do you get the latest order per customer?

**60-second answer.** `ROW_NUMBER() OVER (PARTITION BY customer_id ORDER BY order_date DESC, order_id DESC)` in a subquery,
then keep the rows numbered 1. The second key, the order id, makes the choice unique when two orders share a date.
Without it, the result can change from one run to the next.

**Follow-ups.**
- *Why is the tie-breaker needed?* `ROW_NUMBER` must give each row a unique number, or the "latest" row is not defined.
- *What if the latest row must include ties?* Then use `RANK` or `DENSE_RANK` with the date only, and say that the answer may return several rows.

## Common mistakes

- Writing `WHERE DENSE_RANK() OVER (...) <= 2`, which the engine rejects because windows are evaluated after `WHERE`.
- Using `ROW_NUMBER` for "top N with ties", which drops tied rows at the cut.
- Adding a tie-breaker to `RANK`'s `ORDER BY`, which removes the ties it was meant to keep.
- Leaving the NULL position implicit, and getting a different answer on another engine.
