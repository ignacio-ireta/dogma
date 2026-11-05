# Interview notes: SQL subqueries and CTEs

## Q1. Which customers spend more than the average customer?

**60-second answer.** First fix the population: buyers, meaning customers with at least one completed
order, and say what happens to NULL amounts. Then stage the problem. A CTE computes one total per
buyer, a second CTE computes one average over those totals, and the final query keeps the buyers
whose total is strictly greater than that average. Each stage can be checked on its own, and the
average is taken over customers, not over individual orders.

**Follow-ups.**
- *What if a customer has only NULL amounts?* Its total is NULL. `AVG` skips it, so it is neither in
  the average nor returned.
- *Why strictly greater?* The question says "above". A customer exactly at the average is not above it.

## Q2. Why does my NOT IN return no rows?

**60-second answer.** The subquery contains a NULL. Every comparison with that NULL is NULL, so `WHERE`
keeps no row. The fix is `NOT EXISTS`, which asks whether a matching row exists and is not affected by
a NULL in the subquery. If you do want `NOT IN`, filter the NULLs out of the subquery first.

**Follow-ups.**
- *Is NOT EXISTS always the same as NOT IN?* No. With a NULL on the outer side, `NOT EXISTS` keeps the
  row, because no match exists. Check the intent for that row.
- *Does IN have the same problem?* Not in the same way. A NULL in the list still lets matches through.

## Q3. How do you compare each order with its own customer's average?

**60-second answer.** Use a correlated subquery in the select list, or a window function that keeps
every row: `AVG(amount) OVER (PARTITION BY customer_id)`. A window function cannot appear in `WHERE`,
so filter on its result in an outer query or a CTE. Both forms give the same rows. The correlated
form reads closer to the question, and the window form is shorter when you need the group's value
for other columns as well.

**Follow-ups.**
- *Which one is faster?* It depends on the engine and the plan. Judge the result first, and check the
  plan with `EXPLAIN` only when speed is a problem.
- *What must the average include?* That customer's completed orders with a known amount. Pending,
  refunded and NULL-amount orders are left out.

## Common mistakes
- Using `NOT IN` on a column that can be NULL, and getting an empty result without an error.
- Averaging order amounts when the question asks about customers, or counting customers with no orders
  as 0.
- Writing `>=` where the question says "above".
- Passing a subquery that can return several rows where a single value is expected.
- Putting a window function in `WHERE`.
