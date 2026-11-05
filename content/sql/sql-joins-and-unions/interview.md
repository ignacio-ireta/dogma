# Interview notes: SQL joins and unions

## Q1. Every customer with their order count and total, including customers with no orders. How?

**60-second answer.** I start from the table that defines the answer: customers. The grain is one row
per current customer. I filter to the current version, then `LEFT JOIN` orders onto it. Counting
`COUNT(o.order_id)` gives the number of matched orders, so a customer with none gets 0. The total is
`COALESCE(SUM(o.amount), 0.0)`, because a sum over no rows is NULL. Orders whose customer is not in
the table never appear, because the query never starts from them. An inner join would silently drop
the customers with no orders, and that is the mistake I check for first.

**Follow-ups.**
- *Why not `COUNT(*)`?* After a LEFT JOIN, a customer with no orders still produces one row, with
  NULLs on the right. `COUNT(*)` counts that row and reports 1.
- *What if a dimension has several rows per customer?* The totals double. I filter that dimension to one
  row per key first, or aggregate it before the join.

## Q2. Why did my totals double after I added a join?

**60-second answer.** A join multiplies rows: each left row is repeated once for every match on the
right. If the right side has more than one row per key, such as an old and a current version of a
customer, the left side's rows multiply. I check the grain of every table before I join, and I write
the grain of the answer in one sentence. If the extra table is one-to-many, I reduce it to one row per
key first, with a filter or an aggregate in a subquery. `COUNT(DISTINCT key)` may hide the count
problem, but it does not fix a doubled `SUM`.

**Follow-ups.**
- *How would you find the duplicate?* Count rows per key on the many side. Any count above 1 is a fan-out
  risk.
- *What about a filter on the right table?* In a LEFT JOIN, a filter on the right table belongs in ON.
  In WHERE, it turns the join into an inner join.

## Q3. When do you use UNION and when UNION ALL?

**60-second answer.** `UNION` stacks two result sets and removes duplicates, across both and within each.
`UNION ALL` keeps every row and skips the removal step, so it is cheaper. I use `UNION` when the
question asks for distinct things, and `UNION ALL` when the sources cannot overlap or when duplicates
are the point. Columns are matched by position, so I check the order and the types of both branches.
A NULL is treated as a value for removal, so `UNION` keeps one NULL row.

**Follow-ups.**
- *Can I sort a union?* Yes, with one `ORDER BY` after the last branch. An unparenthesised `ORDER BY` inside a
  branch is a syntax error in DuckDB.
- *Is a join the same as a union?* No. A join adds columns for matching keys. A union adds rows for the
  same columns.

## Common mistakes
- Starting from the orders table, so customers with no orders never appear.
- `COUNT(*)` after a LEFT JOIN, which counts the NULL-extended row as an order.
- A right-table filter in WHERE that quietly turns a LEFT JOIN into an inner join.
- Joining a one-to-many table before aggregating it, so sums double.
- `NOT IN` against a subquery that can return NULL, which returns no rows.
- `UNION ALL` where `UNION` was meant, so a code appears once per source.
