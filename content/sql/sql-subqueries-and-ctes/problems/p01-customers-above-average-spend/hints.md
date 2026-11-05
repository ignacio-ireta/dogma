## Hint 1 — concept

Work in two stages. First compute one total per buyer. Then compare each total with a single
number that summarises all the totals. The comparison can only see the summary if you name it, and
a common table expression (CTE) or a subquery is how you name it.

## Hint 2 — formula / approach

Stage one: `SUM(amount)` per customer, over completed orders that have a customer. The sum of
all-NULL amounts is NULL, and that is the right answer for that buyer. Stage two: `AVG` over those
totals, which skips the NULL totals. Keep the buyers whose total is strictly greater than the
average, join `customers` for the region, and sort.

## Hint 3 — code shape

```sql
WITH buyer_totals AS (
    SELECT ...            -- one row per buyer: the completed orders, grouped
),
average_total AS (
    SELECT ...            -- the one-row average of the buyer totals
)
SELECT ...
FROM buyer_totals ...
CROSS JOIN average_total ...
WHERE ...                 -- strictly above the average
ORDER BY ...;
```
