## Hint 1: concept

Each output row describes one user, so the query must collapse many order rows into one.

## Hint 2: formula / approach

Group by the user and aggregate the amount with `SUM`.

## Hint 3: code shape

`SELECT user_id, SUM(amount) AS total FROM orders GROUP BY user_id;`
