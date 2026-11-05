-- One row per user: GROUP BY collapses the order rows, SUM adds each group's amounts.
SELECT user_id, SUM(amount) AS total
FROM orders
GROUP BY user_id;
