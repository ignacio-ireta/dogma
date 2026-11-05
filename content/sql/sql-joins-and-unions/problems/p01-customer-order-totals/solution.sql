-- Reference query: one row per current customer, with the order count and order total.
--
-- Grain first: the answer has one row per customer. customers has one row per VERSION, so
-- WHERE is_current keeps one row per customer before the join. Without it, every order of a
-- customer with two versions is joined twice, and both count and total double (fan-out).
--
-- LEFT JOIN from customers keeps customers with no orders. Orders whose customer_id is not in
-- customers never appear, because the query starts from customers and only adds orders to them.
SELECT
    c.customer_id,
    c.customer_name,
    COUNT(o.order_id) AS order_count,              -- counts matched orders; 0 for no match
    COALESCE(SUM(o.amount), 0.0) AS total_amount   -- SUM of no rows is NULL; COALESCE gives 0.0
FROM customers AS c
LEFT JOIN orders AS o
    ON o.customer_id = c.customer_id
WHERE c.is_current
GROUP BY c.customer_id, c.customer_name
ORDER BY total_amount DESC, c.customer_id ASC;
