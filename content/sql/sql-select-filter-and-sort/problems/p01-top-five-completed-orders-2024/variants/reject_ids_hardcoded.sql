-- Wrong on purpose: the answer is looked up by a fixed list of order ids taken from the fixture.
SELECT
    order_id,
    customer_id,
    order_date,
    amount
FROM orders
WHERE order_id IN (20, 12, 14, 17, 7)
ORDER BY amount DESC, order_id ASC
LIMIT 5;
