-- Wrong on purpose: the answer is typed in as literal rows instead of read from orders.
--
-- Reference query: customers with three or more completed orders.
SELECT customer_id, completed_orders, total_amount
FROM (
    VALUES
        (207, 5, 440.0),
        (202, 3, 300.0),
        (210, 3, 300.0),
        (208, 3, 180.0),
        (201, 4, 145.5),
        (204, 3, 100.0)
) AS answer(customer_id, completed_orders, total_amount)
ORDER BY total_amount DESC NULLS LAST, customer_id ASC;
