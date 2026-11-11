SELECT category, product_id, revenue, DENSE_RANK() OVER (
    PARTITION BY category ORDER BY revenue DESC NULLS LAST
) AS dense_rank
FROM products
QUALIFY dense_rank <= 2
ORDER BY category, dense_rank, product_id;
