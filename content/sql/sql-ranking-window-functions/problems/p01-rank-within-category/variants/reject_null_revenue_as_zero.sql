-- Wrong on purpose: treats a NULL revenue as 0, so NULL rows tie with zero revenues instead of ranking last.
SELECT
    category,
    product_id,
    revenue,
    ROW_NUMBER() OVER (
        PARTITION BY category
        ORDER BY COALESCE(revenue, 0) DESC, product_id ASC
    ) AS row_number,
    RANK() OVER (
        PARTITION BY category
        ORDER BY COALESCE(revenue, 0) DESC
    ) AS rank,
    DENSE_RANK() OVER (
        PARTITION BY category
        ORDER BY COALESCE(revenue, 0) DESC
    ) AS dense_rank
FROM products
ORDER BY category ASC, row_number ASC;
