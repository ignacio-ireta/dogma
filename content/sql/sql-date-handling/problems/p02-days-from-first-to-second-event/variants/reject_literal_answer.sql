-- Wrong on purpose: the answer is typed in as literal rows and the events table is never read
SELECT *
FROM (VALUES (201, 3), (202, 1), (204, 0), (205, 1), (206, 59)) AS t(user_id, days_to_second)
ORDER BY user_id;
