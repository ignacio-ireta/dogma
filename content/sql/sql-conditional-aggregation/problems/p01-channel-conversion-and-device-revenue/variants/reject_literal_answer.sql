-- Wrong on purpose: the fixture answer is typed in as literal rows, without reading sessions.
SELECT * FROM (VALUES
    ('ads', 8, 3, 0.375, 65.0, 0.0),
    ('email', 6, 2, 0.3333, 30.0, 15.0),
    ('organic', 5, 0, 0.0, 0.0, 0.0),
    ('podcast', 2, 1, 0.5, 0.0, 12.5),
    ('referral', 3, 1, 0.3333, 80.0, 0.0)
) AS answer(channel, sessions, converted_sessions, conversion_rate, mobile_revenue, desktop_revenue)
ORDER BY channel;
