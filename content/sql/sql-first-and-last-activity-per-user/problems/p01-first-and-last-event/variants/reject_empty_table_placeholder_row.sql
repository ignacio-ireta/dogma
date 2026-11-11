-- Wrong on purpose: an empty events table returns one placeholder row instead of no rows.
SELECT
    user_id,
    min(event_time) AS first_event_time,
    arg_min(event_type, struct_pack(t := event_time, id := event_id)) AS first_event_type,
    max(event_time) AS last_event_time,
    arg_max(event_type, struct_pack(t := event_time, id := event_id)) AS last_event_type
FROM events
GROUP BY user_id
UNION ALL
SELECT NULL::BIGINT, NULL::TIMESTAMP, NULL::VARCHAR, NULL::TIMESTAMP, NULL::VARCHAR
WHERE NOT EXISTS (SELECT 1 FROM events)
ORDER BY user_id;
