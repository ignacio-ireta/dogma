-- Wrong on purpose: the users come back in descending user_id order.
SELECT
    user_id,
    min(event_time) AS first_event_time,
    arg_min(event_type, struct_pack(t := event_time, id := event_id)) AS first_event_type,
    max(event_time) AS last_event_time,
    arg_max(event_type, struct_pack(t := event_time, id := event_id)) AS last_event_type
FROM events
GROUP BY user_id
ORDER BY user_id DESC;
