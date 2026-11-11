-- Wrong on purpose: the labels are trimmed and lower-cased before they are returned.
SELECT
    user_id,
    min(event_time) AS first_event_time,
    arg_min(lower(trim(event_type)), struct_pack(t := event_time, id := event_id)) AS first_event_type,
    max(event_time) AS last_event_time,
    arg_max(lower(trim(event_type)), struct_pack(t := event_time, id := event_id)) AS last_event_type
FROM events
GROUP BY user_id
ORDER BY user_id;
