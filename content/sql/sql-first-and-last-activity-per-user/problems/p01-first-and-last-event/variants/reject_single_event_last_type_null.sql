-- Wrong on purpose: a user with one event gets a NULL last_event_type instead of that event's type.
SELECT
    user_id,
    min(event_time) AS first_event_time,
    arg_min(event_type, struct_pack(t := event_time, id := event_id)) AS first_event_type,
    max(event_time) AS last_event_time,
    CASE WHEN count(*) > 1
         THEN arg_max(event_type, struct_pack(t := event_time, id := event_id))
    END AS last_event_type
FROM events
GROUP BY user_id
ORDER BY user_id;
