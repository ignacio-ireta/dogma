-- Keep each event's time and type together in one row, so a tied timestamp never mixes records.
-- The tie-breaker event_id sits inside the key, which makes arg_min and arg_max deterministic.
-- struct_pack orders by event_time first and event_id second, because DuckDB compares structs field by field.
SELECT
    user_id,
    min(event_time) AS first_event_time,
    arg_min(event_type, struct_pack(t := event_time, id := event_id)) AS first_event_type,
    max(event_time) AS last_event_time,
    arg_max(event_type, struct_pack(t := event_time, id := event_id)) AS last_event_type
FROM events
GROUP BY user_id
ORDER BY user_id;
