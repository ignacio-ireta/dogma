-- events(user_id BIGINT, event_id BIGINT, event_time TIMESTAMP, event_type VARCHAR)
-- One row per user: first_event_time, first_event_type, last_event_time, last_event_type.
-- First = smallest event_time, then smallest event_id. Last = largest event_time, then largest event_id.
-- Final order by user_id. Names and types as in problem.md.
SELECT 1;
