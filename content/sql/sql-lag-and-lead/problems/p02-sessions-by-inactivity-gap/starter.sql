-- screen_views(visitor_id BIGINT, view_id BIGINT, viewed_at TIMESTAMP)
-- Order each visitor's views by viewed_at, then view_id. A view starts a session when it is the
-- visitor's first view or its gap from the previous view is greater than 30 minutes.
-- Return visitor_id, view_id, viewed_at, session_number BIGINT (running count of session
-- starts per visitor), and starts_session BOOLEAN. Final order: visitor_id, viewed_at, view_id.
SELECT 1;
