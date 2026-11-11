-- Blind: FILTER clauses instead of CASE, one pass over sessions, no join.
SELECT
    channel,
    COUNT(*) AS sessions,
    COUNT(*) FILTER (WHERE converted) AS converted_sessions,
    ROUND(COUNT(*) FILTER (WHERE converted) * 1.0 / COUNT(*), 4) AS conversion_rate,
    COALESCE(SUM(revenue) FILTER (WHERE device = 'mobile'), 0.0) AS mobile_revenue,
    COALESCE(SUM(revenue) FILTER (WHERE device = 'desktop'), 0.0) AS desktop_revenue
FROM sessions
GROUP BY channel
ORDER BY channel;
