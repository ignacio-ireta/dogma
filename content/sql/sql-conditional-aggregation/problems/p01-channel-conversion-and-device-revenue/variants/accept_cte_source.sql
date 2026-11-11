-- Reference query: one row per channel, with conditional aggregates.
--
-- Grain first: the answer has one row per channel, so GROUP BY channel is the whole structure.
-- Every column is either the group key or an aggregate over that channel's rows.
--
-- The conditions sit inside the aggregates (conditional aggregation). That is one pass over the
-- table, with no join and no second query.
WITH base AS (SELECT * FROM sessions)
SELECT
    channel,
    -- All sessions of the channel, on both devices, converted or not. COUNT(*) counts rows.
    COUNT(*) AS sessions,
    -- Count only the converted rows. CASE gives 1 for a match and 0 otherwise, so the SUM counts
    -- the matches. COUNT(converted) would also count the FALSE rows, because FALSE is not NULL.
    SUM(CASE WHEN converted THEN 1 ELSE 0 END) AS converted_sessions,
    -- The rate. CAST the numerator to DOUBLE so the division is visibly a fraction, never an
    -- integer quotient. NULLIF turns a zero denominator into NULL instead of inf.
    ROUND(
        CAST(SUM(CASE WHEN converted THEN 1 ELSE 0 END) AS DOUBLE) / NULLIF(COUNT(*), 0),
        4
    ) AS conversion_rate,
    -- A SUM over no matching non-NULL value is NULL. COALESCE turns "no revenue on this device"
    -- into 0.0. The CASE without an ELSE gives NULL for the other device, and SUM skips it.
    COALESCE(SUM(CASE WHEN device = 'mobile' THEN revenue END), 0.0) AS mobile_revenue,
    COALESCE(SUM(CASE WHEN device = 'desktop' THEN revenue END), 0.0) AS desktop_revenue
FROM base
GROUP BY channel
ORDER BY channel;
