WITH eligible AS (
    SELECT entity_id, event_id, occurred_at, payload
    FROM events
    WHERE kind = 'success'
),
ranked AS (
    SELECT
        a.entity_id,
        a.event_id,
        a.occurred_at,
        a.payload,
        1 + COUNT(b.event_id) AS rn
    FROM eligible AS a
    LEFT JOIN eligible AS b
        ON b.entity_id = a.entity_id
        AND (
            b.occurred_at > a.occurred_at
            OR (b.occurred_at IS NOT NULL AND a.occurred_at IS NULL)
            OR (b.occurred_at IS NOT DISTINCT FROM a.occurred_at AND b.event_id > a.event_id)
        )
    GROUP BY a.entity_id, a.event_id, a.occurred_at, a.payload
)
SELECT
    entity_id,
    event_id,
    occurred_at,
    payload,
    CAST(rn AS BIGINT) AS rank_in_entity
FROM ranked
WHERE rn <= 2
ORDER BY entity_id ASC, rank_in_entity ASC;
