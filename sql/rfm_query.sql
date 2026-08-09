CREATE OR REPLACE VIEW rfm_base AS
WITH reference_date AS (
    SELECT MAX(invoice_date) AS max_date FROM transactions
),
customer_agg AS (
    SELECT
        t.customer_id,
        MAX(t.invoice_date) AS last_purchase_date,
        COUNT(DISTINCT t.invoice) AS frequency,
        SUM(t.total_amount) AS monetary
    FROM transactions t
    WHERE t.is_return = FALSE
    GROUP BY t.customer_id
)
SELECT
    c.customer_id,
    (SELECT max_date FROM reference_date) - c.last_purchase_date AS recency_days,
    c.frequency,
    c.monetary
FROM customer_agg c;
