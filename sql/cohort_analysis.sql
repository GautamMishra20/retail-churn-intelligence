CREATE OR REPLACE VIEW cohort_analysis AS
WITH first_purchase AS (
    SELECT
        customer_id,
        DATE_TRUNC('month', MIN(invoice_date)) AS cohort_month
    FROM transactions
    WHERE is_return = FALSE
    GROUP BY customer_id
),
customer_activity AS (
    SELECT
        t.customer_id,
        DATE_TRUNC('month', t.invoice_date) AS activity_month
    FROM transactions t
    WHERE t.is_return = FALSE
    GROUP BY t.customer_id, DATE_TRUNC('month', t.invoice_date)
)
SELECT
    fp.cohort_month,
    ca.activity_month,
    (EXTRACT(YEAR FROM ca.activity_month) - EXTRACT(YEAR FROM fp.cohort_month)) * 12 +
    (EXTRACT(MONTH FROM ca.activity_month) - EXTRACT(MONTH FROM fp.cohort_month)) AS month_index,
    COUNT(DISTINCT ca.customer_id) AS active_customers
FROM customer_activity ca
JOIN first_purchase fp ON ca.customer_id = fp.customer_id
GROUP BY fp.cohort_month, ca.activity_month
ORDER BY fp.cohort_month, ca.activity_month;
