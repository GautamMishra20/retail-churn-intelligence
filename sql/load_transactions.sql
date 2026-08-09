INSERT INTO transactions (invoice, stock_code, description, quantity, invoice_date, price, customer_id, country, total_amount, is_return)
SELECT
    invoice,
    stock_code,
    description,
    quantity,
    invoice_date,
    price,
    customer_id,
    country,
    ROUND(quantity * price, 2) AS total_amount,
    CASE WHEN invoice LIKE 'C%' THEN TRUE ELSE FALSE END AS is_return
FROM staging_transactions
WHERE customer_id IS NOT NULL
  AND price > 0;
