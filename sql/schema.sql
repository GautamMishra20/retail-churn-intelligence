CREATE TABLE IF NOT EXISTS staging_transactions (
    invoice         VARCHAR(20),
    stock_code      VARCHAR(20),
    description     TEXT,
    quantity        INTEGER,
    invoice_date    TIMESTAMP,
    price           NUMERIC(10, 2),
    customer_id     VARCHAR(20),
    country         VARCHAR(100)
);

CREATE TABLE IF NOT EXISTS transactions (
    transaction_id  SERIAL PRIMARY KEY,
    invoice         VARCHAR(20),
    stock_code      VARCHAR(20),
    description     TEXT,
    quantity        INTEGER,
    invoice_date    TIMESTAMP,
    price           NUMERIC(10, 2),
    customer_id     VARCHAR(20),
    country         VARCHAR(100),
    total_amount    NUMERIC(12, 2),
    is_return       BOOLEAN DEFAULT FALSE
);

CREATE INDEX IF NOT EXISTS idx_transactions_customer ON transactions(customer_id);
CREATE INDEX IF NOT EXISTS idx_transactions_date ON transactions(invoice_date);
