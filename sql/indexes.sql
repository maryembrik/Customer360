-- Created after the bulk load: building indexes once is much faster than updating them row by row.
SET search_path TO c360;

CREATE INDEX ix_tx_household ON fact_transaction (household_key);
CREATE INDEX ix_tx_day       ON fact_transaction (day);
CREATE INDEX ix_tx_product   ON fact_transaction (product_id);
CREATE INDEX ix_tx_basket    ON fact_transaction (basket_id);
CREATE INDEX ix_redemption_household ON fact_coupon_redemption (household_key);

ANALYZE;
