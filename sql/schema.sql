-- Customer360 star schema (PostgreSQL 16)
-- Grain of the main fact: one product line on one receipt (cleaned, see src/etl/clean.py).
-- Re-runnable: drops and recreates everything in schema c360.

DROP SCHEMA IF EXISTS c360 CASCADE;
CREATE SCHEMA c360;
SET search_path TO c360;

-- ---------- Dimensions ----------

CREATE TABLE dim_household (
    household_key        INTEGER PRIMARY KEY,
    has_demographics     BOOLEAN NOT NULL,
    -- Masked demographic codes (A10); NULL for the 1,699 households without demographics (A04)
    classification_1     TEXT,
    classification_2     TEXT,
    classification_3     TEXT,
    classification_4     TEXT,
    classification_5     TEXT,
    homeowner_desc       TEXT,
    kid_category_desc    TEXT,
    classification_1_ord SMALLINT,
    classification_3_ord SMALLINT,
    classification_4_ord SMALLINT,
    classification_5_ord SMALLINT,
    kids_ord             SMALLINT
);

CREATE TABLE dim_product (
    product_id           BIGINT PRIMARY KEY,
    manufacturer         INTEGER,
    department           TEXT,
    brand                TEXT,
    commodity_desc       TEXT,
    sub_commodity_desc   TEXT,
    curr_size_of_product TEXT,
    is_fuel              BOOLEAN NOT NULL          -- A12
);

CREATE TABLE dim_store (
    store_id INTEGER PRIMARY KEY
);

CREATE TABLE dim_day (
    day           SMALLINT PRIMARY KEY,            -- relative day 1-711 (A01)
    week_no       SMALLINT NOT NULL,
    is_onboarding BOOLEAN  NOT NULL                -- weeks 1-16 (A13)
);

CREATE TABLE dim_campaign (
    campaign        SMALLINT PRIMARY KEY,
    description     TEXT     NOT NULL,             -- TypeA / TypeB / TypeC
    start_day       SMALLINT NOT NULL,
    end_day         SMALLINT NOT NULL,
    ends_after_data BOOLEAN  NOT NULL
);

-- ---------- Facts ----------

CREATE TABLE fact_transaction (
    line_id                    BIGINT PRIMARY KEY,
    household_key              INTEGER  NOT NULL REFERENCES dim_household,
    basket_id                  BIGINT   NOT NULL,
    day                        SMALLINT NOT NULL REFERENCES dim_day,
    product_id                 BIGINT   NOT NULL REFERENCES dim_product,
    store_id                   INTEGER  NOT NULL REFERENCES dim_store,
    quantity                   INTEGER  NOT NULL,
    sales_value                NUMERIC(10,2) NOT NULL,   -- retailer revenue (A02)
    retail_disc                NUMERIC(10,2) NOT NULL,
    coupon_disc                NUMERIC(10,2) NOT NULL,
    coupon_match_disc          NUMERIC(10,2) NOT NULL,
    customer_paid              NUMERIC(10,2) NOT NULL,   -- sales_value + coupon_disc (A02)
    shelf_price                NUMERIC(12,4),            -- A03, NULL when quantity = 0
    card_price                 NUMERIC(12,4),
    trans_time                 SMALLINT NOT NULL,
    hour                       SMALLINT NOT NULL,
    is_fuel                    BOOLEAN  NOT NULL,
    flag_discount_only_line    BOOLEAN  NOT NULL,         -- quantity 0, sales 0, but a discount (A14)
    flag_retail_disc_positive  BOOLEAN  NOT NULL,
    flag_zero_qty_with_sales   BOOLEAN  NOT NULL
);

-- Which households received which campaign (bridge between household and campaign)
CREATE TABLE fact_campaign_exposure (
    household_key INTEGER  NOT NULL REFERENCES dim_household,
    campaign      SMALLINT NOT NULL REFERENCES dim_campaign,
    PRIMARY KEY (household_key, campaign)
);

-- Coupons offered in each campaign and the products they apply to
CREATE TABLE campaign_coupon (
    campaign   SMALLINT NOT NULL REFERENCES dim_campaign,
    coupon_upc BIGINT   NOT NULL,
    product_id BIGINT   NOT NULL REFERENCES dim_product,
    PRIMARY KEY (campaign, coupon_upc, product_id)
);

CREATE TABLE fact_coupon_redemption (
    redemption_id SERIAL PRIMARY KEY,
    household_key INTEGER  NOT NULL REFERENCES dim_household,
    day           SMALLINT NOT NULL REFERENCES dim_day,
    coupon_upc    BIGINT   NOT NULL,
    campaign      SMALLINT NOT NULL REFERENCES dim_campaign
);
