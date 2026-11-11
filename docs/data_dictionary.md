# Data Dictionary

Meanings come from the 2023 user guide unless marked *(EDA)*. Types are the pandas dtypes from `01_dunnhumby_eda`.

## dunnhumby — The Complete Journey

### transaction_data.csv — 2,595,732 rows, one per product line on a receipt
| Column | Type | Meaning | Transformation |
|---|---|---|---|
| household_key | int64 | Household ID | Key to every household table |
| BASKET_ID | int64 | Purchase occasion (one household, one day) *(EDA)* | Order ID for frequency |
| DAY | int64 | Day of transaction, 1–711 *(EDA)* | Recency, intervals (A01) |
| PRODUCT_ID | int64 | Product ID | Join to product |
| QUANTITY | int64 | Units bought | Huge values on fuel/misc lines excluded from quantity features (A12) |
| SALES_VALUE | float64 | Dollars received by the retailer | Retailer revenue (A02) |
| STORE_ID | int64 | Store ID (582 stores) *(EDA)* | |
| RETAIL_DISC | float64 | Loyalty card discount, stored negative | 36 positive rows *(EDA)* |
| TRANS_TIME | int64 | Time of day, HHMM integer 0–2359 *(EDA)* | Hour feature |
| WEEK_NO | int64 | Week, 1–102 | Weekly trends from week 17 (A13) |
| COUPON_DISC | float64 | Manufacturer coupon discount, stored negative | customer paid = SALES_VALUE + COUPON_DISC (A02) |
| COUPON_MATCH_DISC | float64 | Retailer's match of a manufacturer coupon, stored negative | Used in price formulas (A03) |

### product.csv — 92,353 rows
| Column | Type | Meaning | Transformation |
|---|---|---|---|
| PRODUCT_ID | int64 | Product ID (unique) | Primary key |
| MANUFACTURER | int64 | Manufacturer code | |
| DEPARTMENT | str | Top-level grouping (e.g. GROCERY, DRUG GM, KIOSK-GAS) | Category diversity; fuel flag (A12) |
| BRAND | str | National or Private label | Private-label share feature |
| COMMODITY_DESC | str | Mid-level grouping | |
| SUB_COMMODITY_DESC | str | Lowest-level grouping | |
| CURR_SIZE_OF_PRODUCT | str | Package size, not available for all products | Not used |

### hh_demographic.csv — 801 rows (masked labels, see A10)
| Column | Type | Values *(EDA)* | Treatment |
|---|---|---|---|
| classification_1 | str | Age Group1–6 | Ordinal (age band) |
| classification_2 | str | X, Y, Z | Nominal, one-hot |
| classification_3 | str | Level1–12 | Ordinal, meaning not stated |
| HOMEOWNER_DESC | str | Homeowner, Renter, Probable Owner, Probable Renter, Unknown | Nominal |
| classification_5 | str | Group1–6 | Ordinal, meaning not stated |
| classification_4 | str | 1, 2, 3, 4, 5+ | Ordinal |
| KID_CATEGORY_DESC | str | None/Unknown, 1, 2, 3+ | Ordinal (None/Unknown = 0) |
| household_key | int64 | Household ID | Key |

### campaign_table.csv — 7,208 rows (which campaigns each household received)
| Column | Type | Meaning | Transformation |
|---|---|---|---|
| household_key | int64 | Household ID | |
| CAMPAIGN | int64 | Campaign ID, 1–30 | |
| DESCRIPTION | str | TypeA, TypeB or TypeC | |

### campaign_desc.csv — 30 rows
| Column | Type | Meaning | Transformation |
|---|---|---|---|
| CAMPAIGN | int64 | Campaign ID | Primary key |
| DESCRIPTION | str | TypeA, TypeB or TypeC | |
| START_DAY | int64 | First day of the campaign | |
| END_DAY | int64 | Last day (campaign 24 ends on day 719 > 711) *(EDA)* | |

### coupon.csv — 124,548 rows (5,164 duplicates)
| Column | Type | Meaning | Transformation |
|---|---|---|---|
| COUPON_UPC | int64 | Coupon ID | |
| PRODUCT_ID | int64 | Product the coupon is redeemable for | |
| CAMPAIGN | int64 | Campaign ID | Drop duplicate rows |

### coupon_redempt.csv — 2,318 rows
| Column | Type | Meaning | Transformation |
|---|---|---|---|
| household_key | int64 | Household ID | |
| DAY | int64 | Day the coupon was redeemed | |
| COUPON_UPC | int64 | Coupon ID | |
| CAMPAIGN | int64 | Campaign ID | Redemption features |

## Amazon Reviews 2023 — Grocery and Gourmet Food (sample)

| Field | Type | Meaning | Transformation |
|---|---|---|---|
| rating | | Star rating 1–5 | Source of the sentiment label (A08) |
| title | | Review title | |
| text | | Review body | Main NLP input |
| parent_asin | | Product ID | |
| user_id | | Reviewer ID | |
| timestamp | | Review time | |
| verified_purchase | | Verified purchase flag | |
| helpful_vote | | Helpful votes | |
