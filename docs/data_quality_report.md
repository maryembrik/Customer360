# Data Quality Report

Every number comes from `notebooks/01_dunnhumby_eda.ipynb`; the section number of the cell is given in brackets.

## dunnhumby

### Row counts & keys [§1, §2]
| File | Rows | Columns | Memory (MB) | Null cells | Duplicate rows |
|---|---|---|---|---|---|
| transaction_data.csv | 2,595,732 | 12 | ~249 | 0 | 0 |
| product.csv | 92,353 | 7 | ~10 | 0 | 0 |
| hh_demographic.csv | 801 | 8 | <1 | 0 | 0 |
| campaign_table.csv | 7,208 | 3 | <1 | 0 | 0 |
| campaign_desc.csv | 30 | 4 | <1 | 0 | 0 |
| coupon.csv | 124,548 | 3 | ~3 | 0 | **5,164** |
| coupon_redempt.csv | 2,318 | 4 | <1 | 0 | 0 |

### Missing values [§1]
None in any table.

### Duplicates [§1]
Only `coupon.csv` (5,164 exact duplicate rows); drop them in ETL.

### Invalid / suspicious values [§4]
| Check | Rows |
|---|---|
| QUANTITY = 0 | 14,466 |
| QUANTITY = 0 and SALES_VALUE = 0 | 14,399 |
| QUANTITY = 0 but SALES_VALUE > 0 | 67 |
| SALES_VALUE = 0 | 18,850 |
| SALES_VALUE < 0 | 0 |
| QUANTITY > 100 | 23,136 (almost all KIOSK-GAS / MISC SALES TRAN, see A12) |
| RETAIL_DISC > 0 | 36 |

### Discount sign & price-formula check (A03) [§5]
- `COUPON_DISC` and `COUPON_MATCH_DISC` are never positive; `RETAIL_DISC` is positive on 36 rows.
- The guide's three example rows exist and reproduce exactly (1.67 / 1.67 / 1.00; customer paid 2.34).
- `shelf_price >= card_price` on all but 2 rows (both with positive `RETAIL_DISC`).

### Referential integrity [§2]
- Every transaction `PRODUCT_ID` exists in `product.csv` (92,339 of 92,353 catalogue products are bought).
- Each `BASKET_ID` belongs to exactly one household and one day.
- All 801 demographic households and all 1,584 campaign households appear in transactions.

### Time coverage [§3]
- `DAY` 1–711, `WEEK_NO` 1–102, `TRANS_TIME` 0–2359.
- Onboarding ramp: 88 active households in week 1 → ~1,300 by week 16, then flat (A13).
- Campaign 24 ends on day 719, after the last transaction day.

### Demographic coverage & representativeness (A04) [§8]
801 / 2,500 households (32.0%).

| Feature | Median with demo | Median without | SMD |
|---|---|---|---|
| baskets | 139 | 52 | 0.84 |
| sales | $4,676 | $1,397 | 1.11 |
| avg basket | $32.3 | $25.3 | 0.40 |
| active weeks | 73 | 36 | 1.54 |
| tenure days | 635 | 621 | 0.42 |
| last day | 709 | 702 | 0.50 |

All |SMD| > 0.1 → the subset is not representative.

### Churn-definition feasibility (A05) [§10]
Share of eligible households with no purchase in the window after the cutoff:

| Window (days) | Cutoff 500 | Cutoff 550 | Cutoff 600 |
|---|---|---|---|
| 14 | 31.8% | 31.0% | 30.1% |
| 30 | 19.0% | 18.5% | 17.8% |
| 45 | 14.1% | 12.8% | 13.0% |
| 60 | 10.9% | 9.9% | 10.1% |
| 90 | 7.7% | 7.2% | 7.1% |

No purchase in the last 60 / 90 / 120 days of the data: 10.4% / 7.0% / 4.9% of households.

### Business snapshot [§5–§7, §9]
- Retailer sales (`SALES_VALUE`) $8,057,463; customer paid $8,014,852; loyalty discounts $1,398,335.
- Median household: 79 baskets, $2,158 sales, $27.4 average basket. Top 10% of households = 34.3% of sales.
- GROCERY 50.8% of sales, DRUG GM 13.1%, PRODUCE 6.9%, MEAT 6.8%, KIOSK-GAS 6.8%. National brands 72.2%.
- Campaigns: 1,584 households reached; 434 redeemed (2,318 redemptions, 556 coupons). Redemption rate TypeA 16.0%, TypeB 7.9%, TypeC 7.7%.

## Amazon Reviews sample
_Not started (Phase 5)._
