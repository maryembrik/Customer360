# Assumptions & Limitations Log

Every decision that changes how data is read, cleaned or modeled goes here.
Evidence is always labeled as one of three kinds:
**Guide** (stated in the 2023 user guide), **Guide example** (inferred from the guide's worked example),
or **EDA** (measured in a notebook).

Source of truth: `data/raw/dunnhumby/dunnhumby_user_guide.pdf`, © 2023, from the official dunnhumby download.

## Index

| ID | Topic | Assumption / decision | Evidence | Status |
|---|---|---|---|---|
| A01 | Time | `DAY` (1–711) and `WEEK_NO` (1–102) are relative indices; no calendar dates | Guide + EDA | Verified |
| A02 | Revenue | `SALES_VALUE` = what the retailer received; customer paid = `SALES_VALUE + COUPON_DISC` | Guide + Guide example + EDA | Verified |
| A03 | Product pricing | Guide formulas kept as printed; their labels contradict the guide's own example | Guide example + EDA | Verified (labels swapped) |
| A04 | Demographics | 801 of 2,500 households; subset is heavily skewed to heavy shoppers | EDA | Verified |
| A05 | Churn | 60-day inactivity after a cutoff gives ~10% positives | EDA | Measured, final choice in Phase 4 |
| A06 | Spending decline | Not needed as a fallback, since churn is not rare | EDA | Dropped |
| A07 | Review sampling | Uniform reservoir sample of 30K eligible reviews | Sampling design | Accepted |
| A08 | Sentiment | Ratings give weak (proxy) sentiment labels | Modeling assumption | Accepted |
| A09 | Customer identity | Amazon reviewers and dunnhumby households are separate populations | Source schemas | Accepted |
| A10 | Data source | Official release; masked demographics used as generic codes | SHA-256 + Guide + EDA | Accepted |
| A11 | TypeA coupons | Which 16 coupons each TypeA household received is unknown | Guide | Limitation |
| A12 | Fuel & misc lines | `KIOSK-GAS` / `MISC SALES TRAN` lines distort quantities | EDA | Accepted, applied in Phase 2 |
| A13 | Onboarding period | Households were added during weeks 1–16; trends start at week 17 | EDA | Accepted |
| A14 | Zero lines | Drop only all-zero lines; keep 4,980 discount-only lines | EDA + warehouse validation | Applied in Phase 2 |
| A15 | Segmentation | RFM grid + K-Means k=4 on log-scaled R, F, M, avg basket | Silhouette + profiles | Applied in Phase 3 |
| A16 | Campaign targeting | Campaigns were targeted at heavy shoppers, so raw redemption rates are biased | EDA (Phase 3) | Limitation |

---

### A01 — Relative time only
- **Decision:** Use `DAY` and `WEEK_NO`; never invent calendar dates.
- **Evidence:** Guide: `WEEK_NO` "Ranges 1 – 102". EDA: `DAY` 1–711, `WEEK_NO` 1–102, `TRANS_TIME` 0–2359 (HHMM integer).
- **Impact:** Recency and intervals are in day-index units; no month or holiday seasonality. Validation uses chronological splits.
- **Status:** Verified

### A02 — Revenue definition
- **Guide states:** "The variable sales_value in this table is the amount of dollars received by the retailer on the sale of the specific product, taking the coupon match and loyalty card discount into account. It is not the actual price paid by the customer."
- **Guide example (Line 3):** "the customer paid $2.34 ($2.89 - $0.55) … but the retailer will receive $2.89", where 0.55 is `COUPON_DISC`. Inferred general form: **customer paid = `SALES_VALUE + COUPON_DISC`** (discounts are stored negative).
- **EDA:** The guide's example row exists in the data and gives 2.34. `COUPON_DISC` and `COUPON_MATCH_DISC` are never positive. Totals: `SALES_VALUE` $8,057,463; customer paid $8,014,852.
- **Decision:** Retailer revenue KPIs use `SALES_VALUE`. Customer-spend features (RFM monetary) use customer paid. Both are kept as separate columns.
- **Status:** Verified

### A03 — Product price formulas
- **Guide states (exactly as printed):**
  - Loyalty card price = (sales_value – (retail_disc + coupon_match_disc))/quantity
  - Non-loyalty card price = (sales_value – (coupon_match_disc))/quantity
- **Guide example (Line 2):** sales_value 2, retail_disc −1.34, quantity 2. The "loyalty card price" formula gives (2 + 1.34)/2 = **1.67**, which the guide's own text calls the "regular shelf price (exclusive of loyalty card discount)". The "non-loyalty" formula gives 2/2 = **1.00**, which the text calls the "shelf price including loyalty card discount".
- **Conclusion:** The formulas are kept as printed, but their labels contradict the guide's own example. In code they are named by what they compute: `shelf_price` (first formula) and `card_price` (second formula).
- **EDA:** All three example rows reproduce exactly. `shelf_price >= card_price` holds on all but 2 rows; both have a positive `RETAIL_DISC` (36 such rows in total, an anomaly).
- **Status:** Verified (labels swapped)

### A04 — Partial and unrepresentative demographics
- **Guide states:** demographics exist "for a portion of households".
- **EDA:** 801 of 2,500 households (32.0%), all present in transactions. Compared with the other 1,699 households: median baskets 139 vs 52 (SMD 0.84), median sales $4,676 vs $1,397 (SMD 1.11), active weeks 73 vs 36 (SMD 1.54). Every SMD is far above 0.1.
- **Decision:** Model A covers all households (behavioral features). Model B covers the demographic subset only, is evaluated only on it, and its results describe heavier shoppers, not the whole base.
- **Status:** Verified

### A05 — Churn definition
- **Setup:** Cutoff day; features from days ≤ cutoff; label = no purchase in the next `window` days; only households active before the cutoff; `cutoff + window ≤ 711`.
- **EDA (cutoffs 500/550/600):** inactive share ≈ 31% (14 days), 18% (30), 13% (45), 10% (60), 7% (90), stable across cutoffs. 7.0% of households bought nothing in the last 90 days of the data.
- **Proposal:** 60-day window (~10% positives): long enough to skip short breaks, frequent enough to model.
- **Status:** Measured; final choice in Phase 4

### A06 — Spending decline fallback
- **Decision:** Dropped as a fallback. Churn is not rare (A05). It can still be an extra target if time allows.
- **Status:** Dropped

### A07 — Review sampling
- **Decision:** One streaming pass over the compressed JSONL. Validate each record, keep a reservoir of 30,000 eligible reviews, count ratings in the same pass, save as Parquet in `data/sampled/`.
- **Reproducibility:** Fixed random seed, plus the source file name and size recorded here.
- **Status:** Accepted

### A08 — Rating-derived sentiment labels
- **Decision:** 1–2 stars = Negative, 3 = Neutral, 4–5 = Positive.
- **Impact:** Labels are noisy proxies. Splits are stratified, the test set keeps the natural distribution, and class weighting is applied to training data only.
- **Status:** Accepted

### A09 — Separate populations
- **Decision:** Amazon reviewers and dunnhumby households are never joined at the individual level; only category-level comparisons are allowed.
- **Status:** Accepted

### A10 — Data source & masked demographics
- **Decision:** The official dunnhumby download is the source of truth. Its `hh_demographic.csv` is used with masked labels.
- **Evidence (SHA-256, 2026-11-03):** six CSVs are byte-identical to a Kaggle mirror; only `hh_demographic.csv` differs (Kaggle has older readable labels). The masked codes are never decoded with the Kaggle copy.
- **Guide states:** fields have "generic names" and values "have been chosen such that they provide meaningful information: ordinality is important".
- **Guide inconsistencies:** the variable-name column of the guide's demographic table is a copy of the transaction table (`BASKET_ID`, `DAY`, …), and its diagram shows `CLASSIFICATION_1`–`7` while the file has `classification_1`–`5` plus readable `HOMEOWNER_DESC` and `KID_CATEGORY_DESC`.
- **EDA (value sets matched to the guide):**
  - `classification_1`: Age Group1–6, ordered (the values themselves say "Age Group")
  - `classification_2`: X/Y/Z, order not stated (treated as nominal)
  - `classification_3`: Level1–12, ordered (meaning not stated)
  - `classification_4`: 1–5+, ordered
  - `classification_5`: Group1–6, ordered (meaning not stated)
  - The file lists `classification_5` before `classification_4`; columns are always selected by name.
- **Impact:** Ordered codes become ordinal features; `classification_2` is one-hot encoded. Dashboards show the codes.
- **Status:** Accepted

### A11 — Unknown TypeA coupon assignment
- **Guide states:** each TypeA household "received 16 coupons out of the pool … Identifying the specific 16 coupons that each customer received is outside the scope of this database."
- **Impact:** Coupon-level response modeling is not possible for TypeA; campaign-level (household received / redeemed) analysis is.
- **EDA:** Household redemption rate: TypeA 16.0%, TypeB 7.9%, TypeC 7.7%. Campaign 24 ends on day 719, after the last transaction day.
- **Status:** Limitation

### A12 — Fuel and miscellaneous lines
- **EDA:** 23,136 lines with QUANTITY > 100, almost all `KIOSK-GAS` / `MISC SALES TRAN` "COUPON/MISC ITEMS" (median ≈ 10,000, max 89,638). KIOSK-GAS is 6.8% of sales.
- **Decision:** Exclude these departments from quantity and basket-size features. Keep them in total sales, with a separate `fuel_sales` feature.
- **Status:** Accepted, applied in Phase 2

### A13 — Onboarding period
- **EDA:** Active households per week rise from 88 (week 1) to ~1,300 (week 16), then stay flat; 90.8% of households appear by week 16.
- **Decision:** Trend analyses and time-windowed features start at week 17 (day ≥ 113). Tenure is computed from each household's first purchase.
- **Status:** Accepted

### A14 — Zero-quantity, zero-sales lines
- **EDA:** 14,399 lines have QUANTITY = 0 and SALES_VALUE = 0. Of these, 4,955 carry a `COUPON_DISC` (total −$10,328.53) and 25 a `RETAIL_DISC`; 9,419 are all zero. 592 baskets consist only of such lines.
- **How it was found:** the first ETL dropped all 14,399 lines; the warehouse validation then showed customer-paid totals $10,328.53 off the raw file.
- **Decision:** Drop only the 9,419 all-zero lines. Keep the 4,980 discount-only lines, flagged `flag_discount_only_line`, so coupon money is not lost. Money totals in the warehouse match the raw file exactly.
- **Status:** Applied in Phase 2

### A15 — Segmentation design
- **Features:** household feature mart as of day 711. Monetary = `customer_paid` (A02); frequency = grocery baskets (fuel excluded, A12).
- **RFM:** quintile scores 1–5 (ranked first so ties split evenly), FM = ceil((F + M) / 2), standard 10-segment R × FM grid.
- **K-Means:** recency, frequency, monetary, avg basket value; log1p then standardized; `random_state=42`, `n_init=20`.
- **Choice of k = 4:** k = 2 has the best silhouette (0.34) but only splits active from inactive. Among k = 3–8, k = 4 is best (≈0.26) and gives four actionable groups. Clusters are named from their profiles, not from label numbers.
- **Result:** VIP Regulars 31% of households / 65% of spend; Occasional Stock-Up 27% / 19%; Frequent Top-Up 25% / 14%; Lapsing Low-Value 17% / 2%.
- **Status:** Applied in Phase 3

### A16 — Campaigns were targeted, not random
- **EDA:** 96% of VIP Regulars received at least one campaign vs 7% of Lapsing Low-Value households; 41% vs 0% redeemed.
- **Impact:** Comparing redemption or spend between households that did and didn't receive campaigns mixes the campaign effect with who was chosen. Any campaign-effect claim needs a comparison within similar households (e.g. same segment, before vs after).
- **Status:** Limitation

---

<!-- Template for new entries:
### AXX — Title
- **Decision:**
- **Evidence:**
- **Impact:**
- **Status:**
-->
