# Power BI report: build guide

The data is prepared by `python -m src.pipelines.export_bi` (run it after the scoring and sentiment steps).
It writes the same tables to `reports/powerbi/*.csv` and, if PostgreSQL is running, to `c360.bi_*`.

## 1. Load the data

**Option A, CSV (no database needed):** Home → Get data → Text/CSV → load every file in `reports/powerbi/`.

**Option B, PostgreSQL:** Home → Get data → PostgreSQL database. Server `localhost:5433`, database `customer360`,
Import mode. Select the `c360.bi_*` tables. Credentials are in your `.env` (never put them in the .pbix you share).

| Table | Grain | Used on page |
|---|---|---|
| `bi_household` | one household (2,500): segment, RFM, churn risk, 90-day spend, coupon score | 1–4 |
| `bi_weekly_segment` | week × segment: spend, discounts, baskets, households | 1, 2 |
| `bi_department` | department: spend, baskets, private-label share | 1 |
| `bi_campaign` | campaign: type, days, households targeted, redemption rate, targeted per segment | 4 |
| `bi_review_sentiment` | one sampled Amazon review: stars, predicted sentiment, year, theme flags | 5 |
| `bi_review_themes` | theme: share of reviews, negative rate | 5 |

In Power Query, check the types: `week_no`, `rating`, `year` → Whole number; probabilities and money → Decimal;
`is_active`, `is_onboarding`, `mentions_*` → True/False.

## 2. Model

Create a segment dimension (Modeling → New table) and link both tables to it, so one slicer filters everything:

```DAX
Segment = DISTINCT ( bi_household[segment] )
```

Relationships (Model view): `Segment[segment]` 1 → * `bi_household[segment]` and 1 → * `bi_weekly_segment[segment]`.
The review tables stay unrelated: Amazon reviewers are different people from the dunnhumby households (A09).

## 3. Measures (Modeling → New measure, store them in a `_Measures` table)

```DAX
Households = COUNTROWS ( bi_household )
Customer Paid = SUM ( bi_weekly_segment[customer_paid] )
Baskets = SUM ( bi_weekly_segment[baskets] )
Avg Basket = DIVIDE ( [Customer Paid], [Baskets] )
Spend Share = DIVIDE ( [Customer Paid], CALCULATE ( [Customer Paid], ALL ( Segment ) ) )
Household Share = DIVIDE ( [Households], CALCULATE ( [Households], ALL ( Segment ) ) )

Active Households = CALCULATE ( [Households], bi_household[is_active] = TRUE () )
Avg Churn Probability = AVERAGE ( bi_household[churn_probability] )      -- blanks (lapsed) are ignored
High Risk Households = CALCULATE ( [Households], bi_household[churn_risk_band] = "High" )
Predicted Spend 90d = SUM ( bi_household[predicted_spend_90d] )
Spend at Risk 90d = SUM ( bi_household[spend_at_risk_90d] )

Redemption Rate = DIVIDE ( SUM ( bi_campaign[redeeming_households] ), SUM ( bi_campaign[households_targeted] ) )

Reviews = COUNTROWS ( bi_review_sentiment )
Negative Share = DIVIDE ( CALCULATE ( [Reviews], bi_review_sentiment[predicted_label] = "negative" ), [Reviews] )
Delivery or Packaging Negatives =
    CALCULATE ( [Reviews], bi_review_sentiment[predicted_label] = "negative",
                FILTER ( bi_review_sentiment, bi_review_sentiment[mentions_delivery_shipping]
                                            || bi_review_sentiment[mentions_packaging_damage] ) )
```

Check after building page 1: total **Customer Paid = $8,014,851.54**, the figure the warehouse validation reconciles
to the raw files. If it differs, a filter or a relationship is wrong.

## 4. Pages

**1. Overview.** Cards: Households (2,500), Customer Paid, Baskets, Avg Basket. Line chart: Customer Paid by
`week_no`. Add a visual-level filter `is_onboarding = False` (weeks 1–16 are the panel filling up, A13) and say so in the
subtitle. Donut: Customer Paid by segment. Bar: top 10 departments by `customer_paid`.

**2. Segments.** Clustered bar with Household Share and Spend Share by segment (this shows the main finding:
VIP Regulars are about 31% of households but about 65% of spend). Matrix: `r_score` rows × `fm_score` columns, values = Households
(the RFM grid). Table: segment, Households, Avg Basket, Avg Churn Probability, Predicted Spend 90d.

**3. Churn risk.** Cards: Active Households, High Risk Households, Spend at Risk 90d. Stacked bar:
Households by segment and `churn_risk_band` (order Low, Medium, High, Lapsed with the Sort by column feature or a small
mapping table). **Call list** table: `household_key`, segment, `churn_probability`, `predicted_spend_90d`,
`spend_at_risk_90d`, filtered to band = High and sorted by spend at risk. Slicer: segment.
Tooltip text: "Probability of no purchase in the next 60 days, active households only; model ROC-AUC 0.88 on an unseen period."

**4. Campaigns & coupons.** Column chart: Redemption Rate by `campaign_type`. Stacked bar: `targeted_*` columns
by campaign (shows targeting concentrated on VIP Regulars, so raw redemption rates are not campaign effects, A16).
Histogram: Households by `coupon_target_decile`, coloured by `coupon_score_in_training_population`
(false = never targeted, score is an extrapolation, A20).

**5. Customer voice (Amazon reviews).** Donut: Reviews by `predicted_label`. Bar from `bi_review_themes`:
`negative_rate` by theme, with a constant line at the overall negative share. Card: Delivery or Packaging Negatives.
Line: Negative Share by `year`. Note on the page: "30K uniform random sample of 14M Amazon grocery reviews;
not the same customers as pages 1–4."

## 5. Finishing

- One colour per segment, kept on every page (Format → Data colors).
- Sync the segment slicer across pages 1–4 (View → Sync slicers).
- File → Export → PDF to attach to the application; keep the .pbix out of git if it contains credentials.
