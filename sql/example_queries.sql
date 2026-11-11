-- Example analytical queries on the c360 star schema.
SET search_path TO c360;

-- 1. Sales by department, excluding fuel (A12)
SELECT p.department,
       round(sum(f.sales_value), 0)                         AS sales,
       round(100 * sum(f.sales_value) / sum(sum(f.sales_value)) OVER (), 1) AS pct
FROM fact_transaction f
JOIN dim_product p USING (product_id)
WHERE NOT f.is_fuel
GROUP BY p.department
ORDER BY sales DESC
LIMIT 10;

-- 2. Weekly sales after onboarding (A13)
SELECT d.week_no, round(sum(f.sales_value), 0) AS sales, count(DISTINCT f.household_key) AS households
FROM fact_transaction f
JOIN dim_day d USING (day)
WHERE NOT d.is_onboarding
GROUP BY d.week_no
ORDER BY d.week_no;

-- 3. Campaign redemption rate by campaign type (household level)
SELECT c.description,
       count(DISTINCT e.household_key)                       AS reached,
       count(DISTINCT r.household_key)                       AS redeemed,
       round(count(DISTINCT r.household_key)::numeric / count(DISTINCT e.household_key), 3) AS rate
FROM fact_campaign_exposure e
JOIN dim_campaign c USING (campaign)
LEFT JOIN fact_coupon_redemption r ON r.household_key = e.household_key AND r.campaign = e.campaign
GROUP BY c.description
ORDER BY c.description;

-- 4. Households with vs without demographics (A04)
SELECT h.has_demographics,
       count(*)                                   AS households,
       round(avg(t.sales), 0)                     AS avg_sales,
       round(avg(t.baskets), 0)                   AS avg_baskets
FROM dim_household h
JOIN (SELECT household_key, sum(sales_value) AS sales, count(DISTINCT basket_id) AS baskets
      FROM fact_transaction GROUP BY household_key) t USING (household_key)
GROUP BY h.has_demographics;
