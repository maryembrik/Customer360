"""Household-level feature mart: one row per household, computed "as of" a given day.

Only data with day <= as_of_day is used, so the same function builds:
  - the Phase 3 segmentation table (as_of_day = 711, the last day), and
  - leakage-free model features in Phase 4 (as_of_day = the cutoff day).

Rules applied (docs/assumptions_log.md):
  A02 money = customer_paid (sales_value + coupon_disc); retailer revenue kept separately
  A12 fuel / misc lines excluded from basket, quantity and diversity features; kept as fuel_spend
  A13 trend windows only use days after onboarding (week >= 17)
"""
import numpy as np
import pandas as pd

from src.config import PROCESSED

TREND_WINDOW_DAYS = 56  # 8 weeks


def load_inputs() -> dict[str, pd.DataFrame]:
    tx = pd.read_parquet(PROCESSED / "transactions.parquet")
    product = pd.read_parquet(PROCESSED / "product.parquet",
                              columns=["product_id", "department", "commodity_desc", "brand"])
    return {
        "tx": tx.merge(product, on="product_id", how="left"),
        "exposure": pd.read_parquet(PROCESSED / "campaign_table.parquet"),
        "campaigns": pd.read_parquet(PROCESSED / "campaign_desc.parquet"),
        "redemptions": pd.read_parquet(PROCESSED / "coupon_redempt.parquet"),
    }


def build_features(inputs: dict[str, pd.DataFrame], as_of_day: int) -> pd.DataFrame:
    tx = inputs["tx"]
    tx = tx[tx.day <= as_of_day]
    grocery = tx[~tx.is_fuel]

    g = tx.groupby("household_key")
    f = pd.DataFrame({
        "first_day": g.day.min(),
        "last_day": g.day.max(),
        "revenue": g.sales_value.sum(),               # what the retailer received
        "monetary": g.customer_paid.sum(),            # what the household paid (A02)
        "loyalty_savings": -g.retail_disc.sum(),
        "coupon_savings": -g.coupon_disc.sum(),
    })
    f["fuel_spend"] = tx[tx.is_fuel].groupby("household_key").customer_paid.sum()

    gg = grocery.groupby("household_key")
    baskets = grocery.groupby(["household_key", "basket_id"]).agg(
        day=("day", "first"), value=("customer_paid", "sum"), items=("quantity", "sum"))
    bg = baskets.groupby("household_key")
    f["frequency"] = bg.size()                         # grocery baskets
    f["avg_basket_value"] = bg.value.mean()
    f["avg_items_per_basket"] = bg["items"].mean()
    f["n_departments"] = gg.department.nunique()
    f["n_commodities"] = gg.commodity_desc.nunique()
    f["n_stores"] = gg.store_id.nunique()
    f["private_label_share"] = (grocery[grocery.brand == "Private"].groupby("household_key").customer_paid.sum()
                                / gg.customer_paid.sum())

    # Regularity: average and spread of the gap between shopping days
    days = baskets.reset_index()[["household_key", "day"]].drop_duplicates().sort_values(["household_key", "day"])
    gaps = days.groupby("household_key").day.diff()
    f["avg_days_between_trips"] = gaps.groupby(days.household_key).mean()
    f["std_days_between_trips"] = gaps.groupby(days.household_key).std()

    # Spend trend: last 8 weeks vs the 8 weeks before (A13: skip the onboarding period)
    recent = tx[tx.day > as_of_day - TREND_WINDOW_DAYS].groupby("household_key").customer_paid.sum()
    prior_mask = (tx.day > as_of_day - 2 * TREND_WINDOW_DAYS) & (tx.day <= as_of_day - TREND_WINDOW_DAYS) & (tx.week_no >= 17)
    prior = tx[prior_mask].groupby("household_key").customer_paid.sum()
    f["spend_last_8w"] = recent
    f["spend_prev_8w"] = prior

    # Campaigns: only those that started on or before the as-of day
    started = inputs["campaigns"].loc[inputs["campaigns"].start_day <= as_of_day, "campaign"]
    exp = inputs["exposure"][inputs["exposure"].campaign.isin(started)]
    f["campaigns_received"] = exp.groupby("household_key").size()
    f["typea_received"] = exp[exp.description == "TypeA"].groupby("household_key").size()
    red = inputs["redemptions"][inputs["redemptions"].day <= as_of_day]
    f["coupons_redeemed"] = red.groupby("household_key").size()

    count_cols = ["fuel_spend", "frequency", "spend_last_8w", "spend_prev_8w",
                  "campaigns_received", "typea_received", "coupons_redeemed"]
    f[count_cols] = f[count_cols].fillna(0)
    f["private_label_share"] = f.private_label_share.fillna(0)

    # Derived
    f["recency"] = as_of_day - f.last_day
    f["tenure_days"] = as_of_day - f.first_day
    f["trips_per_week"] = f.frequency / np.maximum(f.tenure_days / 7, 1)
    # Log ratio: symmetric, 0 = flat, and does not explode when the previous spend is near 0
    f["spend_trend"] = np.log1p(f.spend_last_8w) - np.log1p(f.spend_prev_8w)
    f["loyalty_savings_rate"] = f.loyalty_savings / (f.revenue + f.loyalty_savings)
    f["redeemed_any"] = f.coupons_redeemed > 0

    f.insert(0, "as_of_day", as_of_day)
    return f.drop(columns=["last_day", "first_day"]).reset_index()


if __name__ == "__main__":
    feats = build_features(load_inputs(), as_of_day=711)
    print(feats.shape)
    print(feats.describe().T.round(2).to_string())
