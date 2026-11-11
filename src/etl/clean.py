"""Phase 2, step 1: raw dunnhumby CSVs -> cleaned Parquet in data/processed/.

Every rule here comes from docs/assumptions_log.md and the Phase 1 EDA.
Run from the project root:  python -m src.etl.clean
"""
import pandas as pd

from src.config import FUEL_DEPARTMENTS, PROCESSED, RAW


def snake(df: pd.DataFrame) -> pd.DataFrame:
    return df.rename(columns=str.lower)


def clean_products() -> pd.DataFrame:
    p = snake(pd.read_csv(RAW / "product.csv"))
    p["is_fuel"] = p.department.isin(FUEL_DEPARTMENTS)  # A12
    return p


def clean_transactions(products: pd.DataFrame) -> pd.DataFrame:
    t = snake(pd.read_csv(RAW / "transaction_data.csv"))
    n_raw = len(t)

    # Of the 14,399 lines with quantity 0 and sales 0, 4,980 still carry a discount (coupon lines on the
    # receipt) and must stay, or coupon totals are lost. Only the 9,419 all-zero lines are dropped (A14).
    zero_line = (t.quantity == 0) & (t.sales_value == 0)
    no_discount = (t.retail_disc == 0) & (t.coupon_disc == 0) & (t.coupon_match_disc == 0)
    t = t[~(zero_line & no_discount)].copy()
    t["flag_discount_only_line"] = (t.quantity == 0) & (t.sales_value == 0)

    t = t.merge(products[["product_id", "is_fuel"]], on="product_id", how="left", validate="many_to_one")

    # A02: retailer revenue vs what the customer paid (discounts are stored negative)
    t["customer_paid"] = t.sales_value + t.coupon_disc

    # A03: guide formulas, named by what they compute (the guide's labels are swapped)
    qty = t.quantity.where(t.quantity > 0)
    t["shelf_price"] = (t.sales_value - (t.retail_disc + t.coupon_match_disc)) / qty
    t["card_price"] = (t.sales_value - t.coupon_match_disc) / qty

    t["hour"] = t.trans_time // 100
    t["flag_retail_disc_positive"] = t.retail_disc > 0
    t["flag_zero_qty_with_sales"] = (t.quantity == 0) & (t.sales_value > 0)

    print(f"transactions: {n_raw:,} raw -> {len(t):,} clean ({n_raw - len(t):,} all-zero lines dropped, "
          f"{t.flag_discount_only_line.sum():,} discount-only lines kept)")
    return t


def clean_demographics() -> pd.DataFrame:
    d = snake(pd.read_csv(RAW / "hh_demographic.csv"))
    # A10: masked codes kept as-is; ordered ones also get an integer version
    d["classification_1_ord"] = d.classification_1.str.extract(r"(\d+)$").astype(int)
    d["classification_3_ord"] = d.classification_3.str.extract(r"(\d+)$").astype(int)
    d["classification_4_ord"] = d.classification_4.str.rstrip("+").astype(int)
    d["classification_5_ord"] = d.classification_5.str.extract(r"(\d+)$").astype(int)
    d["kids_ord"] = d.kid_category_desc.replace({"None/Unknown": "0"}).str.rstrip("+").astype(int)
    return d


def clean_campaigns(last_day: int) -> dict[str, pd.DataFrame]:
    desc = snake(pd.read_csv(RAW / "campaign_desc.csv"))
    desc["ends_after_data"] = desc.end_day > last_day  # campaign 24
    coupon = snake(pd.read_csv(RAW / "coupon.csv"))
    n = len(coupon)
    coupon = coupon.drop_duplicates()
    print(f"coupon: {n:,} -> {len(coupon):,} ({n - len(coupon):,} duplicates dropped)")
    return {
        "campaign_desc": desc,
        "campaign_table": snake(pd.read_csv(RAW / "campaign_table.csv")),
        "coupon": coupon,
        "coupon_redempt": snake(pd.read_csv(RAW / "coupon_redempt.csv")),
    }


def main() -> None:
    PROCESSED.mkdir(parents=True, exist_ok=True)
    products = clean_products()
    tx = clean_transactions(products)

    # Each day must map to exactly one week, so dim_day can be built from transactions
    assert tx.groupby("day").week_no.nunique().max() == 1, "a day maps to several weeks"

    out = {"product": products, "transactions": tx, "hh_demographic": clean_demographics(),
           **clean_campaigns(int(tx.day.max()))}
    for name, df in out.items():
        df.to_parquet(PROCESSED / f"{name}.parquet", index=False)
        print(f"wrote {name}.parquet  {df.shape}")


if __name__ == "__main__":
    main()
