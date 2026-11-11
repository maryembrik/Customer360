"""Phase 2, step 3: check the warehouse against the raw files and the Phase 1 EDA numbers.

Run from the project root:  python -m src.etl.validate
Exits with an error if any check fails.
"""
import pandas as pd
import psycopg

from src.config import PROCESSED, RAW, db_url

# (description, SQL returning one value, expected value)
CHECKS = [
    ("households", "SELECT count(*) FROM dim_household", 2500),
    ("households with demographics", "SELECT count(*) FROM dim_household WHERE has_demographics", 801),
    ("transaction lines", "SELECT count(*) FROM fact_transaction", None),  # filled from Parquet
    ("baskets", "SELECT count(DISTINCT basket_id) FROM fact_transaction", None),  # filled from Parquet
    ("products bought", "SELECT count(DISTINCT product_id) FROM fact_transaction", None),
    ("days", "SELECT count(*) FROM dim_day", 711),
    ("onboarding days (weeks 1-16)", "SELECT count(*) FROM dim_day WHERE is_onboarding", None),
    ("campaign exposures", "SELECT count(*) FROM fact_campaign_exposure", 7208),
    ("households reached", "SELECT count(DISTINCT household_key) FROM fact_campaign_exposure", 1584),
    ("redemptions", "SELECT count(*) FROM fact_coupon_redemption", 2318),
    ("total sales_value", "SELECT round(sum(sales_value), 2) FROM fact_transaction", None),
    ("total customer_paid", "SELECT round(sum(customer_paid), 2) FROM fact_transaction", None),
    ("fuel lines", "SELECT count(*) FROM fact_transaction WHERE is_fuel", None),
]


def expected_from_files() -> dict[str, object]:
    raw = pd.read_csv(RAW / "transaction_data.csv", usecols=["SALES_VALUE", "COUPON_DISC", "PRODUCT_ID"])
    tx = pd.read_parquet(PROCESSED / "transactions.parquet",
                         columns=["day", "week_no", "is_fuel", "product_id", "basket_id"])
    return {
        "transaction lines": len(tx),
        "baskets": tx.basket_id.nunique(),
        "products bought": tx.product_id.nunique(),
        "onboarding days (weeks 1-16)": tx.loc[tx.week_no <= 16, "day"].nunique(),
        # dropped lines are all-zero, so money totals must match the raw file exactly
        "total sales_value": round(raw.SALES_VALUE.sum(), 2),
        "total customer_paid": round((raw.SALES_VALUE + raw.COUPON_DISC).sum(), 2),
        "fuel lines": int(tx.is_fuel.sum()),
    }


def main() -> None:
    from_files = expected_from_files()
    failures = 0
    with psycopg.connect(db_url(driver="postgresql")) as conn, conn.cursor() as cur:
        cur.execute("SET search_path TO c360")
        for name, sql, expected in CHECKS:
            expected = from_files.get(name, expected)
            cur.execute(sql)
            actual = cur.fetchone()[0]
            ok = float(actual) == float(expected)
            failures += not ok
            print(f"{'OK  ' if ok else 'FAIL'} {name:<30} db={actual:>14,}  expected={expected:>14,}")
    if failures:
        raise SystemExit(f"{failures} check(s) failed")
    print("all checks passed")


if __name__ == "__main__":
    main()
