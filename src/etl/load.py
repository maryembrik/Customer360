"""Phase 2, step 2: cleaned Parquet -> PostgreSQL star schema (schema c360).

Uses COPY for the bulk load (about 20x faster than INSERT for 2.6M rows).
Run from the project root, with the Postgres container up:  python -m src.etl.load
"""
import io

import pandas as pd
import psycopg

from src.config import PROCESSED, SQL, db_url


def read(name: str) -> pd.DataFrame:
    return pd.read_parquet(PROCESSED / f"{name}.parquet")


def copy_df(cur: psycopg.Cursor, table: str, df: pd.DataFrame, chunk: int = 500_000) -> None:
    cols = ", ".join(df.columns)
    with cur.copy(f"COPY c360.{table} ({cols}) FROM STDIN WITH (FORMAT csv, NULL '')") as cp:
        for start in range(0, len(df), chunk):
            buf = io.StringIO()
            df.iloc[start:start + chunk].to_csv(buf, index=False, header=False)
            cp.write(buf.getvalue())
    print(f"  {table:<24} {len(df):>10,} rows")


def build_tables() -> dict[str, pd.DataFrame]:
    tx = read("transactions")
    product = read("product")
    demo = read("hh_demographic")
    desc = read("campaign_desc")

    households = pd.DataFrame({"household_key": sorted(tx.household_key.unique())})
    dim_household = households.merge(demo, on="household_key", how="left")
    dim_household.insert(1, "has_demographics", dim_household.household_key.isin(demo.household_key))
    for c in [c for c in dim_household.columns if c.endswith("_ord")]:
        dim_household[c] = dim_household[c].astype("Int16")

    dim_day = tx[["day", "week_no"]].drop_duplicates().sort_values("day")
    dim_day["is_onboarding"] = dim_day.week_no <= 16

    fact = tx.reset_index(drop=True)
    fact.insert(0, "line_id", fact.index + 1)
    fact = fact.drop(columns="week_no")

    exposure = read("campaign_table")[["household_key", "campaign"]]
    coupon = read("coupon")[["campaign", "coupon_upc", "product_id"]]
    redemption = read("coupon_redempt")[["household_key", "day", "coupon_upc", "campaign"]]

    return {
        "dim_household": dim_household,
        "dim_product": product,
        "dim_store": pd.DataFrame({"store_id": sorted(tx.store_id.unique())}),
        "dim_day": dim_day,
        "dim_campaign": desc[["campaign", "description", "start_day", "end_day", "ends_after_data"]],
        "fact_transaction": fact,
        "fact_campaign_exposure": exposure,
        "campaign_coupon": coupon,
        "fact_coupon_redemption": redemption,
    }


def main() -> None:
    tables = build_tables()
    with psycopg.connect(db_url(driver="postgresql")) as conn, conn.cursor() as cur:
        print("creating schema c360")
        cur.execute((SQL / "schema.sql").read_text())
        print("loading")
        for name, df in tables.items():   # dimensions first, so foreign keys resolve
            copy_df(cur, name, df)
        print("building indexes")
        conn.commit()
        conn.autocommit = True
        cur.execute((SQL / "indexes.sql").read_text())
    print("done")


if __name__ == "__main__":
    main()
