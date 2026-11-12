"""Phase 7: tables for the Power BI report.

Writes reports/powerbi/*.csv (Power BI: Get data -> Text/CSV) and, when PostgreSQL is reachable, the same
tables as c360.bi_* (Power BI: Get data -> PostgreSQL, localhost:5433). Both sources hold identical data.

    python -m src.pipelines.export_bi
"""
import json

import numpy as np
import pandas as pd
from sqlalchemy import create_engine

from src.config import PROCESSED, ROOT, db_url

OUT = ROOT / "reports" / "powerbi"


def households() -> pd.DataFrame:
    pred = pd.read_parquet(PROCESSED / "household_predictions.parquet")
    seg = pd.read_parquet(PROCESSED / "household_segments.parquet").rename(columns={"cluster_name": "segment"})
    feat = pd.read_parquet(PROCESSED / "household_features.parquet")
    demo = pd.read_parquet(PROCESSED / "hh_demographic.parquet")[["household_key"]].assign(has_demographics=True)
    df = (pred.merge(seg.drop(columns=["cluster"]), on="household_key")
              .merge(feat[["household_key", "monetary", "frequency", "recency", "avg_basket_value", "trips_per_week",
                           "spend_last_8w", "spend_trend", "campaigns_received", "coupons_redeemed", "redeemed_any",
                           "private_label_share"]], on="household_key")
              .merge(demo, on="household_key", how="left"))
    df["has_demographics"] = df.has_demographics.fillna(False).astype(bool)
    return df


def weekly(tx: pd.DataFrame, seg: pd.DataFrame) -> pd.DataFrame:
    t = tx.merge(seg, on="household_key")
    g = (t.groupby(["week_no", "segment"])
          .agg(customer_paid=("customer_paid", "sum"), sales_value=("sales_value", "sum"),
               retail_disc=("retail_disc", "sum"), coupon_disc=("coupon_disc", "sum"),
               baskets=("basket_id", "nunique"), households=("household_key", "nunique"))
          .reset_index())
    g["is_onboarding"] = g.week_no <= 16   # A13: households were still being added
    return g


def departments(tx: pd.DataFrame, product: pd.DataFrame) -> pd.DataFrame:
    t = tx.merge(product[["product_id", "department", "brand"]], on="product_id", how="left")
    t["private"] = t.brand.eq("Private") * t.customer_paid
    g = (t.groupby("department", dropna=False)
          .agg(customer_paid=("customer_paid", "sum"), sales_value=("sales_value", "sum"),
               private_label_paid=("private", "sum"), baskets=("basket_id", "nunique"),
               households=("household_key", "nunique"), lines=("product_id", "size"))
          .reset_index())
    g["private_label_share"] = g.private_label_paid / g.customer_paid.where(g.customer_paid != 0)
    return g.sort_values("customer_paid", ascending=False)


def campaigns(seg: pd.DataFrame) -> pd.DataFrame:
    desc = pd.read_parquet(PROCESSED / "campaign_desc.parquet")
    exp = pd.read_parquet(PROCESSED / "campaign_table.parquet")
    red = pd.read_parquet(PROCESSED / "coupon_redempt.parquet")
    g = desc.merge(exp.groupby("campaign").household_key.nunique().rename("households_targeted"), on="campaign", how="left")
    # Rate = targeted households that redeemed / targeted households (redeemers outside the list don't count)
    red_t = red.merge(exp[["household_key", "campaign"]], on=["household_key", "campaign"])
    g = g.merge(red.groupby("campaign").size().rename("redemptions"), on="campaign", how="left")
    g = g.merge(red_t.groupby("campaign").household_key.nunique().rename("redeeming_households"),
                on="campaign", how="left").fillna({"redemptions": 0, "redeeming_households": 0})
    g["duration_days"] = g.end_day - g.start_day + 1
    g["redemption_rate"] = g.redeeming_households / g.households_targeted
    # Who each campaign reached, by segment (targeting bias, A16)
    reach = (exp.merge(seg, on="household_key").groupby(["campaign", "segment"]).household_key.nunique()
                .unstack(fill_value=0).add_prefix("targeted_"))
    return g.merge(reach, on="campaign", how="left").rename(columns={"description": "campaign_type"})


def sentiment():
    path = ROOT / "data" / "sampled" / "amazon_grocery_reviews_sample.parquet"
    res_path = ROOT / "reports" / "sentiment_results.json"
    model_path = ROOT / "artifacts" / "sentiment_model.joblib"
    if not (path.exists() and res_path.exists() and model_path.exists()):
        return None, None
    import joblib
    from src.nlp.sentiment import THEMES, label, make_doc
    m = joblib.load(model_path)   # trained by src.nlp.sentiment in this project (joblib unpickles code)
    df = pd.read_parquet(path)
    doc = make_doc(df.title, df.text)   # same input as training: star-count titles removed (A23)
    reviews = pd.DataFrame({
        "review_id": np.arange(len(df)), "rating": df.rating, "star_label": df.rating.map(label),
        "predicted_label": m["model"].predict(doc), "verified_purchase": df.verified_purchase,
        "helpful_vote": df.helpful_vote, "year": pd.to_datetime(df.timestamp, unit="ms").dt.year,
        "n_words": doc.str.split().str.len(), "title": df.title.str.slice(0, 200)})
    low = doc.str.lower()
    for name, pattern in THEMES.items():
        reviews[f"mentions_{name}"] = low.str.contains(pattern, regex=True)
    themes = pd.DataFrame(json.loads(res_path.read_text())["themes"]).T.rename_axis("theme").reset_index()
    ci = themes.pop("negative_rate_ci")
    themes["negative_rate_ci_low"], themes["negative_rate_ci_high"] = ci.str[0], ci.str[1]
    themes = themes.infer_objects()
    return reviews, themes


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    tx = pd.read_parquet(PROCESSED / "transactions.parquet",
                         columns=["household_key", "basket_id", "week_no", "product_id", "sales_value",
                                  "customer_paid", "retail_disc", "coupon_disc"])
    product = pd.read_parquet(PROCESSED / "product.parquet")
    hh = households()
    seg = hh[["household_key", "segment"]]
    tables = {"bi_household": hh, "bi_weekly_segment": weekly(tx, seg), "bi_department": departments(tx, product),
              "bi_campaign": campaigns(seg)}
    reviews, themes = sentiment()
    if reviews is not None:
        tables["bi_review_sentiment"], tables["bi_review_themes"] = reviews, themes

    for name, df in tables.items():
        df.to_csv(OUT / f"{name}.csv", index=False)
        print(f"{name:<22} {len(df):>7,} rows")
    try:
        engine = create_engine(db_url(), connect_args={"connect_timeout": 3})
        for name, df in tables.items():
            df.to_sql(name, engine, schema="c360", if_exists="replace", index=False)
        print("also written to PostgreSQL c360.bi_*")
    except Exception as e:
        print(f"PostgreSQL not reachable ({e.__class__.__name__}): CSVs only")

    # Totals the report must reproduce (reconciliation card on the overview page)
    print(f"check: customer_paid total {tables['bi_weekly_segment'].customer_paid.sum():,.2f} "
          f"(warehouse validation: 8,014,851.54)")


if __name__ == "__main__":
    main()
