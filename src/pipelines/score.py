"""Phase 4 scoring: apply the saved production models to every household as of the last day (711).

Run from the project root (after python -m src.models.train):  python -m src.pipelines.score

Outputs data/processed/household_predictions.parquet and c360.household_predictions.
"""
import joblib
import numpy as np
import pandas as pd
from sqlalchemy import create_engine

from src.config import PROCESSED, ROOT, db_url
from src.features.household_features import build_features, load_inputs
from src.models.datasets import ACTIVE_WINDOW, DROP_COLS, ID_COLS

AS_OF_DAY = 711
ARTIFACTS = ROOT / "artifacts"


def load(task: str) -> dict:
    # joblib unpickles code: only ever load artifacts this project wrote itself (src.models.train), never downloaded ones
    return joblib.load(ARTIFACTS / f"{task}_model.joblib")


def matrix(features: pd.DataFrame, columns: list) -> pd.DataFrame:
    X = features.drop(columns=[c for c in ID_COLS + DROP_COLS if c in features])
    X = X.astype({c: float for c in X.columns if X[c].dtype == bool})
    return X[columns]


def main() -> None:
    features = build_features(load_inputs(), AS_OF_DAY)
    out = features[["household_key"]].copy()
    out["as_of_day"] = AS_OF_DAY

    churn = load("churn")
    active = (features.recency <= ACTIVE_WINDOW).values
    p = churn["model"].predict_proba(matrix(features, churn["features"]))[:, 1]
    # The churn model was trained only on active households; lapsed ones are already gone, not "at risk"
    out["is_active"] = active
    out["churn_probability"] = np.where(active, p, np.nan)
    out["churn_flag"] = np.where(active, (p >= churn["threshold"]).astype(int), pd.NA)

    clv = load("clv")
    out["predicted_spend_90d"] = np.clip(clv["model"].predict(matrix(features, clv["features"])), 0, None)

    # Scenario score: the data has no campaigns after day 711, so we ask
    # "if this household receives one campaign (not TypeA) next quarter, how likely is a redemption?"
    coupon = load("coupon_propensity")
    scenario = features.assign(upcoming_campaigns=1.0, upcoming_typea=0.0)
    out["coupon_redeem_probability"] = coupon["model"].predict_proba(matrix(scenario, coupon["features"]))[:, 1]
    # The coupon model only ever saw households marketing had targeted (heavy shoppers, A16);
    # scores for never-targeted households are an extrapolation and are flagged as such
    out["coupon_score_in_training_population"] = (features.campaigns_received > 0).values

    out["spend_at_risk_90d"] = out.churn_probability * out.predicted_spend_90d
    out["churn_risk_band"] = pd.cut(out.churn_probability, [-0.01, 0.05, churn["threshold"], 1.0],
                                    labels=["Low", "Medium", "High"]).astype("string").fillna("Lapsed")
    out["coupon_target_decile"] = pd.qcut(out.coupon_redeem_probability.rank(method="first", ascending=False),
                                          10, labels=range(1, 11)).astype(int)

    print(out.describe(include="all").T[["count", "mean", "min", "max"]].to_string())
    print("\nchurn risk bands:\n", out.churn_risk_band.value_counts().to_string())
    out.to_parquet(PROCESSED / "household_predictions.parquet", index=False)
    out.to_sql("household_predictions", create_engine(db_url()), schema="c360", if_exists="replace", index=False)
    print(f"\nwrote {len(out):,} households to c360.household_predictions")


if __name__ == "__main__":
    main()
