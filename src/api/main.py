"""Phase 6: Customer360 prediction API.

    uvicorn src.api.main:app --port 8000        # docs at http://localhost:8000/docs

Household endpoints serve the batch scores written by `python -m src.pipelines.score` (as of day 711),
joined with the segments. `/predict/*` endpoints score new feature rows / review texts in real time with the
same saved models. Data source: PostgreSQL (star schema c360) when reachable, otherwise the Parquet copies
in data/processed/ (set C360_DATA_SOURCE=parquet to force it).
"""
import json
import os
from contextlib import asynccontextmanager
from typing import Literal

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field

from src.config import PROCESSED, ROOT, db_url

ARTIFACTS = ROOT / "artifacts"
REPORTS = ROOT / "reports"
STATE: dict = {}


def load_tables() -> tuple[pd.DataFrame, str]:
    if os.environ.get("C360_DATA_SOURCE", "auto") != "parquet":
        try:
            from sqlalchemy import create_engine
            engine = create_engine(db_url(), connect_args={"connect_timeout": 3})
            df = pd.read_sql("""select p.*, s.rfm_segment, s.cluster_name as segment, s.r_score, s.f_score, s.m_score
                                from c360.household_predictions p join c360.household_segment s using (household_key)""",
                             engine)
            return df, "postgres"
        except Exception as e:  # DB down or not configured: the API still serves the same scores from Parquet
            if os.environ.get("C360_DATA_SOURCE") == "postgres":
                raise
            print(f"Postgres unavailable ({e.__class__.__name__}); using Parquet")
    pred = pd.read_parquet(PROCESSED / "household_predictions.parquet")
    seg = pd.read_parquet(PROCESSED / "household_segments.parquet").rename(columns={"cluster_name": "segment"})
    return pred.merge(seg[["household_key", "rfm_segment", "segment", "r_score", "f_score", "m_score"]],
                      on="household_key"), "parquet"


def load_model(name: str):
    # joblib unpickles code: only load artifacts this project trained itself, never downloaded ones
    path = ARTIFACTS / f"{name}_model.joblib"
    return joblib.load(path) if path.exists() else None


@asynccontextmanager
async def lifespan(app: FastAPI):
    STATE["households"], STATE["source"] = load_tables()
    STATE["households"] = STATE["households"].set_index("household_key").sort_index()
    for name in ["churn", "clv", "coupon_propensity", "sentiment"]:
        STATE[name] = load_model(name)
    yield
    STATE.clear()


app = FastAPI(title="Customer360 API", version="1.0.0", lifespan=lifespan,
              description="Segments, churn risk, 90-day spend, coupon propensity and review sentiment.")


# ---------- schemas ----------

class Household(BaseModel):
    household_key: int
    segment: str
    rfm_segment: str
    is_active: bool
    churn_probability: float | None = Field(None, description="P(no purchase in next 60 days); null if lapsed")
    churn_risk_band: Literal["Low", "Medium", "High", "Lapsed"]
    predicted_spend_90d: float
    spend_at_risk_90d: float | None
    coupon_redeem_probability: float
    coupon_target_decile: int
    coupon_score_in_training_population: bool


class FeatureRow(BaseModel):
    features: dict[str, float] = Field(..., description="Household features as of a day, see GET /model-info")


class ScoreOut(BaseModel):
    model: str
    value: float
    flag: int | None = None


class Reviews(BaseModel):
    texts: list[str] = Field(..., min_length=1, max_length=1000)


class SentimentOut(BaseModel):
    text: str
    label: Literal["negative", "neutral", "positive"]
    probabilities: dict[str, float]


# ---------- helpers ----------

def _none(v):
    return None if v is None or (isinstance(v, float) and np.isnan(v)) else v


def _household(key: int, row: pd.Series) -> Household:
    return Household(household_key=key, **{k: _none(row[k]) for k in Household.model_fields if k != "household_key"})


def _matrix(model: dict, features: dict[str, float]) -> pd.DataFrame:
    missing = sorted(set(model["features"]) - set(features))
    extra = sorted(set(features) - set(model["features"]))
    if missing or extra:
        raise HTTPException(422, {"missing_features": missing, "unknown_features": extra})
    return pd.DataFrame([features])[model["features"]].astype(float)


def _require(name: str) -> dict:
    if STATE.get(name) is None:
        raise HTTPException(503, f"{name} model not trained yet")
    return STATE[name]


# ---------- endpoints ----------

@app.get("/health")
def health():
    return {"status": "ok", "data_source": STATE["source"], "households": len(STATE["households"]),
            "models_loaded": [n for n in ["churn", "clv", "coupon_propensity", "sentiment"] if STATE.get(n)]}


@app.get("/households/{household_key}", response_model=Household)
def get_household(household_key: int):
    df = STATE["households"]
    if household_key not in df.index:
        raise HTTPException(404, f"household {household_key} not found")
    return _household(household_key, df.loc[household_key])


@app.get("/households", response_model=list[Household])
def list_households(risk_band: Literal["Low", "Medium", "High", "Lapsed"] | None = None,
                    segment: str | None = None,
                    sort_by: Literal["spend_at_risk_90d", "churn_probability", "predicted_spend_90d",
                                     "coupon_redeem_probability"] = "spend_at_risk_90d",
                    limit: int = Query(20, ge=1, le=500)):
    """Call lists, e.g. ?risk_band=High&sort_by=spend_at_risk_90d gives who to contact first."""
    df = STATE["households"]
    if risk_band:
        df = df[df.churn_risk_band == risk_band]
    if segment:
        df = df[df.segment == segment]
    df = df.sort_values(sort_by, ascending=False, na_position="last").head(limit)
    return [_household(k, r) for k, r in df.iterrows()]


@app.get("/segments")
def segments():
    df = STATE["households"]
    g = df.groupby("segment").agg(households=("is_active", "size"), share_active=("is_active", "mean"),
                                  mean_churn_probability=("churn_probability", "mean"),
                                  predicted_spend_90d=("predicted_spend_90d", "sum"),
                                  spend_at_risk_90d=("spend_at_risk_90d", "sum"),
                                  mean_coupon_probability=("coupon_redeem_probability", "mean"))
    return g.round(4).reset_index().to_dict(orient="records")


@app.post("/predict/churn", response_model=ScoreOut)
def predict_churn(row: FeatureRow):
    m = _require("churn")
    p = float(m["model"].predict_proba(_matrix(m, row.features))[:, 1][0])
    return ScoreOut(model="churn_60d", value=p, flag=int(p >= m["threshold"]))


@app.post("/predict/clv", response_model=ScoreOut)
def predict_clv(row: FeatureRow):
    m = _require("clv")
    return ScoreOut(model="spend_90d", value=max(float(m["model"].predict(_matrix(m, row.features))[0]), 0.0))


@app.post("/predict/coupon", response_model=ScoreOut)
def predict_coupon(row: FeatureRow):
    m = _require("coupon_propensity")
    return ScoreOut(model="coupon_redemption_90d",
                    value=float(m["model"].predict_proba(_matrix(m, row.features))[:, 1][0]))


@app.post("/predict/sentiment", response_model=list[SentimentOut])
def predict_sentiment(body: Reviews):
    m = _require("sentiment")
    proba = m["model"].predict_proba(body.texts)
    classes = list(m["classes"])
    return [SentimentOut(text=t, label=classes[int(np.argmax(p))],
                         probabilities={c: round(float(v), 4) for c, v in zip(classes, p)})
            for t, p in zip(body.texts, proba)]


@app.get("/model-info")
def model_info():
    info = {n: {"features": STATE[n]["features"], "threshold": STATE[n].get("threshold")}
            for n in ["churn", "clv", "coupon_propensity"] if STATE.get(n)}
    for f in ["model_results.json", "sentiment_results.json"]:
        if (REPORTS / f).exists():
            info[f.removesuffix(".json")] = json.loads((REPORTS / f).read_text())
    return info
