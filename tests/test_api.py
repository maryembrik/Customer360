"""API tests on the Parquet data source (no database needed):  python -m pytest -q"""
import os

import pandas as pd
import pytest
from fastapi.testclient import TestClient

os.environ["C360_DATA_SOURCE"] = "parquet"
from src.api.main import app  # noqa: E402
from src.config import PROCESSED  # noqa: E402


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


@pytest.fixture(scope="module")
def features():
    return pd.read_parquet(PROCESSED / "household_features.parquet")


def test_health(client):
    r = client.get("/health").json()
    assert r["status"] == "ok" and r["households"] == 2500 and r["data_source"] == "parquet"
    assert {"churn", "clv", "coupon_propensity"} <= set(r["models_loaded"])


def test_household_found_and_missing(client):
    r = client.get("/households/1")
    assert r.status_code == 200 and r.json()["household_key"] == 1
    assert client.get("/households/999999").status_code == 404


def test_lapsed_household_has_no_churn_score(client):
    lapsed = client.get("/households", params={"risk_band": "Lapsed", "limit": 5}).json()
    assert lapsed and all(h["churn_probability"] is None and not h["is_active"] for h in lapsed)


def test_call_list_is_sorted_by_spend_at_risk(client):
    rows = client.get("/households", params={"risk_band": "High", "limit": 50}).json()
    values = [h["spend_at_risk_90d"] for h in rows]
    assert values == sorted(values, reverse=True) and all(h["churn_risk_band"] == "High" for h in rows)


def test_segments_cover_all_households(client):
    assert sum(s["households"] for s in client.get("/segments").json()) == 2500


def test_realtime_churn_matches_batch_score(client, features):
    """Same model + same features must give the same probability as the batch scoring job."""
    info = client.get("/model-info").json()["churn"]
    hh = client.get("/households", params={"risk_band": "High", "limit": 1}).json()[0]
    row = features[features.household_key == hh["household_key"]].iloc[0]
    payload = {f: float(row[f]) for f in info["features"]}
    r = client.post("/predict/churn", json={"features": payload}).json()
    assert r["value"] == pytest.approx(hh["churn_probability"], abs=1e-6) and r["flag"] == 1


def test_missing_feature_is_rejected(client):
    r = client.post("/predict/clv", json={"features": {"recency": 3.0}})
    assert r.status_code == 422 and "missing_features" in r.json()["detail"]


def test_sentiment_endpoint(client):
    if "sentiment" not in client.get("/health").json()["models_loaded"]:
        pytest.skip("sentiment model not trained yet")
    out = client.post("/predict/sentiment", json={"texts": ["Delicious, will buy again!",
                                                            "Arrived crushed and stale. Waste of money."]}).json()
    assert out[0]["label"] == "positive" and out[1]["label"] == "negative"
    assert abs(sum(out[0]["probabilities"].values()) - 1) < 1e-3
