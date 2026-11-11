"""Out-of-time datasets for the three predictive targets.

For a cutoff day T:
  features = household features computed with data up to T (build_features(as_of_day=T))
  label    = what happens in (T, T + horizon]

Several training cutoffs are stacked to get more rows; validation and test use later cutoffs, so every
model is always evaluated on a future it has never seen (no random split, no leakage).
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.features.household_features import build_features

ID_COLS = ["household_key", "as_of_day"]
# Pure bookkeeping or redundant columns kept out of the models
DROP_COLS = ["revenue"]

ACTIVE_WINDOW = 60  # only households that bought in the 60 days before the cutoff are scored for churn


@dataclass(frozen=True)
class Task:
    name: str
    kind: str              # "classification" or "regression"
    horizon: int           # label window length in days
    train_cutoffs: tuple
    valid_cutoff: int
    test_cutoff: int
    description: str


TASKS = {
    "churn": Task("churn", "classification", 60, (260, 290, 320, 350, 380, 410, 440, 470, 500), 560, 620,
                  "Active household (bought in last 60 days) makes no purchase in the next 60 days"),
    "clv": Task("clv", "regression", 90, (230, 260, 290, 320, 350, 380, 410), 500, 590,
                "Customer spend (customer_paid) in the next 90 days"),
    "coupon_propensity": Task("coupon_propensity", "classification", 90, (230, 260, 290, 320, 350, 380), 470, 560,
                              "Household receiving a campaign in the next 90 days redeems at least one coupon"),
}

LAST_DAY = 711
for _t in TASKS.values():
    # Label windows must never overlap the next split: train labels end before validation starts, etc.
    assert max(_t.train_cutoffs) + _t.horizon <= _t.valid_cutoff, _t.name
    assert _t.valid_cutoff + _t.horizon <= _t.test_cutoff, _t.name
    assert _t.test_cutoff + _t.horizon <= LAST_DAY, _t.name
    assert min(_t.train_cutoffs) >= 224, _t.name  # >= 16 weeks after onboarding so spend_prev_8w is real


def backtest_folds(task: Task) -> list[tuple[list, int]]:
    """Rolling-origin folds for model selection: (train cutoffs whose labels end before the fold, fold cutoff)."""
    folds = []
    for c in list(task.train_cutoffs[1:]) + [task.valid_cutoff]:
        prior = [x for x in task.train_cutoffs if x + task.horizon <= c]
        if len(prior) >= 2:
            folds.append((prior, c))
    return folds


def _future(tx: pd.DataFrame, start: int, end: int) -> pd.DataFrame:
    return tx[(tx.day > start) & (tx.day <= end)]


def make_rows(inputs: dict, task: Task, cutoff: int, feature_cache: dict) -> pd.DataFrame:
    if cutoff not in feature_cache:
        feature_cache[cutoff] = build_features(inputs, cutoff)
    f = feature_cache[cutoff].copy()
    end = cutoff + task.horizon
    tx = inputs["tx"]

    if task.name == "churn":
        f = f[f.recency <= ACTIVE_WINDOW]
        buyers = _future(tx, cutoff, end).household_key.unique()
        f["target"] = (~f.household_key.isin(buyers)).astype(int)

    elif task.name == "clv":
        spend = _future(tx, cutoff, end).groupby("household_key").customer_paid.sum()
        f["target"] = f.household_key.map(spend).fillna(0.0)

    elif task.name == "coupon_propensity":
        camps = inputs["campaigns"]
        upcoming = camps.loc[(camps.start_day > cutoff) & (camps.start_day <= end), "campaign"]
        exposed = inputs["exposure"][inputs["exposure"].campaign.isin(upcoming)]
        # How many upcoming campaigns a household is scheduled to receive is known to marketing at T
        f["upcoming_campaigns"] = f.household_key.map(exposed.groupby("household_key").size())
        f["upcoming_typea"] = f.household_key.map(exposed[exposed.description == "TypeA"].groupby("household_key").size())
        f = f[f.upcoming_campaigns.notna()].fillna({"upcoming_typea": 0})
        red = inputs["redemptions"]
        redeemers = red[(red.day > cutoff) & (red.day <= end)].household_key.unique()
        f["target"] = f.household_key.isin(redeemers).astype(int)

    return f


def build_split(inputs: dict, task: Task, cutoffs, cache: dict) -> pd.DataFrame:
    return pd.concat([make_rows(inputs, task, c, cache) for c in cutoffs], ignore_index=True)


def xy(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    X = df.drop(columns=ID_COLS + DROP_COLS + ["target"])
    X = X.astype({c: float for c in X.columns if X[c].dtype == bool})
    return X, df.target


def summary(df: pd.DataFrame, task: Task) -> dict:
    if task.kind == "classification":
        return {"rows": len(df), "positive_rate": round(df.target.mean(), 3)}
    return {"rows": len(df), "target_mean": round(df.target.mean(), 1), "share_zero": round((df.target == 0).mean(), 3)}
