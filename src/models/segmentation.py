"""Phase 3: customer segmentation, two complementary ways.

1. RFM scoring: quintile scores 1-5 and the standard 10 named RFM segments (rule-based, easy to explain).
2. K-Means: clusters discovered from behaviour (log-scaled, standardized features).
"""
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

KMEANS_FEATURES = ["recency", "frequency", "monetary", "avg_basket_value"]
RANDOM_STATE = 42


# ---------- 1. RFM ----------

def quintile(s: pd.Series, ascending: bool = True) -> pd.Series:
    """Score 1-5 by quintile; rank first so heavy ties (e.g. recency 0-2 days) still split evenly."""
    ranks = s.rank(method="first", ascending=ascending)
    return pd.qcut(ranks, 5, labels=[1, 2, 3, 4, 5]).astype(int)


def rfm_segment(r: int, fm: int) -> str:
    """Recency score x combined frequency/monetary score -> segment (standard RFM grid)."""
    if r <= 2:
        return "Hibernating" if fm <= 2 else ("At Risk" if fm <= 4 else "Can't Lose Them")
    if r == 3:
        return "Need Attention" if fm <= 2 else ("About to Sleep" if fm == 3 else "Loyal Customers")
    if fm >= 4:
        return "Champions" if r == 5 else "Loyal Customers"
    if fm >= 2:
        return "Potential Loyalist"
    return "New Customers" if r == 5 else "Promising"


def rfm_scores(features: pd.DataFrame) -> pd.DataFrame:
    out = features[["household_key"]].copy()
    out["r_score"] = quintile(features.recency, ascending=False)   # recent = high score
    out["f_score"] = quintile(features.frequency)
    out["m_score"] = quintile(features.monetary)
    out["fm_score"] = np.ceil((out.f_score + out.m_score) / 2).astype(int)
    out["rfm_segment"] = [rfm_segment(r, fm) for r, fm in zip(out.r_score, out.fm_score)]
    return out


# ---------- 2. K-Means ----------

def kmeans_matrix(features: pd.DataFrame) -> tuple[np.ndarray, StandardScaler]:
    # log1p tames the long right tails (monetary up to $38K); scaling stops dollars dominating distances
    x = np.log1p(features[KMEANS_FEATURES].clip(lower=0))
    scaler = StandardScaler().fit(x)
    return scaler.transform(x), scaler


def evaluate_k(x: np.ndarray, ks=range(2, 9)) -> pd.DataFrame:
    rows = []
    for k in ks:
        km = KMeans(n_clusters=k, n_init=10, random_state=RANDOM_STATE).fit(x)
        rows.append({"k": k, "inertia": km.inertia_,
                     "silhouette": silhouette_score(x, km.labels_, sample_size=2500, random_state=RANDOM_STATE)})
    return pd.DataFrame(rows).set_index("k")


def fit_kmeans(x: np.ndarray, k: int) -> KMeans:
    return KMeans(n_clusters=k, n_init=20, random_state=RANDOM_STATE).fit(x)


def name_clusters(features: pd.DataFrame, labels: np.ndarray) -> pd.Series:
    """Name the k=4 clusters from their profiles, so names don't depend on K-Means label numbers."""
    med = features.assign(c=labels).groupby("c")[["monetary", "avg_basket_value"]].median()
    order = med.monetary.sort_values()
    names = {order.index[0]: "Lapsing Low-Value", order.index[-1]: "VIP Regulars"}
    middle = med.loc[order.index[1:-1]].avg_basket_value.sort_values()
    names[middle.index[0]] = "Frequent Top-Up Shoppers"   # many small baskets
    names[middle.index[-1]] = "Occasional Stock-Up Shoppers"  # fewer, bigger baskets
    return pd.Series(labels).map(names)


def profile(features: pd.DataFrame, labels: pd.Series, by: str) -> pd.DataFrame:
    df = features.assign(**{by: labels.values})
    p = df.groupby(by).agg(
        households=("household_key", "size"),
        recency_median=("recency", "median"),
        frequency_median=("frequency", "median"),
        monetary_median=("monetary", "median"),
        avg_basket_median=("avg_basket_value", "median"),
        trips_per_week=("trips_per_week", "median"),
        total_monetary=("monetary", "sum"),
    )
    p["pct_households"] = (p.households / p.households.sum()).round(3)
    p["pct_monetary"] = (p.total_monetary / p.total_monetary.sum()).round(3)
    return p.drop(columns="total_monetary").sort_values("monetary_median", ascending=False)
