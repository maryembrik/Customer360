"""Phase 3 pipeline: features as of the last day -> RFM + K-Means segments -> Parquet + Postgres.

Run from the project root:  python -m src.pipelines.segment
"""
import joblib
import pandas as pd
from sqlalchemy import create_engine

from src.config import PROCESSED, ROOT, db_url
from src.features.household_features import build_features, load_inputs
from src.models import segmentation as seg

AS_OF_DAY = 711
K = 4
ARTIFACTS = ROOT / "artifacts"


def main() -> None:
    features = build_features(load_inputs(), AS_OF_DAY)

    rfm = seg.rfm_scores(features)
    x, scaler = seg.kmeans_matrix(features)
    km = seg.fit_kmeans(x, K)
    segments = rfm.assign(cluster=km.labels_, cluster_name=seg.name_clusters(features, km.labels_).values)

    print(seg.profile(features, segments.cluster_name, "cluster_name").to_string(), "\n")
    print(pd.crosstab(segments.rfm_segment, segments.cluster_name).to_string())

    features.to_parquet(PROCESSED / "household_features.parquet", index=False)
    segments.to_parquet(PROCESSED / "household_segments.parquet", index=False)
    ARTIFACTS.mkdir(exist_ok=True)
    joblib.dump({"scaler": scaler, "kmeans": km, "features": seg.KMEANS_FEATURES}, ARTIFACTS / "kmeans_segments.joblib")

    engine = create_engine(db_url())
    features.to_sql("mart_household_features", engine, schema="c360", if_exists="replace", index=False)
    segments.to_sql("household_segment", engine, schema="c360", if_exists="replace", index=False)
    print(f"\nwrote {len(features):,} households to c360.mart_household_features and c360.household_segment")


if __name__ == "__main__":
    main()
