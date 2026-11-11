# Customer360

AI-powered customer intelligence and marketing analytics platform: transactional, promotional and review data turned into segments, predictions and dashboards.

## Datasets

| Dataset | Role | Location |
|---|---|---|
| dunnhumby — The Complete Journey | Households, transactions, products, campaigns, coupons | `data/raw/dunnhumby/` |
| Amazon Reviews 2023 — Grocery and Gourmet Food | Review text for NLP (30K sample in `data/sampled/`) | `data/raw/amazon_reviews/` |

The two datasets describe different people. They run as separate pipelines and are never joined at the customer level.

Raw data is not committed. Download it yourself and place the files in the folders above.

## Setup

```bash
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
copy .env.example .env        # then set a password
docker compose up -d          # PostgreSQL 16 on localhost:5433
```

## Data pipeline (Phase 2)

```bash
python -m src.etl.clean      # raw CSV -> data/processed/*.parquet
python -m src.etl.load       # Parquet -> PostgreSQL star schema c360 (COPY + indexes)
python -m src.etl.validate   # row counts and money totals vs raw files
```

Star schema (`sql/schema.sql`):

| Table | Grain | Rows |
|---|---|---|
| `fact_transaction` | one product line on a receipt | 2,586,313 |
| `fact_campaign_exposure` | household × campaign received | 7,208 |
| `fact_coupon_redemption` | one coupon redemption | 2,318 |
| `campaign_coupon` | campaign × coupon × product | 119,384 |
| `dim_household` | household (with masked demographics when available) | 2,500 |
| `dim_product` | product, with `is_fuel` flag | 92,353 |
| `dim_day` | relative day 1–711, week, onboarding flag | 711 |
| `dim_store` / `dim_campaign` | store / campaign | 582 / 30 |

Example queries: `sql/example_queries.sql`.

## Segmentation (Phase 3)

```bash
python -m src.pipelines.segment   # features -> RFM + K-Means -> c360.household_segment
```

| Segment (K-Means, k=4) | Households | Share of spend |
|---|---|---|
| VIP Regulars | 31% | 65% |
| Occasional Stock-Up Shoppers | 27% | 19% |
| Frequent Top-Up Shoppers | 25% | 14% |
| Lapsing Low-Value | 17% | 2% |

Tables written to Postgres: `c360.mart_household_features` (27 features per household) and
`c360.household_segment` (RFM scores, RFM segment, cluster name).

## Predictive models (Phase 4)

```bash
python -m src.models.train      # out-of-time datasets, backtest selection, test report, plots, SHAP, MLflow
python -m src.pipelines.score   # score all households as of day 711 -> c360.household_predictions
mlflow ui --backend-store-uri sqlite:///mlruns/mlflow.db   # browse every run
```

All evaluation is out-of-time: features up to a cutoff day, labels from the days after it, no label overlap
between train / validation / test, model choice on a rolling backtest, test touched once (A17).

| Task | Selected model | Test result (95% CI) | Baseline |
|---|---|---|---|
| Churn: active household buys nothing in next 60 days (6% positive) | Avg. of logistic regression + XGBoost | PR-AUC 0.366 (0.29–0.44), ROC-AUC 0.88, top-decile lift 5.3× | random PR-AUC 0.062 |
| Spend in next 90 days | LightGBM, Tweedie loss | MAE $166.5 (160–174), R² 0.77 | run-rate rule MAE $179.6 |
| Coupon redemption given a campaign (20% positive) | Avg. of logistic regression + XGBoost | PR-AUC 0.606 (0.54–0.66), ROC-AUC 0.83, top-decile lift 3.6× | random PR-AUC 0.196 |

Diagnostic plots (ROC, PR, calibration, lift, gains, confusion matrix, predicted vs actual, residuals) and SHAP
plots are in `reports/figures/`; the walkthrough is `notebooks/03_predictive_models.ipynb`.
Known limitations are logged in A20–A21.

## Project status

- [x] Phase 1 — Business understanding & EDA (`notebooks/01_dunnhumby_eda.ipynb`)
- [x] Phase 2 — ETL + PostgreSQL star schema (`src/etl/`, `sql/`)
- [x] Phase 3 — RFM & segmentation (`notebooks/02_segmentation.ipynb`, `python -m src.pipelines.segment`)
- [x] Phase 4 — Predictive models + SHAP + MLflow tracking (`notebooks/03_predictive_models.ipynb`)
- [ ] Phase 5 — NLP (sentiment)
- [ ] Phase 6 — MLflow, FastAPI, Docker
- [ ] Phase 7 — Power BI

## Documentation

- [Data dictionary](docs/data_dictionary.md)
- [Data quality report](docs/data_quality_report.md)
- [Assumptions & limitations log](docs/assumptions_log.md)
