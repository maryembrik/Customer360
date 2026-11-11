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
jupyter lab
```

## Project status

- [x] Phase 1 — Business understanding & EDA (`notebooks/01_dunnhumby_eda.ipynb`)
- [ ] Phase 2 — ETL + PostgreSQL star schema
- [ ] Phase 3 — RFM & segmentation
- [ ] Phase 4 — Predictive models + SHAP
- [ ] Phase 5 — NLP (sentiment)
- [ ] Phase 6 — MLflow, FastAPI, Docker
- [ ] Phase 7 — Power BI

## Documentation

- [Data dictionary](docs/data_dictionary.md)
- [Data quality report](docs/data_quality_report.md)
- [Assumptions & limitations log](docs/assumptions_log.md)
