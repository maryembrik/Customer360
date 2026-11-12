# Customer360 API image. Build after the pipelines have produced artifacts/ and data/processed/:
#   docker compose up -d --build api      ->  http://localhost:8000/docs
FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
# LightGBM needs the OpenMP runtime
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements-api.txt .
RUN pip install --no-cache-dir -r requirements-api.txt

COPY src/ src/
COPY artifacts/*_model.joblib artifacts/
COPY reports/*.json reports/
# Parquet fallback when the database is not reachable (2,500 rows each)
COPY data/processed/household_predictions.parquet data/processed/household_segments.parquet data/processed/

RUN useradd --create-home api && chown -R api /app
USER api
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
