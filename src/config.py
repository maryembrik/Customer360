"""Project paths and database settings, shared by every pipeline step."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "dunnhumby"
PROCESSED = ROOT / "data" / "processed"
SQL = ROOT / "sql"

# Departments whose lines are fuel or non-product transactions (A12)
FUEL_DEPARTMENTS = {"KIOSK-GAS", "MISC SALES TRAN"}


def _load_env(path: Path = ROOT / ".env") -> None:
    """Read KEY=VALUE lines from .env into os.environ without overriding real env vars."""
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip())


def db_url(driver: str = "postgresql+psycopg") -> str:
    _load_env()
    e = os.environ
    return (f"{driver}://{e['POSTGRES_USER']}:{e['POSTGRES_PASSWORD']}"
            f"@{e.get('POSTGRES_HOST', 'localhost')}:{e.get('POSTGRES_PORT', '5433')}/{e['POSTGRES_DB']}")
