from __future__ import annotations

import json
from pathlib import Path
from typing import Optional
import warnings

import pandas as pd
import pyodbc

CONFIG_PATH = Path.home() / ".cbond_daily" / "mssql.json"

DATE_COLUMNS = {
    "market_cbond.daily_price": "trade_date",
    "market_cbond.daily_twap": "trade_date",
    "market_cbond.daily_vwap": "trade_date",
    "market_cbond.daily_deriv": "trade_date",
    "market_cbond.daily_base": "trade_date",
    "market_cbond.daily_rating": "trade_date",
    "market_stock.daily_price": "trade_date",
    "market_stock.daily_twap": "trade_date",
    "market_stock.daily_vwap": "trade_date",
    "market_index.daily_price": "trade_date",
}


def load_db_config() -> dict:
    if not CONFIG_PATH.exists():
        raise FileNotFoundError(f"missing config: {CONFIG_PATH}")
    with CONFIG_PATH.open("r", encoding="utf-8-sig") as handle:
        return json.load(handle)


def _build_conn_str(cfg: dict) -> str:
    driver = cfg.get("driver", "ODBC Driver 18 for SQL Server")
    database = cfg.get("database") or ""
    db_part = f"DATABASE={database};" if database else ""
    return (
        f"DRIVER={{{driver}}};"
        f"SERVER={cfg['server']};"
        + db_part
        + f"UID={cfg['username']};"
        + f"PWD={cfg['password']};"
        + "Encrypt=yes;"
        + "TrustServerCertificate=yes;"
    )


def connect() -> pyodbc.Connection:
    cfg = load_db_config()
    return pyodbc.connect(_build_conn_str(cfg), timeout=10)


def fetch_table(
    table: str,
    *,
    start: Optional[str] = None,
    end: Optional[str] = None,
) -> pd.DataFrame:
    date_col = DATE_COLUMNS.get(table)
    if start and end and date_col:
        sql = f"SELECT * FROM {table} WHERE {date_col} >= ? AND {date_col} <= ?"
        params = [start, end]
    else:
        sql = f"SELECT * FROM {table}"
        params = []
    with connect() as conn:
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="pandas only supports SQLAlchemy connectable",
                category=UserWarning,
            )
            return pd.read_sql(sql, conn, params=params)
