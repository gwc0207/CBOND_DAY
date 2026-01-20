from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd


def load_scores_by_date(score_path: str | Path) -> dict[date, pd.DataFrame]:
    path = Path(score_path)
    if not path.exists() or path.stat().st_size == 0:
        raise FileNotFoundError(f"score file not found or empty: {path}")
    try:
        df = pd.read_csv(path, parse_dates=["trade_date"])
    except pd.errors.EmptyDataError as exc:
        raise FileNotFoundError(f"score file not found or empty: {path}") from exc
    required = {"trade_date", "code", "score"}
    if not required.issubset(df.columns):
        missing = sorted(required - set(df.columns))
        raise KeyError(f"score file missing columns: {missing}")
    df = df.copy()
    df["trade_date"] = pd.to_datetime(df["trade_date"]).dt.date
    cache: dict[date, pd.DataFrame] = {}
    for day, group in df.groupby("trade_date"):
        cache[day] = group[["code", "score"]].copy()
    return cache
