from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Iterable, Optional

import pandas as pd


def _table_dir(root: str | Path, table: str) -> Path:
    safe = table.replace(".", "__")
    return Path(root) / safe


def _day_path(root: str | Path, table: str, day: date) -> Path:
    base = _table_dir(root, table)
    month = f"{day.year:04d}-{day.month:02d}"
    filename = f"{day.strftime('%Y%m%d')}.parquet"
    return base / month / filename


def write_table_by_date(
    df: pd.DataFrame,
    root: str | Path,
    table: str,
    *,
    date_col: str = "trade_date",
) -> None:
    if date_col not in df.columns:
        path = _table_dir(root, table) / "all.parquet"
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(path, index=False)
        return

    work = df.copy()
    work[date_col] = pd.to_datetime(work[date_col]).dt.date
    for day, group in work.groupby(date_col):
        path = _day_path(root, table, day)
        path.parent.mkdir(parents=True, exist_ok=True)
        group.to_parquet(path, index=False)


def read_table_range(
    root: str | Path,
    table: str,
    start: date,
    end: date,
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for day in pd.date_range(start, end, freq="D"):
        path = _day_path(root, table, day.date())
        if path.exists():
            frames.append(pd.read_parquet(path))
    if frames:
        return pd.concat(frames, ignore_index=True)
    return pd.DataFrame()


def read_table_all(root: str | Path, table: str) -> pd.DataFrame:
    path = _table_dir(root, table) / "all.parquet"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_parquet(path)


def write_dwd_by_date(
    df: pd.DataFrame,
    root: str | Path,
    *,
    date_col: str = "trade_date",
) -> None:
    if date_col not in df.columns:
        raise KeyError(f"missing {date_col}")
    work = df.copy()
    work[date_col] = pd.to_datetime(work[date_col]).dt.date
    for day, group in work.groupby(date_col):
        month = f"{day.year:04d}-{day.month:02d}"
        filename = f"{day.strftime('%Y%m%d')}.parquet"
        path = Path(root) / "dwd_daily" / month / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        group.to_parquet(path, index=False)


def read_dwd_daily(root: str | Path, day: date) -> pd.DataFrame:
    month = f"{day.year:04d}-{day.month:02d}"
    filename = f"{day.strftime('%Y%m%d')}.parquet"
    path = Path(root) / "dwd_daily" / month / filename
    if not path.exists():
        return pd.DataFrame()
    return pd.read_parquet(path)


def write_dws_factors_by_date(
    df: pd.DataFrame,
    root: str | Path,
    *,
    date_col: str = "trade_date",
) -> None:
    if date_col not in df.columns:
        raise KeyError(f"missing {date_col}")
    work = df.copy()
    work[date_col] = pd.to_datetime(work[date_col]).dt.date
    for day, group in work.groupby(date_col):
        month = f"{day.year:04d}-{day.month:02d}"
        filename = f"{day.strftime('%Y%m%d')}.parquet"
        path = Path(root) / "dws_factors" / month / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        group.to_parquet(path, index=False)


def read_dws_factors_daily(root: str | Path, day: date) -> pd.DataFrame:
    month = f"{day.year:04d}-{day.month:02d}"
    filename = f"{day.strftime('%Y%m%d')}.parquet"
    path = Path(root) / "dws_factors" / month / filename
    if not path.exists():
        return pd.DataFrame()
    return pd.read_parquet(path)
