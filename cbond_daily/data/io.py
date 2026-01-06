from __future__ import annotations

from datetime import date, datetime
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
        path = Path(root) / month / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        group.to_parquet(path, index=False)


def read_dwd_daily(root: str | Path, day: date) -> pd.DataFrame:
    month = f"{day.year:04d}-{day.month:02d}"
    filename = f"{day.strftime('%Y%m%d')}.parquet"
    path = Path(root) / month / filename
    if not path.exists():
        return pd.DataFrame()
    return pd.read_parquet(path)


def get_latest_table_date(root: str | Path, table: str) -> date | None:
    base = _table_dir(root, table)
    if not base.exists():
        return None
    latest: date | None = None
    for path in base.glob("**/*.parquet"):
        if path.name == "all.parquet":
            continue
        try:
            day = datetime.strptime(path.stem, "%Y%m%d").date()
        except ValueError:
            continue
        if latest is None or day > latest:
            latest = day
    return latest


def table_has_data(root: str | Path, table: str) -> bool:
    base = _table_dir(root, table)
    if not base.exists():
        return False
    return any(base.glob("**/*.parquet"))


def get_latest_dwd_date(root: str | Path) -> date | None:
    base = Path(root)
    if not base.exists():
        return None
    latest: date | None = None
    for path in base.glob("**/*.parquet"):
        try:
            day = datetime.strptime(path.stem, "%Y%m%d").date()
        except ValueError:
            continue
        if latest is None or day > latest:
            latest = day
    return latest


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
        path = Path(root) / month / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        group.to_parquet(path, index=False)


def read_dws_factors_daily(root: str | Path, day: date) -> pd.DataFrame:
    month = f"{day.year:04d}-{day.month:02d}"
    filename = f"{day.strftime('%Y%m%d')}.parquet"
    path = Path(root) / month / filename
    if not path.exists():
        return pd.DataFrame()
    return pd.read_parquet(path)
