from __future__ import annotations

from datetime import date
from typing import Dict, Iterable

import pandas as pd

from .io import read_table_all, read_table_range, write_dwd_by_date

PREFIX_MAP = {
    "market_cbond.daily_price": "",
    "market_cbond.daily_twap": "",
    "market_cbond.daily_vwap": "",
    "market_cbond.daily_deriv": "deriv_",
    "market_cbond.daily_base": "base_",
    "market_cbond.daily_rating": "rating_",
    "metadata.cbond_info": "info_",
}


def build_dwd_daily(
    ods_root: str,
    dwd_root: str,
    start: date,
    end: date,
    *,
    primary_table: str,
    merge_tables: Iterable[str],
) -> None:
    primary = _load_table(ods_root, primary_table, start, end)
    if primary.empty:
        return
    primary = _standardize_daily(primary, prefix=PREFIX_MAP.get(primary_table, ""))

    merged = primary
    for table in merge_tables:
        if table == primary_table:
            continue
        other = _load_table(ods_root, table, start, end)
        if other.empty:
            continue
        other = _standardize_daily(other, prefix=PREFIX_MAP.get(table, ""))
        if table == "metadata.cbond_info":
            other = other.drop_duplicates(subset=["code"])
            merged = merged.merge(other, on="code", how="left")
        else:
            merged = merged.merge(other, on=["trade_date", "code"], how="left")

    write_dwd_by_date(merged, dwd_root, date_col="trade_date")


def _load_table(root: str, table: str, start: date, end: date) -> pd.DataFrame:
    if table == "metadata.cbond_info":
        return read_table_all(root, table)
    return read_table_range(root, table, start, end)


def _standardize_daily(df: pd.DataFrame, *, prefix: str) -> pd.DataFrame:
    work = df.copy()
    if "instrument_code" in work.columns and "exchange_code" in work.columns:
        work["code"] = work["instrument_code"].astype(str) + "." + work["exchange_code"].astype(str)
        work = work.drop(columns=["instrument_code", "exchange_code"])
    if prefix:
        skip = {"trade_date", "code"}
        rename = {col: f"{prefix}{col}" for col in work.columns if col not in skip}
        work = work.rename(columns=rename)
    return work
