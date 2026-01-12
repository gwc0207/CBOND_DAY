from __future__ import annotations

from datetime import date
from typing import Dict, Iterable, Mapping

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
    table_schemas: Mapping[str, dict] | None = None,
) -> None:
    schema = (table_schemas or {}).get(primary_table, {})
    primary = _load_table(ods_root, primary_table, start, end, schema)
    if primary.empty:
        return
    primary = _standardize_daily(primary, schema)

    merged = primary
    for table in merge_tables:
        if table == primary_table:
            continue
        schema = (table_schemas or {}).get(table, {})
        other = _load_table(ods_root, table, start, end, schema)
        if other.empty:
            continue
        other = _standardize_daily(other, schema)
        if "code" in other.columns:
            other = other.drop(columns=["code"])
        if table == "metadata.cbond_info":
            other = other.drop_duplicates(subset=["instrument_code", "exchange_code"])
            merged = merged.merge(other, on=["instrument_code", "exchange_code"], how="left")
        else:
            merged = merged.merge(
                other,
                on=["instrument_code", "exchange_code", "trade_date"],
                how="left",
            )

    write_dwd_by_date(merged, dwd_root, date_col="trade_date")


def _load_table(
    root: str,
    table: str,
    start: date,
    end: date,
    schema: Mapping[str, object] | None,
) -> pd.DataFrame:
    if table == "metadata.cbond_info":
        df = read_table_all(root, table)
    else:
        df = read_table_range(root, table, start, end)
    return _apply_schema(df, schema or {})


def _standardize_daily(df: pd.DataFrame, schema: Mapping[str, object]) -> pd.DataFrame:
    work = df.copy()
    if "instrument_code" in work.columns and "exchange_code" in work.columns:
        if "code" not in work.columns:
            work["code"] = (
                work["instrument_code"].astype(str) + "." + work["exchange_code"].astype(str)
            )
    prefix = str(schema.get("prefix", ""))
    if prefix:
        skip = {"trade_date", "instrument_code", "exchange_code", "code"}
        rename = {col: f"{prefix}{col}" for col in work.columns if col not in skip}
        work = work.rename(columns=rename)
    return work


def _apply_schema(df: pd.DataFrame, schema: Mapping[str, object]) -> pd.DataFrame:
    if df.empty:
        return df
    select_cols = schema.get("select_cols")
    if select_cols:
        missing = [c for c in select_cols if c not in df.columns]
        if missing:
            raise KeyError(f"missing columns in source: {missing}")
        df = df[list(select_cols)]
    rename_map = schema.get("rename_map") or {}
    if rename_map:
        df = df.rename(columns=rename_map)
    return df
