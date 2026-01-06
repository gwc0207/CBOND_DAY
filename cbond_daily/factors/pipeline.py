from __future__ import annotations

from datetime import date
from typing import Iterable, Sequence

import pandas as pd

from cbond_daily.data.io import (
    read_dwd_daily,
    read_dws_factors_daily,
    write_dws_factors_by_date,
)
from cbond_daily.core.naming import build_factor_col
from .base import FactorRegistry


def run_factor_pipeline(
    dwd_root: str,
    dws_root: str,
    start: date,
    end: date,
    factor_defs: Sequence[dict],
    *,
    update_only: Iterable[str] | None = None,
    overwrite: bool = False,
) -> None:
    update_only = set(update_only or [])
    for day in pd.date_range(start, end, freq="D"):
        df = read_dwd_daily(dwd_root, day.date())
        if df.empty:
            continue
        new_cols: dict[str, pd.Series] = {}
        for item in factor_defs:
            name = item["name"]
            col_name = build_factor_col(name, item.get("params"))
            if update_only and col_name not in update_only:
                continue
            factor_cls = FactorRegistry.get(name)
            factor = factor_cls(**(item.get("params") or {}))
            new_cols[col_name] = factor.compute(df)
        if not new_cols:
            continue
        if "code" not in df.columns:
            raise KeyError("missing code column in dwd data")
        if "trade_date" in df.columns:
            trade_dates = df["trade_date"]
        else:
            trade_dates = pd.Series([day.date()] * len(df), index=df.index)
        out_df = pd.DataFrame(
            {
                "trade_date": trade_dates,
                "code": df["code"],
                **new_cols,
            }
        )
        existing = read_dws_factors_daily(dws_root, day.date())
        if not existing.empty:
            existing_keyed = existing.set_index(["trade_date", "code"])
            out_keyed = out_df.set_index(["trade_date", "code"])
            if overwrite:
                out_df = out_keyed.combine_first(existing_keyed).reset_index()
            else:
                out_df = existing_keyed.combine_first(out_keyed).reset_index()
        write_dws_factors_by_date(out_df, dws_root, date_col="trade_date")
