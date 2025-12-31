from __future__ import annotations

from datetime import date
from typing import Iterable, Sequence

import pandas as pd

from cbond_daily.data.io import read_dwd_daily, write_dwd_by_date
from .base import FactorRegistry


def run_factor_pipeline(
    dwd_root: str,
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
            if update_only and name not in update_only:
                continue
            factor_cls = FactorRegistry.get(name)
            factor = factor_cls(**(item.get("params") or {}))
            new_cols[name] = factor.compute(df)
        if not new_cols:
            continue
        factor_df = pd.DataFrame(new_cols)
        merged = df.copy()
        for col in factor_df.columns:
            if overwrite or col not in merged.columns:
                merged[col] = factor_df[col]
            else:
                merged[col] = merged[col].combine_first(factor_df[col])
        write_dwd_by_date(merged, dwd_root, date_col="trade_date")
