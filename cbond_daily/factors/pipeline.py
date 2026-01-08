from __future__ import annotations

from datetime import date, datetime
from pathlib import Path
from typing import Iterable, Sequence

import pandas as pd
from bisect import bisect_right

from cbond_daily.data.io import (
    read_dwd_daily,
    read_dws_factors_daily,
    write_dws_factors_by_date,
)
from cbond_daily.core.naming import build_factor_col
from .base import FactorRegistry


def _available_dwd_dates(root: str) -> list[date]:
    base = Path(root)
    dates: list[date] = []
    if not base.exists():
        return dates
    for path in base.glob("**/*.parquet"):
        try:
            day = datetime.strptime(path.stem, "%Y%m%d").date()
        except ValueError:
            continue
        dates.append(day)
    dates.sort()
    return dates


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
    factor_items: list[dict] = []
    for item in factor_defs:
        name = item["name"]
        col_name = build_factor_col(name, item.get("params"))
        if update_only and col_name not in update_only:
            continue
        factor_cls = FactorRegistry.get(name)
        factor_items.append(
            {
                "name": name,
                "params": item.get("params") or {},
                "col_name": col_name,
                "factor_cls": factor_cls,
            }
        )
    if not factor_items:
        return

    available_dates = _available_dwd_dates(dwd_root)
    day_list = [d for d in available_dates if start <= d <= end]
    total_tasks = len(day_list) * len(factor_items)
    done = 0
    last_pct = -1

    for day_date in day_list:
        df = read_dwd_daily(dwd_root, day_date)
        if df.empty:
            continue
        df = df.copy()
        df["_row_id"] = range(len(df))
        df["_is_target"] = True
        new_cols: dict[str, pd.Series] = {}
        for item in factor_items:
            col_name = item["col_name"]
            factor = item["factor_cls"](**item["params"])
            lookback = 1
            if hasattr(factor, "required_lookback"):
                lookback = max(1, int(factor.required_lookback()))
            if lookback <= 1:
                new_cols[col_name] = factor.compute(df)
                done += 1
                pct = int(done * 100 / total_tasks) if total_tasks else 100
                if pct != last_pct:
                    last_pct = pct
                    print(f"factor_pipeline progress: {pct}% ({done}/{total_tasks})")
                continue

            idx = bisect_right(available_dates, day_date) - 1
            if idx < 0:
                continue
            start_idx = max(0, idx - lookback + 1)
            window_dates = available_dates[start_idx : idx + 1]
            frames: list[pd.DataFrame] = []
            for wday in window_dates:
                if wday == day_date:
                    frames.append(df)
                    continue
                wdf = read_dwd_daily(dwd_root, wday)
                if wdf.empty:
                    continue
                wdf = wdf.copy()
                wdf["_row_id"] = pd.NA
                wdf["_is_target"] = False
                frames.append(wdf)
            if not frames:
                continue
            window_df = pd.concat(frames, ignore_index=True)
            series_all = factor.compute(window_df)
            mask = window_df["_is_target"] == True
            day_vals = series_all[mask]
            row_ids = window_df.loc[mask, "_row_id"].astype(int).to_numpy()
            series_day = pd.Series(day_vals.values, index=row_ids)
            series_day = series_day.reindex(df["_row_id"]).reset_index(drop=True)
            series_day.index = df.index
            new_cols[col_name] = series_day
            done += 1
            pct = int(done * 100 / total_tasks) if total_tasks else 100
            if pct != last_pct:
                last_pct = pct
                print(f"factor_pipeline progress: {pct}% ({done}/{total_tasks})")
        if not new_cols:
            continue
        if "code" not in df.columns:
            raise KeyError("missing code column in dwd data")
        if "trade_date" in df.columns:
            trade_dates = df["trade_date"]
        else:
            trade_dates = pd.Series([day_date] * len(df), index=df.index)
        out_df = pd.DataFrame(
            {
                "trade_date": trade_dates,
                "code": df["code"],
                **new_cols,
            }
        )
        existing = read_dws_factors_daily(dws_root, day_date)
        if not existing.empty:
            existing_keyed = existing.set_index(["trade_date", "code"])
            out_keyed = out_df.set_index(["trade_date", "code"])
            if overwrite:
                out_df = out_keyed.combine_first(existing_keyed).reset_index()
            else:
                out_df = existing_keyed.combine_first(out_keyed).reset_index()
        write_dws_factors_by_date(out_df, dws_root, date_col="trade_date")
