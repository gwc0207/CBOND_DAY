from __future__ import annotations

from datetime import date
from typing import Iterable, Sequence
from bisect import bisect_right

import pandas as pd

from cbond_daily.data.io import (
    read_dwd_daily,
    read_dws_factors_daily,
    read_trading_calendar,
    write_dws_factors_by_date,
)
from cbond_daily.core.naming import build_factor_col
from .base import FactorRegistry


def _load_trading_days(ods_root: str) -> list[date]:
    cal = read_trading_calendar(ods_root)
    if cal.empty:
        raise ValueError("trading_calendar is empty; sync raw_data first")
    if "calendar_date" not in cal.columns:
        raise KeyError("trading_calendar missing calendar_date column")
    if "is_open" in cal.columns:
        cal = cal[cal["is_open"].astype(bool)]
    days = pd.to_datetime(cal["calendar_date"]).dt.date.dropna().unique().tolist()
    days.sort()
    return days


def _align_to_next_trading_day(trading_days: list[date], day: date, label: str) -> date:
    if day in trading_days:
        return day
    idx = bisect_right(trading_days, day)
    if idx >= len(trading_days):
        raise ValueError(f"{label} {day} is after last trading day")
    return trading_days[idx]


def run_factor_pipeline(
    ods_root: str,
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
        factor = factor_cls(**(item.get("params") or {}))
        lookback = 1
        if hasattr(factor, "required_lookback"):
            lookback = max(1, int(factor.required_lookback()))
        factor_items.append(
            {
                "name": name,
                "params": item.get("params") or {},
                "col_name": col_name,
                "factor_cls": factor_cls,
                "lookback": lookback,
            }
        )
    if not factor_items:
        return

    trading_days = _load_trading_days(ods_root)
    start = _align_to_next_trading_day(trading_days, start, "start date")
    end = _align_to_next_trading_day(trading_days, end, "end date")
    if end < start:
        raise ValueError(f"end date {end} is before start date {start}")
    target_days = [d for d in trading_days if start <= d <= end]
    if not target_days:
        raise ValueError("no trading days in requested range")

    max_lookback = max(item["lookback"] for item in factor_items)
    start_idx = trading_days.index(start)
    end_idx = trading_days.index(end)
    need_start_idx = start_idx - max_lookback + 1
    if need_start_idx < 0:
        raise ValueError(
            f"insufficient trading days for lookback={max_lookback} before {start}"
        )
    window_days = trading_days[need_start_idx : end_idx + 1]

    frames: list[pd.DataFrame] = []
    for day_date in window_days:
        df = read_dwd_daily(dwd_root, day_date)
        if df.empty:
            raise ValueError(f"missing dwd data for {day_date}")
        frames.append(df)
    full_df = pd.concat(frames, ignore_index=True)
    if "code" not in full_df.columns:
        raise KeyError("missing code column in dwd data")
    if "trade_date" not in full_df.columns:
        raise KeyError("missing trade_date column in dwd data")

    full_df = full_df.copy()
    full_df["trade_date"] = pd.to_datetime(full_df["trade_date"]).dt.date
    target_set = set(target_days)
    mask = full_df["trade_date"].isin(target_set)
    if not mask.any():
        raise ValueError("no rows found for target trading days")

    out_df = full_df.loc[mask, ["trade_date", "code"]].copy()

    total_tasks = len(factor_items)
    done = 0
    last_pct = -1
    for item in factor_items:
        col_name = item["col_name"]
        factor = item["factor_cls"](**item["params"])
        series_all = factor.compute(full_df)
        if len(series_all) != len(full_df):
            raise ValueError(f"factor {item['name']} returned invalid length")
        series_vals = pd.Series(series_all).loc[mask].to_numpy()
        out_df[col_name] = series_vals
        done += 1
        pct = int(done * 100 / total_tasks) if total_tasks else 100
        if pct % 20 == 0 and pct != last_pct:
            last_pct = pct
            print(f"factor_pipeline progress: {pct}% ({done}/{total_tasks})")

    for day_date in target_days:
        day_df = out_df[out_df["trade_date"] == day_date]
        if day_df.empty:
            raise ValueError(f"missing factor output for {day_date}")
        existing = read_dws_factors_daily(dws_root, day_date)
        if not existing.empty:
            existing_keyed = existing.set_index(["trade_date", "code"])
            out_keyed = day_df.set_index(["trade_date", "code"])
            if overwrite:
                day_df = out_keyed.combine_first(existing_keyed).reset_index()
            else:
                day_df = existing_keyed.combine_first(out_keyed).reset_index()
        write_dws_factors_by_date(day_df, dws_root, date_col="trade_date")
