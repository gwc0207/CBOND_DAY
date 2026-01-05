from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import pandas as pd

from cbond_daily.data.io import read_dwd_daily, read_dws_factors_daily
from .execution import apply_twap_bps


@dataclass
class BacktestResult:
    days: int = 0


def run_backtest(
    dwd_root: str,
    dws_root: str,
    start: date,
    end: date,
    *,
    factor_col: str,
    buy_twap_col: str,
    sell_twap_col: str,
    target_count: int,
    min_count: int,
    max_weight: float,
    twap_bps: float,
) -> BacktestResult:
    result = BacktestResult()
    for day in pd.date_range(start, end, freq="D"):
        df = read_dwd_daily(dwd_root, day.date())
        if df.empty:
            continue
        factors = read_dws_factors_daily(dws_root, day.date())
        if factors.empty:
            continue
        if "trade_date" not in factors.columns:
            factors["trade_date"] = day.date()
        if "trade_date" not in df.columns:
            df["trade_date"] = day.date()
        merged = df.merge(factors, on=["trade_date", "code"], how="left")
        if factor_col not in merged.columns:
            continue
        ranked = merged.sort_values(factor_col, ascending=False)
        picks = ranked.head(target_count)
        if len(picks) < min_count:
            continue
        buy_px = apply_twap_bps(picks[buy_twap_col], twap_bps, side="buy")
        sell_px = apply_twap_bps(picks[sell_twap_col], twap_bps, side="sell")
        _ = buy_px, sell_px, max_weight
        result.days += 1
    return result
