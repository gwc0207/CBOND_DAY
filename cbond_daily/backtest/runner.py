from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import pandas as pd

from cbond_daily.data.io import read_dwd_daily
from .execution import apply_twap_bps


@dataclass
class BacktestResult:
    days: int = 0


def run_backtest(
    dwd_root: str,
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
        if df.empty or factor_col not in df.columns:
            continue
        ranked = df.sort_values(factor_col, ascending=False)
        picks = ranked.head(target_count)
        if len(picks) < min_count:
            continue
        buy_px = apply_twap_bps(picks[buy_twap_col], twap_bps, side="buy")
        sell_px = apply_twap_bps(picks[sell_twap_col], twap_bps, side="sell")
        _ = buy_px, sell_px, max_weight
        result.days += 1
    return result
