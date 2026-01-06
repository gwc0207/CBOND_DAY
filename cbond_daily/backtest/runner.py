from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from bisect import bisect_left
from pathlib import Path

import pandas as pd

from cbond_daily.data.io import read_dwd_daily, read_dws_factors_daily
from .execution import apply_twap_bps


@dataclass
class BacktestResult:
    days: int = 0
    daily_returns: pd.DataFrame | None = None
    nav_curve: pd.DataFrame | None = None
    benchmark_curve: pd.DataFrame | None = None
    positions: pd.DataFrame | None = None
    diagnostics: pd.DataFrame | None = None


def _available_factor_dates(dws_root: str) -> list[date]:
    base = Path(dws_root)
    dates: list[date] = []
    for path in base.glob("*/*.parquet"):
        try:
            day = datetime.strptime(path.stem, "%Y%m%d").date()
        except ValueError:
            continue
        dates.append(day)
    dates.sort()
    return dates


def _prev_available_date(dates: list[date], day: date) -> date | None:
    idx = bisect_left(dates, day)
    if idx <= 0:
        return None
    return dates[idx - 1]


def run_backtest(
    dwd_root: str,
    dws_root: str,
    start: date,
    end: date,
    *,
    factor_col: str,
    buy_twap_col: str,
    sell_twap_col: str,
    min_count: int,
    max_weight: float,
    twap_bps: float,
    bin_count: int | None = None,
    bin_select: list[int] | None = None,
) -> BacktestResult:
    result = BacktestResult()
    daily_records: list[dict] = []
    position_records: list[dict] = []
    diagnostics: list[dict] = []
    factor_dates = _available_factor_dates(dws_root)
    for day in pd.date_range(start, end, freq="D"):
        df = read_dwd_daily(dwd_root, day.date())
        if df.empty:
            diagnostics.append({"trade_date": day.date(), "status": "skip", "reason": "missing_dwd"})
            continue
        factor_day = _prev_available_date(factor_dates, day.date())
        if factor_day is None:
            diagnostics.append({"trade_date": day.date(), "status": "skip", "reason": "missing_factor"})
            continue
        factors = read_dws_factors_daily(dws_root, factor_day)
        if factors.empty:
            diagnostics.append({"trade_date": day.date(), "status": "skip", "reason": "missing_factor"})
            continue
        if "code" not in factors.columns:
            raise KeyError("missing code column in factor data")
        if "trade_date" not in df.columns:
            df["trade_date"] = day.date()
        factor_cols = [c for c in factors.columns if c != "trade_date"]
        merged = df.merge(factors[factor_cols], on="code", how="left")
        if factor_col not in merged.columns:
            diagnostics.append(
                {"trade_date": day.date(), "status": "skip", "reason": "factor_col_missing"}
            )
            continue
        missing_cols = [c for c in (buy_twap_col, sell_twap_col) if c not in merged.columns]
        if missing_cols:
            raise KeyError(f"missing price columns: {missing_cols}")
        tradable = merged[
            merged[buy_twap_col].notna()
            & merged[sell_twap_col].notna()
            & (merged[buy_twap_col] > 0)
            & (merged[sell_twap_col] > 0)
        ]
        if tradable.empty:
            diagnostics.append({"trade_date": day.date(), "status": "skip", "reason": "no_tradable"})
            continue
        if bin_select is None:
            raise ValueError("bin_select is required for bin-based selection")
        if bin_count is None or bin_count <= 1:
            raise ValueError("bin_count must be > 1 for bin-based selection")
        try:
            bins_cat = pd.qcut(
                tradable[factor_col],
                bin_count,
                labels=False,
                duplicates="drop",
            )
        except ValueError:
            diagnostics.append({"trade_date": day.date(), "status": "skip", "reason": "binning_failed"})
            continue
        available_bins = sorted(bins_cat.dropna().unique().tolist())
        if not available_bins:
            diagnostics.append({"trade_date": day.date(), "status": "skip", "reason": "binning_failed"})
            continue
        n_bins = len(available_bins)
        if max(bin_select) >= n_bins:
            raise ValueError(
                f"bin_select out of range: have {n_bins} bins, select {bin_select}"
            )
        effective_bins = bin_select
        picks = tradable[bins_cat.isin(effective_bins)]
        if len(picks) < min_count:
            diagnostics.append(
                {
                    "trade_date": day.date(),
                    "status": "skip",
                    "reason": "min_count_not_met",
                    "picked": int(len(picks)),
                    "bin_used": ",".join(str(x) for x in effective_bins),
                    "bin_count_actual": n_bins,
                }
            )
            continue
        buy_px = apply_twap_bps(picks[buy_twap_col], twap_bps, side="buy")
        sell_px = apply_twap_bps(picks[sell_twap_col], twap_bps, side="sell")
        returns = (sell_px - buy_px) / buy_px
        weight = min(1.0 / len(picks), max_weight)
        day_return = float((returns * weight).sum())
        total_weight = float(weight * len(picks))
        bench_buy = apply_twap_bps(tradable[buy_twap_col], twap_bps, side="buy")
        bench_sell = apply_twap_bps(tradable[sell_twap_col], twap_bps, side="sell")
        bench_returns = (bench_sell - bench_buy) / bench_buy
        bench_weight = 1.0 / len(tradable)
        bench_day_return = float((bench_returns * bench_weight).sum())
        daily_records.append(
            {
                "trade_date": day.date(),
                "count": int(len(picks)),
                "avg_return": float(returns.mean()),
                "day_return": day_return,
                "benchmark_count": int(len(tradable)),
                "benchmark_return": bench_day_return,
                "total_weight": total_weight,
            }
        )
        for idx, row in picks.iterrows():
            position_records.append(
                {
                    "trade_date": day.date(),
                    "code": row["code"],
                    "factor": float(row[factor_col]) if pd.notna(row[factor_col]) else None,
                    "weight": weight,
                    "buy_price": float(buy_px.loc[idx]),
                    "sell_price": float(sell_px.loc[idx]),
                    "return": float(returns.loc[idx]),
                }
            )
        result.days += 1
        diagnostics.append(
            {
                "trade_date": day.date(),
                "status": "ok",
                "reason": "",
                "bin_used": ",".join(str(x) for x in effective_bins),
                "bin_count_actual": n_bins,
            }
        )
    if daily_records:
        daily_df = pd.DataFrame(daily_records).sort_values("trade_date")
        nav = (1.0 + daily_df["day_return"]).cumprod()
        nav_df = pd.DataFrame({"trade_date": daily_df["trade_date"], "nav": nav})
        bench_nav = (1.0 + daily_df["benchmark_return"]).cumprod()
        bench_df = pd.DataFrame({"trade_date": daily_df["trade_date"], "nav": bench_nav})
        result.daily_returns = daily_df
        result.nav_curve = nav_df
        result.benchmark_curve = bench_df
        result.positions = pd.DataFrame(position_records)
    if diagnostics:
        result.diagnostics = pd.DataFrame(diagnostics).sort_values("trade_date")
    return result
