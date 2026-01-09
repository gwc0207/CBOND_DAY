from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from bisect import bisect_left
from pathlib import Path

import pandas as pd
import numpy as np

from cbond_daily.data.io import read_dwd_daily, read_dws_factors_daily
from .execution import apply_twap_bps
from .weight_fit import fit_linear_weights


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


def _available_dwd_dates(dwd_root: str) -> list[date]:
    base = Path(dwd_root)
    dates: list[date] = []
    for path in base.glob("*/*.parquet"):
        try:
            day = datetime.strptime(path.stem, "%Y%m%d").date()
        except ValueError:
            continue
        dates.append(day)
    dates.sort()
    return dates


def _select_bins_by_mean_return(
    *,
    dwd_root: str,
    dws_root: str,
    train_days: list[date],
    factor_items: list[dict],
    weights: pd.Series,
    buy_twap_col: str,
    sell_twap_col: str,
    twap_bps: float,
    bin_count: int,
) -> list[int]:
    factor_dates = _available_factor_dates(dws_root)
    cols = [item["col"] for item in factor_items]
    per_bin_returns: dict[int, list[float]] = {}
    for day in train_days:
        factor_day = _prev_available_date(factor_dates, day)
        if factor_day is None:
            continue
        df = read_dwd_daily(dwd_root, day)
        if df.empty:
            continue
        factors = read_dws_factors_daily(dws_root, factor_day)
        if factors.empty:
            continue
        if "code" not in factors.columns:
            continue
        if any(col not in factors.columns for col in cols):
            continue
        merged = df.merge(factors[["code"] + cols], on="code", how="left")
        missing_cols = [c for c in (buy_twap_col, sell_twap_col) if c not in merged.columns]
        if missing_cols:
            continue
        tradable = merged[
            merged[buy_twap_col].notna()
            & merged[sell_twap_col].notna()
            & (merged[buy_twap_col] > 0)
            & (merged[sell_twap_col] > 0)
        ]
        if tradable.empty:
            continue
        work = tradable[cols].copy().apply(_zscore)
        valid_mask = work.notna()
        denom = valid_mask.mul(weights.abs(), axis=1).sum(axis=1)
        weighted = work.mul(weights, axis=1).sum(axis=1)
        composite = weighted.where(denom > 0).div(denom)
        if composite.isna().all():
            continue
        tradable = tradable.copy()
        tradable["composite_factor"] = composite
        try:
            bins_cat = pd.qcut(
                tradable["composite_factor"],
                bin_count,
                labels=False,
                duplicates="drop",
            )
        except ValueError:
            continue
        if bins_cat.dropna().empty:
            continue
        buy_px = apply_twap_bps(tradable[buy_twap_col], twap_bps, side="buy")
        sell_px = apply_twap_bps(tradable[sell_twap_col], twap_bps, side="sell")
        returns = (sell_px - buy_px) / buy_px
        for bin_id in bins_cat.dropna().unique():
            mask = bins_cat == bin_id
            if mask.sum() == 0:
                continue
            per_bin_returns.setdefault(int(bin_id), []).append(float(returns[mask].mean()))
    if not per_bin_returns:
        return []
    mean_ret = {k: float(np.mean(v)) for k, v in per_bin_returns.items() if v}
    ranked = sorted(mean_ret.items(), key=lambda x: x[1], reverse=True)
    return [b for b, _ in ranked]

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
    total_days = len(pd.date_range(start, end, freq="D"))
    done = 0
    last_pct = -1
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
        done += 1
        pct = int(done * 100 / total_days) if total_days else 100
        if pct % 20 == 0 and pct != last_pct:
            last_pct = pct
            print(f"backtest progress: {pct}% ({done}/{total_days})")
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


def _zscore(series: pd.Series) -> pd.Series:
    mean = series.mean()
    std = series.std(ddof=0)
    if std == 0 or pd.isna(std):
        return pd.Series([pd.NA] * len(series), index=series.index)
    return (series - mean) / std


def run_backtest_linear(
    *,
    dwd_root: str,
    dws_root: str,
    start: date,
    end: date,
    factor_items: list[dict],
    buy_twap_col: str,
    sell_twap_col: str,
    min_count: int,
    max_weight: float,
    twap_bps: float,
    bin_count: int,
    bin_select: list[int],
    normalize: str = "zscore",
    weight_source: str = "manual",
    regression_cfg: dict | None = None,
    bin_source: str = "manual",
    bin_top_k: int = 2,
    weights_output_dir: Path | None = None,
) -> BacktestResult:
    result = BacktestResult()
    daily_records: list[dict] = []
    position_records: list[dict] = []
    diagnostics: list[dict] = []
    factor_dates = _available_factor_dates(dws_root)
    dwd_dates = _available_dwd_dates(dwd_root)

    cols = [item["col"] for item in factor_items]
    manual_weights = pd.Series([item["w"] for item in factor_items], index=cols, dtype=float)
    weights = manual_weights.copy()
    last_refit_idx: int | None = None

    total_days = len(pd.date_range(start, end, freq="D"))
    done = 0
    last_pct = -1
    for idx, day in enumerate(pd.date_range(start, end, freq="D")):
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
        missing_factor_cols = [c for c in cols if c not in factors.columns]
        if missing_factor_cols:
            diagnostics.append(
                {
                    "trade_date": day.date(),
                    "status": "skip",
                    "reason": "factor_col_missing",
                    "missing": ",".join(missing_factor_cols),
                }
            )
            continue
        merged = df.merge(factors[["code"] + cols], on="code", how="left")
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
        if weight_source == "regression":
            cfg = regression_cfg or {}
            lookback_days = int(cfg.get("lookback_days", 60))
            refit_freq = int(cfg.get("refit_freq", 1))
            need_refit = last_refit_idx is None or (idx - last_refit_idx) >= refit_freq
            if need_refit:
                train_days = [d for d in dwd_dates if d < day.date()][-lookback_days:]
                label_cfg = {
                    "buy_twap_col": buy_twap_col,
                    "sell_twap_col": sell_twap_col,
                    "twap_bps": twap_bps,
                    "fee_bps": float(cfg.get("fee_bps", 0.0)),
                    "include_cost_in_label": bool(cfg.get("include_cost_in_label", False)),
                }
                fit = fit_linear_weights(
                    train_days,
                    cols,
                    label_cfg,
                    cfg,
                    dwd_root=dwd_root,
                    dws_root=dws_root,
                    factor_dates={d: _prev_available_date(factor_dates, d) for d in train_days},
                )
                if fit is None:
                    fallback = cfg.get("fallback", "manual")
                    if fallback == "equal":
                        weights = pd.Series(1.0, index=cols) / len(cols)
                    else:
                        weights = manual_weights.copy()
                else:
                    weights = fit.weights.copy()
                max_w = float(cfg.get("max_weight", 3.0))
                weights = weights.clip(-max_w, max_w)
                if cfg.get("normalize", "l1") == "l1":
                    denom = weights.abs().sum()
                    if denom > 0:
                        weights = weights / denom
                if weights_output_dir is not None:
                    weights_output_dir.mkdir(parents=True, exist_ok=True)
                    out_path = weights_output_dir / "weights_history.csv"
                    out_df = pd.DataFrame(
                        {
                            "trade_date": [day.date()] * len(weights),
                            "factor": weights.index,
                            "weight": weights.values,
                        }
                    )
                    header = not out_path.exists()
                    out_df.to_csv(out_path, index=False, mode="a", header=header)
                last_refit_idx = idx
                if bin_source == "auto":
                    ranked_bins = _select_bins_by_mean_return(
                        dwd_root=dwd_root,
                        dws_root=dws_root,
                        train_days=train_days,
                        factor_items=factor_items,
                        weights=weights,
                        buy_twap_col=buy_twap_col,
                        sell_twap_col=sell_twap_col,
                        twap_bps=twap_bps,
                        bin_count=bin_count,
                    )
                    if len(ranked_bins) >= bin_top_k:
                        bin_select = ranked_bins[:bin_top_k]
        work = tradable[cols].copy()
        if normalize == "zscore":
            work = work.apply(_zscore)
        else:
            raise ValueError(f"unknown normalize: {normalize}")
        valid_mask = work.notna()
        denom = valid_mask.mul(weights.abs(), axis=1).sum(axis=1)
        weighted = work.mul(weights, axis=1).sum(axis=1)
        composite = weighted.where(denom > 0).div(denom)
        if composite.isna().all():
            diagnostics.append(
                {"trade_date": day.date(), "status": "skip", "reason": "composite_all_nan"}
            )
            continue
        tradable = tradable.copy()
        tradable["composite_factor"] = composite

        try:
            bins_cat = pd.qcut(
                tradable["composite_factor"],
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
        picks = tradable[bins_cat.isin(bin_select)]
        if len(picks) < min_count:
            diagnostics.append(
                {
                    "trade_date": day.date(),
                    "status": "skip",
                    "reason": "min_count_not_met",
                    "picked": int(len(picks)),
                    "bin_used": ",".join(str(x) for x in bin_select),
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
                    "factor": float(row["composite_factor"])
                    if pd.notna(row["composite_factor"])
                    else None,
                    "weight": weight,
                    "buy_price": float(buy_px.loc[idx]),
                    "sell_price": float(sell_px.loc[idx]),
                    "return": float(returns.loc[idx]),
                }
            )
        result.days += 1
        done += 1
        pct = int(done * 100 / total_days) if total_days else 100
        if pct % 20 == 0 and pct != last_pct:
            last_pct = pct
            print(f"backtest progress: {pct}% ({done}/{total_days})")
        diagnostics.append(
            {
                "trade_date": day.date(),
                "status": "ok",
                "reason": "",
                "bin_used": ",".join(str(x) for x in bin_select),
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
