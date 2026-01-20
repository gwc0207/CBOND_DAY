from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date, datetime
from bisect import bisect_left, bisect_right
from pathlib import Path

import numpy as np
import pandas as pd

from cbond_daily.backtest.weight_fit import fit_linear_weights
from cbond_daily.core.naming import build_factor_col
from cbond_daily.data.io import read_dwd_daily, read_dws_factors_daily, read_trading_calendar


@dataclass
class ScoreResult:
    scores: pd.DataFrame
    weights_history: pd.DataFrame


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


def _align_start_end(trading_days: list[date], start: date, end: date) -> tuple[date, date]:
    if start not in trading_days:
        idx = bisect_right(trading_days, start)
        if idx >= len(trading_days):
            raise ValueError(f"start date {start} is after last trading day")
        start = trading_days[idx]
    if end not in trading_days:
        idx = bisect_left(trading_days, end) - 1
        if idx < 0:
            raise ValueError(f"end date {end} is before first trading day")
        end = trading_days[idx]
    if end < start:
        raise ValueError(f"end date {end} is before start date {start}")
    return start, end


def _zscore(series: pd.Series) -> pd.Series:
    mean = series.mean()
    std = series.std(ddof=0)
    if std == 0 or pd.isna(std):
        return pd.Series([pd.NA] * len(series), index=series.index)
    return (series - mean) / std


def _build_weights(
    *,
    train_days: list[date],
    cols: list[str],
    label_cfg: dict,
    regression_cfg: dict,
    dwd_root: str,
    dws_root: str,
    factor_dates: dict[date, date],
    manual_weights: pd.Series,
) -> pd.Series:
    fit = fit_linear_weights(
        train_days,
        cols,
        label_cfg,
        regression_cfg,
        dwd_root=dwd_root,
        dws_root=dws_root,
        factor_dates=factor_dates,
    )
    if fit is None:
        fallback = regression_cfg.get("fallback", "manual")
        if fallback == "equal":
            weights = pd.Series(1.0, index=cols) / len(cols)
        else:
            weights = manual_weights.copy()
    else:
        weights = fit.weights.copy()
    max_w = float(regression_cfg.get("max_weight", 3.0))
    weights = weights.clip(-max_w, max_w)
    if regression_cfg.get("normalize", "l1") == "l1":
        denom = weights.abs().sum()
        if denom > 0:
            weights = weights / denom
    return weights


def run_linear_score(
    *,
    ods_root: str,
    dwd_root: str,
    dws_root: str,
    start: date,
    end: date,
    factors: list[dict],
    label_cfg: dict,
    regression_cfg: dict,
    weight_source: str = "regression",
    normalize: str = "zscore",
) -> ScoreResult:
    factor_dates = _available_factor_dates(dws_root)
    trading_days = _load_trading_days(ods_root)
    start, end = _align_start_end(trading_days, start, end)
    day_list = [d for d in trading_days if start <= d <= end]

    cols: list[str] = []
    manual_w = []
    for item in factors:
        col = build_factor_col(item["name"], item.get("params"))
        cols.append(col)
        manual_w.append(float(item.get("w", 0.0)))
    manual_weights = pd.Series(manual_w, index=cols, dtype=float)

    scores_rows: list[dict] = []
    weight_rows: list[dict] = []
    last_refit_idx: int | None = None
    weights = manual_weights.copy()

    for idx, day_date in enumerate(day_list):
        df = read_dwd_daily(dwd_root, day_date)
        if df.empty:
            continue
        factor_day = _prev_available_date(factor_dates, day_date)
        if factor_day is None:
            continue
        factors_df = read_dws_factors_daily(dws_root, factor_day)
        if factors_df.empty:
            continue
        if "code" not in factors_df.columns:
            raise KeyError("missing code column in factor data")
        missing_factor_cols = [c for c in cols if c not in factors_df.columns]
        if missing_factor_cols:
            continue
        merged = df.merge(factors_df[["code"] + cols], on="code", how="left")
        buy_col = label_cfg["buy_twap_col"]
        sell_col = label_cfg["sell_twap_col"]
        missing_cols = [c for c in (buy_col, sell_col) if c not in merged.columns]
        if missing_cols:
            raise KeyError(f"missing price columns: {missing_cols}")
        tradable = merged[
            merged[buy_col].notna()
            & merged[sell_col].notna()
            & (merged[buy_col] > 0)
            & (merged[sell_col] > 0)
        ]
        if tradable.empty:
            continue

        if weight_source == "regression":
            lookback_days = int(regression_cfg.get("lookback_days", 60))
            refit_freq = int(regression_cfg.get("refit_freq", 1))
            need_refit = last_refit_idx is None or (idx - last_refit_idx) >= refit_freq
            if need_refit:
                train_days = [d for d in trading_days if d < day_date][-lookback_days:]
                factor_map = {d: _prev_available_date(factor_dates, d) for d in train_days}
                weights = _build_weights(
                    train_days=train_days,
                    cols=cols,
                    label_cfg=label_cfg,
                    regression_cfg=regression_cfg,
                    dwd_root=dwd_root,
                    dws_root=dws_root,
                    factor_dates=factor_map,
                    manual_weights=manual_weights,
                )
                for factor, weight in weights.items():
                    weight_rows.append(
                        {"trade_date": day_date, "factor": factor, "weight": float(weight)}
                    )
                last_refit_idx = idx

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
            continue

        for code, score in composite.items():
            scores_rows.append(
                {"trade_date": day_date, "code": code, "score": float(score)}
            )

    scores_df = pd.DataFrame(scores_rows)
    weights_df = pd.DataFrame(weight_rows)
    return ScoreResult(scores=scores_df, weights_history=weights_df)


def write_score_outputs(
    *,
    result: ScoreResult,
    score_path: Path,
    weights_path: Path | None,
    meta_path: Path | None,
    meta_payload: dict,
    overwrite: bool,
) -> None:
    score_path.parent.mkdir(parents=True, exist_ok=True)
    if overwrite and score_path.exists():
        score_path.unlink()
    if not result.scores.empty:
        result.scores.to_csv(score_path, index=False)

    if weights_path is not None:
        weights_path.parent.mkdir(parents=True, exist_ok=True)
        if overwrite and weights_path.exists():
            weights_path.unlink()
        if not result.weights_history.empty:
            result.weights_history.to_csv(weights_path, index=False)

    if meta_path is not None:
        meta_path.parent.mkdir(parents=True, exist_ok=True)
        with meta_path.open("w", encoding="utf-8") as handle:
            json.dump(meta_payload, handle, ensure_ascii=False, indent=2, default=str)
