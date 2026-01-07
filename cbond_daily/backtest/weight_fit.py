from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from cbond_daily.backtest.execution import apply_twap_bps
from cbond_daily.data.io import read_dwd_daily, read_dws_factors_daily


@dataclass
class FitResult:
    weights: pd.Series
    sample_count: int


def _zscore(series: pd.Series) -> pd.Series:
    mean = series.mean()
    std = series.std(ddof=0)
    if std == 0 or pd.isna(std):
        return pd.Series([pd.NA] * len(series), index=series.index)
    return (series - mean) / std


def _build_train_panel(
    train_days: list[date],
    *,
    dwd_root: str,
    dws_root: str,
    cols: list[str],
    buy_twap_col: str,
    sell_twap_col: str,
    twap_bps: float,
    fee_bps: float,
    include_cost: bool,
    factor_dates: dict[date, date],
) -> tuple[pd.DataFrame, pd.Series]:
    frames: list[pd.DataFrame] = []
    ys: list[pd.Series] = []
    for day in train_days:
        factor_day = factor_dates.get(day)
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
        y = (tradable[sell_twap_col] / tradable[buy_twap_col]) - 1.0
        if include_cost:
            cost_bps = twap_bps + fee_bps
            buy_px = apply_twap_bps(tradable[buy_twap_col], cost_bps, side="buy")
            sell_px = apply_twap_bps(tradable[sell_twap_col], cost_bps, side="sell")
            y = (sell_px - buy_px) / buy_px
        panel = work.copy()
        panel["y"] = y
        panel = panel.replace([np.inf, -np.inf], np.nan).dropna()
        if panel.empty:
            continue
        frames.append(panel[cols])
        ys.append(panel["y"])
    if not frames:
        return pd.DataFrame(), pd.Series(dtype=float)
    X = pd.concat(frames, ignore_index=True)
    y = pd.concat(ys, ignore_index=True)
    return X, y


def fit_linear_weights(
    train_days: list[date],
    cols: list[str],
    label_cfg: dict,
    method_cfg: dict,
    *,
    dwd_root: str,
    dws_root: str,
    factor_dates: dict[date, date],
) -> FitResult | None:
    X, y = _build_train_panel(
        train_days,
        dwd_root=dwd_root,
        dws_root=dws_root,
        cols=cols,
        buy_twap_col=label_cfg["buy_twap_col"],
        sell_twap_col=label_cfg["sell_twap_col"],
        twap_bps=float(label_cfg.get("twap_bps", 0.0)),
        fee_bps=float(label_cfg.get("fee_bps", 0.0)),
        include_cost=bool(label_cfg.get("include_cost_in_label", False)),
        factor_dates=factor_dates,
    )
    if X.empty or y.empty:
        return None
    min_samples = int(method_cfg.get("min_samples", 500))
    if len(X) < min_samples:
        return None
    method = method_cfg.get("method", "ridge")
    fit_intercept = bool(method_cfg.get("fit_intercept", True))
    X_mat = X.to_numpy()
    y_vec = y.to_numpy()
    if fit_intercept:
        X_mat = np.column_stack([np.ones(len(X_mat)), X_mat])
    if method == "ridge":
        alpha = float(method_cfg.get("ridge_alpha", 10.0))
        eye = np.eye(X_mat.shape[1])
        if fit_intercept:
            eye[0, 0] = 0.0
        beta = np.linalg.solve(X_mat.T @ X_mat + alpha * eye, X_mat.T @ y_vec)
    elif method == "ols":
        beta, *_ = np.linalg.lstsq(X_mat, y_vec, rcond=None)
    else:
        raise ValueError(f"unknown method: {method}")
    if fit_intercept:
        beta = beta[1:]
    weights = pd.Series(beta, index=cols, dtype=float)
    return FitResult(weights=weights, sample_count=int(len(X)))
