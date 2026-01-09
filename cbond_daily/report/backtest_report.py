from __future__ import annotations

from bisect import bisect_left
import re
from datetime import date, datetime, timedelta
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from cbond_daily.backtest.execution import apply_twap_bps
from cbond_daily.core.config import load_config_file, parse_date
from cbond_daily.core.naming import build_factor_col
from cbond_daily.data.io import read_dwd_daily, read_dws_factors_daily


def _latest_run_dir(base: Path) -> Path | None:
    if not base.exists():
        return None
    dirs = [path for path in base.iterdir() if path.is_dir()]
    if not dirs:
        return None
    ts_dirs = [p for p in dirs if re.match(r"^\d{8}_\d{6}$", p.name)]
    if ts_dirs:
        return max(ts_dirs, key=lambda p: p.name)
    try_dirs = [p for p in dirs if p.name.startswith("Try_")]
    if try_dirs:
        return max(try_dirs, key=lambda p: p.name)
    return max(dirs, key=lambda p: p.name)


def _calc_daily_return(df: pd.DataFrame, buy_col: str, sell_col: str, bps: float) -> pd.Series:
    buy_px = apply_twap_bps(df[buy_col], bps, side="buy")
    sell_px = apply_twap_bps(df[sell_col], bps, side="sell")
    return (sell_px - buy_px) / buy_px


def _rank_ic(x: pd.Series, y: pd.Series) -> float:
    x_rank = x.rank(method="average")
    y_rank = y.rank(method="average")
    return float(x_rank.corr(y_rank, method="pearson"))


def _zscore(series: pd.Series) -> pd.Series:
    mean = series.mean()
    std = series.std(ddof=0)
    if std == 0 or pd.isna(std):
        return pd.Series([pd.NA] * len(series), index=series.index)
    return (series - mean) / std


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


def _load_weights_history(out_dir: Path, cols: list[str]) -> tuple[list[date], dict[date, pd.Series]]:
    path = out_dir / "weights_history.csv"
    if not path.exists() or path.stat().st_size == 0:
        return [], {}
    try:
        df = pd.read_csv(path, parse_dates=["trade_date"])
    except pd.errors.EmptyDataError:
        return [], {}
    if df.empty or not {"trade_date", "factor", "weight"}.issubset(df.columns):
        return [], {}
    df["trade_date"] = pd.to_datetime(df["trade_date"]).dt.date
    weights_map: dict[date, pd.Series] = {}
    for day, group in df.groupby("trade_date"):
        series = pd.Series(group["weight"].astype(float).values, index=group["factor"])
        series = series.reindex(cols).fillna(0.0)
        weights_map[day] = series
    dates = sorted(weights_map)
    return dates, weights_map


def _weights_for_day(
    day: date,
    *,
    weight_dates: list[date],
    weights_map: dict[date, pd.Series],
    default_weights: pd.Series,
) -> pd.Series:
    if not weight_dates:
        return default_weights
    idx = bisect_left(weight_dates, day)
    if idx == 0:
        return weights_map[weight_dates[0]]
    if idx >= len(weight_dates):
        return weights_map[weight_dates[-1]]
    if weight_dates[idx] == day:
        return weights_map[weight_dates[idx]]
    return weights_map[weight_dates[idx - 1]]


def _calc_turnover(positions: pd.DataFrame) -> float:
    if positions.empty:
        return 0.0
    work = positions.copy()
    work["trade_date"] = pd.to_datetime(work["trade_date"]).dt.date
    turnover_vals: list[float] = []
    prev = pd.Series(dtype=float)
    for _, group in work.groupby("trade_date"):
        weights = group.set_index("code")["weight"].astype(float)
        all_codes = prev.index.union(weights.index)
        diff = (weights.reindex(all_codes).fillna(0.0) - prev.reindex(all_codes).fillna(0.0)).abs()
        turnover_vals.append(0.5 * float(diff.sum()))
        prev = weights
    if not turnover_vals:
        return 0.0
    return float(pd.Series(turnover_vals).mean())


def _calc_performance_metrics(daily_returns: pd.DataFrame, nav_curve: pd.DataFrame) -> dict:
    if daily_returns.empty or nav_curve.empty:
        return {"sharpe": 0.0, "maxdd": 0.0, "win_rate": 0.0}
    daily = daily_returns["day_return"].astype(float)
    mean_ret = float(daily.mean())
    vol = float(daily.std(ddof=0))
    sharpe = float((mean_ret / vol) * (252.0**0.5)) if vol else 0.0
    nav = nav_curve["nav"].astype(float)
    running_max = nav.cummax()
    drawdown = (nav / running_max) - 1.0
    maxdd = float(drawdown.min()) if not drawdown.empty else 0.0
    win_rate = float((daily > 0).mean())
    return {"sharpe": sharpe, "maxdd": maxdd, "win_rate": win_rate}


def _render_report(
    *,
    dwd_root: str,
    dws_root: str,
    out_dir: Path,
    start: date,
    end: date,
    factor_items: list[dict],
    buy_col: str,
    sell_col: str,
    bps: float,
    bins: int,
) -> None:
    ic_records: list[dict] = []
    bin_records: list[dict] = []
    bin_time_records: list[dict] = []

    factor_dates = _available_factor_dates(dws_root)
    cols = [item["col"] for item in factor_items]
    static_weights = pd.Series([item["w"] for item in factor_items], index=cols, dtype=float)
    weight_dates, weights_map = _load_weights_history(out_dir, cols)

    for day in pd.date_range(start, end, freq="D"):
        df = read_dwd_daily(dwd_root, day.date())
        if df.empty:
            continue
        factor_day = _prev_available_date(factor_dates, day.date())
        if factor_day is None:
            continue
        factors = read_dws_factors_daily(dws_root, factor_day)
        if factors.empty:
            continue
        if "code" not in factors.columns:
            raise KeyError("missing code column in factor data")
        missing_factor_cols = [c for c in cols if c not in factors.columns]
        if missing_factor_cols:
            continue
        merged = df.merge(factors[["code"] + cols], on="code", how="left")
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
        weights = _weights_for_day(
            day.date(),
            weight_dates=weight_dates,
            weights_map=weights_map,
            default_weights=static_weights,
        )
        work = tradable[cols].copy().apply(_zscore)
        valid_mask = work.notna()
        denom = valid_mask.mul(weights.abs(), axis=1).sum(axis=1)
        weighted = work.mul(weights, axis=1).sum(axis=1)
        composite = weighted.where(denom > 0).div(denom)
        if composite.isna().all():
            continue
        tradable = tradable.copy()
        tradable["composite_factor"] = composite
        returns = _calc_daily_return(tradable, buy_col, sell_col, bps)

        factors_vals = tradable["composite_factor"]
        ic = factors_vals.corr(returns, method="pearson")
        rank_ic = _rank_ic(factors_vals, returns)
        ic_records.append(
            {"trade_date": day.date(), "ic": ic, "rank_ic": rank_ic, "count": len(tradable)}
        )

        try:
            bins_cat = pd.qcut(factors_vals, bins, labels=False, duplicates="drop")
        except ValueError:
            continue
        bin_df = pd.DataFrame(
            {"bin": bins_cat, "factor": factors_vals.values, "return": returns.values}
        ).dropna()
        if bin_df.empty:
            continue
        grouped = bin_df.groupby("bin", dropna=True)
        for bin_id, group in grouped:
            bin_records.append(
                {
                    "bin": int(bin_id),
                    "factor_mean": float(group["factor"].mean()),
                    "return_mean": float(group["return"].mean()),
                    "count": int(len(group)),
                }
            )
            bin_time_records.append(
                {
                    "trade_date": day.date(),
                    "bin": int(bin_id),
                    "return_mean": float(group["return"].mean()),
                }
            )

    if not ic_records:
        return

    ic_df = pd.DataFrame(ic_records).sort_values("trade_date")
    ic_df["ic_cum"] = ic_df["ic"].cumsum()
    ic_df["rank_ic_cum"] = ic_df["rank_ic"].cumsum()
    ic_mean = float(ic_df["ic"].mean())
    rank_ic_mean = float(ic_df["rank_ic"].mean())
    ic_ir = float(ic_df["ic"].mean() / ic_df["ic"].std()) if ic_df["ic"].std() else 0.0
    rank_ic_ir = (
        float(ic_df["rank_ic"].mean() / ic_df["rank_ic"].std())
        if ic_df["rank_ic"].std()
        else 0.0
    )

    bin_time_df = pd.DataFrame(bin_time_records)
    if not bin_time_df.empty:
        pivot = bin_time_df.pivot_table(
            index="trade_date", columns="bin", values="return_mean", aggfunc="mean"
        ).sort_index()
        nav = (1.0 + pivot.fillna(0.0)).cumprod()
        nav_end = nav.tail(1).T.reset_index()
        nav_end.columns = ["bin", "nav_end"]
        nav_end["total_return"] = nav_end["nav_end"] - 1.0
        bin_stats = nav_end
    else:
        bin_stats = pd.DataFrame(columns=["bin", "nav_end", "total_return"])

    daily_returns_path = out_dir / "daily_returns.csv"
    if daily_returns_path.exists() and daily_returns_path.stat().st_size > 0:
        try:
            daily_returns_df = pd.read_csv(daily_returns_path, parse_dates=["trade_date"])
        except pd.errors.EmptyDataError:
            daily_returns_df = pd.DataFrame()
    else:
        daily_returns_df = pd.DataFrame()
    positions_path = out_dir / "positions.csv"
    if positions_path.exists() and positions_path.stat().st_size > 0:
        try:
            positions_df = pd.read_csv(positions_path)
        except pd.errors.EmptyDataError:
            positions_df = pd.DataFrame()
    else:
        positions_df = pd.DataFrame()
    nav_curve_path = out_dir / "nav_curve.csv"
    if nav_curve_path.exists() and nav_curve_path.stat().st_size > 0:
        try:
            nav_curve_df = pd.read_csv(nav_curve_path, parse_dates=["trade_date"])
        except pd.errors.EmptyDataError:
            nav_curve_df = pd.DataFrame()
    else:
        nav_curve_df = pd.DataFrame()

    out_dir.mkdir(parents=True, exist_ok=True)
    ic_df.to_csv(out_dir / "ic_series.csv", index=False)
    bin_stats.to_csv(out_dir / "factor_bins.csv", index=False)
    metrics = _calc_performance_metrics(daily_returns_df, nav_curve_df)
    metrics["turnover"] = _calc_turnover(positions_df)
    pd.DataFrame([metrics]).to_csv(out_dir / "factor_metrics.csv", index=False)

    fig, axes = plt.subplots(2, 3, figsize=(13.5, 7.5))
    ax = axes[0, 0]
    ax.plot(ic_df["trade_date"], ic_df["ic"], label="Mean Daily IC")
    ax.plot(ic_df["trade_date"], ic_df["ic_cum"], label="Accumulative IC")
    ax.plot(ic_df["trade_date"], ic_df["rank_ic"], label="Mean Daily Rank IC")
    ax.plot(ic_df["trade_date"], ic_df["rank_ic_cum"], label="Accumulative Rank IC")
    ax.set_title("Mean IC & Accumulative IC by Date")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    metric_labels = ["IC", "IR", "Rank_IC", "Rank_IR"]
    values = [ic_mean, ic_ir, rank_ic_mean, rank_ic_ir]
    ax.bar(metric_labels, values, color=["#3C8DBC", "#1F77B4", "#9467BD", "#FF7F0E"])
    for idx, val in enumerate(values):
        ax.text(idx, val, f"{val:.4f}", ha="center", va="bottom", fontsize=8)
    ax.set_title("IC & IR (mean)")
    ax.grid(True, axis="y", alpha=0.3)

    ax = axes[0, 2]
    if not bin_time_df.empty:
        colors = plt.cm.viridis(np.linspace(0, 1, len(nav.columns)))
        for color, col in zip(colors, nav.columns):
            ax.plot(
                nav.index,
                nav[col],
                linewidth=1.0,
                color=color,
                alpha=0.8,
                label=f"bin {col}",
            )
    ax.set_title("Bin cumulative NAV")
    ax.set_xlabel("Date")
    ax.set_ylabel("NAV")
    if not bin_time_df.empty:
        ax.legend(fontsize=7, ncol=2)
    ax.grid(True, alpha=0.3)

    ax = axes[1, 0]
    if not bin_stats.empty:
        ax.bar(bin_stats["bin"].astype(int), bin_stats["total_return"], color="#1F77B4")
    ax.set_title("Bin total return")
    ax.set_xlabel("Bin")
    ax.set_ylabel("Total Return")
    ax.grid(True, axis="y", alpha=0.3)

    ax = axes[1, 1]
    metric_names = ["sharpe", "maxdd", "win_rate", "turnover"]
    metric_vals = [metrics[m] for m in metric_names]
    ax.bar(metric_names, metric_vals, color=["#4C78A8", "#F58518", "#54A24B", "#B279A2"])
    for idx, val in enumerate(metric_vals):
        ax.text(idx, val, f"{val:.4f}", ha="center", va="bottom", fontsize=8)
    ax.set_title("Performance metrics")
    ax.grid(True, axis="y", alpha=0.3)

    ax = axes[1, 2]
    if not daily_returns_df.empty:
        daily_returns_df = daily_returns_df.sort_values("trade_date")
        factor_nav = (1.0 + daily_returns_df["day_return"]).cumprod()
        bench_nav = (1.0 + daily_returns_df["benchmark_return"]).cumprod()
        ax.plot(daily_returns_df["trade_date"], factor_nav, label="Factor")
        ax.plot(daily_returns_df["trade_date"], bench_nav, label="Benchmark")
    ax.set_title("Factor vs Benchmark NAV")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_dir / "factor_report.png", dpi=150)
    plt.close(fig)


def run_backtest_report() -> None:
    paths_cfg = load_config_file("paths")
    cfg = load_config_file("backtest")

    dwd_root = paths_cfg["dwd_root"]
    dws_root = paths_cfg["dws_root"]
    logs_root = paths_cfg.get("results")
    if not logs_root:
        raise KeyError("missing results in paths_config.json5")

    start = parse_date(cfg["start"])
    end = parse_date(cfg["end"])
    buy_col = cfg["buy_twap_col"]
    sell_col = cfg["sell_twap_col"]
    bps = float(cfg["twap_bps"])
    bins = int(cfg.get("ic_bins", 20))

    signals = cfg.get("signals", [])
    if not signals:
        raise ValueError("backtest_config.json5 missing signals")

    date_dir = f"{start:%Y-%m-%d}_{end:%Y-%m-%d}"
    batch_id = cfg.get("batch_id", "Backtest")
    base_dir = Path(logs_root) / date_dir / batch_id
    run_dir = _latest_run_dir(base_dir)
    if run_dir is None:
        raise FileNotFoundError(f"missing backtest output under {base_dir}")

    for signal in signals:
        signal_name = signal.get("name") or "signal"
        items = signal.get("items", [])
        if not items:
            continue
        factor_items = []
        for it in items:
            col = build_factor_col(it["name"], it.get("params"))
            factor_items.append({"col": col, "w": float(it.get("w", 0.0))})
        out_dir = run_dir / signal_name
        _render_report(
            dwd_root=dwd_root,
            dws_root=dws_root,
            out_dir=out_dir,
            start=start,
            end=end,
            factor_items=factor_items,
            buy_col=buy_col,
            sell_col=sell_col,
            bps=bps,
            bins=bins,
        )
    print(f"saved: {run_dir}")
