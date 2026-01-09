from __future__ import annotations

import re
import sys
from datetime import timedelta
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cbond_daily.backtest.execution import apply_twap_bps
from cbond_daily.data.io import read_dwd_daily, read_dws_factors_daily
from cbond_daily.core.config import load_config_file, parse_date
from cbond_daily.core.naming import build_factor_col


def _calc_daily_return(df: pd.DataFrame, buy_col: str, sell_col: str, bps: float) -> pd.Series:
    buy_px = apply_twap_bps(df[buy_col], bps, side="buy")
    sell_px = apply_twap_bps(df[sell_col], bps, side="sell")
    return (sell_px - buy_px) / buy_px


def _rank_ic(x: pd.Series, y: pd.Series) -> float:
    x_rank = x.rank(method="average")
    y_rank = y.rank(method="average")
    return float(x_rank.corr(y_rank, method="pearson"))


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


def _params_to_str(params: dict) -> str:
    if not params:
        return "default"
    parts = []
    for key in sorted(params):
        value = str(params[key]).replace(" ", "")
        parts.append(value.replace("/", "_").replace("\\", "_").replace(":", "_"))
    return "_".join(parts)


def _render_report(
    *,
    dwd_root: str,
    dws_root: str,
    out_dir: Path,
    start: date,
    end: date,
    factor_col: str,
    buy_col: str,
    sell_col: str,
    bps: float,
    bins: int,
) -> None:
    ic_records: list[dict] = []
    bin_records: list[dict] = []
    bin_time_records: list[dict] = []

    for day in pd.date_range(start, end, freq="D"):
        factor_day = (day - timedelta(days=1)).date()
        df = read_dwd_daily(dwd_root, day.date())
        if df.empty:
            continue
        factors = read_dws_factors_daily(dws_root, factor_day)
        if factors.empty or factor_col not in factors.columns:
            continue
        if "code" not in factors.columns:
            raise KeyError("missing code column in factor data")
        merged = df.merge(factors[["code", factor_col]], on="code", how="left")
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
        tradable = tradable.dropna(subset=[factor_col])
        if tradable.empty:
            continue
        returns = _calc_daily_return(tradable, buy_col, sell_col, bps)
        factors_vals = tradable[factor_col]

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


def run_factor_report() -> None:
    paths_cfg = load_config_file("paths")
    cfg = load_config_file("factor_batch")

    dwd_root = paths_cfg["dwd_root"]
    dws_root = paths_cfg["dws_root"]
    logs_root = paths_cfg.get("results")
    if not logs_root:
        raise KeyError("missing results in paths_config.json")

    start = parse_date(cfg["start"])
    end = parse_date(cfg["end"])
    buy_col = cfg["buy_twap_col"]
    sell_col = cfg["sell_twap_col"]
    bps = float(cfg["twap_bps"])
    bins = int(cfg.get("ic_bins", 20))

    factors = cfg.get("factors", [])
    factor_meta: dict[str, dict] = {}
    for item in factors:
        col = build_factor_col(item["name"], item.get("params"))
        factor_meta[col] = {
            "name": item["name"],
            "params": item.get("params", {}),
            "params_str": _params_to_str(item.get("params", {})),
        }

    signals_cfg = cfg.get("signals", [])
    if signals_cfg:
        signals = []
        for item in signals_cfg:
            if "col" in item:
                col = item["col"]
            else:
                col = build_factor_col(item["name"], item.get("params"))
            signals.append((item.get("signal_id") or col, col))
    else:
        signals = [(col, col) for col in factor_meta]

    date_dir = f"{start:%Y-%m-%d}_{end:%Y-%m-%d}"
    batch_id = cfg.get("batch_id", "Signal_Factor")
    base_dir = Path(logs_root) / date_dir / batch_id
    run_dir = _latest_run_dir(base_dir)
    if run_dir is None:
        raise FileNotFoundError(f"missing batch output under {base_dir}")

    for signal_id, col in signals:
        meta = factor_meta.get(col, {})
        factor_name = meta.get("name") or col
        params_str = meta.get("params_str") or "default"
        out_dir = run_dir / factor_name / params_str
        _render_report(
            dwd_root=dwd_root,
            dws_root=dws_root,
            out_dir=out_dir,
            start=start,
            end=end,
            factor_col=col,
            buy_col=buy_col,
            sell_col=sell_col,
            bps=bps,
            bins=bins,
        )
    print(f"saved: {run_dir}")


