from __future__ import annotations

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


def _latest_try_dir(base: Path) -> Path | None:
    if not base.exists():
        return None
    latest: Path | None = None
    max_idx = -1
    for path in base.iterdir():
        if not path.is_dir() or not path.name.startswith("Try_"):
            continue
        try:
            idx = int(path.name.split("_", 1)[1])
        except (IndexError, ValueError):
            continue
        if idx > max_idx:
            max_idx = idx
            latest = path
    return latest


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
    daily_stats: list[dict] = []
    zero_one_records: list[dict] = []

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

        daily_stats.append(
            {
                "trade_date": day.date(),
                "mean": float(factors_vals.mean()),
                "std": float(factors_vals.std()),
                "max": float(factors_vals.max()),
                "min": float(factors_vals.min()),
            }
        )
        zero_one_records.append(
            {
                "trade_date": day.date(),
                "zero_pct": float((factors_vals == 0).mean()),
                "one_pct": float((factors_vals == 1).mean()),
            }
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

    bin_df_all = pd.DataFrame(bin_records)
    if not bin_df_all.empty:
        bin_stats = (
            bin_df_all.groupby("bin")
            .agg(factor_mean=("factor_mean", "mean"), return_mean=("return_mean", "mean"))
            .reset_index()
        )
    else:
        bin_stats = pd.DataFrame(columns=["bin", "factor_mean", "return_mean"])

    daily_stats_df = pd.DataFrame(daily_stats).sort_values("trade_date")
    zero_one_df = pd.DataFrame(zero_one_records).sort_values("trade_date")

    daily_returns_path = out_dir / "daily_returns.csv"
    if daily_returns_path.exists():
        daily_returns_df = pd.read_csv(daily_returns_path, parse_dates=["trade_date"])
    else:
        daily_returns_df = pd.DataFrame()

    out_dir.mkdir(parents=True, exist_ok=True)
    ic_df.to_csv(out_dir / "ic_series.csv", index=False)
    bin_stats.to_csv(out_dir / "factor_bins.csv", index=False)
    daily_stats_df.to_csv(out_dir / "factor_daily_stats.csv", index=False)
    zero_one_df.to_csv(out_dir / "factor_zero_one.csv", index=False)

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
    metrics = ["IC", "IR", "Rank_IC", "Rank_IR"]
    values = [ic_mean, ic_ir, rank_ic_mean, rank_ic_ir]
    ax.bar(metrics, values, color=["#3C8DBC", "#1F77B4", "#9467BD", "#FF7F0E"])
    for idx, val in enumerate(values):
        ax.text(idx, val, f"{val:.4f}", ha="center", va="bottom", fontsize=8)
    ax.set_title("IC & IR (mean)")
    ax.grid(True, axis="y", alpha=0.3)

    ax = axes[0, 2]
    bin_time_df = pd.DataFrame(bin_time_records)
    if not bin_time_df.empty:
        pivot = bin_time_df.pivot_table(
            index="trade_date", columns="bin", values="return_mean", aggfunc="mean"
        ).sort_index()
        nav = (1.0 + pivot.fillna(0.0)).cumprod()
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
        ax.scatter(bin_stats["factor_mean"], bin_stats["return_mean"], color="#1F77B4", s=18)
    ax.set_title("factor_equal_cut (mean)")
    ax.set_xlabel("Factor Mean in bin")
    ax.set_ylabel("Return")
    ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    if not daily_stats_df.empty:
        ax.plot(daily_stats_df["trade_date"], daily_stats_df["mean"], label="mean")
        ax.plot(daily_stats_df["trade_date"], daily_stats_df["std"], label="std")
        ax.plot(daily_stats_df["trade_date"], daily_stats_df["max"], label="max")
        ax.plot(daily_stats_df["trade_date"], daily_stats_df["min"], label="min")
    ax.set_title("Factor daily stats")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

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
    logs_root = paths_cfg.get("logs")
    if not logs_root:
        raise KeyError("missing logs in paths_config.json")

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
    try_dir = _latest_try_dir(base_dir)
    if try_dir is None:
        raise FileNotFoundError(f"missing batch output under {base_dir}")

    for signal_id, col in signals:
        meta = factor_meta.get(col, {})
        factor_name = meta.get("name") or col
        params_str = meta.get("params_str") or "default"
        out_dir = try_dir / factor_name / params_str
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
    print(f"saved: {try_dir}")


if __name__ == "__main__":
    run_factor_report()
