from __future__ import annotations

import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Iterable

import pandas as pd

from cbond_daily.backtest.metrics import compute_metrics
from cbond_daily.backtest.runner import run_backtest


@dataclass
class SignalSpec:
    signal_id: str
    col: str
    bin_select: list[int] | None
    max_weight: float
    bin_source: str = "manual"
    bin_top_k: int = 2
    bin_lookback_days: int = 60


def _safe_part(value: str) -> str:
    return value.replace(" ", "").replace("/", "_").replace("\\", "_").replace(":", "_")


def _params_to_str(params: dict) -> str:
    if not params:
        return "default"
    parts = []
    for key in sorted(params):
        parts.append(_safe_part(str(params[key])))
    return "_".join(parts)


def _next_run_dir(base: Path) -> Path:
    ts = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    return base / ts


def _write_result(out_dir: Path, result) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    daily = result.daily_returns if result.daily_returns is not None else pd.DataFrame()
    nav = result.nav_curve if result.nav_curve is not None else pd.DataFrame()
    positions = result.positions if result.positions is not None else pd.DataFrame()
    diagnostics = result.diagnostics if result.diagnostics is not None else pd.DataFrame()
    daily.to_csv(out_dir / "daily_returns.csv", index=False)
    nav.to_csv(out_dir / "nav_curve.csv", index=False)
    positions.to_csv(out_dir / "positions.csv", index=False)
    diagnostics.to_csv(out_dir / "diagnostics.csv", index=False)


def _plot_nav_compare(nav_map: dict[str, pd.DataFrame], out_path: Path) -> None:
    import matplotlib.pyplot as plt

    if not nav_map:
        return
    fig, ax = plt.subplots(figsize=(10, 4.2))
    has_label = False
    for signal_id, nav_df in nav_map.items():
        if nav_df is None or nav_df.empty:
            continue
        if "trade_date" not in nav_df.columns or "nav" not in nav_df.columns:
            continue
        ax.plot(nav_df["trade_date"], nav_df["nav"], linewidth=1.2, label=signal_id)
        has_label = True
    ax.set_title("Signal NAV Compare")
    ax.set_xlabel("Date")
    ax.set_ylabel("NAV")
    ax.grid(True, alpha=0.3)
    if has_label:
        ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def run_factor_batch(
    *,
    ods_root: str,
    dwd_root: str,
    dws_root: str,
    logs_root: str,
    start: date,
    end: date,
    buy_twap_col: str,
    sell_twap_col: str,
    twap_bps: float,
    min_count: int,
    signals: Iterable[SignalSpec],
    factor_meta: dict[str, dict],
    batch_id: str = "Signal_Factor",
    max_workers: int = 4,
    bin_count: int | None = None,
) -> Path:
    date_dir = f"{start:%Y-%m-%d}_{end:%Y-%m-%d}"
    base_dir = Path(logs_root) / date_dir / batch_id
    try_dir = _next_run_dir(base_dir)
    try_dir.mkdir(parents=True, exist_ok=True)

    summary_rows: list[dict] = []
    nav_map: dict[str, pd.DataFrame] = {}

    def _run_signal(spec: SignalSpec) -> dict:
        meta = factor_meta.get(spec.col, {})
        factor_name = meta.get("name") or spec.col
        params_str = meta.get("params_str") or "default"
        signal_dir = try_dir / factor_name / params_str

        result = run_backtest(
            ods_root,
            dwd_root,
            dws_root,
            start,
            end,
            factor_col=spec.col,
            buy_twap_col=buy_twap_col,
            sell_twap_col=sell_twap_col,
            min_count=min_count,
            max_weight=spec.max_weight,
            twap_bps=twap_bps,
            bin_count=bin_count,
            bin_select=spec.bin_select,
            bin_source=spec.bin_source,
            bin_top_k=spec.bin_top_k,
            bin_lookback_days=spec.bin_lookback_days,
        )
        _write_result(signal_dir, result)
        daily = result.daily_returns if result.daily_returns is not None else pd.DataFrame()
        nav = result.nav_curve if result.nav_curve is not None else pd.DataFrame()
        metrics = compute_metrics(daily, nav)
        metrics.update(
            {
                "signal_id": spec.signal_id,
                "factor_col": spec.col,
                "factor_name": factor_name,
                "params": json.dumps(meta.get("params", {}), ensure_ascii=False),
            }
        )
        return {"metrics": metrics, "nav": nav, "signal_id": spec.signal_id}

    workers = max(1, int(max_workers))
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(_run_signal, spec) for spec in signals]
        for fut in as_completed(futures):
            out = fut.result()
            summary_rows.append(out["metrics"])
            nav_map[out["signal_id"]] = out["nav"]

    summary_df = pd.DataFrame(summary_rows).sort_values("signal_id")
    summary_df.to_csv(try_dir / "summary.csv", index=False)
    _plot_nav_compare(nav_map, try_dir / "nav_compare.png")
    return try_dir
