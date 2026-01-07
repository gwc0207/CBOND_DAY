from __future__ import annotations

import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cbond_daily.backtest.metrics import compute_metrics
from cbond_daily.backtest.runner import run_backtest_linear
from cbond_daily.report.backtest_report import run_backtest_report
from cbond_daily.core.config import load_config_file, parse_date
from cbond_daily.core.naming import build_factor_col


def _next_try_dir(base: Path) -> Path:
    if not base.exists():
        return base / "Try_1"
    max_idx = 0
    for path in base.iterdir():
        if not path.is_dir() or not path.name.startswith("Try_"):
            continue
        try:
            idx = int(path.name.split("_", 1)[1])
        except (IndexError, ValueError):
            continue
        max_idx = max(max_idx, idx)
    return base / f"Try_{max_idx + 1}"


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


def main() -> None:
    paths_cfg = load_config_file("paths")
    cfg = load_config_file("backtest")

    start = parse_date(cfg["start"])
    end = parse_date(cfg["end"])
    batch_id = cfg.get("batch_id", "Backtest")

    logs_root = Path(paths_cfg["logs"])
    date_dir = f"{start:%Y-%m-%d}_{end:%Y-%m-%d}"
    base_dir = logs_root / date_dir / batch_id
    try_dir = _next_try_dir(base_dir)
    try_dir.mkdir(parents=True, exist_ok=True)

    signals = cfg.get("signals", [])
    if not signals:
        raise ValueError("backtest_config.json missing signals")

    def _run_signal(item: dict) -> dict:
        signal_name = item.get("name") or "signal"
        items = item.get("items", [])
        if not items:
            raise ValueError("signal missing items")
        bin_select = item.get("bin_select")
        if not bin_select:
            raise ValueError("signal missing bin_select")
        normalize = item.get("normalize", "zscore")

        factor_items = []
        for it in items:
            col = build_factor_col(it["name"], it.get("params"))
            factor_items.append({"col": col, "w": float(it.get("w", 0.0))})

        out_dir = try_dir / signal_name
        result = run_backtest_linear(
            dwd_root=paths_cfg["dwd_root"],
            dws_root=paths_cfg["dws_root"],
            start=start,
            end=end,
            factor_items=factor_items,
            buy_twap_col=cfg["buy_twap_col"],
            sell_twap_col=cfg["sell_twap_col"],
            min_count=int(cfg["min_count"]),
            max_weight=float(cfg["max_weight"]),
            twap_bps=float(cfg["twap_bps"]) + float(cfg.get("fee_bps", 0.0)),
            bin_count=int(cfg.get("ic_bins", 20)),
            bin_select=[int(x) for x in bin_select],
            normalize=normalize,
            weight_source=cfg.get("weight_source", "manual"),
            regression_cfg=cfg.get("regression_cfg"),
            weights_output_dir=out_dir,
        )
        _write_result(out_dir, result)
        daily = result.daily_returns if result.daily_returns is not None else pd.DataFrame()
        nav = result.nav_curve if result.nav_curve is not None else pd.DataFrame()
        metrics = compute_metrics(daily, nav)
        metrics.update(
            {
                "signal_id": signal_name,
                "items": json.dumps(items, ensure_ascii=False),
                
            }
        )
        return metrics

    summary_rows = []
    workers = max(1, int(cfg.get("max_workers", 4)))
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(_run_signal, item) for item in signals]
        for fut in as_completed(futures):
            summary_rows.append(fut.result())

    summary_df = pd.DataFrame(summary_rows).sort_values("signal_id")
    summary_df.to_csv(try_dir / "summary.csv", index=False)
    run_backtest_report()


if __name__ == "__main__":
    main()
