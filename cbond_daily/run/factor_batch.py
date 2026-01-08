from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cbond_daily.factor_batch.runner import SignalSpec, run_factor_batch
from cbond_daily.report.factor_report import run_factor_report
from cbond_daily.factors.pipeline import run_factor_pipeline
from cbond_daily.core.config import load_config_file, parse_date
from cbond_daily.core.naming import build_factor_col


def main() -> None:
    paths_cfg = load_config_file("paths")
    exp_cfg = load_config_file("factor_batch")

    start = parse_date(exp_cfg["start"])
    end = parse_date(exp_cfg["end"])

    factors = exp_cfg.get("factors", [])
    factor_meta: dict[str, dict] = {}
    for item in factors:
        col = build_factor_col(item["name"], item.get("params"))
        factor_meta[col] = {
            "name": item["name"],
            "params": item.get("params", {}),
            "params_str": _params_to_str(item.get("params", {})),
        }

    signals_cfg = exp_cfg.get("signals", [])
    if signals_cfg:
        signals = []
        for item in signals_cfg:
            if "col" in item:
                col = item["col"]
            else:
                col = build_factor_col(item["name"], item.get("params"))
            bin_select = item.get("bin_select")
            if not bin_select:
                raise ValueError("each signal needs bin_select")
            max_weight = float(exp_cfg.get("max_weight", 0.05))
            signals.append(
                SignalSpec(
                    signal_id=item.get("signal_id") or col,
                    col=col,
                    bin_select=[int(x) for x in bin_select],
                    max_weight=max_weight,
                )
            )
    else:
        raise ValueError("factor_batch_config.json missing signals")
    if not signals:
        raise ValueError("factor_batch_config.json missing signals")

    if bool(exp_cfg.get("write_factors", True)):
        run_factor_pipeline(
            paths_cfg["dwd_root"],
            paths_cfg["dws_root"],
            start,
            end,
            factors,
            update_only=exp_cfg.get("update_only"),
            overwrite=bool(exp_cfg.get("overwrite", False)),
        )

    run_factor_batch(
        dwd_root=paths_cfg["dwd_root"],
        dws_root=paths_cfg["dws_root"],
        logs_root=paths_cfg["results"],
        start=start,
        end=end,
        buy_twap_col=exp_cfg["buy_twap_col"],
        sell_twap_col=exp_cfg["sell_twap_col"],
        twap_bps=float(exp_cfg["twap_bps"]),
        min_count=int(exp_cfg["min_count"]),
        signals=signals,
        factor_meta=factor_meta,
        batch_id=exp_cfg.get("batch_id", "Signal_Factor"),
        max_workers=int(exp_cfg.get("max_workers", 4)),
        bin_count=int(exp_cfg.get("ic_bins", 20)),
    )
    run_factor_report()


def _params_to_str(params: dict) -> str:
    if not params:
        return "default"
    parts = []
    for key in sorted(params):
        value = str(params[key]).replace(" ", "")
        parts.append(value.replace("/", "_").replace("\\", "_").replace(":", "_"))
    return "_".join(parts)


if __name__ == "__main__":
    main()
