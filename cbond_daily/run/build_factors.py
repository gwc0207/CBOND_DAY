from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cbond_daily.run.common import load_config_file, parse_date
from cbond_daily.factors.pipeline import run_factor_pipeline
from cbond_daily.factors import builtin  # noqa: F401


def main() -> None:
    paths_cfg = load_config_file("paths")
    factors_cfg = load_config_file("factors")
    bt_cfg = load_config_file("backtest")

    dwd_root = paths_cfg["dwd_root"]
    dws_root = paths_cfg["dws_root"]
    start = parse_date(bt_cfg["start"])
    end = parse_date(bt_cfg["end"])

    run_factor_pipeline(
        dwd_root,
        dws_root,
        start,
        end,
        factors_cfg.get("factors", []),
        update_only=factors_cfg.get("update_only"),
        overwrite=bool(factors_cfg.get("overwrite", False)),
    )


if __name__ == "__main__":
    main()
