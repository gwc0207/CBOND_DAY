from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cbond_daily.run.common import load_config_file, parse_date
from cbond_daily.backtest.runner import run_backtest


def main() -> None:
    paths_cfg = load_config_file("paths")
    bt_cfg = load_config_file("backtest")

    factor_col = bt_cfg.get("factor_col")
    if not factor_col:
        raise ValueError("backtest_config.json missing factor_col")

    result = run_backtest(
        paths_cfg["dwd_root"],
        parse_date(bt_cfg["start"]),
        parse_date(bt_cfg["end"]),
        factor_col=factor_col,
        buy_twap_col=bt_cfg["buy_twap_col"],
        sell_twap_col=bt_cfg["sell_twap_col"],
        target_count=int(bt_cfg["target_count"]),
        min_count=int(bt_cfg["min_count"]),
        max_weight=float(bt_cfg["max_weight"]),
        twap_bps=float(bt_cfg["twap_bps"]),
    )
    print(result)


if __name__ == "__main__":
    main()
