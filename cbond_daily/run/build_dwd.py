from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cbond_daily.run.common import load_config_file, parse_date
from cbond_daily.data.dwd_builder import build_dwd_daily


def main() -> None:
    paths_cfg = load_config_file("paths")
    dwd_cfg = load_config_file("dwd")
    bt_cfg = load_config_file("backtest")

    ods_root = paths_cfg["ods_root"]
    dwd_root = paths_cfg["dwd_root"]
    start = parse_date(bt_cfg["start"])
    end = parse_date(bt_cfg["end"])

    build_dwd_daily(
        ods_root,
        dwd_root,
        start,
        end,
        primary_table=dwd_cfg["primary_table"],
        merge_tables=dwd_cfg["merge_tables"],
    )


if __name__ == "__main__":
    main()
