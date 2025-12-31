from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cbond_daily.run.common import load_config_file, parse_date
from cbond_daily.data.extract import fetch_table
from cbond_daily.data.io import write_table_by_date


def main() -> None:
    paths_cfg = load_config_file("paths")
    ods_cfg = load_config_file("ods")
    bt_cfg = load_config_file("backtest")

    ods_root = paths_cfg["ods_root"]
    start = parse_date(bt_cfg["start"])
    end = parse_date(bt_cfg["end"])

    for table in ods_cfg.get("sync_tables", []):
        df = fetch_table(table, start=str(start), end=str(end))
        if df.empty:
            continue
        write_table_by_date(df, ods_root, table, date_col="trade_date")
        print(f"synced {table}: {len(df)}")


if __name__ == "__main__":
    main()
