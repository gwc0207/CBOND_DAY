from __future__ import annotations

import sys
from datetime import timedelta
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cbond_daily.core.config import load_config_file, parse_date
from cbond_daily.data.extract import DATE_COLUMNS, fetch_table
from cbond_daily.data.io import get_latest_table_date, table_has_data, write_table_by_date


def main(*, full: bool | None = None) -> None:
    paths_cfg = load_config_file("paths")
    raw_cfg = load_config_file("raw_data")
    bt_cfg = load_config_file("factor_batch")
    if full is None:
        full = bool(raw_cfg.get("full_refresh", False))

    ods_root = paths_cfg["ods_root"]
    start = parse_date(bt_cfg["start"])
    end = parse_date(bt_cfg["end"])

    for table in raw_cfg.get("sync_tables", []):
        last_date = None if full else get_latest_table_date(ods_root, table)
        date_based = table in DATE_COLUMNS
        if date_based:
            fetch_start = max(start, last_date + timedelta(days=1)) if last_date else start
            if fetch_start > end:
                continue
            df = fetch_table(table, start=str(fetch_start), end=str(end))
        else:
            if not full and table_has_data(ods_root, table):
                continue
            df = fetch_table(table)
        if df.empty:
            continue
        write_table_by_date(df, ods_root, table, date_col="trade_date")
        print(f"synced {table}: {len(df)}")


if __name__ == "__main__":
    main(full=None)
