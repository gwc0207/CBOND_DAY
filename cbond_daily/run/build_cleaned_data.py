from __future__ import annotations

import sys
from datetime import timedelta
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cbond_daily.core.config import load_config_file, parse_date
from cbond_daily.data.dwd_builder import build_dwd_daily
from cbond_daily.data.io import get_latest_dwd_date


def main() -> None:
    paths_cfg = load_config_file("paths")
    cleaned_cfg = load_config_file("cleaned_data")
    bt_cfg = load_config_file("factor_batch")

    ods_root = paths_cfg["ods_root"]
    dwd_root = paths_cfg["dwd_root"]
    start = parse_date(bt_cfg["start"])
    end = parse_date(bt_cfg["end"])
    full_refresh = bool(cleaned_cfg.get("full_refresh", False))
    if not full_refresh:
        last_date = get_latest_dwd_date(dwd_root)
        if last_date is not None:
            start = max(start, last_date + timedelta(days=1))
        if start > end:
            return

    build_dwd_daily(
        ods_root,
        dwd_root,
        start,
        end,
        primary_table=cleaned_cfg["primary_table"],
        merge_tables=cleaned_cfg["merge_tables"],
    )


if __name__ == "__main__":
    main()
