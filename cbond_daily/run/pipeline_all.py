from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cbond_daily.run.sync_ods import main as sync_ods
from cbond_daily.run.build_dwd import main as build_dwd
from cbond_daily.run.build_factors import main as build_factors
from cbond_daily.run.backtest import main as backtest
from cbond_daily.report.factor_report import run_factor_report


def main(*, full: bool | None = None) -> None:
    sync_ods(full=full)
    build_dwd()
    build_factors()
    backtest()
    run_factor_report()


if __name__ == "__main__":
    main(full=None)
