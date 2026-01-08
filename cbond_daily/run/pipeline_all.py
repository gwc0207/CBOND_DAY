from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cbond_daily.run.sync_raw_data import main as sync_raw_data
from cbond_daily.run.build_cleaned_data import main as build_cleaned_data
from cbond_daily.run.build_factors import main as build_factors
from cbond_daily.run.factor_batch import main as factor_batch


def main(*, full: bool | None = None) -> None:
    sync_raw_data(full=full)
    build_cleaned_data()
    build_factors()
    factor_batch()


if __name__ == "__main__":
    main(full=None)
