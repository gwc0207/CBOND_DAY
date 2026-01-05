from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd

CONFIG_DIR = Path(__file__).resolve().parents[1] / "config"


def load_config_file(name: str) -> dict[str, Any]:
    filename = name if name.endswith("_config.json") else f"{name}_config.json"
    path = CONFIG_DIR / filename
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def parse_date(value: str | date) -> date:
    if isinstance(value, date):
        return value
    return pd.to_datetime(value).date()
